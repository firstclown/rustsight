//! sqlite-vec KNN index for semantic search (RS-7).
//!
//! Every embedded item (memory, observation, opinion) is mirrored into a
//! `vec0` virtual table — one per embedding dimension (`vec_f32_<dim>`),
//! partitioned by bank, with the item kind as a filterable metadata column
//! and its id as an auxiliary column. `vec_rows` maps item ids to the vec0
//! rowid so updates and deletes are point operations.
//!
//! The base tables stay the source of truth: the index is write-through
//! (maintained in the same transaction as the row it mirrors) and a bank
//! only answers semantic queries from it once `vec_indexed_banks` lists
//! it — set when the bank is created by an index-maintaining store, or by
//! [`crate::SqliteStore::reindex_vectors`]. A store that cannot maintain
//! the index (feature off, extension failed to load) clears that table on
//! open, so a later indexed open never trusts a stale mirror.
//!
//! sqlite-vec 0.1.x is an exact (SIMD flat-scan) KNN, not an approximate
//! one: results match the brute-force scan except for float rounding and
//! the order of exact ties.

use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use rustsight_core::types::{MemoryId, MemoryKind};

/// Largest `k` a vec0 KNN query accepts (`SQLITE_VEC_VEC0_K_MAX`).
pub(crate) const MAX_K: usize = 4096;
/// Widest vector a vec0 column accepts (`SQLITE_VEC_VEC0_MAX_DIMENSIONS`).
pub(crate) const MAX_DIM: usize = 8192;

/// Plain bookkeeping tables, created whether or not the extension is
/// available (a non-indexing store must still be able to clear the markers).
pub(crate) const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS vec_rows(
  rowid INTEGER PRIMARY KEY,
  ref_id TEXT NOT NULL UNIQUE,
  bank_id TEXT NOT NULL,
  dim INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vec_rows_bank ON vec_rows(bank_id);
CREATE TABLE IF NOT EXISTS vec_indexed_banks(
  bank_id TEXT PRIMARY KEY
);
"#;

/// Initialize sqlite-vec on this connection; returns its version, or why it
/// is unavailable.
#[cfg(feature = "sqlite-vec")]
pub(crate) fn load(conn: &Connection) -> Result<String, String> {
    use rusqlite::ffi;
    use std::ffi::{c_char, c_int, c_void, CStr};
    type InitFn = unsafe extern "C" fn(*mut ffi::sqlite3, *mut *mut c_char, *const c_void) -> c_int;
    // SAFETY: `sqlite3_vec_init` is the extension entry point with the
    // standard `sqlite3_extension_init` signature; the crate declares it
    // argument-less only as an opaque symbol. Compiled with SQLITE_CORE, it
    // ignores the API-routines pointer and registers on the given handle,
    // which stays valid for the duration of the call.
    unsafe {
        let init: InitFn = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = init(conn.handle(), &mut err, std::ptr::null());
        if rc != ffi::SQLITE_OK {
            let msg = if err.is_null() {
                format!("sqlite3_vec_init returned {rc}")
            } else {
                let m = CStr::from_ptr(err).to_string_lossy().into_owned();
                ffi::sqlite3_free(err.cast());
                m
            };
            return Err(msg);
        }
    }
    conn.query_row("SELECT vec_version()", [], |r| r.get(0))
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "sqlite-vec"))]
pub(crate) fn load(_conn: &Connection) -> Result<String, String> {
    Err("built without the `sqlite-vec` feature".into())
}

fn table(dim: usize) -> String {
    format!("vec_f32_{dim}")
}

fn table_exists(conn: &Connection, dim: usize) -> rusqlite::Result<bool> {
    conn.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1")?
        .exists(params![table(dim)])
}

fn ensure_table(conn: &Connection, dim: usize) -> rusqlite::Result<()> {
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING vec0(
           bank_id text partition key,
           kind text,
           embedding float[{dim}] distance_metric=cosine,
           +ref_id text
         );",
        table(dim)
    ))
}

fn blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Drop one item from the index (no-op when it is not indexed).
pub(crate) fn remove(conn: &Connection, ref_id: &str) -> rusqlite::Result<()> {
    let row: Option<(i64, usize)> = conn
        .query_row(
            "SELECT rowid, dim FROM vec_rows WHERE ref_id = ?1",
            params![ref_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    if let Some((rowid, dim)) = row {
        if table_exists(conn, dim)? {
            conn.execute(
                &format!("DELETE FROM {} WHERE rowid = ?1", table(dim)),
                params![rowid],
            )?;
        }
        conn.execute("DELETE FROM vec_rows WHERE rowid = ?1", params![rowid])?;
    }
    Ok(())
}

/// Drop many items from the index.
pub(crate) fn remove_many<'a>(
    conn: &Connection,
    ref_ids: impl IntoIterator<Item = &'a String>,
) -> rusqlite::Result<()> {
    for id in ref_ids {
        remove(conn, id)?;
    }
    Ok(())
}

/// Mirror an item's current embedding: replaces whatever was indexed for
/// it, and leaves it unindexed when the embedding is absent or empty.
/// Vectors wider than vec0 accepts cannot be indexed, so their bank falls
/// back to brute force (its marker is cleared) instead of failing the write.
pub(crate) fn put(
    conn: &Connection,
    bank_id: &str,
    kind: MemoryKind,
    ref_id: &str,
    embedding: Option<&[f32]>,
) -> rusqlite::Result<()> {
    remove(conn, ref_id)?;
    let Some(v) = embedding.filter(|v| !v.is_empty()) else {
        return Ok(());
    };
    if v.len() > MAX_DIM {
        unmark(conn, bank_id)?;
        return Ok(());
    }
    ensure_table(conn, v.len())?;
    conn.execute(
        "INSERT INTO vec_rows(ref_id, bank_id, dim) VALUES (?1, ?2, ?3)",
        params![ref_id, bank_id, v.len()],
    )?;
    let rowid = conn.last_insert_rowid();
    conn.execute(
        &format!(
            "INSERT INTO {}(rowid, bank_id, kind, embedding, ref_id) VALUES (?1, ?2, ?3, ?4, ?5)",
            table(v.len())
        ),
        params![rowid, bank_id, kind.as_str(), blob(v), ref_id],
    )?;
    Ok(())
}

/// Every vec0 table in the database (one per dimension ever indexed).
fn vec_tables(conn: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'vec\\_f32\\_%' ESCAPE '\\'
           AND sql LIKE 'CREATE VIRTUAL TABLE%'",
    )?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
    rows.collect()
}

/// Drop every indexed item of a bank, and its marker. Deletes by partition
/// key rather than through `vec_rows`, so rows orphaned by a store that ran
/// without the extension go too.
pub(crate) fn remove_bank(conn: &Connection, bank_id: &str) -> rusqlite::Result<()> {
    for t in vec_tables(conn)? {
        conn.execute(
            &format!("DELETE FROM {t} WHERE bank_id = ?1"),
            params![bank_id],
        )?;
    }
    conn.execute("DELETE FROM vec_rows WHERE bank_id = ?1", params![bank_id])?;
    unmark(conn, bank_id)
}

pub(crate) fn is_indexed(conn: &Connection, bank_id: &str) -> rusqlite::Result<bool> {
    conn.prepare("SELECT 1 FROM vec_indexed_banks WHERE bank_id = ?1")?
        .exists(params![bank_id])
}

pub(crate) fn mark(conn: &Connection, bank_id: &str) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO vec_indexed_banks(bank_id) VALUES (?1)",
        params![bank_id],
    )
    .map(|_| ())
}

pub(crate) fn unmark(conn: &Connection, bank_id: &str) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM vec_indexed_banks WHERE bank_id = ?1",
        params![bank_id],
    )
    .map(|_| ())
}

/// Forget every marker: called by a store that cannot maintain the index,
/// since its writes will not be mirrored.
pub(crate) fn invalidate_all(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM vec_indexed_banks", [])
        .map(|_| ())
}

/// The `k` nearest items of `kinds` in a bank, nearest first, as
/// `(id, kind, cosine similarity)`. Empty when nothing of this dimension
/// was ever indexed.
pub(crate) fn knn(
    conn: &Connection,
    bank_id: &str,
    query: &[f32],
    kinds: &[MemoryKind],
    k: usize,
) -> rusqlite::Result<Vec<(MemoryId, MemoryKind, f32)>> {
    if query.is_empty() || query.len() > MAX_DIM || k == 0 || !table_exists(conn, query.len())? {
        return Ok(vec![]);
    }
    let mut kinds: Vec<&str> = kinds.iter().map(|k| k.as_str()).collect();
    kinds.sort_unstable();
    kinds.dedup();
    if kinds.is_empty() {
        return Ok(vec![]);
    }
    let kind_filter = if kinds.len() >= 4 {
        String::new()
    } else {
        format!(" AND kind IN ({})", vec!["?"; kinds.len()].join(","))
    };
    let sql = format!(
        "SELECT ref_id, kind, distance FROM {} WHERE embedding MATCH ? AND k = ? AND bank_id = ?{kind_filter} ORDER BY distance",
        table(query.len())
    );
    let qblob = blob(query);
    let k = k.min(MAX_K) as i64;
    let mut args: Vec<rusqlite::types::Value> = vec![
        rusqlite::types::Value::Blob(qblob),
        rusqlite::types::Value::Integer(k),
        rusqlite::types::Value::Text(bank_id.to_string()),
    ];
    if !kind_filter.is_empty() {
        args.extend(
            kinds
                .iter()
                .map(|s| rusqlite::types::Value::Text(s.to_string())),
        );
    }
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(args), |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, f64>(2)?,
        ))
    })?;
    let mut out = vec![];
    for row in rows {
        let (id, kind, distance) = row?;
        out.push((
            id,
            MemoryKind::parse(&kind).unwrap_or(MemoryKind::World),
            (1.0 - distance) as f32,
        ));
    }
    Ok(out)
}
