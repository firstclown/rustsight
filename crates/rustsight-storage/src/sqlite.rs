use crate::vector;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, Row};
use rustsight_core::lang::cjk_segment;
use rustsight_core::types::*;
use rustsight_core::{
    brute_force_semantic, sort_hits, Admit, Clock, Error, MemoryStore, Result, SemanticHit,
    SystemClock,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

fn store_err(e: rusqlite::Error) -> Error {
    Error::Store(e.to_string())
}

pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    clock: Arc<dyn Clock>,
    /// sqlite-vec version when the KNN index is live on this connection
    /// (RS-7); `None` means every semantic query is a brute-force scan.
    vec_version: Option<String>,
    /// Banks already warned about answering by brute force while the index
    /// is live (pre-index banks awaiting `rustsight reindex`).
    warned_unindexed: Mutex<HashSet<String>>,
}

/// How a [`SqliteStore`] opens its database.
#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// Maintain and query the sqlite-vec KNN index (default true). Off, or
    /// when the extension cannot initialize, semantic search scans every
    /// embedding and the index markers are cleared so a later indexed open
    /// does not trust a mirror this store's writes skipped.
    pub vector_index: bool,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self { vector_index: true }
    }
}

/// Outcome of [`SqliteStore::reindex_vectors`] for one bank.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ReindexReport {
    pub bank_id: String,
    /// Embedded items written to the index.
    pub indexed: usize,
    /// Items whose embedding is too wide for vec0; when non-zero the bank
    /// stays on brute force.
    pub skipped: usize,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS banks(
  id TEXT PRIMARY KEY,
  mission TEXT, directives TEXT, background TEXT,
  skepticism INTEGER NOT NULL DEFAULT 3,
  literalism INTEGER NOT NULL DEFAULT 3,
  empathy INTEGER NOT NULL DEFAULT 3,
  bias_strength REAL NOT NULL DEFAULT 0.5,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memories(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  confidence REAL,
  occurred_start_ms INTEGER,
  occurred_end_ms INTEGER,
  mentioned_at_ms INTEGER NOT NULL,
  embedding BLOB,
  source TEXT,
  context TEXT,
  created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_bank_time ON memories(bank_id, mentioned_at_ms);
CREATE INDEX IF NOT EXISTS idx_mem_occurred ON memories(bank_id, occurred_start_ms, occurred_end_ms);
CREATE TABLE IF NOT EXISTS entities(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  name TEXT NOT NULL,
  etype TEXT NOT NULL,
  last_mentioned_ms INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ent_bank ON entities(bank_id);
CREATE TABLE IF NOT EXISTS entity_aliases(
  bank_id TEXT NOT NULL,
  alias TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  PRIMARY KEY(bank_id, alias)
);
CREATE TABLE IF NOT EXISTS memory_entities(
  memory_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  mention TEXT NOT NULL,
  PRIMARY KEY(memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);
CREATE TABLE IF NOT EXISTS edges(
  bank_id TEXT NOT NULL,
  a_id TEXT NOT NULL,
  b_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  label TEXT,
  weight REAL NOT NULL,
  PRIMARY KEY(a_id, b_id, kind, label)
);
CREATE INDEX IF NOT EXISTS idx_edge_a ON edges(a_id);
CREATE INDEX IF NOT EXISTS idx_edge_b ON edges(b_id);
CREATE VIRTUAL TABLE IF NOT EXISTS texts_fts USING fts5(
  text, bank_id UNINDEXED, kind UNINDEXED, ref_id UNINDEXED,
  tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS dirty_entities(
  bank_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  marked_at_ms INTEGER NOT NULL,
  PRIMARY KEY(bank_id, entity_id)
);
CREATE TABLE IF NOT EXISTS observations(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  text TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  proof_count INTEGER NOT NULL,
  history_json TEXT NOT NULL DEFAULT '[]',
  embedding BLOB,
  updated_at_ms INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity_id);
CREATE INDEX IF NOT EXISTS idx_obs_bank ON observations(bank_id);
CREATE TABLE IF NOT EXISTS opinions(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  text TEXT NOT NULL,
  confidence REAL NOT NULL,
  evidence_json TEXT NOT NULL DEFAULT '[]',
  history_json TEXT NOT NULL DEFAULT '[]',
  embedding BLOB,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS opinion_entities(
  opinion_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  PRIMARY KEY(opinion_id, entity_id)
);
CREATE TABLE IF NOT EXISTS documents(
  bank_id TEXT NOT NULL,
  id TEXT NOT NULL,
  original_text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  tags_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT,
  retain_params_json TEXT NOT NULL DEFAULT '{}',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY(bank_id, id)
);
CREATE TABLE IF NOT EXISTS operations(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  payload_json TEXT NOT NULL DEFAULT 'null',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ops_bank ON operations(bank_id, created_at_ms);
CREATE TABLE IF NOT EXISTS chunks(
  bank_id TEXT NOT NULL,
  id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  PRIMARY KEY(bank_id, id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(bank_id, document_id, chunk_index);
CREATE TABLE IF NOT EXISTS webhooks(
  id TEXT PRIMARY KEY,
  bank_id TEXT,
  url TEXT NOT NULL,
  secret TEXT,
  event_types_json TEXT NOT NULL DEFAULT '[]',
  enabled INTEGER NOT NULL DEFAULT 1,
  http_config_json TEXT NOT NULL DEFAULT '{}',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_webhooks_bank ON webhooks(bank_id, created_at_ms);
CREATE TABLE IF NOT EXISTS webhook_deliveries(
  id TEXT PRIMARY KEY,
  webhook_id TEXT,
  bank_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  url TEXT NOT NULL,
  secret TEXT,
  http_config_json TEXT NOT NULL DEFAULT '{}',
  payload TEXT NOT NULL,
  status TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  next_attempt_at_ms INTEGER NOT NULL,
  last_error TEXT,
  last_response_status INTEGER,
  last_response_body TEXT,
  last_attempt_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deliveries_due ON webhook_deliveries(status, next_attempt_at_ms);
CREATE INDEX IF NOT EXISTS idx_deliveries_webhook ON webhook_deliveries(bank_id, webhook_id, created_at_ms);
CREATE TABLE IF NOT EXISTS attachments(
  bank_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  id TEXT NOT NULL,
  filename TEXT NOT NULL,
  content_type TEXT,
  size INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  bytes BLOB NOT NULL,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY(bank_id, document_id, id)
);
CREATE INDEX IF NOT EXISTS idx_attachments_id ON attachments(bank_id, id);
CREATE TABLE IF NOT EXISTS directives(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  name TEXT NOT NULL,
  content TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  is_active INTEGER NOT NULL DEFAULT 1,
  tags_json TEXT NOT NULL DEFAULT '[]',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  UNIQUE(bank_id, name)
);
CREATE TABLE IF NOT EXISTS store_flags(
  name TEXT PRIMARY KEY,
  set_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS mental_models(
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  question TEXT NOT NULL,
  path TEXT,
  content TEXT NOT NULL DEFAULT '',
  last_refreshed_ms INTEGER,
  created_at_ms INTEGER NOT NULL
);
"#;

const WEBHOOK_COLS: &str = "id, bank_id, url, secret, event_types_json, enabled, http_config_json, created_at_ms, updated_at_ms";

const DELIVERY_COLS: &str = "id, webhook_id, bank_id, event_id, event_type, url, secret, http_config_json, payload, status, retry_count, next_attempt_at_ms, last_error, last_response_status, last_response_body, last_attempt_at_ms, created_at_ms, updated_at_ms";

fn json_err(e: serde_json::Error) -> Error {
    Error::Store(format!("json: {e}"))
}

/// A JSON column that fails to decode surfaces as a row error rather
/// than a silently defaulted value.
fn json_col<T: serde::de::DeserializeOwned>(
    row: &rusqlite::Row,
    idx: usize,
) -> rusqlite::Result<T> {
    let raw: String = row.get(idx)?;
    serde_json::from_str(&raw).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(idx, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn row_webhook(r: &rusqlite::Row) -> rusqlite::Result<Webhook> {
    Ok(Webhook {
        id: r.get(0)?,
        bank_id: r.get(1)?,
        url: r.get(2)?,
        secret: r.get(3)?,
        event_types: json_col(r, 4)?,
        enabled: r.get(5)?,
        http_config: json_col(r, 6)?,
        created_at_ms: r.get(7)?,
        updated_at_ms: r.get(8)?,
    })
}

fn row_delivery(r: &rusqlite::Row) -> rusqlite::Result<WebhookDelivery> {
    Ok(WebhookDelivery {
        id: r.get(0)?,
        webhook_id: r.get(1)?,
        bank_id: r.get(2)?,
        event_id: r.get(3)?,
        event_type: r.get(4)?,
        url: r.get(5)?,
        secret: r.get(6)?,
        http_config: json_col(r, 7)?,
        payload: r.get(8)?,
        status: r.get(9)?,
        retry_count: r.get(10)?,
        next_attempt_at_ms: r.get(11)?,
        last_error: r.get(12)?,
        last_response_status: r.get(13)?,
        last_response_body: r.get(14)?,
        last_attempt_at_ms: r.get(15)?,
        created_at_ms: r.get(16)?,
        updated_at_ms: r.get(17)?,
    })
}

fn emb_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_emb(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_tags(raw: Option<String>) -> Vec<String> {
    raw.and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn tags_to_json(tags: &[String]) -> String {
    serde_json::to_string(tags).unwrap_or_else(|_| "[]".into())
}

fn parse_metadata(raw: Option<String>) -> Option<BTreeMap<String, String>> {
    raw.and_then(|s| serde_json::from_str(&s).ok())
}

fn metadata_to_json(metadata: &Option<BTreeMap<String, String>>) -> Option<String> {
    metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".into()))
}

fn placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

impl SqliteStore {
    fn init(conn: Connection, clock: Arc<dyn Clock>, opts: &StoreOptions) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(store_err)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(store_err)?;
        conn.pragma_update(None, "busy_timeout", 5000)
            .map_err(store_err)?;
        conn.execute_batch(SCHEMA).map_err(store_err)?;
        conn.execute_batch(vector::SCHEMA).map_err(store_err)?;
        Self::migrate(&conn)?;
        let vec_version = if opts.vector_index {
            match vector::load(&conn) {
                Ok(version) => Some(version),
                Err(reason) => {
                    eprintln!(
                        "rustsight: sqlite-vec unavailable ({reason}); semantic search falls back to brute force"
                    );
                    None
                }
            }
        } else {
            None
        };
        if vec_version.is_none() {
            vector::invalidate_all(&conn).map_err(store_err)?;
        }
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            clock,
            vec_version,
            warned_unindexed: Mutex::new(HashSet::new()),
        })
    }

    /// Additive column migrations for databases created before a column
    /// existed (`CREATE TABLE IF NOT EXISTS` never alters an existing
    /// table). Each entry is idempotent: the column is added only when
    /// `pragma table_info` does not already list it.
    fn migrate(conn: &Connection) -> Result<()> {
        const COLUMNS: &[(&str, &str, &str)] = &[
            ("memories", "document_id", "TEXT"),
            ("memories", "tags_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("memories", "metadata_json", "TEXT"),
            ("memories", "chunk_id", "TEXT"),
            ("banks", "config_json", "TEXT"),
            // RS-10d: bank-template fields on mental models.
            ("mental_models", "name", "TEXT"),
            ("mental_models", "tags_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("mental_models", "max_tokens", "INTEGER"),
            ("mental_models", "trigger_json", "TEXT"),
        ];
        for (table, column, decl) in COLUMNS {
            let exists = conn
                .prepare(&format!(
                    "SELECT 1 FROM pragma_table_info('{table}') WHERE name = ?1"
                ))
                .and_then(|mut st| st.exists(params![column]))
                .map_err(store_err)?;
            if !exists {
                conn.execute(
                    &format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"),
                    [],
                )
                .map_err(store_err)?;
            }
        }
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_document ON memories(bank_id, document_id)",
            [],
        )
        .map_err(store_err)?;
        Self::reindex_cjk_once(conn)?;
        Ok(())
    }

    /// RS-10e: FTS rows written before CJK bigram indexing hold whole CJK
    /// runs as single tokens. Rewrite those rows once (non-CJK rows are
    /// unchanged by the segmenter and left alone), then record the flag.
    fn reindex_cjk_once(conn: &Connection) -> Result<()> {
        const FLAG: &str = "fts_cjk_bigrams";
        let done = conn
            .prepare("SELECT 1 FROM store_flags WHERE name = ?1")
            .and_then(|mut st| st.exists(params![FLAG]))
            .map_err(store_err)?;
        if done {
            return Ok(());
        }
        let rows: Vec<(i64, String)> = {
            let mut stmt = conn
                .prepare("SELECT rowid, text FROM texts_fts")
                .map_err(store_err)?;
            let rows = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        let tx = conn.unchecked_transaction().map_err(store_err)?;
        for (rowid, text) in rows {
            let segmented = cjk_segment(&text);
            if segmented != text {
                tx.execute(
                    "UPDATE texts_fts SET text = ?1 WHERE rowid = ?2",
                    params![segmented, rowid],
                )
                .map_err(store_err)?;
            }
        }
        tx.execute(
            "INSERT INTO store_flags(name, set_at_ms) VALUES (?1, ?2)",
            params![
                FLAG,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis() as i64)
            ],
        )
        .map_err(store_err)?;
        tx.commit().map_err(store_err)?;
        Ok(())
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_clock(path, Arc::new(SystemClock))
    }

    pub fn open_with_clock(path: impl AsRef<Path>, clock: Arc<dyn Clock>) -> Result<Self> {
        Self::open_with_options(path, clock, &StoreOptions::default())
    }

    pub fn open_with_options(
        path: impl AsRef<Path>,
        clock: Arc<dyn Clock>,
        opts: &StoreOptions,
    ) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Store(format!("create db dir: {e}")))?;
            }
        }
        let conn = Connection::open(path).map_err(store_err)?;
        Self::init(conn, clock, opts)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::open_in_memory_with_clock(Arc::new(SystemClock))
    }

    pub fn open_in_memory_with_clock(clock: Arc<dyn Clock>) -> Result<Self> {
        Self::open_in_memory_with_options(clock, &StoreOptions::default())
    }

    pub fn open_in_memory_with_options(clock: Arc<dyn Clock>, opts: &StoreOptions) -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(store_err)?;
        Self::init(conn, clock, opts)
    }

    /// The sqlite-vec version when the KNN index is live, `None` when
    /// semantic search is brute force (feature off, disabled, or the
    /// extension failed to initialize).
    pub fn vector_index_version(&self) -> Option<&str> {
        self.vec_version.as_deref()
    }

    /// Whether a bank's semantic queries are answered from the KNN index.
    pub fn bank_vector_indexed(&self, bank_id: &str) -> Result<bool> {
        if self.vec_version.is_none() {
            return Ok(false);
        }
        vector::is_indexed(&self.lock_conn(), bank_id).map_err(store_err)
    }

    /// Rebuild the KNN index for one bank (or every bank) from the stored
    /// embeddings, then mark it indexed. The migration path for databases
    /// written before the index existed or by a store that did not maintain
    /// it. Runs in one transaction per bank, holding the connection lock
    /// for the duration.
    pub fn reindex_vectors(&self, bank_id: Option<&str>) -> Result<Vec<ReindexReport>> {
        if self.vec_version.is_none() {
            return Err(Error::Store(
                "vector index unavailable: sqlite-vec is disabled or failed to load".into(),
            ));
        }
        let banks: Vec<String> = match bank_id {
            Some(b) => vec![b.to_string()],
            None => self.list_banks()?.into_iter().map(|b| b.id).collect(),
        };
        let mut reports = vec![];
        let mut conn = self.lock_conn();
        for bank in banks {
            let tx = conn.transaction().map_err(store_err)?;
            vector::remove_bank(&tx, &bank).map_err(store_err)?;
            let all = Self::all_embeddings_on(&tx, &bank)?;
            let mut report = ReindexReport {
                bank_id: bank.clone(),
                indexed: 0,
                skipped: 0,
            };
            for (id, kind, v) in &all {
                if v.is_empty() {
                    continue;
                }
                if v.len() > vector::MAX_DIM {
                    report.skipped += 1;
                    continue;
                }
                vector::put(&tx, &bank, *kind, id, Some(v)).map_err(store_err)?;
                report.indexed += 1;
            }
            if report.skipped == 0 {
                vector::mark(&tx, &bank).map_err(store_err)?;
            }
            tx.commit().map_err(store_err)?;
            reports.push(report);
        }
        Ok(reports)
    }

    /// Mirror an item's embedding into the KNN index when it is live.
    fn vec_put(
        &self,
        conn: &Connection,
        bank_id: &str,
        kind: MemoryKind,
        ref_id: &str,
        embedding: Option<&[f32]>,
    ) -> Result<()> {
        if self.vec_version.is_some() {
            vector::put(conn, bank_id, kind, ref_id, embedding).map_err(store_err)?;
        }
        Ok(())
    }

    /// Drop items from the KNN index when it is live.
    fn vec_remove<'a>(
        &self,
        conn: &Connection,
        ref_ids: impl IntoIterator<Item = &'a String>,
    ) -> Result<()> {
        if self.vec_version.is_some() {
            vector::remove_many(conn, ref_ids).map_err(store_err)?;
        }
        Ok(())
    }

    /// One query's top-k from the index, widening the KNN fetch while an
    /// `admit` predicate keeps rejecting rows. `None` when even the widest
    /// fetch vec0 allows cannot fill `k` — the caller scans instead.
    fn knn_top_k(
        conn: &Connection,
        bank_id: &str,
        query: &[f32],
        kinds: &[MemoryKind],
        k: usize,
        admit: Admit<'_>,
    ) -> Result<Option<Vec<SemanticHit>>> {
        if k == 0 {
            return Ok(Some(vec![]));
        }
        let mut fetch = match admit {
            Some(_) => (k * 4).max(64),
            None => k,
        }
        .min(vector::MAX_K);
        loop {
            let rows = vector::knn(conn, bank_id, query, kinds, fetch).map_err(store_err)?;
            let exhausted = rows.len() < fetch;
            let mut hits: Vec<SemanticHit> = rows
                .into_iter()
                .filter(|(id, _, _)| admit.is_none_or(|f| f(id)))
                .collect();
            if hits.len() >= k || exhausted {
                sort_hits(&mut hits);
                hits.truncate(k);
                return Ok(Some(hits));
            }
            if fetch >= vector::MAX_K {
                return Ok(None);
            }
            fetch = (fetch * 4).min(vector::MAX_K);
        }
    }

    /// Deliberate poisoning-recovery policy: SQLite's real state lives in the
    /// database file, not in the guard, so a panic mid-query leaves the
    /// `Connection` usable and we take the lock anyway instead of failing
    /// every subsequent store call.
    fn lock_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn all_embeddings_on(
        conn: &Connection,
        bank_id: &str,
    ) -> Result<Vec<(MemoryId, MemoryKind, Vec<f32>)>> {
        let mut out = vec![];
        let mut stmt = conn
            .prepare("SELECT id, kind, embedding FROM memories WHERE bank_id=?1 AND embedding IS NOT NULL")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Vec<u8>>(2)?,
                ))
            })
            .map_err(store_err)?;
        for r in rows {
            let (id, kind, blob) = r.map_err(store_err)?;
            out.push((
                id,
                MemoryKind::parse(&kind).unwrap_or(MemoryKind::World),
                blob_to_emb(&blob),
            ));
        }
        let mut stmt = conn
            .prepare(
                "SELECT id, embedding FROM observations WHERE bank_id=?1 AND embedding IS NOT NULL",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .map_err(store_err)?;
        for r in rows {
            let (id, blob) = r.map_err(store_err)?;
            out.push((id, MemoryKind::Observation, blob_to_emb(&blob)));
        }
        let mut stmt = conn
            .prepare(
                "SELECT id, embedding FROM opinions WHERE bank_id=?1 AND embedding IS NOT NULL",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .map_err(store_err)?;
        for r in rows {
            let (id, blob) = r.map_err(store_err)?;
            out.push((id, MemoryKind::Opinion, blob_to_emb(&blob)));
        }
        Ok(out)
    }

    fn row_memory(r: &Row) -> rusqlite::Result<Memory> {
        Ok(Memory {
            id: r.get(0)?,
            bank_id: r.get(1)?,
            kind: MemoryKind::parse(&r.get::<_, String>(2)?).unwrap_or(MemoryKind::World),
            text: r.get(3)?,
            confidence: r.get(4)?,
            occurred_start_ms: r.get(5)?,
            occurred_end_ms: r.get(6)?,
            mentioned_at_ms: r.get(7)?,
            source: r.get(8)?,
            context: r.get(9)?,
            document_id: r.get(10)?,
            tags: parse_tags(r.get::<_, Option<String>>(11)?),
            metadata: parse_metadata(r.get::<_, Option<String>>(12)?),
        })
    }

    fn row_entity(r: &Row) -> rusqlite::Result<Entity> {
        Ok(Entity {
            id: r.get(0)?,
            bank_id: r.get(1)?,
            name: r.get(2)?,
            etype: EntityType::parse(&r.get::<_, String>(3)?),
            last_mentioned_ms: r.get(4)?,
        })
    }

    fn fts_upsert(
        conn: &Connection,
        bank_id: &str,
        kind: &str,
        ref_id: &str,
        text: &str,
    ) -> Result<()> {
        conn.execute(
            "DELETE FROM texts_fts WHERE bank_id = ?1 AND ref_id = ?2",
            params![bank_id, ref_id],
        )
        .map_err(store_err)?;
        conn.execute(
            "INSERT INTO texts_fts(text, bank_id, kind, ref_id) VALUES (?1, ?2, ?3, ?4)",
            params![cjk_segment(text), bank_id, kind, ref_id],
        )
        .map_err(store_err)?;
        Ok(())
    }
}

impl MemoryStore for SqliteStore {
    fn ensure_bank(&self, bank: &Bank) -> Result<()> {
        let conn = self.lock_conn();
        let now = self.clock.now_ms();
        let created = conn.execute(
            "INSERT OR IGNORE INTO banks(id, mission, directives, background, skepticism, literalism, empathy, bias_strength, created_at_ms, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)",
            params![
                bank.id,
                bank.mission,
                bank.directives,
                bank.background,
                bank.disposition.skepticism,
                bank.disposition.literalism,
                bank.disposition.empathy,
                bank.disposition.bias_strength,
                now
            ],
        )
        .map_err(store_err)?;
        // A bank born under a live index has every embedding mirrored from
        // its first write; older banks wait for `reindex_vectors`.
        if created > 0 && self.vec_version.is_some() {
            vector::mark(&conn, &bank.id).map_err(store_err)?;
        }
        Ok(())
    }

    fn get_bank(&self, bank_id: &str) -> Result<Option<Bank>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT id, mission, directives, background, skepticism, literalism, empathy, bias_strength FROM banks WHERE id = ?1",
            params![bank_id],
            |r| {
                Ok(Bank {
                    id: r.get(0)?,
                    mission: r.get(1)?,
                    directives: r.get(2)?,
                    background: r.get(3)?,
                    disposition: Disposition {
                        skepticism: r.get::<_, i64>(4)?.clamp(1, 5) as u8,
                        literalism: r.get::<_, i64>(5)?.clamp(1, 5) as u8,
                        empathy: r.get::<_, i64>(6)?.clamp(1, 5) as u8,
                        bias_strength: r.get(7)?,
                    },
                })
            },
        )
        .optional()
        .map_err(store_err)
    }

    fn update_bank(&self, bank: &Bank) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "UPDATE banks SET mission=?2, directives=?3, background=?4, skepticism=?5, literalism=?6, empathy=?7, bias_strength=?8, updated_at_ms=?9 WHERE id=?1",
            params![
                bank.id,
                bank.mission,
                bank.directives,
                bank.background,
                bank.disposition.skepticism,
                bank.disposition.literalism,
                bank.disposition.empathy,
                bank.disposition.bias_strength,
                self.clock.now_ms()
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn list_banks(&self) -> Result<Vec<Bank>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, mission, directives, background, skepticism, literalism, empathy, bias_strength FROM banks ORDER BY id")
            .map_err(store_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok(Bank {
                    id: r.get(0)?,
                    mission: r.get(1)?,
                    directives: r.get(2)?,
                    background: r.get(3)?,
                    disposition: Disposition {
                        skepticism: r.get::<_, i64>(4)?.clamp(1, 5) as u8,
                        literalism: r.get::<_, i64>(5)?.clamp(1, 5) as u8,
                        empathy: r.get::<_, i64>(6)?.clamp(1, 5) as u8,
                        bias_strength: r.get(7)?,
                    },
                })
            })
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn insert_memories(&self, bank_id: &str, items: &[NewMemoryRecord]) -> Result<Vec<Memory>> {
        let mut conn = self.lock_conn();
        let now = self.clock.now_ms();
        let mut out = vec![];
        let tx = conn.transaction().map_err(store_err)?;
        for item in items {
            let id = uuid::Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO memories(id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, embedding, source, context, created_at_ms, document_id, tags_json, metadata_json, chunk_id)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
                params![
                    id,
                    bank_id,
                    item.kind.as_str(),
                    item.text,
                    item.confidence,
                    item.occurred_start_ms,
                    item.occurred_end_ms,
                    item.mentioned_at_ms,
                    emb_to_blob(&item.embedding),
                    item.source,
                    item.context,
                    now,
                    item.document_id,
                    tags_to_json(&item.tags),
                    metadata_to_json(&item.metadata),
                    item.chunk_id
                ],
            )
            .map_err(store_err)?;
            tx.execute(
                "INSERT INTO texts_fts(text, bank_id, kind, ref_id) VALUES (?1, ?2, ?3, ?4)",
                params![cjk_segment(&item.text), bank_id, item.kind.as_str(), id],
            )
            .map_err(store_err)?;
            self.vec_put(&tx, bank_id, item.kind, &id, Some(&item.embedding))?;
            out.push(Memory {
                id,
                bank_id: bank_id.to_string(),
                kind: item.kind,
                text: item.text.clone(),
                confidence: item.confidence,
                occurred_start_ms: item.occurred_start_ms,
                occurred_end_ms: item.occurred_end_ms,
                mentioned_at_ms: item.mentioned_at_ms,
                source: item.source.clone(),
                context: item.context.clone(),
                document_id: item.document_id.clone(),
                tags: item.tags.clone(),
                metadata: item.metadata.clone(),
            });
        }
        tx.commit().map_err(store_err)?;
        Ok(out)
    }

    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context, document_id, tags_json, metadata_json FROM memories WHERE id = ?1",
            params![id],
            Self::row_memory,
        )
        .optional()
        .map_err(store_err)
    }

    fn recent_memories(&self, bank_id: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context, document_id, tags_json, metadata_json FROM memories WHERE bank_id=?1 ORDER BY mentioned_at_ms DESC, id LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, limit], Self::row_memory)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn list_memories(&self, bank_id: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context, document_id, tags_json, metadata_json FROM memories WHERE bank_id=?1 ORDER BY created_at_ms DESC, id LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, limit], Self::row_memory)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn all_embeddings(&self, bank_id: &str) -> Result<Vec<(MemoryId, MemoryKind, Vec<f32>)>> {
        Self::all_embeddings_on(&self.lock_conn(), bank_id)
    }

    fn search_semantic(
        &self,
        bank_id: &str,
        queries: &[&[f32]],
        kinds: &[MemoryKind],
        k: usize,
        admit: Admit<'_>,
    ) -> Result<Vec<Vec<SemanticHit>>> {
        let conn = self.lock_conn();
        let indexed =
            self.vec_version.is_some() && vector::is_indexed(&conn, bank_id).map_err(store_err)?;
        if !indexed {
            if self.vec_version.is_some() {
                let mut warned = self
                    .warned_unindexed
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if warned.insert(bank_id.to_string()) {
                    eprintln!(
                        "rustsight: bank {bank_id} has no vector index yet; semantic search scans every embedding until `rustsight reindex --bank {bank_id}`"
                    );
                }
            }
            let all = Self::all_embeddings_on(&conn, bank_id)?;
            return Ok(brute_force_semantic(&all, queries, kinds, k, admit));
        }
        let mut scan: Option<Vec<(MemoryId, MemoryKind, Vec<f32>)>> = None;
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            match Self::knn_top_k(&conn, bank_id, q, kinds, k, admit)? {
                Some(hits) => out.push(hits),
                None => {
                    if scan.is_none() {
                        scan = Some(Self::all_embeddings_on(&conn, bank_id)?);
                    }
                    let all = scan.as_deref().unwrap_or_default();
                    out.extend(brute_force_semantic(all, &[q], kinds, k, admit));
                }
            }
        }
        Ok(out)
    }

    fn search_bm25(
        &self,
        bank_id: &str,
        terms: &[String],
        limit: usize,
    ) -> Result<Vec<(MemoryId, MemoryKind, f32)>> {
        if terms.is_empty() {
            return Ok(vec![]);
        }
        let query = terms
            .iter()
            .filter(|t| t.chars().all(|c| c.is_alphanumeric()))
            .cloned()
            .collect::<Vec<_>>()
            .join(" OR ");
        if query.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT ref_id, kind, bm25(texts_fts) AS b FROM texts_fts WHERE texts_fts MATCH ?1 AND bank_id = ?2 ORDER BY b LIMIT ?3",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![query, bank_id, limit], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                ))
            })
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            let (id, kind, b) = r.map_err(store_err)?;
            if let Some(k) = MemoryKind::parse(&kind) {
                out.push((id, k, -(b as f32)));
            }
        }
        Ok(out)
    }

    fn memories_in_window(
        &self,
        bank_id: &str,
        start_ms: i64,
        end_ms: i64,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context, document_id, tags_json, metadata_json FROM memories
                 WHERE bank_id=?1 AND (
                   (occurred_start_ms IS NOT NULL AND occurred_start_ms <= ?2 AND COALESCE(occurred_end_ms, occurred_start_ms) >= ?3)
                   OR (occurred_start_ms IS NULL AND mentioned_at_ms BETWEEN ?3 AND ?2)
                 ) ORDER BY mentioned_at_ms DESC LIMIT ?4",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, end_ms, start_ms, limit], Self::row_memory)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn get_refs(&self, bank_id: &str, ids: &[MemoryId]) -> Result<Vec<MemoryRef>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.lock_conn();
        let mut map: HashMap<String, MemoryRef> = HashMap::new();
        for chunk in ids.chunks(400) {
            let ph = placeholders(chunk.len());
            let sql = format!(
                "SELECT id, kind, text, occurred_start_ms, occurred_end_ms, mentioned_at_ms, context, document_id, tags_json, metadata_json FROM memories WHERE bank_id = ? AND id IN ({ph})"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(
                    params_from_iter(
                        std::iter::once(bank_id.to_string()).chain(chunk.iter().cloned()),
                    ),
                    |r| {
                        Ok(MemoryRef {
                            id: r.get(0)?,
                            kind: MemoryKind::parse(&r.get::<_, String>(1)?)
                                .unwrap_or(MemoryKind::World),
                            text: r.get(2)?,
                            score: 0.0,
                            occurred_start_ms: r.get(3)?,
                            occurred_end_ms: r.get(4)?,
                            mentioned_at_ms: Some(r.get(5)?),
                            entity_ids: vec![],
                            proof_count: None,
                            context: r.get(6)?,
                            document_id: r.get(7)?,
                            tags: parse_tags(r.get::<_, Option<String>>(8)?),
                            metadata: parse_metadata(r.get::<_, Option<String>>(9)?),
                        })
                    },
                )
                .map_err(store_err)?;
            for row in rows {
                let m = row.map_err(store_err)?;
                map.insert(m.id.clone(), m);
            }

            let sql = format!(
                "SELECT id, text, entity_id, updated_at_ms, proof_count FROM observations WHERE bank_id = ? AND id IN ({ph})"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(
                    params_from_iter(
                        std::iter::once(bank_id.to_string()).chain(chunk.iter().cloned()),
                    ),
                    |r| {
                        Ok(MemoryRef {
                            id: r.get(0)?,
                            kind: MemoryKind::Observation,
                            text: r.get(1)?,
                            score: 0.0,
                            occurred_start_ms: None,
                            occurred_end_ms: None,
                            mentioned_at_ms: Some(r.get(3)?),
                            entity_ids: vec![r.get(2)?],
                            proof_count: Some(r.get(4)?),
                            context: None,
                            document_id: None,
                            tags: vec![],
                            metadata: None,
                        })
                    },
                )
                .map_err(store_err)?;
            for row in rows {
                let m = row.map_err(store_err)?;
                map.insert(m.id.clone(), m);
            }

            let sql = format!(
                "SELECT id, text, updated_at_ms FROM opinions WHERE bank_id = ? AND id IN ({ph})"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(
                    params_from_iter(
                        std::iter::once(bank_id.to_string()).chain(chunk.iter().cloned()),
                    ),
                    |r| {
                        Ok(MemoryRef {
                            id: r.get(0)?,
                            kind: MemoryKind::Opinion,
                            text: r.get(1)?,
                            score: 0.0,
                            occurred_start_ms: None,
                            occurred_end_ms: None,
                            mentioned_at_ms: Some(r.get(2)?),
                            entity_ids: vec![],
                            proof_count: None,
                            context: None,
                            document_id: None,
                            tags: vec![],
                            metadata: None,
                        })
                    },
                )
                .map_err(store_err)?;
            for row in rows {
                let m = row.map_err(store_err)?;
                map.insert(m.id.clone(), m);
            }

            // memory_entities has no bank column, but it is only consulted
            // for ids already admitted to `map` by the bank-filtered
            // memories query above, so no cross-bank rows can attach.
            let sql = format!(
                "SELECT memory_id, entity_id FROM memory_entities WHERE memory_id IN ({ph})"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })
                .map_err(store_err)?;
            for r in rows {
                let (mid, eid) = r.map_err(store_err)?;
                if let Some(m) = map.get_mut(&mid) {
                    m.entity_ids.push(eid);
                }
            }
        }
        Ok(ids.iter().filter_map(|id| map.get(id).cloned()).collect())
    }

    fn neighbors(
        &self,
        bank_id: &str,
        seeds: &[MemoryId],
        hops: usize,
        limit: usize,
    ) -> Result<Vec<MemoryId>> {
        if seeds.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.lock_conn();
        let mut visited: HashSet<String> = seeds.iter().cloned().collect();
        let mut frontier: Vec<String> = seeds.to_vec();
        for _ in 0..hops {
            if frontier.is_empty() || visited.len() >= limit {
                break;
            }
            let mut next = vec![];
            for chunk in frontier.chunks(400) {
                if visited.len() >= limit {
                    break;
                }
                let ph = placeholders(chunk.len());
                let sql = format!(
                    "SELECT b_id FROM edges WHERE bank_id = ? AND a_id IN ({ph}) UNION SELECT a_id FROM edges WHERE bank_id = ? AND b_id IN ({ph})",
                );
                let mut stmt = conn.prepare(&sql).map_err(store_err)?;
                let mut bind: Vec<String> = Vec::with_capacity(chunk.len() * 2 + 2);
                bind.push(bank_id.to_string());
                bind.extend(chunk.iter().cloned());
                bind.push(bank_id.to_string());
                bind.extend(chunk.iter().cloned());
                let rows = stmt
                    .query_map(params_from_iter(bind), |r| r.get::<_, String>(0))
                    .map_err(store_err)?;
                for r in rows {
                    let id = r.map_err(store_err)?;
                    if visited.insert(id.clone()) {
                        next.push(id);
                        if visited.len() >= limit {
                            break;
                        }
                    }
                }
            }
            frontier = next;
        }
        Ok(visited.into_iter().collect())
    }

    fn create_entity(&self, e: &NewEntity) -> Result<Entity> {
        let conn = self.lock_conn();
        let id = uuid::Uuid::new_v4().to_string();
        let now = self.clock.now_ms();
        conn.execute(
            "INSERT INTO entities(id, bank_id, name, etype, last_mentioned_ms, created_at_ms) VALUES (?1,?2,?3,?4,?5,?5)",
            params![id, e.bank_id, e.name, e.etype.as_str(), now],
        )
        .map_err(store_err)?;
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases(bank_id, alias, entity_id) VALUES (?1, ?2, ?3)",
            params![e.bank_id, rustsight_core::normalize_name(&e.name), id],
        )
        .map_err(store_err)?;
        Ok(Entity {
            id,
            bank_id: e.bank_id.clone(),
            name: e.name.clone(),
            etype: e.etype,
            last_mentioned_ms: now,
        })
    }

    fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT id, bank_id, name, etype, last_mentioned_ms FROM entities WHERE id = ?1",
            params![id],
            Self::row_entity,
        )
        .optional()
        .map_err(store_err)
    }

    fn find_by_alias(&self, bank_id: &str, alias: &str) -> Result<Option<Entity>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT e.id, e.bank_id, e.name, e.etype, e.last_mentioned_ms FROM entity_aliases a JOIN entities e ON e.id = a.entity_id WHERE a.bank_id = ?1 AND a.alias = ?2",
            params![bank_id, alias],
            Self::row_entity,
        )
        .optional()
        .map_err(store_err)
    }

    fn entities_for_bank(&self, bank_id: &str) -> Result<Vec<Entity>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, name, etype, last_mentioned_ms FROM entities WHERE bank_id=?1 ORDER BY name")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], Self::row_entity)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn add_alias(&self, bank_id: &str, alias: &str, entity_id: &str) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases(bank_id, alias, entity_id) VALUES (?1, ?2, ?3)",
            params![bank_id, alias, entity_id],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn link_memory_entities(
        &self,
        memory_id: &MemoryId,
        links: &[(String, String)],
        at_ms: i64,
    ) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        for (entity_id, mention) in links {
            tx.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, mention) VALUES (?1, ?2, ?3)",
                params![memory_id, entity_id, mention],
            )
            .map_err(store_err)?;
            tx.execute(
                "UPDATE entities SET last_mentioned_ms = MAX(last_mentioned_ms, ?2) WHERE id = ?1",
                params![entity_id, at_ms],
            )
            .map_err(store_err)?;
        }
        tx.commit().map_err(store_err)
    }

    fn memory_entity_ids(&self, memory_id: &MemoryId) -> Result<Vec<String>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT entity_id FROM memory_entities WHERE memory_id = ?1")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![memory_id], |r| r.get::<_, String>(0))
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn coentities(&self, entity_id: &str, limit: usize) -> Result<Vec<String>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT me2.entity_id FROM memory_entities me1 JOIN memory_entities me2 ON me1.memory_id = me2.memory_id WHERE me1.entity_id = ?1 AND me2.entity_id != ?1 LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id, limit], |r| r.get::<_, String>(0))
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn coentities_batch(
        &self,
        entity_ids: &[String],
        limit: usize,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        if entity_ids.is_empty() {
            return Ok(out);
        }
        let conn = self.lock_conn();
        for chunk in entity_ids.chunks(400) {
            let ph = placeholders(chunk.len());
            let sql = format!(
                "SELECT DISTINCT me1.entity_id, me2.entity_id FROM memory_entities me1
                 JOIN memory_entities me2 ON me1.memory_id = me2.memory_id
                 WHERE me1.entity_id IN ({ph}) AND me2.entity_id != me1.entity_id"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })
                .map_err(store_err)?;
            for r in rows {
                let (a, b) = r.map_err(store_err)?;
                // Truncate in Rust: per-entity LIMIT inside the IN-query has
                // no meaningful ordering to key on (matching the arbitrary
                // truncation of the per-id variant).
                let list = out.entry(a).or_default();
                if list.len() < limit {
                    list.push(b);
                }
            }
        }
        Ok(out)
    }

    fn cooccurrence_index(&self, bank_id: &str) -> Result<CooccurrenceIndex> {
        // Every directed pair of entities sharing a stored memory, with both
        // canonical names — the same pair set upstream persists in
        // `entity_cooccurrences` (one row per fact's entity pair), computed
        // here from the per-memory links. Degree is counted over ALL
        // partners of an entity, exactly like upstream's index build.
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT me1.entity_id, e1.name, me2.entity_id, e2.name
                 FROM memory_entities me1
                 JOIN memory_entities me2
                      ON me1.memory_id = me2.memory_id AND me1.entity_id != me2.entity_id
                 JOIN entities e1 ON e1.id = me1.entity_id
                 JOIN entities e2 ON e2.id = me2.entity_id
                 WHERE e1.bank_id = ?1",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .map_err(store_err)?;
        let mut neighbours: HashMap<String, HashSet<String>> = HashMap::new();
        let mut partners_by_entity: HashMap<String, HashSet<String>> = HashMap::new();
        let mut name_by_entity: HashMap<String, String> = HashMap::new();
        for r in rows {
            let (eid, name, other_id, other_name) = r.map_err(store_err)?;
            neighbours.entry(eid.clone()).or_default().insert(other_id);
            partners_by_entity
                .entry(eid.clone())
                .or_default()
                .insert(other_name.to_lowercase());
            name_by_entity
                .entry(eid)
                .or_insert_with(|| name.to_lowercase());
        }
        let mut degree_by_name: HashMap<String, usize> = HashMap::new();
        for (eid, name) in &name_by_entity {
            let degree = neighbours.get(eid).map(|s| s.len()).unwrap_or(0);
            // Same name from two ids can only happen mid-merge; keep the
            // larger degree (upstream does the same).
            let slot = degree_by_name.entry(name.clone()).or_insert(0);
            *slot = (*slot).max(degree);
        }
        Ok(CooccurrenceIndex {
            partners_by_entity,
            degree_by_name,
        })
    }

    fn facts_for_entity(&self, entity_id: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.bank_id, m.kind, m.text, m.confidence, m.occurred_start_ms, m.occurred_end_ms, m.mentioned_at_ms, m.source, m.context, m.document_id, m.tags_json, m.metadata_json
                 FROM memories m JOIN memory_entities me ON me.memory_id = m.id
                 WHERE me.entity_id = ?1 ORDER BY m.mentioned_at_ms DESC LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id, limit], Self::row_memory)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn facts_for_entities(
        &self,
        entity_ids: &[String],
        per_entity_limit: usize,
    ) -> Result<HashMap<String, Vec<Memory>>> {
        let mut out: HashMap<String, Vec<Memory>> = HashMap::new();
        if entity_ids.is_empty() || per_entity_limit == 0 {
            return Ok(out);
        }
        let conn = self.lock_conn();
        for chunk in entity_ids.chunks(400) {
            let ph = placeholders(chunk.len());
            // ROW_NUMBER caps each entity's rows at per_entity_limit,
            // reproducing facts_for_entity's per-id
            // `ORDER BY mentioned_at_ms DESC LIMIT ?` in one query.
            let sql = format!(
                "SELECT m.id, m.bank_id, m.kind, m.text, m.confidence, m.occurred_start_ms, m.occurred_end_ms, m.mentioned_at_ms, m.source, m.context, m.document_id, m.tags_json, m.metadata_json, sub.eid
                 FROM (
                   SELECT me.entity_id AS eid, me.memory_id AS mid,
                          ROW_NUMBER() OVER (PARTITION BY me.entity_id ORDER BY m2.mentioned_at_ms DESC) AS rn
                   FROM memory_entities me JOIN memories m2 ON m2.id = me.memory_id
                   WHERE me.entity_id IN ({ph})
                 ) sub
                 JOIN memories m ON m.id = sub.mid
                 WHERE sub.rn <= ?
                 ORDER BY sub.eid, m.mentioned_at_ms DESC"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            // Value mixes TEXT ids and the INTEGER limit in one iterator.
            let mut bind: Vec<rusqlite::types::Value> = chunk
                .iter()
                .map(|id| rusqlite::types::Value::Text(id.clone()))
                .collect();
            bind.push(rusqlite::types::Value::Integer(per_entity_limit as i64));
            let rows = stmt
                .query_map(params_from_iter(bind), |r| {
                    Ok((Self::row_memory(r)?, r.get::<_, String>(13)?))
                })
                .map_err(store_err)?;
            for r in rows {
                let (m, eid) = r.map_err(store_err)?;
                out.entry(eid).or_default().push(m);
            }
        }
        Ok(out)
    }

    fn add_edges(&self, edges: &[Edge]) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        for e in edges {
            tx.execute(
                "INSERT OR IGNORE INTO edges(bank_id, a_id, b_id, kind, label, weight) VALUES (?1,?2,?3,?4,?5,?6)",
                params![e.bank_id, e.a_id, e.b_id, e.kind.as_str(), e.label.clone().unwrap_or_default(), e.weight],
            )
            .map_err(store_err)?;
        }
        tx.commit().map_err(store_err)
    }

    fn edges_for_bank(&self, bank_id: &str, limit: usize) -> Result<Vec<Edge>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT bank_id, a_id, b_id, kind, label, weight FROM edges WHERE bank_id = ?1 LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, limit], |r| {
                let label: String = r.get(4)?;
                Ok(Edge {
                    bank_id: r.get(0)?,
                    a_id: r.get(1)?,
                    b_id: r.get(2)?,
                    kind: EdgeKind::parse(&r.get::<_, String>(3)?).unwrap_or(EdgeKind::Semantic),
                    label: if label.is_empty() { None } else { Some(label) },
                    weight: r.get(5)?,
                })
            })
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn mark_entities_dirty(
        &self,
        bank_id: &str,
        entity_ids: &[String],
        reason: &str,
    ) -> Result<()> {
        let mut conn = self.lock_conn();
        let now = self.clock.now_ms();
        let tx = conn.transaction().map_err(store_err)?;
        for id in entity_ids {
            tx.execute(
                "INSERT OR REPLACE INTO dirty_entities(bank_id, entity_id, reason, marked_at_ms) VALUES (?1,?2,?3,?4)",
                params![bank_id, id, reason, now],
            )
            .map_err(store_err)?;
        }
        tx.commit().map_err(store_err)
    }

    fn list_dirty_entities(&self, bank_id: &str, limit: usize) -> Result<Vec<String>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT entity_id FROM dirty_entities WHERE bank_id = ?1 ORDER BY marked_at_ms LIMIT ?2")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, limit], |r| r.get::<_, String>(0))
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn count_dirty_entities(&self, bank_id: &str) -> Result<usize> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT COUNT(*) FROM dirty_entities WHERE bank_id = ?1",
            params![bank_id],
            |r| r.get::<_, i64>(0),
        )
        .map(|n| n as usize)
        .map_err(store_err)
    }

    fn clear_dirty_entities(&self, bank_id: &str, entity_ids: &[String]) -> Result<()> {
        let conn = self.lock_conn();
        for chunk in entity_ids.chunks(400) {
            let ph = placeholders(chunk.len());
            let sql =
                format!("DELETE FROM dirty_entities WHERE bank_id = ? AND entity_id IN ({ph})");
            conn.execute(
                &sql,
                params_from_iter(
                    std::iter::once(bank_id as &str).chain(chunk.iter().map(|s| s.as_str())),
                ),
            )
            .map_err(store_err)?;
        }
        Ok(())
    }

    fn banks_with_dirty(&self) -> Result<Vec<String>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT DISTINCT bank_id FROM dirty_entities ORDER BY bank_id")
            .map_err(store_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn upsert_observation(&self, o: &Observation) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            "INSERT INTO observations(id, bank_id, entity_id, text, evidence_json, proof_count, history_json, embedding, updated_at_ms, created_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
             ON CONFLICT(id) DO UPDATE SET text=excluded.text, evidence_json=excluded.evidence_json, proof_count=excluded.proof_count, history_json=excluded.history_json, embedding=excluded.embedding, updated_at_ms=excluded.updated_at_ms",
            params![
                o.id,
                o.bank_id,
                o.entity_id,
                o.text,
                serde_json::to_string(&o.evidence).map_err(|e| Error::Store(e.to_string()))?,
                o.proof_count,
                serde_json::to_string(&o.history).map_err(|e| Error::Store(e.to_string()))?,
                o.embedding.as_deref().map(emb_to_blob),
                o.updated_at_ms,
                o.created_at_ms
            ],
        )
        .map_err(store_err)?;
        Self::fts_upsert(&tx, &o.bank_id, "observation", &o.id, &o.text)?;
        self.vec_put(
            &tx,
            &o.bank_id,
            MemoryKind::Observation,
            &o.id,
            o.embedding.as_deref(),
        )?;
        tx.commit().map_err(store_err)
    }

    fn observations_for_entity(&self, entity_id: &str) -> Result<Vec<Observation>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, entity_id, text, evidence_json, proof_count, history_json, embedding, updated_at_ms, created_at_ms FROM observations WHERE entity_id = ?1 ORDER BY updated_at_ms DESC")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id], Self::row_observation)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn observations_for_bank(&self, bank_id: &str) -> Result<Vec<Observation>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, entity_id, text, evidence_json, proof_count, history_json, embedding, updated_at_ms, created_at_ms FROM observations WHERE bank_id = ?1 ORDER BY updated_at_ms DESC")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], Self::row_observation)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn set_observation_embedding(&self, id: &str, embedding: Option<Vec<f32>>) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        let bank_id: Option<String> = tx
            .query_row(
                "UPDATE observations SET embedding = ?2 WHERE id = ?1 RETURNING bank_id",
                params![id, embedding.as_deref().map(emb_to_blob)],
                |r| r.get(0),
            )
            .optional()
            .map_err(store_err)?;
        if let Some(bank_id) = bank_id {
            self.vec_put(
                &tx,
                &bank_id,
                MemoryKind::Observation,
                id,
                embedding.as_deref(),
            )?;
        }
        tx.commit().map_err(store_err)
    }

    fn delete_observation(&self, bank_id: &str, id: &str) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            "DELETE FROM observations WHERE id = ?1 AND bank_id = ?2",
            params![id, bank_id],
        )
        .map_err(store_err)?;
        tx.execute(
            "DELETE FROM texts_fts WHERE bank_id = ?1 AND ref_id = ?2",
            params![bank_id, id],
        )
        .map_err(store_err)?;
        self.vec_remove(&tx, [&id.to_string()])?;
        tx.commit().map_err(store_err)
    }

    fn upsert_opinion(&self, o: &Opinion) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            "INSERT INTO opinions(id, bank_id, text, confidence, evidence_json, history_json, embedding, created_at_ms, updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
             ON CONFLICT(id) DO UPDATE SET text=excluded.text, confidence=excluded.confidence, evidence_json=excluded.evidence_json, history_json=excluded.history_json, embedding=excluded.embedding, updated_at_ms=excluded.updated_at_ms",
            params![
                o.id,
                o.bank_id,
                o.text,
                o.confidence,
                serde_json::to_string(&o.evidence).map_err(|e| Error::Store(e.to_string()))?,
                serde_json::to_string(&o.history).map_err(|e| Error::Store(e.to_string()))?,
                o.embedding.as_deref().map(emb_to_blob),
                o.created_at_ms,
                o.updated_at_ms
            ],
        )
        .map_err(store_err)?;
        for eid in &o.entity_ids {
            tx.execute(
                "INSERT OR IGNORE INTO opinion_entities(opinion_id, entity_id) VALUES (?1, ?2)",
                params![o.id, eid],
            )
            .map_err(store_err)?;
        }
        Self::fts_upsert(&tx, &o.bank_id, "opinion", &o.id, &o.text)?;
        self.vec_put(
            &tx,
            &o.bank_id,
            MemoryKind::Opinion,
            &o.id,
            o.embedding.as_deref(),
        )?;
        tx.commit().map_err(store_err)
    }

    fn list_opinions(&self, bank_id: &str) -> Result<Vec<Opinion>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, text, confidence, evidence_json, history_json, embedding, created_at_ms, updated_at_ms FROM opinions WHERE bank_id = ?1 ORDER BY updated_at_ms DESC")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, f32>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, Option<Vec<u8>>>(6)?,
                    r.get::<_, i64>(7)?,
                    r.get::<_, i64>(8)?,
                ))
            })
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            let (id, bank_id, text, confidence, evj, hj, emb, created, updated) =
                r.map_err(store_err)?;
            let evidence: Vec<Evidence> =
                serde_json::from_str(&evj).map_err(|e| Error::Store(e.to_string()))?;
            let history: Vec<ConfidencePoint> =
                serde_json::from_str(&hj).map_err(|e| Error::Store(e.to_string()))?;
            out.push(Opinion {
                id,
                bank_id,
                text,
                confidence,
                evidence,
                history,
                entity_ids: vec![],
                updated_at_ms: updated,
                created_at_ms: created,
                embedding: emb.map(|b| blob_to_emb(&b)),
            });
        }
        drop(stmt);
        if !out.is_empty() {
            // Batched entity-link fetch: one IN query per 400 opinions instead
            // of a prepared statement per row (CR-17).
            let ids: Vec<String> = out.iter().map(|o| o.id.clone()).collect();
            let mut links: HashMap<String, Vec<String>> = HashMap::new();
            for chunk in ids.chunks(400) {
                let ph = placeholders(chunk.len());
                let sql = format!(
                    "SELECT opinion_id, entity_id FROM opinion_entities WHERE opinion_id IN ({ph}) ORDER BY opinion_id, entity_id"
                );
                let mut links_stmt = conn.prepare(&sql).map_err(store_err)?;
                let link_rows = links_stmt
                    .query_map(params_from_iter(chunk.iter()), |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                    })
                    .map_err(store_err)?;
                for r in link_rows {
                    let (oid, eid) = r.map_err(store_err)?;
                    links.entry(oid).or_default().push(eid);
                }
            }
            for o in out.iter_mut() {
                if let Some(eids) = links.remove(&o.id) {
                    o.entity_ids = eids;
                }
            }
        }
        Ok(out)
    }

    fn bank_config(&self, bank_id: &str) -> Result<serde_json::Value> {
        let conn = self.lock_conn();
        let raw: Option<String> = conn
            .query_row(
                "SELECT config_json FROM banks WHERE id = ?1",
                params![bank_id],
                |r| r.get::<_, Option<String>>(0),
            )
            .optional()
            .map(Option::flatten)
            .map_err(store_err)?;
        Ok(raw
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .filter(|v| v.is_object())
            .unwrap_or_else(|| serde_json::json!({})))
    }

    fn set_bank_config(&self, bank_id: &str, config: &serde_json::Value) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "UPDATE banks SET config_json = ?2 WHERE id = ?1",
            params![bank_id, config.to_string()],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn upsert_document(&self, doc: &Document) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO documents(bank_id, id, original_text, content_hash, tags_json, metadata_json, retain_params_json, created_at_ms, updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
             ON CONFLICT(bank_id, id) DO UPDATE SET original_text=excluded.original_text, content_hash=excluded.content_hash, tags_json=excluded.tags_json, metadata_json=excluded.metadata_json, retain_params_json=excluded.retain_params_json, updated_at_ms=excluded.updated_at_ms",
            params![
                doc.bank_id,
                doc.id,
                doc.original_text,
                doc.content_hash,
                tags_to_json(&doc.tags),
                metadata_to_json(&doc.metadata),
                doc.retain_params.to_string(),
                doc.created_at_ms,
                doc.updated_at_ms
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn get_document(&self, bank_id: &str, id: &str) -> Result<Option<Document>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT id, bank_id, original_text, content_hash, tags_json, metadata_json, retain_params_json, created_at_ms, updated_at_ms FROM documents WHERE bank_id = ?1 AND id = ?2",
            params![bank_id, id],
            Self::row_document,
        )
        .optional()
        .map_err(store_err)
    }

    fn list_documents(&self, bank_id: &str, limit: usize) -> Result<Vec<Document>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, original_text, content_hash, tags_json, metadata_json, retain_params_json, created_at_ms, updated_at_ms FROM documents WHERE bank_id = ?1 ORDER BY updated_at_ms DESC, id LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, limit], Self::row_document)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn count_document_memories(&self, bank_id: &str, id: &str) -> Result<usize> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE bank_id = ?1 AND document_id = ?2",
            params![bank_id, id],
            |r| r.get::<_, i64>(0),
        )
        .map(|n| n as usize)
        .map_err(store_err)
    }

    fn delete_document_memories(&self, bank_id: &str, id: &str) -> Result<(usize, Vec<String>)> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        let ids: Vec<String> = {
            let mut stmt = tx
                .prepare("SELECT id FROM memories WHERE bank_id = ?1 AND document_id = ?2")
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![bank_id, id], |r| r.get::<_, String>(0))
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        tx.execute(
            "DELETE FROM chunks WHERE bank_id = ?1 AND document_id = ?2",
            params![bank_id, id],
        )
        .map_err(store_err)?;
        if ids.is_empty() {
            tx.commit().map_err(store_err)?;
            return Ok((0, vec![]));
        }
        let gone: HashSet<&str> = ids.iter().map(String::as_str).collect();
        let mut affected: HashSet<String> = HashSet::new();
        for chunk in ids.chunks(400) {
            let ph = placeholders(chunk.len());
            {
                let mut stmt = tx
                    .prepare(&format!(
                        "SELECT DISTINCT entity_id FROM memory_entities WHERE memory_id IN ({ph})"
                    ))
                    .map_err(store_err)?;
                let rows = stmt
                    .query_map(params_from_iter(chunk.iter()), |r| r.get::<_, String>(0))
                    .map_err(store_err)?;
                for r in rows {
                    affected.insert(r.map_err(store_err)?);
                }
            }
            for sql in [
                format!("DELETE FROM memory_entities WHERE memory_id IN ({ph})"),
                format!("DELETE FROM edges WHERE a_id IN ({ph})"),
                format!("DELETE FROM edges WHERE b_id IN ({ph})"),
                format!("DELETE FROM texts_fts WHERE ref_id IN ({ph})"),
                format!("DELETE FROM memories WHERE id IN ({ph})"),
            ] {
                tx.execute(&sql, params_from_iter(chunk.iter()))
                    .map_err(store_err)?;
            }
        }
        self.vec_remove(&tx, &ids)?;
        // Observations must not keep citing facts that no longer exist.
        let observations: Vec<(String, String, String)> = {
            let mut stmt = tx
                .prepare("SELECT id, entity_id, evidence_json FROM observations WHERE bank_id = ?1")
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![bank_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        for (oid, entity_id, evidence_json) in observations {
            let evidence: Vec<Evidence> = serde_json::from_str(&evidence_json).unwrap_or_default();
            if !evidence
                .iter()
                .any(|ev| gone.contains(ev.memory_id.as_str()))
            {
                continue;
            }
            affected.insert(entity_id);
            let kept: Vec<Evidence> = evidence
                .into_iter()
                .filter(|ev| !gone.contains(ev.memory_id.as_str()))
                .collect();
            if kept.is_empty() {
                tx.execute("DELETE FROM observations WHERE id = ?1", params![oid])
                    .map_err(store_err)?;
                self.vec_remove(&tx, [&oid])?;
                tx.execute(
                    "DELETE FROM texts_fts WHERE bank_id = ?1 AND ref_id = ?2",
                    params![bank_id, oid],
                )
                .map_err(store_err)?;
            } else {
                tx.execute(
                    "UPDATE observations SET evidence_json = ?2, proof_count = ?3 WHERE id = ?1",
                    params![
                        oid,
                        serde_json::to_string(&kept).map_err(|e| Error::Store(e.to_string()))?,
                        kept.len() as i64
                    ],
                )
                .map_err(store_err)?;
            }
        }
        tx.commit().map_err(store_err)?;
        let mut affected: Vec<String> = affected.into_iter().collect();
        affected.sort();
        Ok((ids.len(), affected))
    }

    fn delete_document(&self, bank_id: &str, id: &str) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        for sql in [
            "DELETE FROM documents WHERE bank_id = ?1 AND id = ?2",
            "DELETE FROM attachments WHERE bank_id = ?1 AND document_id = ?2",
        ] {
            tx.execute(sql, params![bank_id, id]).map_err(store_err)?;
        }
        tx.commit().map_err(store_err)?;
        Ok(())
    }

    fn put_attachment(&self, a: &rustsight_core::files::Attachment) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT OR REPLACE INTO attachments(bank_id, document_id, id, filename, content_type, size, sha256, bytes, created_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                a.bank_id,
                a.document_id,
                a.id,
                a.filename,
                a.content_type,
                a.bytes.len() as i64,
                a.sha256,
                a.bytes,
                a.created_at_ms
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn get_attachment(
        &self,
        bank_id: &str,
        id: &str,
    ) -> Result<Option<rustsight_core::files::Attachment>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT bank_id, document_id, id, filename, content_type, sha256, bytes, created_at_ms
             FROM attachments WHERE bank_id = ?1 AND id = ?2
             ORDER BY created_at_ms DESC LIMIT 1",
            params![bank_id, id],
            |r| {
                Ok(rustsight_core::files::Attachment {
                    bank_id: r.get(0)?,
                    document_id: r.get(1)?,
                    id: r.get(2)?,
                    filename: r.get(3)?,
                    content_type: r.get(4)?,
                    sha256: r.get(5)?,
                    bytes: r.get(6)?,
                    created_at_ms: r.get(7)?,
                })
            },
        )
        .optional()
        .map_err(store_err)
    }

    fn prune_document_attachments(
        &self,
        bank_id: &str,
        document_id: &str,
        keep: &str,
    ) -> Result<usize> {
        let conn = self.lock_conn();
        conn.execute(
            "DELETE FROM attachments WHERE bank_id = ?1 AND document_id = ?2 AND id <> ?3",
            params![bank_id, document_id, keep],
        )
        .map_err(store_err)
    }

    fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        for c in chunks {
            tx.execute(
                "INSERT OR REPLACE INTO chunks(bank_id, id, document_id, chunk_index, text)
                 VALUES (?1,?2,?3,?4,?5)",
                params![c.bank_id, c.id, c.document_id, c.chunk_index as i64, c.text],
            )
            .map_err(store_err)?;
        }
        tx.commit().map_err(store_err)
    }

    fn next_chunk_index(&self, bank_id: &str, document_id: &str) -> Result<usize> {
        let conn = self.lock_conn();
        let next: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(chunk_index) + 1, 0) FROM chunks
                 WHERE bank_id = ?1 AND document_id = ?2",
                params![bank_id, document_id],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        Ok(next as usize)
    }

    fn get_chunks(&self, bank_id: &str, ids: &[String]) -> Result<Vec<Chunk>> {
        let conn = self.lock_conn();
        let mut out = Vec::with_capacity(ids.len());
        for batch in ids.chunks(400) {
            let sql = format!(
                "SELECT c.id, c.document_id, c.chunk_index, c.text, d.tags_json
                 FROM chunks c JOIN documents d ON d.bank_id = c.bank_id AND d.id = c.document_id
                 WHERE c.bank_id = ? AND c.id IN ({})",
                placeholders(batch.len())
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(
                    params_from_iter(
                        std::iter::once(bank_id.to_string()).chain(batch.iter().cloned()),
                    ),
                    |r| {
                        Ok(Chunk {
                            id: r.get(0)?,
                            bank_id: bank_id.to_string(),
                            document_id: r.get(1)?,
                            chunk_index: r.get::<_, i64>(2)? as usize,
                            text: r.get(3)?,
                            document_tags: parse_tags(r.get::<_, Option<String>>(4)?),
                        })
                    },
                )
                .map_err(store_err)?;
            for row in rows {
                out.push(row.map_err(store_err)?);
            }
        }
        Ok(out)
    }

    fn memory_chunk_ids(
        &self,
        bank_id: &str,
        memory_ids: &[MemoryId],
    ) -> Result<HashMap<String, String>> {
        let conn = self.lock_conn();
        let mut out = HashMap::with_capacity(memory_ids.len());
        for batch in memory_ids.chunks(400) {
            let sql = format!(
                "SELECT id, chunk_id FROM memories
                 WHERE bank_id = ? AND chunk_id IS NOT NULL AND id IN ({})",
                placeholders(batch.len())
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(
                    params_from_iter(
                        std::iter::once(bank_id.to_string()).chain(batch.iter().cloned()),
                    ),
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                )
                .map_err(store_err)?;
            for row in rows {
                let (memory, chunk) = row.map_err(store_err)?;
                out.insert(memory, chunk);
            }
        }
        Ok(out)
    }

    fn delete_bank(&self, bank_id: &str) -> Result<BankDeletion> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        for sql in [
            "DELETE FROM operations WHERE bank_id = ?1",
            "DELETE FROM webhooks WHERE bank_id = ?1",
            "DELETE FROM webhook_deliveries WHERE bank_id = ?1",
            "DELETE FROM attachments WHERE bank_id = ?1",
            "DELETE FROM directives WHERE bank_id = ?1",
        ] {
            tx.execute(sql, params![bank_id]).map_err(store_err)?;
        }
        let exists = tx
            .prepare("SELECT 1 FROM banks WHERE id = ?1")
            .and_then(|mut st| st.exists(params![bank_id]))
            .map_err(store_err)?;
        if !exists {
            // Every write path creates the bank row first, so there is
            // nothing else to find (and no full FTS scan to pay for).
            tx.commit().map_err(store_err)?;
            return Ok(BankDeletion::default());
        }
        let count = |table: &str| -> Result<usize> {
            tx.query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE bank_id = ?1"),
                params![bank_id],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n as usize)
            .map_err(store_err)
        };
        let deleted = BankDeletion {
            memories: count("memories")?,
            entities: count("entities")?,
            documents: count("documents")?,
        };
        for sql in [
            "DELETE FROM memory_entities WHERE memory_id IN (SELECT id FROM memories WHERE bank_id = ?1)",
            "DELETE FROM opinion_entities WHERE opinion_id IN (SELECT id FROM opinions WHERE bank_id = ?1)",
            "DELETE FROM edges WHERE bank_id = ?1",
            "DELETE FROM texts_fts WHERE bank_id = ?1",
            "DELETE FROM dirty_entities WHERE bank_id = ?1",
            "DELETE FROM observations WHERE bank_id = ?1",
            "DELETE FROM opinions WHERE bank_id = ?1",
            "DELETE FROM chunks WHERE bank_id = ?1",
            "DELETE FROM documents WHERE bank_id = ?1",
            "DELETE FROM mental_models WHERE bank_id = ?1",
            "DELETE FROM memories WHERE bank_id = ?1",
            "DELETE FROM entity_aliases WHERE bank_id = ?1",
            "DELETE FROM entities WHERE bank_id = ?1",
            "DELETE FROM banks WHERE id = ?1",
        ] {
            tx.execute(sql, params![bank_id]).map_err(store_err)?;
        }
        // Without the extension the vec0 rows cannot be touched (and the
        // marker is already gone); a later reindex clears them by partition.
        if self.vec_version.is_some() {
            vector::remove_bank(&tx, bank_id).map_err(store_err)?;
        }
        tx.commit().map_err(store_err)?;
        Ok(deleted)
    }

    fn tag_index(&self, bank_id: &str) -> Result<HashMap<String, Vec<String>>> {
        let conn = self.lock_conn();
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT id, tags_json FROM memories WHERE bank_id = ?1 AND tags_json IS NOT NULL AND tags_json != '[]'")
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![bank_id], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
                })
                .map_err(store_err)?;
            for r in rows {
                let (id, raw) = r.map_err(store_err)?;
                let tags = parse_tags(raw);
                if !tags.is_empty() {
                    out.insert(id, tags);
                }
            }
        }
        if out.is_empty() {
            return Ok(out);
        }
        let mut stmt = conn
            .prepare("SELECT id, evidence_json FROM observations WHERE bank_id = ?1")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(store_err)?;
        let mut derived: Vec<(String, Vec<String>)> = vec![];
        for r in rows {
            let (id, evidence_json) = r.map_err(store_err)?;
            let evidence: Vec<Evidence> = serde_json::from_str(&evidence_json).unwrap_or_default();
            let mut tags: Vec<String> = vec![];
            for ev in &evidence {
                for t in out.get(&ev.memory_id).into_iter().flatten() {
                    if !tags.contains(t) {
                        tags.push(t.clone());
                    }
                }
            }
            if !tags.is_empty() {
                derived.push((id, tags));
            }
        }
        out.extend(derived);
        Ok(out)
    }

    fn upsert_webhook(&self, w: &Webhook) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT OR REPLACE INTO webhooks(id, bank_id, url, secret, event_types_json, enabled, http_config_json, created_at_ms, updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                w.id,
                w.bank_id,
                w.url,
                w.secret,
                serde_json::to_string(&w.event_types).map_err(json_err)?,
                w.enabled,
                serde_json::to_string(&w.http_config).map_err(json_err)?,
                w.created_at_ms,
                w.updated_at_ms
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn get_webhook(&self, bank_id: &str, id: &str) -> Result<Option<Webhook>> {
        let conn = self.lock_conn();
        conn.query_row(
            &format!("SELECT {WEBHOOK_COLS} FROM webhooks WHERE bank_id = ?1 AND id = ?2"),
            params![bank_id, id],
            row_webhook,
        )
        .optional()
        .map_err(store_err)
    }

    fn list_webhooks(
        &self,
        bank_id: &str,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Webhook>, usize)> {
        let conn = self.lock_conn();
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM webhooks WHERE bank_id = ?1",
                params![bank_id],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {WEBHOOK_COLS} FROM webhooks WHERE bank_id = ?1
                 ORDER BY created_at_ms, id LIMIT ?2 OFFSET ?3"
            ))
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id, limit as i64, offset as i64], row_webhook)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok((out, total as usize))
    }

    fn delete_webhook(&self, bank_id: &str, id: &str) -> Result<bool> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        let n = tx
            .execute(
                "DELETE FROM webhooks WHERE bank_id = ?1 AND id = ?2",
                params![bank_id, id],
            )
            .map_err(store_err)?;
        // Nothing left to deliver to; finished rows stay as history.
        tx.execute(
            "DELETE FROM webhook_deliveries WHERE bank_id = ?1 AND webhook_id = ?2 AND status = 'pending'",
            params![bank_id, id],
        )
        .map_err(store_err)?;
        tx.commit().map_err(store_err)?;
        Ok(n > 0)
    }

    fn upsert_webhook_delivery(&self, d: &WebhookDelivery) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO webhook_deliveries({DELIVERY_COLS})
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)"
            ),
            params![
                d.id,
                d.webhook_id,
                d.bank_id,
                d.event_id,
                d.event_type,
                d.url,
                d.secret,
                serde_json::to_string(&d.http_config).map_err(json_err)?,
                d.payload,
                d.status,
                d.retry_count,
                d.next_attempt_at_ms,
                d.last_error,
                d.last_response_status,
                d.last_response_body,
                d.last_attempt_at_ms,
                d.created_at_ms,
                d.updated_at_ms
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn due_webhook_deliveries(&self, now_ms: i64, limit: usize) -> Result<Vec<WebhookDelivery>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {DELIVERY_COLS} FROM webhook_deliveries
                 WHERE status = 'pending' AND next_attempt_at_ms <= ?1
                 ORDER BY next_attempt_at_ms, created_at_ms, id LIMIT ?2"
            ))
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![now_ms, limit as i64], row_delivery)
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn next_webhook_delivery_at(&self) -> Result<Option<i64>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT MIN(next_attempt_at_ms) FROM webhook_deliveries WHERE status = 'pending'",
            [],
            |r| r.get(0),
        )
        .map_err(store_err)
    }

    fn list_webhook_deliveries(
        &self,
        bank_id: &str,
        webhook_id: &str,
        limit: usize,
        before: Option<(i64, String)>,
    ) -> Result<Vec<WebhookDelivery>> {
        let conn = self.lock_conn();
        let (before_ms, before_id) = before.unwrap_or((i64::MAX, String::new()));
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {DELIVERY_COLS} FROM webhook_deliveries
                 WHERE bank_id = ?1 AND webhook_id = ?2
                   AND (created_at_ms < ?3 OR (created_at_ms = ?3 AND id < ?4))
                 ORDER BY created_at_ms DESC, id DESC LIMIT ?5"
            ))
            .map_err(store_err)?;
        // `id < ''` is never true, so the unbounded first page relies on
        // created_at_ms < i64::MAX alone.
        let rows = stmt
            .query_map(
                params![bank_id, webhook_id, before_ms, before_id, limit as i64],
                row_delivery,
            )
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn upsert_operation(&self, op: &Operation) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO operations(id, bank_id, kind, status, error, payload_json, created_at_ms, updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(id) DO UPDATE SET status=excluded.status, error=excluded.error, payload_json=excluded.payload_json, updated_at_ms=excluded.updated_at_ms",
            params![
                op.id,
                op.bank_id,
                op.kind,
                op.status,
                op.error,
                op.payload.to_string(),
                op.created_at_ms,
                op.updated_at_ms
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn get_operation(&self, id: &str) -> Result<Option<Operation>> {
        let conn = self.lock_conn();
        conn.query_row(
            "SELECT id, bank_id, kind, status, error, payload_json, created_at_ms, updated_at_ms FROM operations WHERE id = ?1",
            params![id],
            Self::row_operation,
        )
        .optional()
        .map_err(store_err)
    }

    fn pending_operations(&self) -> Result<Vec<Operation>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, kind, status, error, payload_json, created_at_ms, updated_at_ms FROM operations WHERE status = 'pending' ORDER BY created_at_ms, id")
            .map_err(store_err)?;
        let rows = stmt.query_map([], Self::row_operation).map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn list_operations(
        &self,
        bank_id: &str,
        status: Option<&str>,
        kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Operation>, usize)> {
        const FILTER: &str =
            "bank_id = ?1 AND (?2 IS NULL OR status = ?2) AND (?3 IS NULL OR kind = ?3)";
        let conn = self.lock_conn();
        let total: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM operations WHERE {FILTER}"),
                params![bank_id, status, kind],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT id, bank_id, kind, status, error, payload_json, created_at_ms, updated_at_ms
                 FROM operations WHERE {FILTER}
                 ORDER BY created_at_ms DESC, id LIMIT ?4 OFFSET ?5"
            ))
            .map_err(store_err)?;
        let rows = stmt
            .query_map(
                params![bank_id, status, kind, limit as i64, offset as i64],
                Self::row_operation,
            )
            .map_err(store_err)?;
        let operations: Vec<Operation> =
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?;
        Ok((operations, total as usize))
    }

    fn upsert_mental_model(&self, m: &MentalModel) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT INTO mental_models(id, bank_id, question, path, content, last_refreshed_ms, created_at_ms, name, tags_json, max_tokens, trigger_json)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
             ON CONFLICT(id) DO UPDATE SET question=excluded.question, path=excluded.path, content=excluded.content, last_refreshed_ms=excluded.last_refreshed_ms,
               name=excluded.name, tags_json=excluded.tags_json, max_tokens=excluded.max_tokens, trigger_json=excluded.trigger_json",
            params![
                m.id,
                m.bank_id,
                m.question,
                m.path,
                m.content,
                m.last_refreshed_ms,
                m.created_at_ms,
                m.name,
                serde_json::to_string(&m.tags).map_err(json_err)?,
                m.max_tokens,
                m.trigger
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()
                    .map_err(json_err)?
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn list_mental_models(&self, bank_id: &str) -> Result<Vec<MentalModel>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, question, path, content, last_refreshed_ms, created_at_ms, name, tags_json, max_tokens, trigger_json FROM mental_models WHERE bank_id = ?1 ORDER BY created_at_ms")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok(MentalModel {
                    id: r.get(0)?,
                    bank_id: r.get(1)?,
                    question: r.get(2)?,
                    path: r.get(3)?,
                    content: r.get(4)?,
                    last_refreshed_ms: r.get(5)?,
                    created_at_ms: r.get(6)?,
                    name: r.get(7)?,
                    tags: json_col(r, 8)?,
                    max_tokens: r.get(9)?,
                    trigger: r
                        .get::<_, Option<String>>(10)?
                        .and_then(|raw| serde_json::from_str(&raw).ok()),
                })
            })
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn upsert_directive(&self, d: &BankDirective) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute(
            "INSERT OR REPLACE INTO directives(id, bank_id, name, content, priority, is_active, tags_json, created_at_ms, updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                d.id,
                d.bank_id,
                d.name,
                d.content,
                d.priority,
                d.is_active,
                serde_json::to_string(&d.tags).map_err(json_err)?,
                d.created_at_ms,
                d.updated_at_ms
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn list_directives(&self, bank_id: &str) -> Result<Vec<BankDirective>> {
        let conn = self.lock_conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, name, content, priority, is_active, tags_json, created_at_ms, updated_at_ms
                 FROM directives WHERE bank_id = ?1 ORDER BY priority DESC, created_at_ms, name",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![bank_id], |r| {
                Ok(BankDirective {
                    id: r.get(0)?,
                    bank_id: r.get(1)?,
                    name: r.get(2)?,
                    content: r.get(3)?,
                    priority: r.get(4)?,
                    is_active: r.get(5)?,
                    tags: json_col(r, 6)?,
                    created_at_ms: r.get(7)?,
                    updated_at_ms: r.get(8)?,
                })
            })
            .map_err(store_err)?;
        let mut out = vec![];
        for r in rows {
            out.push(r.map_err(store_err)?);
        }
        Ok(out)
    }

    fn delete_mental_model(&self, id: &str) -> Result<()> {
        let conn = self.lock_conn();
        conn.execute("DELETE FROM mental_models WHERE id = ?1", params![id])
            .map_err(store_err)?;
        Ok(())
    }
}

impl SqliteStore {
    fn row_document(r: &Row) -> rusqlite::Result<Document> {
        Ok(Document {
            id: r.get(0)?,
            bank_id: r.get(1)?,
            original_text: r.get(2)?,
            content_hash: r.get(3)?,
            tags: parse_tags(r.get::<_, Option<String>>(4)?),
            metadata: parse_metadata(r.get::<_, Option<String>>(5)?),
            retain_params: serde_json::from_str(&r.get::<_, String>(6)?)
                .unwrap_or(serde_json::Value::Null),
            created_at_ms: r.get(7)?,
            updated_at_ms: r.get(8)?,
        })
    }

    fn row_operation(r: &Row) -> rusqlite::Result<Operation> {
        Ok(Operation {
            id: r.get(0)?,
            bank_id: r.get(1)?,
            kind: r.get(2)?,
            status: r.get(3)?,
            error: r.get(4)?,
            payload: serde_json::from_str(&r.get::<_, String>(5)?)
                .unwrap_or(serde_json::Value::Null),
            created_at_ms: r.get(6)?,
            updated_at_ms: r.get(7)?,
        })
    }

    fn row_observation(r: &Row) -> rusqlite::Result<Observation> {
        let evj: String = r.get(4)?;
        let hj: String = r.get(6)?;
        Ok(Observation {
            id: r.get(0)?,
            bank_id: r.get(1)?,
            entity_id: r.get(2)?,
            text: r.get(3)?,
            evidence: serde_json::from_str(&evj).unwrap_or_default(),
            proof_count: r.get::<_, i64>(5)? as usize,
            history: serde_json::from_str(&hj).unwrap_or_default(),
            embedding: r.get::<_, Option<Vec<u8>>>(7)?.map(|b| blob_to_emb(&b)),
            updated_at_ms: r.get(8)?,
            created_at_ms: r.get(9)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SqliteStore {
        SqliteStore::open_in_memory().unwrap()
    }

    #[test]
    fn store_uses_injected_clock_for_write_timestamps() {
        use rustsight_core::FixedClock;
        let s = SqliteStore::open_in_memory_with_clock(Arc::new(FixedClock::new(1234))).unwrap();
        let b = Bank {
            id: "b1".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        };
        s.ensure_bank(&b).unwrap();
        s.insert_memories("b1", &[new_mem("hello", 1234)]).unwrap();
        let conn = s.lock_conn();
        let bank_created: i64 = conn
            .query_row("SELECT created_at_ms FROM banks WHERE id = 'b1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let mem_created: i64 = conn
            .query_row("SELECT created_at_ms FROM memories LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        drop(conn);
        assert_eq!(bank_created, 1234);
        assert_eq!(mem_created, 1234);
    }

    fn new_mem(text: &str, t: i64) -> NewMemoryRecord {
        NewMemoryRecord {
            kind: MemoryKind::World,
            text: text.into(),
            confidence: None,
            occurred_start_ms: Some(t),
            occurred_end_ms: None,
            mentioned_at_ms: t,
            embedding: vec![1.0, 0.0],
            source: None,
            context: None,
            document_id: None,
            tags: vec![],
            metadata: None,
            chunk_id: None,
        }
    }

    #[test]
    fn bank_roundtrip() {
        let s = store();
        let b = Bank {
            id: "b1".into(),
            mission: Some("test".into()),
            directives: None,
            background: None,
            disposition: Disposition::default(),
        };
        s.ensure_bank(&b).unwrap();
        assert!(s.get_bank("b1").unwrap().is_some());
        let mut b2 = b.clone();
        b2.mission = Some("updated".into());
        s.update_bank(&b2).unwrap();
        assert_eq!(
            s.get_bank("b1").unwrap().unwrap().mission.as_deref(),
            Some("updated")
        );
        assert_eq!(s.list_banks().unwrap().len(), 1);
    }

    #[test]
    fn insert_and_bm25() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let mems = s
            .insert_memories(
                "b",
                &[
                    new_mem("Alice works at Google as a software engineer", 100),
                    new_mem("Bob enjoys hiking in Yosemite on weekends", 100),
                ],
            )
            .unwrap();
        assert_eq!(mems.len(), 2);
        let hits = s
            .search_bm25("b", &["yosemite".to_string(), "hiking".to_string()], 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, mems[1].id);
        assert_eq!(s.all_embeddings("b").unwrap().len(), 2);
    }

    #[test]
    fn time_window() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let may = 1_683_648_000_000i64;
        s.insert_memories("b", &[new_mem("trip to Rome", may)])
            .unwrap();
        s.insert_memories("b", &[new_mem("later party", 1_800_000_000_000)])
            .unwrap();
        let w = s
            .memories_in_window("b", may - 1000, may + 1000, 10)
            .unwrap();
        assert_eq!(w.len(), 1);
        assert!(w[0].text.contains("Rome"));
    }

    #[test]
    fn graph_neighbors() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let m = s
            .insert_memories("b", &[new_mem("a", 1), new_mem("b", 2), new_mem("c", 3)])
            .unwrap();
        s.add_edges(&[
            Edge {
                bank_id: "b".into(),
                a_id: m[0].id.clone(),
                b_id: m[1].id.clone(),
                kind: EdgeKind::Entity,
                label: None,
                weight: 1.0,
            },
            Edge {
                bank_id: "b".into(),
                a_id: m[1].id.clone(),
                b_id: m[2].id.clone(),
                kind: EdgeKind::Semantic,
                label: None,
                weight: 0.9,
            },
        ])
        .unwrap();
        let n = s.neighbors("b", &[m[0].id.clone()], 2, 10).unwrap();
        assert!(n.contains(&m[1].id));
        assert!(n.contains(&m[2].id));
    }

    #[test]
    fn neighbors_and_refs_are_bank_isolated() {
        let s = store();
        for id in ["a", "b"] {
            s.ensure_bank(&Bank {
                id: id.into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            })
            .unwrap();
        }
        let ma = s
            .insert_memories("a", &[new_mem("fact a1", 1), new_mem("fact a2", 2)])
            .unwrap();
        s.insert_memories("b", &[new_mem("fact b1", 1)]).unwrap();
        s.add_edges(&[Edge {
            bank_id: "a".into(),
            a_id: ma[0].id.clone(),
            b_id: ma[1].id.clone(),
            kind: EdgeKind::Entity,
            label: None,
            weight: 1.0,
        }])
        .unwrap();

        // graph expansion must not cross banks: bank b sees only its seed
        let n = s.neighbors("b", &[ma[0].id.clone()], 2, 10).unwrap();
        assert_eq!(
            n,
            vec![ma[0].id.clone()],
            "bank b must not expand into bank a's edges"
        );

        // refs must not cross banks either
        assert!(s.get_refs("b", &[ma[0].id.clone()]).unwrap().is_empty());
        // sanity: the same lookups under bank a still work
        let refs = s.get_refs("a", &[ma[0].id.clone()]).unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].text, "fact a1");
        let n = s.neighbors("a", &[ma[0].id.clone()], 2, 10).unwrap();
        assert!(n.contains(&ma[1].id));
    }

    #[test]
    fn dirty_cycle_and_observations() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let ent = s
            .create_entity(&NewEntity {
                bank_id: "b".into(),
                name: "Alice".into(),
                etype: EntityType::Person,
            })
            .unwrap();
        s.mark_entities_dirty("b", std::slice::from_ref(&ent.id), "new_facts")
            .unwrap();
        let d = s.list_dirty_entities("b", 10).unwrap();
        assert_eq!(d, vec![ent.id.clone()]);
        let o = Observation {
            id: "o1".into(),
            bank_id: "b".into(),
            entity_id: ent.id.clone(),
            text: "Alice works at Google".into(),
            evidence: vec![],
            proof_count: 0,
            history: vec![],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: Some(vec![0.5, 0.5]),
        };
        s.upsert_observation(&o).unwrap();
        assert_eq!(s.observations_for_entity(&ent.id).unwrap().len(), 1);
        assert_eq!(s.observations_for_bank("b").unwrap().len(), 1);
        let hits = s.search_bm25("b", &["google".into()], 5).unwrap();
        assert_eq!(hits.len(), 1);
        s.clear_dirty_entities("b", &d).unwrap();
        assert!(s.list_dirty_entities("b", 10).unwrap().is_empty());
    }

    #[test]
    fn banks_with_dirty_lists_distinct_banks() {
        let s = store();
        for id in ["a", "b", "c"] {
            s.ensure_bank(&Bank {
                id: id.into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            })
            .unwrap();
        }
        assert!(s.banks_with_dirty().unwrap().is_empty());
        for bank in ["a", "a", "b"] {
            let ent = s
                .create_entity(&NewEntity {
                    bank_id: bank.into(),
                    name: "Alice".into(),
                    etype: EntityType::Person,
                })
                .unwrap();
            s.mark_entities_dirty(bank, std::slice::from_ref(&ent.id), "new_facts")
                .unwrap();
        }
        assert_eq!(
            s.banks_with_dirty().unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        let dirty_a = s.list_dirty_entities("a", 10).unwrap();
        s.clear_dirty_entities("a", &dirty_a).unwrap();
        assert_eq!(s.banks_with_dirty().unwrap(), vec!["b".to_string()]);
    }

    #[test]
    fn refs_preserve_order_and_kinds() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let m = s
            .insert_memories("b", &[new_mem("one", 1), new_mem("two", 2)])
            .unwrap();
        let o = Observation {
            id: "obs".into(),
            bank_id: "b".into(),
            entity_id: "e".into(),
            text: "an observation".into(),
            evidence: vec![],
            proof_count: 0,
            history: vec![],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: None,
        };
        s.upsert_observation(&o).unwrap();
        let ids = vec![o.id.clone(), m[1].id.clone(), m[0].id.clone()];
        let refs = s.get_refs("b", &ids).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].kind, MemoryKind::Observation);
        assert_eq!(refs[1].text, "two");
        assert_eq!(refs[2].text, "one");
    }

    #[test]
    fn observation_fts_upsert_is_atomic_dedup() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let o = Observation {
            id: "o1".into(),
            bank_id: "b".into(),
            entity_id: "e".into(),
            text: "Alice works at Google".into(),
            evidence: vec![],
            proof_count: 0,
            history: vec![],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: None,
        };
        s.upsert_observation(&o).unwrap();
        let mut updated = o.clone();
        updated.text = "Alice works at Amazon".into();
        updated.updated_at_ms = 2;
        s.upsert_observation(&updated).unwrap();

        assert!(
            s.search_bm25("b", &["google".into()], 5)
                .unwrap()
                .is_empty(),
            "stale text must be deleted from the fts index"
        );
        let hits = s.search_bm25("b", &["alice".into()], 5).unwrap();
        assert_eq!(hits.len(), 1, "re-upsert must not duplicate fts rows");
        assert_eq!(hits[0].0, "o1");
        assert_eq!(hits[0].1, MemoryKind::Observation);
    }

    #[test]
    fn opinion_fts_upsert_is_atomic_dedup() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let op = Opinion {
            id: "op1".into(),
            bank_id: "b".into(),
            text: "Bob is enthusiastic about rust".into(),
            confidence: 0.7,
            evidence: vec![],
            history: vec![],
            entity_ids: vec!["e1".into(), "e2".into()],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: None,
        };
        s.upsert_opinion(&op).unwrap();
        let mut updated = op.clone();
        updated.text = "Bob is enthusiastic about zig".into();
        updated.updated_at_ms = 2;
        s.upsert_opinion(&updated).unwrap();

        assert!(
            s.search_bm25("b", &["rust".into()], 5).unwrap().is_empty(),
            "stale text must be deleted from the fts index"
        );
        let hits = s.search_bm25("b", &["zig".into()], 5).unwrap();
        assert_eq!(hits.len(), 1, "re-upsert must not duplicate fts rows");
        assert_eq!(hits[0].0, "op1");
        assert_eq!(hits[0].1, MemoryKind::Opinion);

        let listed = s.list_opinions("b").unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(
            listed[0].entity_ids.len(),
            2,
            "entity links survive re-upsert"
        );
        assert_eq!(listed[0].text, "Bob is enthusiastic about zig");
    }

    #[test]
    fn set_observation_embedding_writes_by_id() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let o = Observation {
            id: "o1".into(),
            bank_id: "b".into(),
            entity_id: "e".into(),
            text: "Alice works at Google".into(),
            evidence: vec![],
            proof_count: 0,
            history: vec![],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: None,
        };
        s.upsert_observation(&o).unwrap();

        s.set_observation_embedding("o1", Some(vec![0.25, 0.75]))
            .unwrap();
        let obs = s.observations_for_entity("e").unwrap();
        assert_eq!(obs[0].embedding.as_deref(), Some(&[0.25f32, 0.75][..]));

        // clearing works too
        s.set_observation_embedding("o1", None).unwrap();
        assert!(s.observations_for_entity("e").unwrap()[0]
            .embedding
            .is_none());

        // unknown ids are a no-op, not an error
        s.set_observation_embedding("missing", Some(vec![1.0]))
            .unwrap();
    }

    #[test]
    fn list_opinions_batches_entity_links() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let mk = |id: &str, eids: &[&str]| Opinion {
            id: id.into(),
            bank_id: "b".into(),
            text: format!("opinion {id}"),
            confidence: 0.5,
            evidence: vec![],
            history: vec![],
            entity_ids: eids.iter().map(|e| e.to_string()).collect(),
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: None,
        };
        s.upsert_opinion(&mk("op1", &["e1", "e2"])).unwrap();
        s.upsert_opinion(&mk("op2", &["e2"])).unwrap();
        s.upsert_opinion(&mk("op3", &[])).unwrap();

        let listed = s.list_opinions("b").unwrap();
        assert_eq!(listed.len(), 3);
        let by = |id: &str| listed.iter().find(|o| o.id == id).unwrap();
        assert_eq!(
            by("op1").entity_ids,
            vec!["e1".to_string(), "e2".to_string()]
        );
        assert_eq!(by("op2").entity_ids, vec!["e2".to_string()]);
        assert!(by("op3").entity_ids.is_empty());
    }

    /// CR-14: the batched entity queries return the same results as their
    /// per-id counterparts on the same fixture.
    #[test]
    fn batched_entity_queries_match_per_id() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let alice = s
            .create_entity(&NewEntity {
                bank_id: "b".into(),
                name: "Alice".into(),
                etype: EntityType::Person,
            })
            .unwrap();
        let bob = s
            .create_entity(&NewEntity {
                bank_id: "b".into(),
                name: "Bob".into(),
                etype: EntityType::Person,
            })
            .unwrap();

        let m0 = s
            .insert_memories("b", &[new_mem("Alice met Bob", 10)])
            .unwrap();
        let m1 = s
            .insert_memories("b", &[new_mem("Alice fact one", 20)])
            .unwrap();
        let m2 = s
            .insert_memories("b", &[new_mem("Alice fact two", 30)])
            .unwrap();
        let m3 = s
            .insert_memories("b", &[new_mem("Alice fact three", 40)])
            .unwrap();
        s.link_memory_entities(
            &m0[0].id,
            &[
                (alice.id.clone(), "Alice".into()),
                (bob.id.clone(), "Bob".into()),
            ],
            10,
        )
        .unwrap();
        for (i, m) in [&m1, &m2, &m3].iter().enumerate() {
            s.link_memory_entities(
                &m[0].id,
                &[(alice.id.clone(), "Alice".into())],
                20 + i as i64,
            )
            .unwrap();
        }

        let texts = |v: &[Memory]| v.iter().map(|m| m.text.clone()).collect::<Vec<_>>();
        let batch = s
            .facts_for_entities(&[alice.id.clone(), bob.id.clone()], 2)
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(
            texts(&batch[&alice.id]),
            texts(&s.facts_for_entity(&alice.id, 2).unwrap()),
            "batch must match the per-id query (newest first, per-entity limit)"
        );
        assert_eq!(batch[&alice.id][0].text, "Alice fact three");
        assert_eq!(batch[&alice.id][1].text, "Alice fact two");
        assert_eq!(
            texts(&batch[&bob.id]),
            vec!["Alice met Bob".to_string()],
            "bob only co-occurs via m0"
        );

        let co = s
            .coentities_batch(&[alice.id.clone(), bob.id.clone()], 50)
            .unwrap();
        assert_eq!(co[&alice.id], s.coentities(&alice.id, 50).unwrap());
        assert_eq!(co[&bob.id], s.coentities(&bob.id, 50).unwrap());
        assert_eq!(co[&alice.id], vec![bob.id.clone()]);
        assert_eq!(co[&bob.id], vec![alice.id.clone()]);

        // empty input and unknown ids are well-behaved
        assert!(s.facts_for_entities(&[], 5).unwrap().is_empty());
        assert!(s.coentities_batch(&[], 50).unwrap().is_empty());
        assert!(s
            .facts_for_entities(&["ghost".into()], 5)
            .unwrap()
            .is_empty());
        assert!(s
            .coentities_batch(&["ghost".into()], 50)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn migrates_databases_created_before_document_columns() {
        // The schema as it shipped before documents/tags/metadata existed.
        let dir = std::env::temp_dir().join(format!("rustsight-migrate-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE banks(id TEXT PRIMARY KEY, mission TEXT, directives TEXT, background TEXT,
                   skepticism INTEGER NOT NULL DEFAULT 3, literalism INTEGER NOT NULL DEFAULT 3,
                   empathy INTEGER NOT NULL DEFAULT 3, bias_strength REAL NOT NULL DEFAULT 0.5,
                   created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL);
                 CREATE TABLE memories(id TEXT PRIMARY KEY, bank_id TEXT NOT NULL, kind TEXT NOT NULL,
                   text TEXT NOT NULL, confidence REAL, occurred_start_ms INTEGER, occurred_end_ms INTEGER,
                   mentioned_at_ms INTEGER NOT NULL, embedding BLOB, source TEXT, context TEXT,
                   created_at_ms INTEGER NOT NULL);
                 INSERT INTO banks(id, created_at_ms, updated_at_ms) VALUES ('b', 1, 1);
                 INSERT INTO memories(id, bank_id, kind, text, mentioned_at_ms, created_at_ms)
                   VALUES ('m1', 'b', 'world', 'an old fact', 5, 5);
                 CREATE TABLE mental_models(id TEXT PRIMARY KEY, bank_id TEXT NOT NULL,
                   question TEXT NOT NULL, path TEXT, content TEXT NOT NULL DEFAULT '',
                   last_refreshed_ms INTEGER, created_at_ms INTEGER NOT NULL);
                 INSERT INTO mental_models(id, bank_id, question, content, created_at_ms)
                   VALUES ('mm', 'b', 'What matters?', 'old content', 1);",
            )
            .unwrap();
        }
        // Opening twice proves the migration is idempotent.
        for _ in 0..2 {
            let s = SqliteStore::open(&path).unwrap();
            let old = s.get_memory(&"m1".to_string()).unwrap().unwrap();
            assert_eq!(old.text, "an old fact");
            assert!(old.tags.is_empty() && old.document_id.is_none() && old.metadata.is_none());
            assert_eq!(s.bank_config("b").unwrap(), serde_json::json!({}));
            assert!(s.memory_chunk_ids("b", &["m1".into()]).unwrap().is_empty());
            let models = s.list_mental_models("b").unwrap();
            assert_eq!(models[0].content, "old content");
            assert!(models[0].name.is_none() && models[0].tags.is_empty());
            assert!(models[0].max_tokens.is_none() && models[0].trigger.is_none());
        }
        let s = SqliteStore::open(&path).unwrap();
        let mut rec = new_mem("a new fact", 9);
        rec.document_id = Some("d".into());
        rec.tags = vec!["t".into()];
        let stored = s.insert_memories("b", &[rec]).unwrap();
        let back = s.get_memory(&stored[0].id).unwrap().unwrap();
        assert_eq!(back.document_id.as_deref(), Some("d"));
        assert_eq!(back.tags, vec!["t".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reindexes_cjk_rows_written_before_bigram_indexing() {
        let dir = std::env::temp_dir().join(format!("rustsight-cjk-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            // An FTS table as it was: whole CJK runs indexed as one token.
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE VIRTUAL TABLE texts_fts USING fts5(
                   text, bank_id UNINDEXED, kind UNINDEXED, ref_id UNINDEXED,
                   tokenize='porter unicode61');
                 INSERT INTO texts_fts VALUES ('爱丽丝加入了谷歌', 'b', 'world', 'm1');
                 INSERT INTO texts_fts VALUES ('Bob moved to Berlin', 'b', 'world', 'm2');",
            )
            .unwrap();
            let before: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM texts_fts WHERE texts_fts MATCH '谷歌'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(before, 0, "the old index cannot find a CJK word");
        }
        for _ in 0..2 {
            let s = SqliteStore::open(&path).unwrap();
            let hits = s
                .search_bm25("b", &rustsight_core::bm25_terms("谷歌"), 10)
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, "m1");
            let hits = s
                .search_bm25("b", &rustsight_core::bm25_terms("berlin"), 10)
                .unwrap();
            assert_eq!(hits.len(), 1, "non-CJK rows untouched");
            let conn = s.lock_conn();
            let text: String = conn
                .query_row("SELECT text FROM texts_fts WHERE ref_id = 'm1'", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(text, "爱丽 丽丝 丝加 加入 入了 了谷 谷歌", "rewritten once");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn document_delete_cascades_and_repairs_observations() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        let mut doomed = new_mem("doomed fact", 1);
        doomed.document_id = Some("d1".into());
        doomed.tags = vec!["t".into()];
        let mut kept = new_mem("kept fact", 2);
        kept.document_id = Some("d2".into());
        let mems = s.insert_memories("b", &[doomed, kept]).unwrap();
        let ent = s
            .create_entity(&NewEntity {
                bank_id: "b".into(),
                name: "Alice".into(),
                etype: EntityType::Person,
            })
            .unwrap();
        s.link_memory_entities(&mems[0].id, &[(ent.id.clone(), "Alice".into())], 1)
            .unwrap();
        let obs = |id: &str, evidence: Vec<&Memory>| Observation {
            id: id.into(),
            bank_id: "b".into(),
            entity_id: ent.id.clone(),
            text: format!("observation {id}"),
            evidence: evidence
                .into_iter()
                .map(|m| Evidence {
                    memory_id: m.id.clone(),
                    quote: m.text.clone(),
                })
                .collect(),
            proof_count: 2,
            history: vec![],
            updated_at_ms: 3,
            created_at_ms: 3,
            embedding: None,
        };
        s.upsert_observation(&obs("o-mixed", vec![&mems[0], &mems[1]]))
            .unwrap();
        s.upsert_observation(&obs("o-only", vec![&mems[0]]))
            .unwrap();

        // Observations inherit the union of their evidence facts' tags.
        let index = s.tag_index("b").unwrap();
        assert_eq!(index.get("o-mixed"), Some(&vec!["t".to_string()]));
        assert_eq!(index.get(&mems[1].id), None, "untagged items are absent");

        let (deleted, affected) = s.delete_document_memories("b", "d1").unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(affected, vec![ent.id.clone()]);
        assert!(s.get_memory(&mems[0].id).unwrap().is_none());
        assert!(s.get_memory(&mems[1].id).unwrap().is_some());
        assert!(s.memory_entity_ids(&mems[0].id).unwrap().is_empty());
        let left = s.observations_for_bank("b").unwrap();
        assert_eq!(left.len(), 1, "evidence-less observation removed: {left:?}");
        assert_eq!(left[0].id, "o-mixed");
        assert_eq!(left[0].evidence.len(), 1);
        assert_eq!(left[0].evidence[0].memory_id, mems[1].id);
        assert_eq!(left[0].proof_count, 1);
        // The deleted fact no longer matches a keyword search.
        assert!(s
            .search_bm25("b", &["doomed".into()], 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn documents_and_operations_roundtrip() {
        let s = store();
        let doc = Document {
            id: "tracker:config:water".into(),
            bank_id: "b".into(),
            original_text: "v1".into(),
            content_hash: "h1".into(),
            tags: vec!["t".into()],
            metadata: Some([("k".to_string(), "v".to_string())].into()),
            retain_params: serde_json::json!({"context": "c"}),
            created_at_ms: 10,
            updated_at_ms: 10,
        };
        s.upsert_document(&doc).unwrap();
        s.upsert_document(&Document {
            original_text: "v2".into(),
            created_at_ms: 99,
            updated_at_ms: 20,
            ..doc.clone()
        })
        .unwrap();
        let back = s
            .get_document("b", "tracker:config:water")
            .unwrap()
            .unwrap();
        assert_eq!(back.original_text, "v2");
        assert_eq!(back.created_at_ms, 10, "created_at survives an update");
        assert_eq!(back.updated_at_ms, 20);
        assert_eq!(back.metadata, doc.metadata);
        assert!(s
            .get_document("other", "tracker:config:water")
            .unwrap()
            .is_none());
        assert_eq!(s.list_documents("b", 10).unwrap().len(), 1);

        let mut op = Operation {
            id: "op1".into(),
            bank_id: "b".into(),
            kind: "retain".into(),
            status: "pending".into(),
            error: None,
            payload: serde_json::json!({"items": [{"content": "x"}]}),
            created_at_ms: 1,
            updated_at_ms: 1,
        };
        s.upsert_operation(&op).unwrap();
        assert_eq!(s.pending_operations().unwrap().len(), 1);
        op.status = "completed".into();
        s.upsert_operation(&op).unwrap();
        assert!(s.pending_operations().unwrap().is_empty());
        let back = s.get_operation("op1").unwrap().unwrap();
        assert_eq!(back.status, "completed");
        assert_eq!(back.payload["items"][0]["content"], "x");

        s.upsert_operation(&Operation {
            id: "op2".into(),
            status: "failed".into(),
            created_at_ms: 2,
            ..op.clone()
        })
        .unwrap();
        let (newest, total) = s.list_operations("b", None, None, 1, 0).unwrap();
        assert_eq!((newest[0].id.as_str(), total), ("op2", 2));
        let (rest, _) = s.list_operations("b", None, None, 5, 1).unwrap();
        assert_eq!(rest[0].id, "op1");
        let (failed, total) = s
            .list_operations("b", Some("failed"), Some("retain"), 5, 0)
            .unwrap();
        assert_eq!((failed.len(), total), (1, 1));
        assert_eq!(s.list_operations("x", None, None, 5, 0).unwrap().1, 0);
    }

    #[test]
    fn delete_bank_leaves_no_rows_behind_and_spares_other_banks() {
        let s = store();
        for bank in ["gone", "kept"] {
            s.ensure_bank(&Bank {
                id: bank.into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            })
            .unwrap();
            let entity = s
                .create_entity(&NewEntity {
                    bank_id: bank.into(),
                    name: format!("{bank} entity"),
                    etype: EntityType::Other,
                })
                .unwrap();
            let memory = s.insert_memories(bank, &[new_mem("a fact", 1)]).unwrap();
            s.link_memory_entities(&memory[0].id, &[(entity.id.clone(), "e".into())], 1)
                .unwrap();
            s.mark_entities_dirty(bank, &[entity.id], "new_facts")
                .unwrap();
            s.upsert_document(&Document {
                id: "d".into(),
                bank_id: bank.into(),
                original_text: "a fact".into(),
                content_hash: "h".into(),
                tags: vec![],
                metadata: None,
                retain_params: serde_json::json!({}),
                created_at_ms: 1,
                updated_at_ms: 1,
            })
            .unwrap();
            s.insert_chunks(&[Chunk {
                id: format!("{bank}_d_0"),
                bank_id: bank.into(),
                document_id: "d".into(),
                chunk_index: 0,
                text: "a fact".into(),
                document_tags: vec![],
            }])
            .unwrap();
        }
        let deleted = s.delete_bank("gone").unwrap();
        assert_eq!(
            (deleted.memories, deleted.entities, deleted.documents),
            (1, 1, 1)
        );
        let conn = s.lock_conn();
        for table in [
            "memories",
            "entities",
            "entity_aliases",
            "dirty_entities",
            "documents",
            "chunks",
            "texts_fts",
        ] {
            let count = |bank: &str| -> i64 {
                conn.query_row(
                    &format!("SELECT COUNT(*) FROM {table} WHERE bank_id = ?1"),
                    params![bank],
                    |r| r.get(0),
                )
                .unwrap()
            };
            assert_eq!(count("gone"), 0, "{table}");
            if table != "entity_aliases" {
                assert!(count("kept") > 0, "{table}");
            }
        }
        let links: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory_entities", [], |r| r.get(0))
            .unwrap();
        assert_eq!(links, 1);
        drop(conn);
        assert_eq!(s.delete_bank("gone").unwrap(), BankDeletion::default());
    }

    #[test]
    fn chunks_follow_their_document() {
        let s = store();
        s.ensure_bank(&Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
        s.upsert_document(&Document {
            id: "d".into(),
            bank_id: "b".into(),
            original_text: "text".into(),
            content_hash: "h".into(),
            tags: vec!["t".into()],
            metadata: None,
            retain_params: serde_json::json!({}),
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .unwrap();
        assert_eq!(s.next_chunk_index("b", "d").unwrap(), 0);
        s.insert_chunks(&[Chunk {
            id: "b_d_0".into(),
            bank_id: "b".into(),
            document_id: "d".into(),
            chunk_index: 0,
            text: "text".into(),
            document_tags: vec![],
        }])
        .unwrap();
        assert_eq!(s.next_chunk_index("b", "d").unwrap(), 1);
        let chunks = s.get_chunks("b", &["b_d_0".into()]).unwrap();
        assert_eq!(chunks[0].document_tags, vec!["t".to_string()]);
        s.delete_document_memories("b", "d").unwrap();
        assert!(s.get_chunks("b", &["b_d_0".into()]).unwrap().is_empty());
    }

    // ---- RS-7: sqlite-vec KNN index -------------------------------------

    fn vbank(s: &SqliteStore, id: &str) {
        s.ensure_bank(&Bank {
            id: id.into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
    }

    /// Deterministic, tie-free 4-d vectors.
    fn vvec(i: usize) -> Vec<f32> {
        let x = i as f32;
        vec![
            (x * 0.37).sin(),
            (x * 0.91).cos(),
            (x * 1.3 + 0.2).sin(),
            0.5 + x * 0.01,
        ]
    }

    fn vmem(i: usize, kind: MemoryKind, doc: Option<&str>) -> NewMemoryRecord {
        NewMemoryRecord {
            kind,
            embedding: vvec(i),
            document_id: doc.map(str::to_string),
            ..new_mem(&format!("m{i}"), i as i64)
        }
    }

    #[cfg(feature = "sqlite-vec")]
    fn vobs(id: &str, bank: &str, emb: Option<Vec<f32>>) -> Observation {
        Observation {
            id: id.into(),
            bank_id: bank.into(),
            entity_id: "e".into(),
            text: format!("obs {id}"),
            evidence: vec![],
            proof_count: 0,
            history: vec![],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: emb,
        }
    }

    const ALL: [MemoryKind; 4] = [
        MemoryKind::World,
        MemoryKind::Experience,
        MemoryKind::Observation,
        MemoryKind::Opinion,
    ];

    fn ids(hits: &[SemanticHit]) -> Vec<String> {
        hits.iter().map(|(id, _, _)| id.clone()).collect()
    }

    /// The index answer must equal the reference scan for every query,
    /// kind set, and admit predicate tried.
    fn assert_matches_scan(s: &SqliteStore, bank: &str) {
        let all = s.all_embeddings(bank).unwrap();
        let queries: Vec<Vec<f32>> = (100..106).map(vvec).collect();
        let qrefs: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
        let even = |id: &str| id.bytes().map(u32::from).sum::<u32>() % 2 == 0;
        let kind_sets: [&[MemoryKind]; 3] = [
            &ALL,
            &[MemoryKind::World, MemoryKind::Experience],
            &[MemoryKind::Observation],
        ];
        for kinds in kind_sets {
            for k in [1, 3, 50] {
                for admit in [None, Some(&even as &dyn Fn(&str) -> bool)] {
                    let got = s.search_semantic(bank, &qrefs, kinds, k, admit).unwrap();
                    let want = brute_force_semantic(&all, &qrefs, kinds, k, admit);
                    assert_eq!(got.len(), want.len());
                    for (g, w) in got.iter().zip(&want) {
                        assert_eq!(ids(g), ids(w), "kinds {kinds:?} k {k}");
                        for (a, b) in g.iter().zip(w) {
                            assert!((a.2 - b.2).abs() < 1e-4, "{a:?} vs {b:?}");
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "sqlite-vec")]
    #[test]
    fn vector_index_mirrors_every_embedding_write_and_delete() {
        let s = store();
        assert!(s.vector_index_version().unwrap().starts_with('v'));
        vbank(&s, "b");
        vbank(&s, "other");
        assert!(s.bank_vector_indexed("b").unwrap());
        let mut recs = vec![];
        for i in 0..30 {
            let kind = if i % 3 == 0 {
                MemoryKind::Experience
            } else {
                MemoryKind::World
            };
            recs.push(vmem(i, kind, Some(if i < 10 { "d1" } else { "d2" })));
        }
        let stored = s.insert_memories("b", &recs).unwrap();
        s.insert_memories("other", &[vmem(101, MemoryKind::World, None)])
            .unwrap();
        s.upsert_observation(&vobs("o1", "b", Some(vvec(40))))
            .unwrap();
        s.upsert_observation(&vobs("o2", "b", Some(vvec(41))))
            .unwrap();
        s.upsert_observation(&vobs("o3", "b", None)).unwrap();
        s.upsert_opinion(&Opinion {
            id: "p1".into(),
            bank_id: "b".into(),
            text: "op".into(),
            confidence: 0.5,
            evidence: vec![],
            history: vec![],
            entity_ids: vec![],
            updated_at_ms: 1,
            created_at_ms: 1,
            embedding: Some(vvec(42)),
        })
        .unwrap();
        assert_matches_scan(&s, "b");

        // Updates replace the mirrored vector; clearing one unindexes it.
        s.upsert_observation(&vobs("o1", "b", Some(vvec(102))))
            .unwrap();
        s.set_observation_embedding("o3", Some(vvec(43))).unwrap();
        s.set_observation_embedding("o2", None).unwrap();
        assert_matches_scan(&s, "b");

        // Deletes: one observation, a whole document's memories (with an
        // observation that loses all its evidence), then the other bank.
        s.delete_observation("b", "o3").unwrap();
        let mut citing = vobs("o4", "b", Some(vvec(44)));
        citing.evidence = vec![Evidence {
            memory_id: stored[0].id.clone(),
            quote: String::new(),
        }];
        s.upsert_observation(&citing).unwrap();
        s.delete_document_memories("b", "d1").unwrap();
        assert!(s
            .all_embeddings("b")
            .unwrap()
            .iter()
            .all(|(id, _, _)| id != "o4"));
        assert_matches_scan(&s, "b");

        s.delete_bank("other").unwrap();
        let conn = s.lock_conn();
        let left: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vec_rows WHERE bank_id = 'other'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let in_vec0: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vec_f32_4 WHERE bank_id = 'other'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let total_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_rows", [], |r| r.get(0))
            .unwrap();
        let total_vec0: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_f32_4", [], |r| r.get(0))
            .unwrap();
        drop(conn);
        assert_eq!((left, in_vec0), (0, 0));
        assert_eq!(total_rows as usize, s.all_embeddings("b").unwrap().len());
        assert_eq!(total_rows, total_vec0);
        assert!(!s.bank_vector_indexed("other").unwrap());
    }

    #[cfg(feature = "sqlite-vec")]
    #[test]
    fn vector_index_widens_fetch_until_a_strict_admit_is_filled() {
        let s = store();
        vbank(&s, "b");
        let recs: Vec<_> = (0..400).map(|i| vmem(i, MemoryKind::World, None)).collect();
        let stored = s.insert_memories("b", &recs).unwrap();
        // Admit only the 5 items farthest from the query: the first KNN
        // fetches (64, 256) cannot contain them.
        let q = vvec(7);
        let mut far: Vec<(f32, String)> = stored
            .iter()
            .zip(&recs)
            .map(|(m, r)| (rustsight_core::cosine(&q, &r.embedding), m.id.clone()))
            .collect();
        far.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let allowed: HashSet<String> = far.iter().take(5).map(|(_, id)| id.clone()).collect();
        let admit = |id: &str| allowed.contains(id);
        let got = s
            .search_semantic("b", &[&q], &ALL, 5, Some(&admit))
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(ids(&got).into_iter().collect::<HashSet<_>>(), allowed);
        assert_matches_scan(&s, "b");
    }

    #[cfg(feature = "sqlite-vec")]
    #[test]
    fn reindex_adopts_banks_written_without_the_index() {
        let dir = std::env::temp_dir().join(format!("rustsight-rs7-{}", uuid::Uuid::new_v4()));
        let path = dir.join("db.sqlite");
        let off = StoreOptions {
            vector_index: false,
        };
        {
            // An indexed store creates and marks the bank ...
            let s = SqliteStore::open(&path).unwrap();
            vbank(&s, "b");
            s.insert_memories("b", &[vmem(1, MemoryKind::World, None)])
                .unwrap();
            assert!(s.bank_vector_indexed("b").unwrap());
        }
        {
            // ... then a store without it writes and deletes unmirrored.
            let s = SqliteStore::open_with_options(&path, Arc::new(SystemClock), &off).unwrap();
            assert_eq!(s.vector_index_version(), None);
            assert!(!s.bank_vector_indexed("b").unwrap());
            let recs: Vec<_> = (2..20)
                .map(|i| vmem(i, MemoryKind::World, Some("d")))
                .collect();
            s.insert_memories("b", &recs).unwrap();
            s.insert_memories("b", &[vmem(30, MemoryKind::World, Some("gone"))])
                .unwrap();
            s.delete_document_memories("b", "gone").unwrap();
            assert_matches_scan(&s, "b");
            assert!(s.reindex_vectors(Some("b")).is_err());
        }
        let s = SqliteStore::open(&path).unwrap();
        // The stale mirror is not trusted: brute force until reindex.
        assert!(!s.bank_vector_indexed("b").unwrap());
        assert_matches_scan(&s, "b");
        let report = s.reindex_vectors(None).unwrap();
        assert_eq!(
            report,
            vec![ReindexReport {
                bank_id: "b".into(),
                indexed: 19,
                skipped: 0
            }]
        );
        assert!(s.bank_vector_indexed("b").unwrap());
        assert_matches_scan(&s, "b");
        let vec0: i64 = s
            .lock_conn()
            .query_row("SELECT COUNT(*) FROM vec_f32_4", [], |r| r.get(0))
            .unwrap();
        assert_eq!(vec0, 19, "orphaned rows cleared by the reindex");
        drop(s);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "sqlite-vec")]
    #[test]
    fn vectors_too_wide_for_vec0_fall_back_to_the_scan() {
        let s = store();
        vbank(&s, "b");
        let wide = vec![0.5; vector::MAX_DIM + 1];
        let rec = NewMemoryRecord {
            embedding: wide.clone(),
            ..new_mem("wide", 1)
        };
        s.insert_memories("b", &[rec]).unwrap();
        assert!(!s.bank_vector_indexed("b").unwrap());
        let hits = s.search_semantic("b", &[&wide], &ALL, 5, None).unwrap();
        assert_eq!(hits[0].len(), 1);
        let report = s.reindex_vectors(Some("b")).unwrap();
        assert_eq!(report[0].skipped, 1);
        assert!(!s.bank_vector_indexed("b").unwrap());
    }

    #[cfg(feature = "sqlite-vec")]
    #[test]
    fn indexed_bank_answers_from_vec0_not_the_scan() {
        let s = store();
        vbank(&s, "b");
        s.insert_memories("b", &[vmem(1, MemoryKind::World, None)])
            .unwrap();
        // A row only the index knows about proves the KNN path served it.
        vector::put(
            &s.lock_conn(),
            "b",
            MemoryKind::World,
            "ghost",
            Some(&vvec(2)),
        )
        .unwrap();
        let hits = s
            .search_semantic("b", &[&vvec(2)], &ALL, 5, None)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(hits[0].0, "ghost");
        assert!((hits[0].2 - 1.0).abs() < 1e-5);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn semantic_search_without_the_index_is_the_scan() {
        let s = SqliteStore::open_in_memory_with_options(
            Arc::new(SystemClock),
            &StoreOptions {
                vector_index: false,
            },
        )
        .unwrap();
        assert_eq!(s.vector_index_version(), None);
        vbank(&s, "b");
        assert!(!s.bank_vector_indexed("b").unwrap());
        let recs: Vec<_> = (0..12).map(|i| vmem(i, MemoryKind::World, None)).collect();
        s.insert_memories("b", &recs).unwrap();
        assert_matches_scan(&s, "b");
        let rows: i64 = s
            .lock_conn()
            .query_row("SELECT COUNT(*) FROM vec_rows", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0, "nothing mirrored without the extension");
    }
}
