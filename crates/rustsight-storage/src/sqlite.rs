use rusqlite::{params, params_from_iter, Connection, OptionalExtension, Row};
use rustsight_core::types::*;
use rustsight_core::{Error, MemoryStore, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

fn store_err(e: rusqlite::Error) -> Error {
    Error::Store(e.to_string())
}

pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
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

fn placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

impl SqliteStore {
    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(store_err)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(store_err)?;
        conn.pragma_update(None, "busy_timeout", 5000)
            .map_err(store_err)?;
        conn.execute_batch(SCHEMA).map_err(store_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Store(format!("create db dir: {e}")))?;
            }
        }
        let conn = Connection::open(path).map_err(store_err)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(store_err)?;
        Self::init(conn)
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

    fn fts_upsert(&self, bank_id: &str, kind: &str, ref_id: &str, text: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM texts_fts WHERE ref_id = ?1", params![ref_id])
            .map_err(store_err)?;
        conn.execute(
            "INSERT INTO texts_fts(text, bank_id, kind, ref_id) VALUES (?1, ?2, ?3, ?4)",
            params![text, bank_id, kind, ref_id],
        )
        .map_err(store_err)?;
        Ok(())
    }
}

impl MemoryStore for SqliteStore {
    fn ensure_bank(&self, bank: &Bank) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        conn.execute(
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
        Ok(())
    }

    fn get_bank(&self, bank_id: &str) -> Result<Option<Bank>> {
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
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
                chrono_now_ms()
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn list_banks(&self) -> Result<Vec<Bank>> {
        let conn = self.conn.lock().unwrap();
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
        let mut conn = self.conn.lock().unwrap();
        let now = chrono_now_ms();
        let mut out = vec![];
        let tx = conn.transaction().map_err(store_err)?;
        for item in items {
            let id = uuid::Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO memories(id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, embedding, source, context, created_at_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
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
                    now
                ],
            )
            .map_err(store_err)?;
            tx.execute(
                "INSERT INTO texts_fts(text, bank_id, kind, ref_id) VALUES (?1, ?2, ?3, ?4)",
                params![item.text, bank_id, item.kind.as_str(), id],
            )
            .map_err(store_err)?;
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
            });
        }
        tx.commit().map_err(store_err)?;
        Ok(out)
    }

    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context FROM memories WHERE id = ?1",
            params![id],
            Self::row_memory,
        )
        .optional()
        .map_err(store_err)
    }

    fn recent_memories(&self, bank_id: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context FROM memories WHERE bank_id=?1 ORDER BY mentioned_at_ms DESC, id LIMIT ?2",
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
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context FROM memories WHERE bank_id=?1 ORDER BY created_at_ms DESC, id LIMIT ?2",
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
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context FROM memories
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

    fn get_refs(&self, _bank_id: &str, ids: &[MemoryId]) -> Result<Vec<MemoryRef>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
        let mut map: HashMap<String, MemoryRef> = HashMap::new();
        for chunk in ids.chunks(400) {
            let ph = placeholders(chunk.len());
            let sql = format!(
                "SELECT id, kind, text, occurred_start_ms, occurred_end_ms, mentioned_at_ms FROM memories WHERE id IN ({ph})"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
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
                    })
                })
                .map_err(store_err)?;
            for row in rows {
                let m = row.map_err(store_err)?;
                map.insert(m.id.clone(), m);
            }

            let sql = format!(
                "SELECT id, text, entity_id, updated_at_ms FROM observations WHERE id IN ({ph})"
            );
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok(MemoryRef {
                        id: r.get(0)?,
                        kind: MemoryKind::Observation,
                        text: r.get(1)?,
                        score: 0.0,
                        occurred_start_ms: None,
                        occurred_end_ms: None,
                        mentioned_at_ms: Some(r.get(3)?),
                        entity_ids: vec![r.get(2)?],
                    })
                })
                .map_err(store_err)?;
            for row in rows {
                let m = row.map_err(store_err)?;
                map.insert(m.id.clone(), m);
            }

            let sql = format!("SELECT id, text, updated_at_ms FROM opinions WHERE id IN ({ph})");
            let mut stmt = conn.prepare(&sql).map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok(MemoryRef {
                        id: r.get(0)?,
                        kind: MemoryKind::Opinion,
                        text: r.get(1)?,
                        score: 0.0,
                        occurred_start_ms: None,
                        occurred_end_ms: None,
                        mentioned_at_ms: Some(r.get(2)?),
                        entity_ids: vec![],
                    })
                })
                .map_err(store_err)?;
            for row in rows {
                let m = row.map_err(store_err)?;
                map.insert(m.id.clone(), m);
            }

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
        _bank_id: &str,
        seeds: &[MemoryId],
        hops: usize,
        limit: usize,
    ) -> Result<Vec<MemoryId>> {
        if seeds.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
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
                    "SELECT b_id FROM edges WHERE a_id IN ({ph}) UNION SELECT a_id FROM edges WHERE b_id IN ({ph})",
                );
                let mut stmt = conn.prepare(&sql).map_err(store_err)?;
                let all: Vec<String> = chunk.to_vec();
                let rows = stmt
                    .query_map(params_from_iter(all.iter().chain(all.iter())), |r| {
                        r.get::<_, String>(0)
                    })
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
        let conn = self.conn.lock().unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono_now_ms();
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
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, bank_id, name, etype, last_mentioned_ms FROM entities WHERE id = ?1",
            params![id],
            Self::row_entity,
        )
        .optional()
        .map_err(store_err)
    }

    fn find_by_alias(&self, bank_id: &str, alias: &str) -> Result<Option<Entity>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT e.id, e.bank_id, e.name, e.etype, e.last_mentioned_ms FROM entity_aliases a JOIN entities e ON e.id = a.entity_id WHERE a.bank_id = ?1 AND a.alias = ?2",
            params![bank_id, alias],
            Self::row_entity,
        )
        .optional()
        .map_err(store_err)
    }

    fn entities_for_bank(&self, bank_id: &str) -> Result<Vec<Entity>> {
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases(bank_id, alias, entity_id) VALUES (?1, ?2, ?3)",
            params![bank_id, alias, entity_id],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn link_memory_entities(&self, memory_id: &MemoryId, links: &[(String, String)]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        for (entity_id, mention) in links {
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, mention) VALUES (?1, ?2, ?3)",
                params![memory_id, entity_id, mention],
            )
            .map_err(store_err)?;
            conn.execute(
                "UPDATE entities SET last_mentioned_ms = MAX(last_mentioned_ms, ?2) WHERE id = ?1",
                params![entity_id, chrono_now_ms()],
            )
            .map_err(store_err)?;
        }
        Ok(())
    }

    fn memory_entity_ids(&self, memory_id: &MemoryId) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
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

    fn facts_for_entity(&self, entity_id: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.bank_id, m.kind, m.text, m.confidence, m.occurred_start_ms, m.occurred_end_ms, m.mentioned_at_ms, m.source, m.context
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

    fn add_edges(&self, edges: &[Edge]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        for e in edges {
            conn.execute(
                "INSERT OR IGNORE INTO edges(bank_id, a_id, b_id, kind, label, weight) VALUES (?1,?2,?3,?4,?5,?6)",
                params![e.bank_id, e.a_id, e.b_id, e.kind.as_str(), e.label.clone().unwrap_or_default(), e.weight],
            )
            .map_err(store_err)?;
        }
        Ok(())
    }

    fn mark_entities_dirty(
        &self,
        bank_id: &str,
        entity_ids: &[String],
        reason: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = chrono_now_ms();
        for id in entity_ids {
            conn.execute(
                "INSERT OR REPLACE INTO dirty_entities(bank_id, entity_id, reason, marked_at_ms) VALUES (?1,?2,?3,?4)",
                params![bank_id, id, reason, now],
            )
            .map_err(store_err)?;
        }
        Ok(())
    }

    fn list_dirty_entities(&self, bank_id: &str, limit: usize) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
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

    fn clear_dirty_entities(&self, bank_id: &str, entity_ids: &[String]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
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

    fn upsert_observation(&self, o: &Observation) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
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
        drop(conn);
        self.fts_upsert(&o.bank_id, "observation", &o.id, &o.text)
    }

    fn observations_for_entity(&self, entity_id: &str) -> Result<Vec<Observation>> {
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
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

    fn upsert_opinion(&self, o: &Opinion) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
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
            conn.execute(
                "INSERT OR IGNORE INTO opinion_entities(opinion_id, entity_id) VALUES (?1, ?2)",
                params![o.id, eid],
            )
            .map_err(store_err)?;
        }
        drop(conn);
        self.fts_upsert(&o.bank_id, "opinion", &o.id, &o.text)
    }

    fn list_opinions(&self, bank_id: &str) -> Result<Vec<Opinion>> {
        let conn = self.conn.lock().unwrap();
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
            let mut stmt2 = conn
                .prepare("SELECT entity_id FROM opinion_entities WHERE opinion_id = ?1")
                .map_err(store_err)?;
            let eids = stmt2
                .query_map(params![id], |r| r.get::<_, String>(0))
                .map_err(store_err)?
                .collect::<rusqlite::Result<Vec<String>>>()
                .map_err(store_err)?;
            out.push(Opinion {
                id,
                bank_id,
                text,
                confidence,
                evidence,
                history,
                entity_ids: eids,
                updated_at_ms: updated,
                created_at_ms: created,
                embedding: emb.map(|b| blob_to_emb(&b)),
            });
        }
        Ok(out)
    }

    fn upsert_mental_model(&self, m: &MentalModel) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO mental_models(id, bank_id, question, path, content, last_refreshed_ms, created_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(id) DO UPDATE SET question=excluded.question, path=excluded.path, content=excluded.content, last_refreshed_ms=excluded.last_refreshed_ms",
            params![m.id, m.bank_id, m.question, m.path, m.content, m.last_refreshed_ms, m.created_at_ms],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn list_mental_models(&self, bank_id: &str) -> Result<Vec<MentalModel>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, bank_id, question, path, content, last_refreshed_ms, created_at_ms FROM mental_models WHERE bank_id = ?1 ORDER BY created_at_ms")
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
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM mental_models WHERE id = ?1", params![id])
            .map_err(store_err)?;
        Ok(())
    }
}

impl SqliteStore {
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

fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SqliteStore {
        SqliteStore::open_in_memory().unwrap()
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
}
