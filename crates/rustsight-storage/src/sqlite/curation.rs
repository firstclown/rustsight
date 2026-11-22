//! Memory curation (RS-22): edit, invalidate (move to the archive) and
//! restore a single raw fact, plus the curation read views. Mirrors
//! upstream `memory_engine.py::update_memory_unit` and
//! `memories/pg/curation.py`.

use super::*;
use rusqlite::types::Value as SqlValue;
use rustsight_core::types::{CurationEffect, MemoryListQuery, MemoryRewrite, MemoryState};

/// Columns shared by `memories` and `invalidated_memories`, in the order
/// [`row_unit`] reads them; the two trailing curation columns follow.
const UNIT_COLS: &str = "id, bank_id, kind, text, context, occurred_start_ms, occurred_end_ms, mentioned_at_ms, document_id, chunk_id, tags_json, metadata_json, edited_at_ms, created_at_ms";
const LIVE_EXTRA: &str = "NULL, NULL";
const ARCHIVE_EXTRA: &str = "invalidation_reason, invalidated_at_ms";

/// An entity link carried through the archive.
#[derive(serde::Serialize, serde::Deserialize)]
struct EntitySnap {
    entity_id: String,
    mention: String,
}

/// A causal edge carried through the archive (#2864: nothing re-derives
/// causal edges, so curation must not lose them).
#[derive(serde::Serialize, serde::Deserialize)]
struct CausalSnap {
    a_id: String,
    b_id: String,
    label: String,
    weight: f32,
}

fn row_unit(r: &Row, state: MemoryState) -> rusqlite::Result<MemoryUnit> {
    Ok(MemoryUnit {
        id: r.get(0)?,
        bank_id: r.get(1)?,
        kind: MemoryKind::parse(&r.get::<_, String>(2)?).unwrap_or(MemoryKind::World),
        text: r.get(3)?,
        context: r.get(4)?,
        occurred_start_ms: r.get(5)?,
        occurred_end_ms: r.get(6)?,
        mentioned_at_ms: Some(r.get(7)?),
        document_id: r.get(8)?,
        chunk_id: r.get(9)?,
        tags: parse_tags(r.get::<_, Option<String>>(10)?),
        metadata: parse_metadata(r.get::<_, Option<String>>(11)?),
        edited_at_ms: r.get(12)?,
        created_at_ms: r.get(13)?,
        entities: vec![],
        state,
        invalidation_reason: r.get(14)?,
        invalidated_at_ms: r.get(15)?,
        updated_at_ms: None,
        source_memory_ids: vec![],
        proof_count: 1,
    })
}

fn to_json<T: serde::Serialize>(v: &T) -> Result<String> {
    serde_json::to_string(v).map_err(json_err)
}

fn dedup_sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v.dedup();
    v
}

impl SqliteStore {
    /// Fill each live unit's sorted entity names in one query per 400 ids.
    fn attach_entity_names(conn: &Connection, units: &mut [MemoryUnit]) -> Result<()> {
        let ids: Vec<String> = units
            .iter()
            .filter(|u| u.state == MemoryState::Valid)
            .map(|u| u.id.clone())
            .collect();
        let mut names: HashMap<String, Vec<String>> = HashMap::new();
        for chunk in ids.chunks(400) {
            let ph = placeholders(chunk.len());
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT me.memory_id, e.name FROM memory_entities me JOIN entities e ON e.id = me.entity_id WHERE me.memory_id IN ({ph})"
                ))
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })
                .map_err(store_err)?;
            for r in rows {
                let (mid, name) = r.map_err(store_err)?;
                names.entry(mid).or_default().push(name);
            }
        }
        for u in units.iter_mut() {
            if let Some(mut n) = names.remove(&u.id) {
                n.sort();
                u.entities = n;
            }
        }
        Ok(())
    }

    fn entity_links_of(conn: &Connection, memory_id: &str) -> Result<Vec<EntitySnap>> {
        let mut stmt = conn
            .prepare("SELECT entity_id, mention FROM memory_entities WHERE memory_id = ?1 ORDER BY entity_id")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![memory_id], |r| {
                Ok(EntitySnap {
                    entity_id: r.get(0)?,
                    mention: r.get(1)?,
                })
            })
            .map_err(store_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(store_err)
    }

    /// Delete every observation of the bank whose evidence cites `id`
    /// (upstream `delete_stale_observations_for_memories`: deleted whole,
    /// not trimmed), with its FTS row, vector and edges. Returns the
    /// deleted observations' entity ids and ids.
    fn delete_citing_observations(
        &self,
        tx: &Connection,
        bank_id: &str,
        id: &str,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let candidates: Vec<(String, String, String)> = {
            let mut stmt = tx
                .prepare(
                    "SELECT id, entity_id, evidence_json FROM observations WHERE bank_id = ?1 AND instr(evidence_json, ?2) > 0",
                )
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![bank_id, id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        let mut entities = vec![];
        let mut deleted = vec![];
        for (oid, entity_id, evidence_json) in candidates {
            let evidence: Vec<Evidence> = serde_json::from_str(&evidence_json).unwrap_or_default();
            if !evidence.iter().any(|ev| ev.memory_id == id) {
                continue;
            }
            for sql in [
                "DELETE FROM observations WHERE id = ?1",
                "DELETE FROM edges WHERE a_id = ?1 OR b_id = ?1",
            ] {
                tx.execute(sql, params![oid]).map_err(store_err)?;
            }
            tx.execute(
                "DELETE FROM texts_fts WHERE bank_id = ?1 AND ref_id = ?2",
                params![bank_id, oid],
            )
            .map_err(store_err)?;
            self.vec_remove(tx, [&oid])?;
            entities.push(entity_id);
            deleted.push(oid);
        }
        Ok((entities, deleted))
    }

    /// Strip `id` from every opinion's evidence (rustsight-only: upstream
    /// has no opinions table). Opinions keep their remaining evidence.
    fn trim_opinion_evidence(tx: &Connection, bank_id: &str, id: &str) -> Result<()> {
        let rows: Vec<(String, String)> = {
            let mut stmt = tx
                .prepare(
                    "SELECT id, evidence_json FROM opinions WHERE bank_id = ?1 AND instr(evidence_json, ?2) > 0",
                )
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![bank_id, id], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        for (oid, evidence_json) in rows {
            let evidence: Vec<Evidence> = serde_json::from_str(&evidence_json).unwrap_or_default();
            if !evidence.iter().any(|ev| ev.memory_id == id) {
                continue;
            }
            let kept: Vec<Evidence> = evidence
                .into_iter()
                .filter(|ev| ev.memory_id != id)
                .collect();
            tx.execute(
                "UPDATE opinions SET evidence_json = ?2 WHERE id = ?1",
                params![oid, to_json(&kept)?],
            )
            .map_err(store_err)?;
        }
        Ok(())
    }

    fn live_kind(tx: &Connection, bank_id: &str, id: &str) -> Result<Option<String>> {
        tx.query_row(
            "SELECT kind FROM memories WHERE id = ?1 AND bank_id = ?2",
            params![id, bank_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(store_err)
    }

    pub(super) fn curation_get_unit(&self, bank_id: &str, id: &str) -> Result<Option<MemoryUnit>> {
        let conn = self.lock_conn();
        let live = conn
            .query_row(
                &format!(
                    "SELECT {UNIT_COLS}, {LIVE_EXTRA} FROM memories WHERE id = ?1 AND bank_id = ?2"
                ),
                params![id, bank_id],
                |r| row_unit(r, MemoryState::Valid),
            )
            .optional()
            .map_err(store_err)?;
        if let Some(unit) = live {
            let mut units = [unit];
            Self::attach_entity_names(&conn, &mut units)?;
            let [unit] = units;
            return Ok(Some(unit));
        }
        conn.query_row(
            &format!(
                "SELECT {UNIT_COLS}, {ARCHIVE_EXTRA} FROM invalidated_memories WHERE id = ?1 AND bank_id = ?2"
            ),
            params![id, bank_id],
            |r| row_unit(r, MemoryState::Invalidated),
        )
        .optional()
        .map_err(store_err)
    }

    pub(super) fn curation_list_units(
        &self,
        bank_id: &str,
        query: &MemoryListQuery,
    ) -> Result<Vec<MemoryUnit>> {
        let state = query.state.unwrap_or_default();
        let (table, extra) = match state {
            MemoryState::Valid => ("memories", LIVE_EXTRA),
            MemoryState::Invalidated => ("invalidated_memories", ARCHIVE_EXTRA),
        };
        let mut conds = vec!["bank_id = ?".to_string()];
        let mut args: Vec<SqlValue> = vec![SqlValue::Text(bank_id.to_string())];
        match query.kind {
            None => {}
            Some(k @ (MemoryKind::World | MemoryKind::Experience)) => {
                conds.push("kind = ?".into());
                args.push(SqlValue::Text(k.as_str().into()));
            }
            Some(_) => return Ok(vec![]),
        }
        if let Some(doc) = &query.document_id {
            conds.push("document_id = ?".into());
            args.push(SqlValue::Text(doc.clone()));
        }
        if let Some(q) = query.q.as_deref().filter(|q| !q.is_empty()) {
            conds.push("(instr(lower(text), lower(?)) > 0 OR instr(lower(coalesce(context, '')), lower(?)) > 0)".into());
            args.push(SqlValue::Text(q.to_string()));
            args.push(SqlValue::Text(q.to_string()));
        }
        if let Some(entity) = &query.entity_id {
            // Entity links reference live facts only, so this selects
            // nothing in the archive (as upstream documents).
            conds.push("id IN (SELECT memory_id FROM memory_entities WHERE entity_id = ?)".into());
            args.push(SqlValue::Text(entity.clone()));
        }
        let conn = self.lock_conn();
        let mut units: Vec<MemoryUnit> = {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT {UNIT_COLS}, {extra} FROM {table} WHERE {} ORDER BY mentioned_at_ms DESC, created_at_ms DESC, id",
                    conds.join(" AND ")
                ))
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params_from_iter(args.iter()), |r| row_unit(r, state))
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        Self::attach_entity_names(&conn, &mut units)?;
        Ok(units)
    }

    pub(super) fn curation_update(
        &self,
        bank_id: &str,
        id: &str,
        rw: &MemoryRewrite,
    ) -> Result<CurationEffect> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        if Self::live_kind(&tx, bank_id, id)?.is_none() {
            return Err(Error::NotFound(format!("Memory unit '{id}' not found")));
        }
        let mut dirty: Vec<String> = Self::entity_links_of(&tx, id)?
            .into_iter()
            .map(|l| l.entity_id)
            .collect();
        tx.execute(
            "UPDATE memories SET kind = ?3, text = ?4, context = ?5, occurred_start_ms = ?6, occurred_end_ms = ?7, mentioned_at_ms = ?8, embedding = ?9, edited_at_ms = ?10 WHERE id = ?1 AND bank_id = ?2",
            params![
                id,
                bank_id,
                rw.kind.as_str(),
                rw.text,
                rw.context,
                rw.occurred_start_ms,
                rw.occurred_end_ms,
                rw.mentioned_at_ms,
                emb_to_blob(&rw.embedding),
                rw.edited_at_ms
            ],
        )
        .map_err(store_err)?;
        Self::fts_upsert(&tx, bank_id, rw.kind.as_str(), id, &rw.text)?;
        self.vec_put(&tx, bank_id, rw.kind, id, Some(&rw.embedding))?;
        if let Some(links) = &rw.entity_links {
            tx.execute(
                "DELETE FROM memory_entities WHERE memory_id = ?1",
                params![id],
            )
            .map_err(store_err)?;
            let at_ms = rw.occurred_start_ms.unwrap_or(rw.mentioned_at_ms);
            for (entity_id, mention) in links {
                tx.execute(
                    "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, mention) VALUES (?1, ?2, ?3)",
                    params![id, entity_id, mention],
                )
                .map_err(store_err)?;
                tx.execute(
                    "UPDATE entities SET last_mentioned_ms = MAX(last_mentioned_ms, ?2) WHERE id = ?1",
                    params![entity_id, at_ms],
                )
                .map_err(store_err)?;
                dirty.push(entity_id.clone());
            }
        }
        // Derived links go (the engine rebuilds them); causal edges stay.
        tx.execute(
            "DELETE FROM edges WHERE kind IN ('entity', 'temporal', 'semantic') AND (a_id = ?1 OR b_id = ?1)",
            params![id],
        )
        .map_err(store_err)?;
        let (obs_entities, deleted) = self.delete_citing_observations(&tx, bank_id, id)?;
        tx.commit().map_err(store_err)?;
        dirty.extend(obs_entities);
        Ok(CurationEffect {
            dirty_entities: dedup_sorted(dirty),
            deleted_observations: deleted,
        })
    }

    pub(super) fn curation_invalidate(
        &self,
        bank_id: &str,
        id: &str,
        reason: Option<&str>,
        at_ms: i64,
    ) -> Result<CurationEffect> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        if Self::live_kind(&tx, bank_id, id)?.is_none() {
            return Err(Error::NotFound(format!("Memory unit '{id}' not found")));
        }
        let links = Self::entity_links_of(&tx, id)?;
        let causal: Vec<CausalSnap> = {
            let mut stmt = tx
                .prepare(
                    "SELECT a_id, b_id, label, weight FROM edges WHERE bank_id = ?2 AND kind = 'causal' AND (a_id = ?1 OR b_id = ?1)",
                )
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![id, bank_id], |r| {
                    Ok(CausalSnap {
                        a_id: r.get(0)?,
                        b_id: r.get(1)?,
                        label: r.get(2)?,
                        weight: r.get(3)?,
                    })
                })
                .map_err(store_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(store_err)?
        };
        tx.execute(
            "INSERT OR REPLACE INTO invalidated_memories(id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context, created_at_ms, document_id, tags_json, metadata_json, chunk_id, edited_at_ms, invalidation_reason, invalidated_at_ms, entities_json, causal_edges_json)
             SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, source, context, created_at_ms, document_id, tags_json, metadata_json, chunk_id, edited_at_ms, ?3, ?4, ?5, ?6
             FROM memories WHERE id = ?1 AND bank_id = ?2",
            params![
                id,
                bank_id,
                reason,
                at_ms,
                to_json(&links)?,
                to_json(&causal)?
            ],
        )
        .map_err(store_err)?;
        for sql in [
            "DELETE FROM memory_entities WHERE memory_id = ?1",
            "DELETE FROM edges WHERE a_id = ?1 OR b_id = ?1",
            "DELETE FROM memories WHERE id = ?1",
        ] {
            tx.execute(sql, params![id]).map_err(store_err)?;
        }
        tx.execute(
            "DELETE FROM texts_fts WHERE bank_id = ?1 AND ref_id = ?2",
            params![bank_id, id],
        )
        .map_err(store_err)?;
        self.vec_remove(&tx, [&id.to_string()])?;
        let (obs_entities, deleted) = self.delete_citing_observations(&tx, bank_id, id)?;
        Self::trim_opinion_evidence(&tx, bank_id, id)?;
        tx.commit().map_err(store_err)?;
        let mut dirty: Vec<String> = links.into_iter().map(|l| l.entity_id).collect();
        dirty.extend(obs_entities);
        Ok(CurationEffect {
            dirty_entities: dedup_sorted(dirty),
            deleted_observations: deleted,
        })
    }

    pub(super) fn curation_set_reason(
        &self,
        bank_id: &str,
        id: &str,
        reason: &str,
    ) -> Result<bool> {
        let conn = self.lock_conn();
        let n = conn
            .execute(
                "UPDATE invalidated_memories SET invalidation_reason = ?3 WHERE id = ?1 AND bank_id = ?2",
                params![id, bank_id, reason],
            )
            .map_err(store_err)?;
        Ok(n > 0)
    }

    pub(super) fn curation_restore(
        &self,
        bank_id: &str,
        id: &str,
        embedding: &[f32],
    ) -> Result<CurationEffect> {
        let mut conn = self.lock_conn();
        let tx = conn.transaction().map_err(store_err)?;
        let archived: Option<(String, String, String, String)> = tx
            .query_row(
                "SELECT kind, text, entities_json, causal_edges_json FROM invalidated_memories WHERE id = ?1 AND bank_id = ?2",
                params![id, bank_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()
            .map_err(store_err)?;
        let Some((kind, text, entities_json, causal_json)) = archived else {
            return Err(Error::NotFound(format!("Memory unit '{id}' not found")));
        };
        let kind = MemoryKind::parse(&kind).unwrap_or(MemoryKind::World);
        tx.execute(
            "INSERT INTO memories(id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, embedding, source, context, created_at_ms, document_id, tags_json, metadata_json, chunk_id, edited_at_ms)
             SELECT id, bank_id, kind, text, confidence, occurred_start_ms, occurred_end_ms, mentioned_at_ms, ?3, source, context, created_at_ms, document_id, tags_json, metadata_json, chunk_id, edited_at_ms
             FROM invalidated_memories WHERE id = ?1 AND bank_id = ?2",
            params![id, bank_id, emb_to_blob(embedding)],
        )
        .map_err(store_err)?;
        tx.execute(
            "INSERT INTO texts_fts(text, bank_id, kind, ref_id) VALUES (?1, ?2, ?3, ?4)",
            params![cjk_segment(&text), bank_id, kind.as_str(), id],
        )
        .map_err(store_err)?;
        self.vec_put(&tx, bank_id, kind, id, Some(embedding))?;
        let links: Vec<EntitySnap> = serde_json::from_str(&entities_json).map_err(json_err)?;
        let mut relinked = vec![];
        for link in links {
            let n = tx
                .execute(
                    "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, mention)
                     SELECT ?1, id, ?3 FROM entities WHERE id = ?2 AND bank_id = ?4",
                    params![id, link.entity_id, link.mention, bank_id],
                )
                .map_err(store_err)?;
            if n > 0 {
                relinked.push(link.entity_id);
            }
        }
        let causal: Vec<CausalSnap> = serde_json::from_str(&causal_json).map_err(json_err)?;
        for c in causal {
            let other = if c.a_id == id { &c.b_id } else { &c.a_id };
            if Self::live_kind(&tx, bank_id, other)?.is_some() {
                tx.execute(
                    "INSERT OR IGNORE INTO edges(bank_id, a_id, b_id, kind, label, weight) VALUES (?1, ?2, ?3, 'causal', ?4, ?5)",
                    params![bank_id, c.a_id, c.b_id, c.label, c.weight],
                )
                .map_err(store_err)?;
            }
        }
        tx.execute(
            "DELETE FROM invalidated_memories WHERE id = ?1 AND bank_id = ?2",
            params![id, bank_id],
        )
        .map_err(store_err)?;
        tx.commit().map_err(store_err)?;
        Ok(CurationEffect {
            dirty_entities: dedup_sorted(relinked),
            deleted_observations: vec![],
        })
    }
}
