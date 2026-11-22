//! Memory curation (RS-22): edit, invalidate and restore one raw fact,
//! with upstream's `PATCH /memories/{id}` semantics
//! (`memory_engine.py::update_memory_unit`, hindsight-api 1.2.1).
//!
//! Phase 1 runs every provider call (the embed) and entity resolution with
//! no store write pending; phase 2 is one store transaction per state
//! change, after which the fact's derived edges are rebuilt inline
//! (rustsight has no graph-maintenance sweep — see DIVERGENCES.md) and its
//! entities are re-queued for consolidation.

use crate::engine::retain::{
    build_semantic_edges, build_temporal_edges, resolve_fact_entities, ALL_KINDS,
};
use crate::engine::{with_retry, Engine};
use crate::entity::{collapse_ws, normalize_name};
use crate::types::*;
use crate::{Error, Result};
use std::collections::HashSet;
use std::sync::Arc;

/// Upstream's id check: memory ids are UUIDs.
fn check_uuid(id: &str) -> Result<()> {
    uuid::Uuid::parse_str(id)
        .map(|_| ())
        .map_err(|_| Error::BadInput(format!("Invalid memory_id: '{id}' is not a valid UUID")))
}

/// Trim, drop blanks and de-duplicate case-insensitively (first spelling
/// wins), as upstream normalizes `entities` before resolving.
fn clean_entity_names(names: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = vec![];
    for name in names {
        let cleaned = collapse_ws(name);
        if !cleaned.is_empty() && seen.insert(cleaned.to_lowercase()) {
            out.push(cleaned);
        }
    }
    out
}

impl Engine {
    /// A memory in the curation view: a live or archived raw fact, or an
    /// observation. `Ok(None)` when the bank has no such memory.
    pub fn get_memory_unit(&self, bank_id: &str, id: &str) -> Result<Option<MemoryUnit>> {
        check_uuid(id)?;
        if let Some(unit) = self.store.get_memory_unit(bank_id, id)? {
            return Ok(Some(unit));
        }
        let is_observation = self
            .store
            .get_refs(bank_id, &[id.to_string()])?
            .first()
            .is_some_and(|r| r.kind == MemoryKind::Observation);
        if !is_observation {
            return Ok(None);
        }
        let tags = self.store.tag_index(bank_id)?;
        self.store
            .observations_for_bank(bank_id)?
            .into_iter()
            .find(|o| o.id == id)
            .map(|o| self.observation_unit(o, &tags))
            .transpose()
    }

    fn observation_unit(
        &self,
        o: Observation,
        tags: &std::collections::HashMap<String, Vec<String>>,
    ) -> Result<MemoryUnit> {
        let entity = self.store.get_entity(&o.entity_id)?.map(|e| e.name);
        Ok(MemoryUnit {
            tags: tags.get(&o.id).cloned().unwrap_or_default(),
            id: o.id,
            bank_id: o.bank_id,
            kind: MemoryKind::Observation,
            text: o.text,
            context: None,
            occurred_start_ms: None,
            occurred_end_ms: None,
            mentioned_at_ms: None,
            document_id: None,
            chunk_id: None,
            metadata: None,
            entities: entity.into_iter().collect(),
            state: MemoryState::Valid,
            invalidation_reason: None,
            invalidated_at_ms: None,
            edited_at_ms: None,
            created_at_ms: o.created_at_ms,
            updated_at_ms: Some(o.updated_at_ms),
            source_memory_ids: o.evidence.into_iter().map(|e| e.memory_id).collect(),
            proof_count: o.proof_count,
        })
    }

    /// One page of memory units (upstream `GET /memories/list`) and the
    /// total before paging. Live listings include observations unless a
    /// document filter or another kind is asked for; the archive holds raw
    /// facts only. Newest mention first (observations, which have none,
    /// last), then newest ingest, then id.
    pub fn list_memory_units(
        &self,
        bank_id: &str,
        query: &MemoryListQuery,
    ) -> Result<(Vec<MemoryUnit>, usize)> {
        let mut units = self.store.list_memory_units(bank_id, query)?;
        let with_observations = query.state.unwrap_or_default() == MemoryState::Valid
            && query.document_id.is_none()
            && matches!(query.kind, None | Some(MemoryKind::Observation));
        if with_observations {
            let tags = self.store.tag_index(bank_id)?;
            let q = query.q.as_deref().map(str::to_lowercase);
            for o in self.store.observations_for_bank(bank_id)? {
                if query.entity_id.as_ref().is_some_and(|e| *e != o.entity_id) {
                    continue;
                }
                if q.as_deref()
                    .is_some_and(|q| !o.text.to_lowercase().contains(q))
                {
                    continue;
                }
                units.push(self.observation_unit(o, &tags)?);
            }
        }
        if let Some(filter) = query.tags.as_ref().filter(|f| !f.is_noop()) {
            units.retain(|u| filter.matches(&u.tags));
        }
        units.sort_by(|a, b| {
            b.mentioned_at_ms
                .is_some()
                .cmp(&a.mentioned_at_ms.is_some())
                .then(b.mentioned_at_ms.cmp(&a.mentioned_at_ms))
                .then(b.created_at_ms.cmp(&a.created_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        let total = units.len();
        let page = units
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();
        Ok((page, total))
    }

    /// Curate one raw fact (upstream "Curate memory unit"): edit its
    /// fields, invalidate it into the archive, restore it, or edit then
    /// invalidate. Returns the memory as it now stands.
    ///
    /// Errors: `BadInput` for a non-UUID id, an empty request, blank
    /// text, a non-world/experience `kind`, curating an observation or
    /// opinion, or editing an archived fact; `NotFound` for an unknown id.
    pub async fn curate_memory(&self, req: CurateRequest) -> Result<MemoryUnit> {
        check_uuid(&req.memory_id)?;
        if req.edit.is_empty() && req.state.is_none() {
            return Err(Error::BadInput(
                "Provide at least one field to update.".into(),
            ));
        }
        if req
            .edit
            .text
            .as_deref()
            .is_some_and(|t| t.trim().is_empty())
        {
            return Err(Error::BadInput("text must not be empty.".into()));
        }
        if let Some(k) = req.edit.kind {
            if !matches!(k, MemoryKind::World | MemoryKind::Experience) {
                return Err(Error::BadInput(format!(
                    "Invalid fact_type '{}': expected 'world' or 'experience'.",
                    k.as_str()
                )));
            }
        }
        let guard = self.bank_read(&req.bank_id).await;
        guard.until_deleted(curate(self, &req)).await
    }
}

async fn curate(e: &Engine, req: &CurateRequest) -> Result<MemoryUnit> {
    let bank_id = req.bank_id.as_str();
    let id = req.memory_id.as_str();
    let Some(unit) = e.store.get_memory_unit(bank_id, id)? else {
        if let Some(r) = e.store.get_refs(bank_id, &[id.to_string()])?.first() {
            return Err(Error::BadInput(format!(
                "Memory '{id}' is a {}; only world/experience facts can be curated. Observations are derived and regenerate from their sources.",
                r.kind.as_str()
            )));
        }
        return Err(Error::NotFound(format!("Memory unit '{id}' not found")));
    };
    let live = unit.state == MemoryState::Valid;
    let doing_edit = !req.edit.is_empty();
    if doing_edit && !live {
        return Err(Error::BadInput(
            "Cannot edit an invalidated memory; revert it to 'valid' first.".into(),
        ));
    }
    let mut dirty: Vec<String> = vec![];

    if doing_edit {
        dirty.extend(edit(e, req, &unit).await?);
    }
    match req.state {
        Some(MemoryState::Invalidated) if live => {
            let effect =
                e.store
                    .invalidate_memory(bank_id, id, req.reason.as_deref(), e.clock.now_ms())?;
            dirty.extend(effect.dirty_entities);
        }
        Some(MemoryState::Invalidated) => {
            if let Some(reason) = &req.reason {
                e.store.set_invalidation_reason(bank_id, id, reason)?;
            }
        }
        Some(MemoryState::Valid) if !live => {
            let embed = Arc::clone(&e.embed);
            let text = unit.text.clone();
            let embedding = embed_one(e, &embed, text).await?;
            let effect = e.store.restore_memory(bank_id, id, &embedding)?;
            let entity_ids = e.store.memory_entity_ids(&id.to_string())?;
            rebuild_edges(e, bank_id, id, &entity_ids, &embedding)?;
            dirty.extend(effect.dirty_entities);
        }
        _ => {}
    }

    dirty.sort();
    dirty.dedup();
    if !dirty.is_empty() {
        e.store
            .mark_entities_dirty(bank_id, &dirty, "memory_curated")?;
        e.wake_worker();
    }
    e.store
        .get_memory_unit(bank_id, id)?
        .ok_or_else(|| Error::NotFound(format!("Memory unit '{id}' not found")))
}

async fn embed_one(
    e: &Engine,
    embed: &Arc<dyn crate::traits::EmbedProvider>,
    text: String,
) -> Result<Vec<f32>> {
    let mut out = with_retry(e, || embed.embed(vec![text.clone()])).await?;
    match (out.pop(), out.is_empty()) {
        (Some(v), true) => Ok(v),
        _ => Err(Error::Embed("embedding count mismatch".into())),
    }
}

/// Apply a field edit to a live fact. Provider call first (the embed), then
/// entity resolution, then the store rewrite and the edge rebuild. Returns
/// the entities to re-queue.
async fn edit(e: &Engine, req: &CurateRequest, unit: &MemoryUnit) -> Result<Vec<String>> {
    let ed = &req.edit;
    let bank_id = req.bank_id.as_str();
    let id = unit.id.as_str();
    let text = ed.text.clone().unwrap_or_else(|| unit.text.clone());
    let context = match &ed.context {
        Some(c) => c.clone().filter(|c| !c.is_empty()),
        None => unit.context.clone(),
    };
    let occurred_start_ms = ed.occurred_start_ms.unwrap_or(unit.occurred_start_ms);
    let occurred_end_ms = ed.occurred_end_ms.unwrap_or(unit.occurred_end_ms);
    let mentioned_at_ms = match ed.mentioned_at_ms {
        Some(Some(ms)) => ms,
        // Cleared: rustsight's mention time is never null, so it falls
        // back to the ingest time.
        Some(None) => unit.created_at_ms,
        None => unit.mentioned_at_ms.unwrap_or(unit.created_at_ms),
    };
    let kind = ed.kind.unwrap_or(unit.kind);

    let embed = Arc::clone(&e.embed);
    let embedding = embed_one(e, &embed, text.clone()).await?;

    let entity_links = match &ed.entities {
        None => None,
        Some(names) => {
            let names = clean_entity_names(names);
            let at_ms = occurred_start_ms.unwrap_or(mentioned_at_ms);
            Some(if req.resolve_entities {
                resolve_like_retain(e, bank_id, &text, kind, &names, at_ms)?
            } else {
                resolve_literally(e, bank_id, &names)?
            })
        }
    };

    let effect = e.store.update_memory(
        bank_id,
        id,
        &MemoryRewrite {
            kind,
            text,
            context,
            occurred_start_ms,
            occurred_end_ms,
            mentioned_at_ms,
            embedding: embedding.clone(),
            entity_links,
            edited_at_ms: e.clock.now_ms(),
        },
    )?;
    let entity_ids = e.store.memory_entity_ids(&id.to_string())?;
    rebuild_edges(e, bank_id, id, &entity_ids, &embedding)?;
    Ok(effect.dirty_entities)
}

/// `resolve_entities: true`: the names go through retain's resolver as one
/// fact's mentions (alias, fuzzy scoring, then clustering into new ones).
fn resolve_like_retain(
    e: &Engine,
    bank_id: &str,
    text: &str,
    kind: MemoryKind,
    names: &[String],
    at_ms: i64,
) -> Result<Vec<(String, String)>> {
    let fact = ExtractedFact {
        text: text.to_string(),
        fact_type: kind,
        entities: names
            .iter()
            .map(|n| ExtractedEntity {
                name: n.clone(),
                etype: EntityType::Concept,
            })
            .collect(),
        occurred_start_ms: Some(at_ms),
        occurred_end_ms: None,
        causal: vec![],
        chunk_index: 0,
        labels: None,
        tags: vec![],
    };
    let (per_fact, _) = resolve_fact_entities(e, bank_id, std::slice::from_ref(&fact), at_ms)?;
    let mut seen = HashSet::new();
    Ok(per_fact
        .into_iter()
        .flatten()
        .filter(|(ent, _)| seen.insert(ent.id.clone()))
        .map(|(ent, mention)| (ent.id, mention))
        .collect())
}

/// `resolve_entities: false` (upstream #3479): reuse an entity only when
/// its name matches case-insensitively; any other name becomes its own
/// entity, and names in one request are never merged with each other.
fn resolve_literally(e: &Engine, bank_id: &str, names: &[String]) -> Result<Vec<(String, String)>> {
    let bank_entities = e.store.entities_for_bank(bank_id)?;
    let mut out = vec![];
    for name in names {
        let lower = name.to_lowercase();
        let existing = bank_entities
            .iter()
            .find(|x| collapse_ws(&x.name).to_lowercase() == lower)
            .cloned();
        let ent = match existing {
            Some(x) => x,
            None => {
                let ent = e.store.create_entity(&NewEntity {
                    bank_id: bank_id.to_string(),
                    name: name.clone(),
                    etype: EntityType::Concept,
                })?;
                // Only claim an alias key nobody holds: a literal name
                // must not redirect future retains away from an entity
                // that already answers to it.
                let key = normalize_name(name);
                if !key.is_empty() && e.store.find_by_alias(bank_id, &key)?.is_none() {
                    e.store.add_alias(bank_id, &key, &ent.id)?;
                }
                ent
            }
        };
        if !out.iter().any(|(eid, _): &(String, String)| *eid == ent.id) {
            out.push((ent.id, name.clone()));
        }
    }
    Ok(out)
}

/// Rebuild a curated fact's entity, temporal and semantic edges with
/// retain's builders (upstream leaves this to its graph-maintenance
/// sweep). Causal edges are never touched here.
fn rebuild_edges(
    e: &Engine,
    bank_id: &str,
    id: &str,
    entity_ids: &[String],
    embedding: &[f32],
) -> Result<usize> {
    let Some(mem) = e.store.get_memory(&id.to_string())? else {
        return Ok(0);
    };
    let cfg = &e.config;
    let mut edges: Vec<Edge> = vec![];
    let facts_map = e
        .store
        .facts_for_entities(entity_ids, cfg.entity_backfill + 1)?;
    for eid in entity_ids {
        for sib in facts_map
            .get(eid)
            .into_iter()
            .flatten()
            .filter(|s| s.id != mem.id)
            .take(cfg.entity_backfill)
        {
            edges.push(Edge {
                bank_id: bank_id.to_string(),
                a_id: mem.id.clone(),
                b_id: sib.id.clone(),
                kind: EdgeKind::Entity,
                label: Some(eid.clone()),
                weight: 1.0,
            });
        }
    }
    let recent = e.store.recent_memories(bank_id, 60)?;
    let own: HashSet<String> = HashSet::from([mem.id.clone()]);
    let stored = std::slice::from_ref(&mem);
    edges.extend(build_temporal_edges(cfg, bank_id, stored, &recent, &own));
    let not_self = |x: &str| x != mem.id;
    let candidates = e.store.search_semantic(
        bank_id,
        &[embedding],
        &ALL_KINDS,
        cfg.semantic_edge_cap,
        Some(&not_self),
    )?;
    edges.extend(build_semantic_edges(cfg, bank_id, stored, &candidates));
    e.store.add_edges(&edges)?;
    Ok(edges.len())
}
