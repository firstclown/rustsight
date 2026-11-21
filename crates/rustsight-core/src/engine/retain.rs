use crate::engine::chunks::{merge_chunk_facts, split_chunks, split_half, OVERLAP_DEDUP_THRESHOLD};
use crate::engine::{with_retry, Engine, EngineConfig};
use crate::entity::{collapse_ws, normalize_name, score_candidate};
use crate::events::{EngineEvent, EventData, EventType};
use crate::labels::BankLabels;
use crate::prompts::extraction::{
    build_extraction_messages, parse_extraction_with_reference, ExtractionConfig,
};
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::{temporal_weight, Error, Result, SemanticHit};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// How many retain chunks extract concurrently (CR-19). Bounded on
/// purpose: each chunk is one LLM round trip, and an unbounded fan-out
/// would trip provider rate limits on large retains. `buffered` (not
/// `buffer_unordered`) keeps per-chunk results in chunk order — the
/// causal-index remap in `merge_chunk_facts` depends on it.
const CHUNK_EXTRACT_CONCURRENCY: usize = 2;

/// Boxed per-chunk extraction future; boxing lets retain buffer a bounded
/// number of chunk extractions concurrently (CR-19) and the recursive
/// half-size re-ask return one concrete type.
type ChunkFut<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<ExtractedFact>>> + Send + 'a>>;

pub(super) async fn retain(e: &Engine, mut req: RetainRequest) -> Result<RetainResponse> {
    let operation_id = req
        .operation_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let defense = screen_content(e, &mut req, &operation_id)?;
    let bank_id = req.bank_id.clone();
    let tags = req.tags.clone();
    let mut resp = retain_screened(e, req).await?;
    resp.defense = defense;
    // retain.completed, once the memories are durable. The count is asked
    // of the store so it is the document's total after an append.
    let count = match &resp.document_id {
        Some(doc) => Some(e.store.count_document_memories(&bank_id, doc)?),
        None => None,
    };
    e.emit(EngineEvent::new(
        EventType::RetainCompleted,
        &bank_id,
        &operation_id,
        "completed",
        e.clock.now_ms(),
        EventData::Retain {
            document_id: resp.document_id.clone(),
            tags: Some(tags),
            memory_unit_count: count,
        },
    ));
    Ok(resp)
}

/// `memory_defense.triggered` for a non-allow decision (upstream
/// `_fire_memory_defense_webhook`). Public through `Engine` so a batch
/// front end that drops blocked items before retaining can report them.
pub(crate) fn defense_event(
    e: &Engine,
    bank_id: &str,
    operation_id: &str,
    document_id: Option<&str>,
    decision: &crate::defense::DefenseDecision,
) -> EngineEvent {
    let nonempty = |v: &Vec<String>| (!v.is_empty()).then(|| v.clone());
    EngineEvent::new(
        EventType::MemoryDefenseTriggered,
        bank_id,
        operation_id,
        decision.action.as_str(),
        e.clock.now_ms(),
        EventData::MemoryDefense {
            action: decision.action.as_str().to_string(),
            detector: decision.detector.clone(),
            document_id: document_id.map(str::to_string),
            matched_types: nonempty(&decision.matched_types),
            message: Some(decision.message.clone()).filter(|m| !m.is_empty()),
            hits: Some(decision.hits.clone()).filter(|h| !h.is_empty()),
        },
    )
}

/// Memory Defense pre-parse hook (RS-10a): screen the content under the
/// bank's `memory_defense` policy before anything is chunked, extracted or
/// stored. `redact` rewrites `req.content` in place — the document body,
/// chunks and facts are all derived from it, so none sees the secret;
/// `block` refuses the item. A malformed policy fails the retain rather
/// than skipping screening (a security control must not fail open).
fn screen_content(
    e: &Engine,
    req: &mut RetainRequest,
    operation_id: &str,
) -> Result<Option<crate::defense::DefenseReport>> {
    use crate::defense::{parse_policy, screen, DefenseAction, DefenseReport};
    let config = e.store.bank_config(&req.bank_id)?;
    let policy = parse_policy(config.get("memory_defense"))?;
    if !policy.enabled {
        return Ok(None);
    }
    let decision = screen(&policy, &req.content);
    if decision.action != DefenseAction::Allow {
        e.emit(defense_event(
            e,
            &req.bank_id,
            operation_id,
            req.document_id.as_deref(),
            &decision,
        ));
    }
    match decision.action {
        DefenseAction::Allow => Ok(None),
        DefenseAction::Redact => {
            if let Some(redacted) = &decision.redacted_content {
                req.content = redacted.clone();
            }
            Ok(Some(DefenseReport::from(&decision)))
        }
        DefenseAction::Block => Err(Error::Blocked {
            detector: decision.detector.unwrap_or_default(),
            message: decision.message,
        }),
    }
}

async fn retain_screened(e: &Engine, req: RetainRequest) -> Result<RetainResponse> {
    if req.content.trim().is_empty() {
        return Err(Error::BadInput("content is empty".into()));
    }
    let bank = match e.store.get_bank(&req.bank_id)? {
        Some(b) => b,
        None => {
            let b = Bank {
                id: req.bank_id.clone(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            };
            e.store.ensure_bank(&b)?;
            b
        }
    };
    let now = req.timestamp_ms.unwrap_or_else(|| e.clock.now_ms());

    let chunks = split_chunks(
        &req.content,
        e.config.chunk_size_tokens,
        e.config.chunk_overlap_tokens,
        e.tokens.as_ref(),
    );
    // `timestamp: "unset"` (timeless content): the extractor gets no event
    // date; memories are still stamped with the retain time.
    let event_date = if req.timestamp_unset {
        None
    } else {
        DateTime::from_timestamp_millis(now).map(|dt| dt.with_timezone(&Utc))
    };
    let document_id = req
        .document_id
        .clone()
        .filter(|d| !d.trim().is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let incoming_hash = content_hash(&req.content);
    let existing_doc = e.store.get_document(&bank.id, &document_id)?;
    // Re-retaining a document byte-for-byte (same scope and labels) is a
    // no-op: the stored memories already are its extraction.
    if let Some(doc) = &existing_doc {
        if req.update_mode == UpdateMode::Replace
            && doc.content_hash == incoming_hash
            && doc.tags == req.tags
            && doc.metadata == req.metadata
            && doc.retain_params.get("context").and_then(|v| v.as_str()) == req.context.as_deref()
            && e.store.count_document_memories(&bank.id, &document_id)? > 0
        {
            return Ok(RetainResponse {
                memories_stored: 0,
                entities: vec![],
                edges_created: 0,
                reasks: 0,
                document_id: Some(document_id),
                defense: None,
            });
        }
    }
    let total_chunks = chunks.len();
    let retain_mission = e.scoped_mission(&bank, "retain_mission");
    let bank_mission = retain_mission.as_deref();
    let bank_config = e.store.bank_config(&bank.id)?;
    let free_form = bank_config
        .get("entities_allow_free_form")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let labels = BankLabels::parse(bank_config.get("entity_labels"), free_form)?;
    let mut extract_config = e.config.extract.clone();
    if let Some(labels) = &labels {
        extract_config.entity_labels = Some(labels.groups.clone()).filter(|g| !g.is_empty());
        extract_config.entities_allow_free_form = labels.free_form;
    }
    // Parse-failure re-asks across all chunks of this retain, surfaced in
    // the response so benchmark drivers can attribute retain latency.
    let reasks = Arc::new(AtomicUsize::new(0));
    // Futures are created in a plain loop (not a stream combinator closure
    // returning a borrowing future): axum's Send check on the handler
    // future would otherwise trip rustc's higher-ranked closure
    // limitation ("implementation of FnOnce is not general enough").
    let mut chunk_futs: Vec<ChunkFut<'_>> = Vec::with_capacity(total_chunks);
    for (idx, chunk) in chunks.iter().enumerate() {
        chunk_futs.push(extract_chunk(
            e,
            ChunkExtract {
                content: chunk,
                chunk_index: idx,
                total_chunks,
                context: req.context.as_deref(),
                event_date,
                bank_mission,
                extract: &extract_config,
                reasks: Arc::clone(&reasks),
            },
        ));
    }
    let mut per_chunk: Vec<Vec<ExtractedFact>> = futures::stream::iter(chunk_futs)
        .buffered(CHUNK_EXTRACT_CONCURRENCY)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    for (index, chunk_facts) in per_chunk.iter_mut().enumerate() {
        for fact in chunk_facts {
            fact.chunk_index = index;
        }
    }
    let mut facts = merge_chunk_facts(per_chunk, OVERLAP_DEDUP_THRESHOLD);
    let reasks = reasks.load(Ordering::Relaxed);
    // Caller-supplied entities ride on every fact extracted from this
    // content, next to the extractor's own (upstream `entities`).
    for fact in &mut facts {
        if let Some(labels) = &labels {
            labels.apply(fact);
        }
        for user in &req.entities {
            let key = normalize_name(&user.name);
            if !key.is_empty() && !fact.entities.iter().any(|x| normalize_name(&x.name) == key) {
                fact.entities.push(user.clone());
            }
        }
    }

    let mut embeddings = if facts.is_empty() {
        vec![]
    } else {
        let embed = Arc::clone(&e.embed);
        let texts: Vec<String> = facts.iter().map(|f| f.text.clone()).collect();
        with_retry(e, || embed.embed(texts.clone())).await?
    };
    if embeddings.len() != facts.len() {
        return Err(Error::Embed("embedding count mismatch".into()));
    }

    // Document bookkeeping — only after every provider call succeeded, so a
    // failed extraction or embedding never destroys the previous version.
    // Replace (upstream default) drops the old version's memories; append
    // keeps them and extends the stored body with the new tail.
    let wall_now = e.clock.now_ms();
    let mut stale_entities: Vec<String> = vec![];
    // An append extends the document: labels the new tail does not restate
    // (metadata, context) stay as they were.
    let mut doc_metadata = req.metadata.clone();
    let mut doc_context = req.context.clone();
    let (body, doc_tags, created_at_ms) = match (&existing_doc, req.update_mode) {
        (Some(doc), UpdateMode::Append) => {
            let mut tags = doc.tags.clone();
            for t in &req.tags {
                if !tags.contains(t) {
                    tags.push(t.clone());
                }
            }
            doc_metadata = doc_metadata.or_else(|| doc.metadata.clone());
            doc_context = doc_context.or_else(|| {
                doc.retain_params
                    .get("context")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            });
            (
                join_document_parts(&doc.original_text, &req.content),
                tags,
                doc.created_at_ms,
            )
        }
        (Some(doc), UpdateMode::Replace) => {
            let (_, affected) = e.store.delete_document_memories(&bank.id, &document_id)?;
            stale_entities = affected;
            (req.content.clone(), req.tags.clone(), doc.created_at_ms)
        }
        (None, _) => (req.content.clone(), req.tags.clone(), wall_now),
    };
    e.store.upsert_document(&Document {
        id: document_id.clone(),
        bank_id: bank.id.clone(),
        content_hash: content_hash(&body),
        original_text: body,
        tags: doc_tags,
        metadata: doc_metadata,
        retain_params: serde_json::json!({
            "context": doc_context,
            "event_date": event_date.map(|d| d.to_rfc3339()),
        }),
        created_at_ms,
        updated_at_ms: wall_now,
    })?;
    // An append continues the document's numbering; a replace already
    // dropped the old chunks with the old memories.
    let first_chunk = if existing_doc.is_some() && req.update_mode == UpdateMode::Append {
        e.store.next_chunk_index(&bank.id, &document_id)?
    } else {
        0
    };
    let stored_chunks: Vec<Chunk> = chunks
        .iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            id: format!("{}_{}_{}", bank.id, document_id, first_chunk + i),
            bank_id: bank.id.clone(),
            document_id: document_id.clone(),
            chunk_index: first_chunk + i,
            text: text.clone(),
            document_tags: vec![],
        })
        .collect();
    e.store.insert_chunks(&stored_chunks)?;

    if facts.is_empty() {
        if !stale_entities.is_empty() {
            e.store
                .mark_entities_dirty(&bank.id, &stale_entities, "document_replaced")?;
            e.wake_worker();
        }
        return Ok(RetainResponse {
            memories_stored: 0,
            entities: vec![],
            edges_created: 0,
            reasks,
            document_id: Some(document_id),
            defense: None,
        });
    }

    let mut alias_cache: HashMap<String, Entity> = HashMap::new();
    let mut bank_entities = e.store.entities_for_bank(&bank.id)?;
    // Whole-bank co-occurrence snapshot (partner names + per-name degrees),
    // taken before this batch's links land — the same pre-commit view
    // upstream's resolution reads; this batch's pairs flush with the links.
    let co_index = e.store.cooccurrence_index(&bank.id)?;
    let mut touched: Vec<String> = vec![];
    // Mentions that failed resolution: (normalized alias key, lowercased
    // display, extractor type), in mention order; clustered into entities
    // after the per-mention pass (upstream's in-batch dedup).
    let mut pending: Vec<(String, String, EntityType)> = vec![];
    let mut per_fact_entities: Vec<Vec<(Entity, String)>> = vec![vec![]; facts.len()];

    for (fi, fact) in facts.iter().enumerate() {
        // The extractor's per-fact entity list is the mention's "nearby"
        // set (upstream `nearby_entities`): same-fact siblings matched by
        // name against the candidate's co-occurring partners.
        let fact_names: HashSet<String> = fact
            .entities
            .iter()
            .map(|em| collapse_ws(&em.name).to_lowercase())
            .collect();
        let event_ms = fact.occurred_start_ms.unwrap_or(now);

        for em in &fact.entities {
            let norm = normalize_name(&em.name);
            if norm.is_empty() {
                continue;
            }
            let mention_lower = collapse_ws(&em.name).to_lowercase();
            let mut nearby = fact_names.clone();
            nearby.remove(&mention_lower);

            let hit = if let Some(x) = alias_cache.get(&norm) {
                Some(x.clone())
            } else {
                match e.store.find_by_alias(&bank.id, &norm)? {
                    Some(x) => Some(x),
                    None => resolve_fuzzy(
                        e,
                        &bank.id,
                        &norm,
                        &mention_lower,
                        &em.etype,
                        &nearby,
                        &bank_entities,
                        &co_index,
                        event_ms,
                    )?,
                }
            };

            match hit {
                Some(ent) => {
                    alias_cache.insert(norm, ent.clone());
                    if !bank_entities.iter().any(|x| x.id == ent.id) {
                        bank_entities.push(ent.clone());
                    }
                    if !touched.contains(&ent.id) {
                        touched.push(ent.id.clone());
                    }
                    per_fact_entities[fi].push((ent, em.name.clone()));
                }
                None => pending.push((norm, mention_lower, em.etype)),
            }
        }
    }

    // Cluster the failed names into canonical spellings and find-or-create
    // each cluster's entity (upstream `_intrabatch_canonical_map` + the
    // grouped INSERT's ON CONFLICT semantics: a canonical spelling that
    // exactly names an existing entity reuses it).
    if !pending.is_empty() {
        let display_of = |norm: &str| -> String {
            facts
                .iter()
                .flat_map(|f| f.entities.iter())
                .map(|em| (normalize_name(&em.name), collapse_ws(&em.name)))
                .find(|(n, _)| n == norm)
                .map(|(_, d)| d)
                .unwrap_or_else(|| norm.to_string())
        };
        let names: Vec<(String, usize)> = {
            let mut counts: HashMap<String, (String, usize)> = HashMap::new();
            for (norm, _, _) in &pending {
                let display = display_of(norm);
                let slot = counts.entry(norm.clone()).or_insert((display, 0));
                slot.1 += 1;
            }
            counts.into_values().collect()
        };
        let canonical_by_lower = crate::entity::cluster_new_entity_names(&names, &e.config.entity);
        for (norm, _lower, etype) in &pending {
            let display = display_of(norm);
            let canonical_display = canonical_by_lower.get(&display.to_lowercase()).cloned();
            let ent = match canonical_display {
                Some(canonical) if canonical != display => {
                    // folded into a cluster: find-or-create the canonical
                    let cnorm = normalize_name(&canonical);
                    match e.store.find_by_alias(&bank.id, &cnorm)? {
                        Some(x) => x,
                        None => e.store.create_entity(&NewEntity {
                            bank_id: bank.id.clone(),
                            name: canonical.clone(),
                            etype: *etype,
                        })?,
                    }
                }
                _ => match e.store.find_by_alias(&bank.id, norm)? {
                    Some(x) => x,
                    None => e.store.create_entity(&NewEntity {
                        bank_id: bank.id.clone(),
                        name: display.clone(),
                        etype: *etype,
                    })?,
                },
            };
            e.store.add_alias(&bank.id, norm, &ent.id)?;
            alias_cache.insert(norm.clone(), ent.clone());
            if !touched.contains(&ent.id) {
                touched.push(ent.id.clone());
            }
        }
        // Second pass: fill per_fact_entities for the pending mentions.
        for (fi, fact) in facts.iter().enumerate() {
            for em in &fact.entities {
                let norm = normalize_name(&em.name);
                if norm.is_empty() {
                    continue;
                }
                if let Some(ent) = alias_cache.get(&norm) {
                    let already = per_fact_entities[fi].iter().any(|(x, _)| x.id == ent.id);
                    if !already {
                        per_fact_entities[fi].push((ent.clone(), em.name.clone()));
                    }
                }
            }
        }
    }

    let records: Vec<NewMemoryRecord> = facts
        .iter()
        .enumerate()
        .map(|(i, f)| NewMemoryRecord {
            kind: f.fact_type,
            text: f.text.clone(),
            confidence: None,
            occurred_start_ms: f.occurred_start_ms,
            occurred_end_ms: f.occurred_end_ms,
            mentioned_at_ms: now,
            embedding: embeddings[i].clone(),
            source: None,
            context: req.context.clone(),
            document_id: Some(document_id.clone()),
            tags: {
                let mut tags = req.tags.clone();
                for tag in &f.tags {
                    if !tags.contains(tag) {
                        tags.push(tag.clone());
                    }
                }
                tags
            },
            metadata: req.metadata.clone(),
            chunk_id: stored_chunks.get(f.chunk_index).map(|c| c.id.clone()),
        })
        .collect();
    let stored = e.store.insert_memories(&bank.id, &records)?;
    for ((mem, ents), fact) in stored.iter().zip(&per_fact_entities).zip(&facts) {
        if ents.is_empty() {
            continue;
        }
        let links: Vec<(String, String)> = ents
            .iter()
            .map(|(ent, mention)| (ent.id.clone(), mention.clone()))
            .collect();
        // The entity's last_mentioned advances by the fact's effective
        // date (upstream's last_seen stat flush), not the retain's wall
        // clock.
        let at_ms = fact.occurred_start_ms.unwrap_or(now);
        e.store.link_memory_entities(&mem.id, &links, at_ms)?;
    }

    let new_ids: HashSet<String> = stored.iter().map(|m| m.id.clone()).collect();
    let recent = e.store.recent_memories(&bank.id, 60)?;
    // Nearest pre-existing neighbours per new fact, through the store's
    // semantic search (KNN when the bank is vector-indexed, RS-7; one
    // brute-force scan for the whole batch otherwise).
    let semantic_candidates = {
        let queries: Vec<&[f32]> = embeddings.iter().map(Vec::as_slice).collect();
        let not_new = |id: &str| !new_ids.contains(id);
        e.store.search_semantic(
            &bank.id,
            &queries,
            &ALL_KINDS,
            e.config.semantic_edge_cap,
            Some(&not_new),
        )?
    };
    // One batched facts fetch for every entity the new facts mention
    // (CR-14): the entity-edge builder used to call `facts_for_entity`
    // once per (memory x entity) pair.
    let mut edge_entity_ids: Vec<String> = per_fact_entities
        .iter()
        .flat_map(|ents| ents.iter().map(|(ent, _)| ent.id.clone()))
        .collect();
    edge_entity_ids.sort();
    edge_entity_ids.dedup();
    let facts_map = e
        .store
        .facts_for_entities(&edge_entity_ids, e.config.entity_backfill + 1)?;

    let mut edges: Vec<Edge> = vec![];
    edges.extend(build_entity_edges(
        &e.config,
        &bank.id,
        &stored,
        &per_fact_entities,
        &facts_map,
    ));
    edges.extend(build_temporal_edges(
        &e.config, &bank.id, &stored, &recent, &new_ids,
    ));
    edges.extend(build_semantic_edges(
        &e.config,
        &bank.id,
        &stored,
        &semantic_candidates,
    ));
    edges.extend(build_causal_edges(&bank.id, &stored, &facts));

    e.store.add_edges(&edges)?;
    if !stale_entities.is_empty() {
        e.store
            .mark_entities_dirty(&bank.id, &stale_entities, "document_replaced")?;
        e.wake_worker();
    }
    if !touched.is_empty() {
        e.store
            .mark_entities_dirty(&bank.id, &touched, "new_facts")?;
        // Non-blocking wakeup: the background worker (if any) picks the
        // dirty queue up on its next pass; retain never waits on it.
        e.wake_worker();
    }

    embeddings.clear();
    Ok(RetainResponse {
        memories_stored: stored.len(),
        entities: touched,
        edges_created: edges.len(),
        reasks,
        document_id: Some(document_id),
        defense: None,
    })
}

/// Stable content fingerprint for a document body (FNV-1a 64, hex). Only
/// ever compared for equality against values this function produced.
pub(crate) fn content_hash(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// How a document's stored body and an appended tail become one body
/// (upstream `join_document_parts`): two JSON arrays of objects merge into
/// one array so a conversation stored as JSON stays valid JSON; anything
/// else joins with a newline.
pub(crate) fn join_document_parts(existing: &str, incoming: &str) -> String {
    let as_objects = |text: &str| -> Option<Vec<serde_json::Value>> {
        match serde_json::from_str::<serde_json::Value>(text).ok()? {
            serde_json::Value::Array(items) if items.iter().all(|v| v.is_object()) => Some(items),
            _ => None,
        }
    };
    match (as_objects(existing), as_objects(incoming)) {
        (Some(mut a), Some(b)) => {
            a.extend(b);
            serde_json::Value::Array(a).to_string()
        }
        _ => format!("{existing}\n{incoming}"),
    }
}

/// Entity edges: link each stored fact to its entity's other recent facts
/// (`facts_for` = one batched `facts_for_entities` snapshot per retain).
/// The walk stops at the fact itself or after `entity_backfill` siblings,
/// whichever comes first in the sibling list (newest first).
fn build_entity_edges(
    cfg: &EngineConfig,
    bank_id: &str,
    stored: &[Memory],
    per_fact_entities: &[Vec<(Entity, String)>],
    facts_for: &HashMap<String, Vec<Memory>>,
) -> Vec<Edge> {
    let mut edges = vec![];
    for (mem, ents) in stored.iter().zip(per_fact_entities) {
        for (ent, _) in ents {
            let siblings = facts_for.get(&ent.id).map(Vec::as_slice).unwrap_or(&[]);
            for (n, sib) in siblings.iter().enumerate() {
                if sib.id == mem.id || n >= cfg.entity_backfill {
                    break;
                }
                edges.push(Edge {
                    bank_id: bank_id.to_string(),
                    a_id: mem.id.clone(),
                    b_id: sib.id.clone(),
                    kind: EdgeKind::Entity,
                    label: Some(ent.id.clone()),
                    weight: 1.0,
                });
            }
        }
    }
    edges
}

/// Temporal edges: connect the new facts to each other and to recent
/// memories when their occurrence times are close enough that
/// `temporal_weight` clears `temporal_edge_min_weight`. `new_ids` keeps
/// the recent-memory pass from re-linking facts already paired above.
fn build_temporal_edges(
    cfg: &EngineConfig,
    bank_id: &str,
    stored: &[Memory],
    recent: &[Memory],
    new_ids: &HashSet<String>,
) -> Vec<Edge> {
    let t_of = |m: &Memory| m.occurred_start_ms.unwrap_or(m.mentioned_at_ms);
    let mut edges = vec![];
    for a in stored {
        for b in stored {
            if a.id == b.id {
                continue;
            }
            let dt = (t_of(a) - t_of(b)).abs();
            let w = temporal_weight(dt, cfg.temporal_sigma_ms);
            if w >= cfg.temporal_edge_min_weight {
                edges.push(Edge {
                    bank_id: bank_id.to_string(),
                    a_id: a.id.clone(),
                    b_id: b.id.clone(),
                    kind: EdgeKind::Temporal,
                    label: None,
                    weight: w,
                });
            }
        }
    }
    for a in stored {
        for r in recent {
            if new_ids.contains(&r.id) {
                continue;
            }
            let dt = (t_of(a) - t_of(r)).abs();
            let w = temporal_weight(dt, cfg.temporal_sigma_ms);
            if w >= cfg.temporal_edge_min_weight {
                edges.push(Edge {
                    bank_id: bank_id.to_string(),
                    a_id: a.id.clone(),
                    b_id: r.id.clone(),
                    kind: EdgeKind::Temporal,
                    label: None,
                    weight: w,
                });
            }
        }
    }
    edges
}

/// Every memory kind — semantic edges may land on any embedded item.
const ALL_KINDS: [MemoryKind; 4] = [
    MemoryKind::World,
    MemoryKind::Experience,
    MemoryKind::Observation,
    MemoryKind::Opinion,
];

/// Semantic edges: connect each new fact to its most similar existing
/// memories (cosine >= `semantic_edge_threshold`), keeping the top
/// `semantic_edge_cap` per fact. `candidates[i]` is stored fact `i`'s
/// nearest pre-existing items, best first, with this batch's new ids
/// already excluded (no self-matches).
fn build_semantic_edges(
    cfg: &EngineConfig,
    bank_id: &str,
    stored: &[Memory],
    candidates: &[Vec<SemanticHit>],
) -> Vec<Edge> {
    let mut edges = vec![];
    for (a, hits) in stored.iter().zip(candidates) {
        for (id, _, w) in hits
            .iter()
            .filter(|(_, _, c)| *c >= cfg.semantic_edge_threshold)
            .take(cfg.semantic_edge_cap)
        {
            edges.push(Edge {
                bank_id: bank_id.to_string(),
                a_id: a.id.clone(),
                b_id: id.clone(),
                kind: EdgeKind::Semantic,
                label: None,
                weight: *w,
            });
        }
    }
    edges
}

/// Causal edges: each fact's in-batch causal references (already remapped
/// to global fact indexes by `merge_chunk_facts`) become an edge between
/// the corresponding stored memories. Out-of-range and self refs are
/// dropped.
fn build_causal_edges(bank_id: &str, stored: &[Memory], facts: &[ExtractedFact]) -> Vec<Edge> {
    let mut edges = vec![];
    for (i, fact) in facts.iter().enumerate() {
        for c in &fact.causal {
            if c.fact_index < stored.len() && c.fact_index != i {
                edges.push(Edge {
                    bank_id: bank_id.to_string(),
                    a_id: stored[i].id.clone(),
                    b_id: stored[c.fact_index].id.clone(),
                    kind: EdgeKind::Causal,
                    label: Some(c.relation.clone()),
                    weight: 1.0,
                });
            }
        }
    }
    edges
}

/// Everything one extraction call needs, bundled so the recursive re-ask
/// path can pass it through unchanged.
struct ChunkExtract<'a> {
    content: &'a str,
    chunk_index: usize,
    total_chunks: usize,
    context: Option<&'a str>,
    event_date: Option<DateTime<Utc>>,
    bank_mission: Option<&'a str>,
    /// The engine's extraction settings plus this bank's entity labels.
    extract: &'a ExtractionConfig,
    /// Shared per-retain parse-failure re-ask counter.
    reasks: Arc<AtomicUsize>,
}

/// Extract facts from one retain chunk, retrying transient provider failures.
/// If the extraction output fails to parse (e.g. JSON truncated by an output
/// token limit), the chunk is re-asked once at half size — upstream's
/// auto-split — before the error surfaces. Boxed so retains can buffer a
/// bounded number of chunk extractions concurrently (CR-19).
fn extract_chunk<'a>(e: &'a Engine, args: ChunkExtract<'a>) -> ChunkFut<'a> {
    extract_chunk_at(e, args, 0)
}

/// Boxed because it recurses once (re-ask at half size); the depth cap keeps
/// the recursion at one level.
fn extract_chunk_at<'a>(e: &'a Engine, args: ChunkExtract<'a>, depth: usize) -> ChunkFut<'a> {
    Box::pin(async move {
        let (system, user) = build_extraction_messages(
            args.extract,
            args.bank_mission,
            args.content,
            args.chunk_index,
            args.total_chunks,
            args.event_date,
            args.context,
        );
        let llm = e.llm_for_retain();
        let request = LlmRequest {
            system: Some(system),
            messages: vec![ChatMessage::user(user)],
            json_mode: true,
            temperature: Some(0.0),
            max_tokens: e.config.retain_max_completion_tokens,
            ..Default::default()
        };
        let resp = with_retry(e, || llm.complete(&request)).await?;
        match parse_extraction_with_reference(&resp.content.unwrap_or_default(), args.event_date) {
            Ok(facts) => Ok(facts),
            Err(err) => {
                if depth == 0 {
                    if let Some((head, tail)) = split_half(args.content, e.tokens.as_ref()) {
                        let n = args.reasks.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!(
                            "rustsight: retain chunk {} of {} failed to parse ({err}); \
                             re-asking at half size (re-ask #{n})",
                            args.chunk_index + 1,
                            args.total_chunks
                        );
                        let reasks = Arc::clone(&args.reasks);
                        let first = extract_chunk_at(
                            e,
                            ChunkExtract {
                                content: &head,
                                reasks: Arc::clone(&reasks),
                                ..args
                            },
                            depth + 1,
                        )
                        .await?;
                        let offset = first.len();
                        let mut merged = first;
                        for mut fact in extract_chunk_at(
                            e,
                            ChunkExtract {
                                content: &tail,
                                reasks,
                                ..args
                            },
                            depth + 1,
                        )
                        .await?
                        {
                            for c in &mut fact.causal {
                                c.fact_index += offset;
                            }
                            merged.push(fact);
                        }
                        return Ok(merged);
                    }
                }
                Err(err)
            }
        }
    })
}

/// Fuzzy-resolve one unresolved mention against the bank's entities using
/// upstream's scoring model (trigram floor + word-compat gate, then
/// 0.5·name-similarity + 0.3·co-occurrence + 0.2·recency against the 0.6
/// threshold). Candidates are the bank's whole entity list — upstream's
/// "full" strategy shape; the gates make the scored set identical to the
/// trigram-probe strategy's. Returns `None` when no candidate clears the
/// threshold (the caller queues the name for in-batch clustering).
#[allow(clippy::too_many_arguments)]
fn resolve_fuzzy(
    e: &Engine,
    bank_id: &str,
    norm: &str,
    mention_lower: &str,
    _etype: &EntityType,
    nearby: &HashSet<String>,
    candidates: &[Entity],
    co_index: &CooccurrenceIndex,
    event_ms: i64,
) -> Result<Option<Entity>> {
    let cfg = &e.config.entity;
    let mut best: Option<(f32, Entity)> = None;
    for c in candidates {
        let candidate_lower = collapse_ws(&c.name).to_lowercase();
        if let Some(score) = score_candidate(
            mention_lower,
            &candidate_lower,
            nearby,
            co_index.partners_by_entity.get(&c.id),
            &co_index.degree_by_name,
            event_ms,
            c.last_mentioned_ms,
            cfg,
        ) {
            if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                best = Some((score, c.clone()));
            }
        }
    }
    if let Some((score, ent)) = best {
        // Inclusive threshold on a score rounded to 6 decimal places, so
        // compositions that sum to exactly the threshold compare equal
        // regardless of which signals produced them.
        if (score * 1_000_000.0).round() / 1_000_000.0 >= cfg.link_threshold {
            e.store.add_alias(bank_id, norm, &ent.id)?;
            return Ok(Some(ent));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: &str, mentioned: i64, occurred: Option<i64>) -> Memory {
        Memory {
            id: id.into(),
            bank_id: "b".into(),
            kind: MemoryKind::World,
            text: format!("fact {id}"),
            confidence: None,
            occurred_start_ms: occurred,
            occurred_end_ms: None,
            mentioned_at_ms: mentioned,
            source: None,
            context: None,
            document_id: None,
            tags: vec![],
            metadata: None,
        }
    }

    fn ent(id: &str) -> Entity {
        Entity {
            id: id.into(),
            bank_id: "b".into(),
            name: format!("entity {id}"),
            etype: EntityType::Person,
            last_mentioned_ms: 0,
        }
    }

    fn fact(text: &str, causal: Vec<(usize, &str)>) -> ExtractedFact {
        ExtractedFact {
            text: text.into(),
            fact_type: MemoryKind::World,
            entities: vec![],
            occurred_start_ms: None,
            occurred_end_ms: None,
            causal: causal
                .into_iter()
                .map(|(fact_index, relation)| CausalRef {
                    fact_index,
                    relation: relation.into(),
                })
                .collect(),
            chunk_index: 0,
            labels: None,
            tags: vec![],
        }
    }

    #[test]
    fn entity_edges_stop_at_self_and_backfill_cap() {
        let cfg = EngineConfig {
            entity_backfill: 2,
            ..Default::default()
        };
        let stored = vec![mem("m0", 0, None)];
        let e0 = ent("E");
        let per_fact_entities: Vec<Vec<(Entity, String)>> = vec![vec![(e0.clone(), "E".into())]];
        let mut facts_for = HashMap::new();

        // self at the head of the sibling list -> walk stops before any link
        facts_for.insert(
            e0.id.clone(),
            vec![mem("m0", 0, None), mem("s1", 0, None), mem("s2", 0, None)],
        );
        let edges = build_entity_edges(&cfg, "b", &stored, &per_fact_entities, &facts_for);
        assert!(
            edges.is_empty(),
            "self at head must stop the walk: {edges:?}"
        );

        // self mid-list -> only earlier siblings link
        facts_for.insert(
            e0.id.clone(),
            vec![mem("s1", 0, None), mem("m0", 0, None), mem("s2", 0, None)],
        );
        let edges = build_entity_edges(&cfg, "b", &stored, &per_fact_entities, &facts_for);
        assert_eq!(edges.len(), 1, "{edges:?}");
        assert_eq!(edges[0].b_id, "s1");
        assert_eq!(edges[0].a_id, "m0");
        assert_eq!(edges[0].label.as_deref(), Some("E"));
        assert_eq!(edges[0].kind, EdgeKind::Entity);

        // self absent -> backfill cap applies
        facts_for.insert(
            e0.id.clone(),
            vec![
                mem("s1", 0, None),
                mem("s2", 0, None),
                mem("s3", 0, None),
                mem("s4", 0, None),
            ],
        );
        let edges = build_entity_edges(&cfg, "b", &stored, &per_fact_entities, &facts_for);
        assert_eq!(edges.len(), 2, "{edges:?}");
        assert!(edges.iter().all(|x| x.a_id == "m0"));
        assert!(edges.iter().any(|x| x.b_id == "s1"));
        assert!(edges.iter().any(|x| x.b_id == "s2"));
    }

    #[test]
    fn entity_edges_skip_facts_without_entities() {
        let cfg = EngineConfig::default();
        let stored = vec![mem("m0", 0, None), mem("m1", 0, None)];
        // only m0 mentions an entity; m1 must not produce edges
        let per_fact_entities: Vec<Vec<(Entity, String)>> =
            vec![vec![(ent("E"), "E".into())], vec![]];
        let facts_for = HashMap::new();
        let edges = build_entity_edges(&cfg, "b", &stored, &per_fact_entities, &facts_for);
        assert!(edges.is_empty(), "{edges:?}");
    }

    #[test]
    fn temporal_edges_respect_min_weight_and_new_ids() {
        let cfg = EngineConfig {
            temporal_edge_min_weight: 0.25,
            ..Default::default()
        }; // default sigma: 7 days (Laplacian decay)
        let day = 86_400_000i64;
        let stored = vec![mem("a", 0, Some(0)), mem("b", 0, Some(day))];
        let recent = vec![
            mem("far", 0, Some(400 * day)), // exp(-400/7) ~ 0
            mem("near", 0, Some(2 * day)),  // exp(-2/7) ~ 0.75
            stored[0].clone(),              // in new_ids -> skipped
        ];
        let new_ids: HashSet<String> = stored.iter().map(|m| m.id.clone()).collect();
        let edges = build_temporal_edges(&cfg, "b", &stored, &recent, &new_ids);

        // the stored pair links both ways (exp(-1/7) ~ 0.87 >= 0.25)
        assert!(edges
            .iter()
            .any(|x| x.a_id == "a" && x.b_id == "b" && x.kind == EdgeKind::Temporal));
        assert!(edges.iter().any(|x| x.a_id == "b" && x.b_id == "a"));
        // near recents link from both stored facts; far does not
        assert_eq!(edges.iter().filter(|x| x.b_id == "near").count(), 2);
        assert_eq!(edges.iter().filter(|x| x.b_id == "far").count(), 0);
        // the skipped new_ids member would otherwise self-loop (dt = 0)
        assert!(
            !edges.iter().any(|x| x.a_id == x.b_id),
            "new_ids guard must prevent self-loops: {edges:?}"
        );
    }

    #[test]
    fn semantic_edges_threshold_cap_and_self_exclusion() {
        let cfg = EngineConfig {
            semantic_edge_threshold: 0.82,
            semantic_edge_cap: 2,
            ..Default::default()
        };
        let stored = vec![mem("a", 0, None)];
        let embeddings = [vec![1.0, 0.0]];
        let existing = vec![
            ("x1".to_string(), MemoryKind::World, vec![1.0, 0.0]), // cos 1.0
            ("x2".to_string(), MemoryKind::World, vec![0.0, 1.0]), // cos 0.0
            ("x3".to_string(), MemoryKind::World, vec![0.99, 0.1]), // cos ~0.995
            ("a".to_string(), MemoryKind::World, vec![1.0, 0.0]),  // new id -> excluded
        ];
        let new_ids: HashSet<String> = stored.iter().map(|m| m.id.clone()).collect();
        let not_new = |id: &str| !new_ids.contains(id);
        let candidates = |cap| {
            crate::brute_force_semantic(
                &existing,
                &[embeddings[0].as_slice()],
                &ALL_KINDS,
                cap,
                Some(&not_new),
            )
        };
        let edges = build_semantic_edges(&cfg, "b", &stored, &candidates(2));
        assert_eq!(edges.len(), 2, "{edges:?}");
        // similarity-descending order, x2 below threshold, self excluded
        assert_eq!(edges[0].b_id, "x1");
        assert!((edges[0].weight - 1.0).abs() < 1e-6);
        assert_eq!(edges[1].b_id, "x3");
        assert!(edges[1].weight > 0.98 && edges[1].weight < 1.0);
        assert_eq!(edges[0].kind, EdgeKind::Semantic);

        // cap 1 keeps only the closest match
        let cfg = EngineConfig {
            semantic_edge_threshold: 0.82,
            semantic_edge_cap: 1,
            ..Default::default()
        };
        let edges = build_semantic_edges(&cfg, "b", &stored, &candidates(1));
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].b_id, "x1");
    }

    #[test]
    fn causal_edges_from_remapped_fact_refs() {
        let stored = vec![mem("m0", 0, None), mem("m1", 0, None), mem("m2", 0, None)];
        let facts = vec![
            fact("zero", vec![(0, "self")]), // self-ref dropped
            fact("one", vec![(0, "causes")]),
            fact("two", vec![(9, "enables")]), // out of range dropped
        ];
        let edges = build_causal_edges("b", &stored, &facts);
        assert_eq!(edges.len(), 1, "{edges:?}");
        assert_eq!(edges[0].a_id, "m1");
        assert_eq!(edges[0].b_id, "m0");
        assert_eq!(edges[0].label.as_deref(), Some("causes"));
        assert_eq!(edges[0].kind, EdgeKind::Causal);
    }
}
