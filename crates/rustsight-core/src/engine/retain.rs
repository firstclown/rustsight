use crate::engine::chunks::{merge_chunk_facts, split_chunks, split_half, OVERLAP_DEDUP_THRESHOLD};
use crate::engine::{with_retry, Engine};
use crate::entity::{combined_score, normalize_name, string_similarity};
use crate::prompts::extraction::{build_extraction_messages, parse_extraction_with_reference};
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::{cosine, temporal_weight, Error, Result};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(super) async fn retain(e: &Engine, req: RetainRequest) -> Result<RetainResponse> {
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
    let event_date = DateTime::from_timestamp_millis(now).map(|dt| dt.with_timezone(&Utc));
    let total_chunks = chunks.len();
    let bank_mission = bank.mission.as_deref();
    let mut per_chunk: Vec<Vec<ExtractedFact>> = Vec::with_capacity(total_chunks);
    for (idx, chunk) in chunks.iter().enumerate() {
        per_chunk.push(
            extract_chunk(
                e,
                ChunkExtract {
                    content: chunk,
                    chunk_index: idx,
                    total_chunks,
                    context: req.context.as_deref(),
                    event_date,
                    bank_mission,
                },
            )
            .await?,
        );
    }
    let facts = merge_chunk_facts(per_chunk, OVERLAP_DEDUP_THRESHOLD);
    if facts.is_empty() {
        return Ok(RetainResponse {
            memories_stored: 0,
            entities: vec![],
            edges_created: 0,
        });
    }

    let embed = Arc::clone(&e.embed);
    let texts: Vec<String> = facts.iter().map(|f| f.text.clone()).collect();
    let mut embeddings = with_retry(e, || embed.embed(texts.clone())).await?;
    if embeddings.len() != facts.len() {
        return Err(Error::Embed("embedding count mismatch".into()));
    }

    let mut alias_cache: HashMap<String, Entity> = HashMap::new();
    let mut bank_entities = e.store.entities_for_bank(&bank.id)?;
    let mut touched: Vec<String> = vec![];
    let mut per_fact_entities: Vec<Vec<(Entity, String)>> = Vec::with_capacity(facts.len());

    for fact in &facts {
        let mut resolved = vec![];
        for em in &fact.entities {
            let norm = normalize_name(&em.name);
            if norm.is_empty() {
                continue;
            }
            let ent = if let Some(x) = alias_cache.get(&norm) {
                x.clone()
            } else {
                let found = match e.store.find_by_alias(&bank.id, &norm)? {
                    Some(x) => x,
                    None => resolve_fuzzy(
                        e,
                        &bank.id,
                        &norm,
                        &em.name,
                        &em.etype,
                        &alias_cache,
                        &bank_entities,
                        now,
                    )?,
                };
                alias_cache.insert(norm, found.clone());
                if !bank_entities.iter().any(|x| x.id == found.id) {
                    bank_entities.push(found.clone());
                }
                found
            };
            if !touched.contains(&ent.id) {
                touched.push(ent.id.clone());
            }
            resolved.push((ent, em.name.clone()));
        }
        per_fact_entities.push(resolved);
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
        })
        .collect();
    let stored = e.store.insert_memories(&bank.id, &records)?;
    for (mem, ents) in stored.iter().zip(&per_fact_entities) {
        if ents.is_empty() {
            continue;
        }
        let links: Vec<(String, String)> = ents
            .iter()
            .map(|(ent, mention)| (ent.id.clone(), mention.clone()))
            .collect();
        e.store.link_memory_entities(&mem.id, &links)?;
    }

    let new_ids: HashSet<String> = stored.iter().map(|m| m.id.clone()).collect();
    let mut edges: Vec<Edge> = vec![];

    for (mem, ents) in stored.iter().zip(&per_fact_entities) {
        for (ent, _) in ents {
            let siblings = e
                .store
                .facts_for_entity(&ent.id, e.config.entity_backfill + 1)?;
            for (n, sib) in siblings.iter().enumerate() {
                if sib.id == mem.id || n >= e.config.entity_backfill {
                    break;
                }
                edges.push(Edge {
                    bank_id: bank.id.clone(),
                    a_id: mem.id.clone(),
                    b_id: sib.id.clone(),
                    kind: EdgeKind::Entity,
                    label: Some(ent.id.clone()),
                    weight: 1.0,
                });
            }
        }
    }

    let t_of = |m: &Memory| m.occurred_start_ms.unwrap_or(m.mentioned_at_ms);
    for a in &stored {
        for b in &stored {
            if a.id == b.id {
                continue;
            }
            let dt = (t_of(a) - t_of(b)).abs();
            let w = temporal_weight(dt, e.config.temporal_sigma_ms);
            if w >= e.config.temporal_edge_min_weight {
                edges.push(Edge {
                    bank_id: bank.id.clone(),
                    a_id: a.id.clone(),
                    b_id: b.id.clone(),
                    kind: EdgeKind::Temporal,
                    label: None,
                    weight: w,
                });
            }
        }
    }
    let recent = e.store.recent_memories(&bank.id, 60)?;
    for a in &stored {
        for r in &recent {
            if new_ids.contains(&r.id) {
                continue;
            }
            let dt = (t_of(a) - t_of(r)).abs();
            let w = temporal_weight(dt, e.config.temporal_sigma_ms);
            if w >= e.config.temporal_edge_min_weight {
                edges.push(Edge {
                    bank_id: bank.id.clone(),
                    a_id: a.id.clone(),
                    b_id: r.id.clone(),
                    kind: EdgeKind::Temporal,
                    label: None,
                    weight: w,
                });
            }
        }
    }

    let existing_emb = e.store.all_embeddings(&bank.id)?;
    for (i, a) in stored.iter().enumerate() {
        let mut best: Vec<(f32, String)> = existing_emb
            .iter()
            .filter_map(|(id, _, v)| {
                if new_ids.contains(id) {
                    None
                } else {
                    let c = cosine(&embeddings[i], v);
                    (c >= e.config.semantic_edge_threshold).then(|| (c, id.clone()))
                }
            })
            .collect();
        best.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.1.cmp(&y.1))
        });
        best.truncate(e.config.semantic_edge_cap);
        for (w, id) in best {
            edges.push(Edge {
                bank_id: bank.id.clone(),
                a_id: a.id.clone(),
                b_id: id,
                kind: EdgeKind::Semantic,
                label: None,
                weight: w,
            });
        }
    }

    for (i, fact) in facts.iter().enumerate() {
        for c in &fact.causal {
            if c.fact_index < stored.len() && c.fact_index != i {
                edges.push(Edge {
                    bank_id: bank.id.clone(),
                    a_id: stored[i].id.clone(),
                    b_id: stored[c.fact_index].id.clone(),
                    kind: EdgeKind::Causal,
                    label: Some(c.relation.clone()),
                    weight: 1.0,
                });
            }
        }
    }

    e.store.add_edges(&edges)?;
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
    })
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
}

/// Extract facts from one retain chunk, retrying transient provider failures.
/// If the extraction output fails to parse (e.g. JSON truncated by an output
/// token limit), the chunk is re-asked once at half size — upstream's
/// auto-split — before the error surfaces.
async fn extract_chunk(e: &Engine, args: ChunkExtract<'_>) -> Result<Vec<ExtractedFact>> {
    extract_chunk_at(e, args, 0).await
}

/// Boxed because it recurses once (re-ask at half size); the depth cap keeps
/// the recursion at one level.
fn extract_chunk_at<'a>(
    e: &'a Engine,
    args: ChunkExtract<'a>,
    depth: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<ExtractedFact>>> + Send + 'a>> {
    Box::pin(async move {
        let (system, user) = build_extraction_messages(
            &e.config.extract,
            args.bank_mission,
            args.content,
            args.chunk_index,
            args.total_chunks,
            args.event_date,
            args.context,
        );
        let llm = Arc::clone(&e.llm);
        let request = LlmRequest {
            system: Some(system),
            messages: vec![ChatMessage::user(user)],
            json_mode: true,
            temperature: Some(0.0),
            ..Default::default()
        };
        let resp = with_retry(e, || llm.complete(&request)).await?;
        match parse_extraction_with_reference(&resp.content.unwrap_or_default(), args.event_date) {
            Ok(facts) => Ok(facts),
            Err(err) => {
                if depth == 0 {
                    if let Some((head, tail)) = split_half(args.content, e.tokens.as_ref()) {
                        let first = extract_chunk_at(
                            e,
                            ChunkExtract {
                                content: &head,
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

#[allow(clippy::too_many_arguments)]
fn resolve_fuzzy(
    e: &Engine,
    bank_id: &str,
    norm: &str,
    display: &str,
    etype: &EntityType,
    alias_cache: &HashMap<String, Entity>,
    candidates: &[Entity],
    now: i64,
) -> Result<Entity> {
    let cfg = &e.config.entity;
    let siblings: Vec<String> = alias_cache.values().map(|x| x.id.clone()).collect();
    let mut best: Option<(f32, Entity)> = None;
    for c in candidates {
        let s = string_similarity(display, &c.name);
        let co = if siblings.is_empty() {
            0.0
        } else {
            let co = e.store.coentities(&c.id, 50).unwrap_or_default();
            let hits = co
                .iter()
                .filter(|id| siblings.iter().any(|s| s == *id))
                .count() as f32;
            hits / siblings.len() as f32
        };
        let t = 1.0 - (((now - c.last_mentioned_ms).abs() as f32) / (90.0 * 86_400_000.0)).min(1.0);
        let score = combined_score(s, co, t, cfg);
        if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
            best = Some((score, c.clone()));
        }
    }
    if let Some((score, ent)) = best {
        if score >= cfg.link_threshold {
            e.store.add_alias(bank_id, norm, &ent.id)?;
            return Ok(ent);
        }
    }
    let ne = e.store.create_entity(&NewEntity {
        bank_id: bank_id.to_string(),
        name: display.to_string(),
        etype: *etype,
    })?;
    e.store.add_alias(bank_id, norm, &ne.id)?;
    Ok(ne)
}
