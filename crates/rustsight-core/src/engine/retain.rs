use crate::engine::chunks::{merge_chunk_facts, split_chunks, split_half, OVERLAP_DEDUP_THRESHOLD};
use crate::engine::{with_retry, Engine, EngineConfig};
use crate::entity::{combined_score, normalize_name, string_similarity};
use crate::prompts::extraction::{build_extraction_messages, parse_extraction_with_reference};
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::{cosine, temporal_weight, Error, Result};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// How many retain chunks extract concurrently (CR-19). Bounded on
/// purpose: each chunk is one LLM round trip, and an unbounded fan-out
/// would trip provider rate limits on large retains. `buffered` (not
/// `buffer_unordered`) keeps per-chunk results in chunk order — the
/// causal-index remap in `merge_chunk_facts` depends on it.
const CHUNK_EXTRACT_CONCURRENCY: usize = 2;

/// Co-occurrence probe window for fuzzy entity-resolution candidates.
const COENTITY_LIMIT: usize = 50;

/// Boxed per-chunk extraction future; boxing lets retain buffer a bounded
/// number of chunk extractions concurrently (CR-19) and the recursive
/// half-size re-ask return one concrete type.
type ChunkFut<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<ExtractedFact>>> + Send + 'a>>;

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
            },
        ));
    }
    let per_chunk: Vec<Vec<ExtractedFact>> = futures::stream::iter(chunk_futs)
        .buffered(CHUNK_EXTRACT_CONCURRENCY)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
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
    // One co-occurrence snapshot per retain (CR-14): resolution used to
    // fire `coentities` per candidate per unresolved mention, each taking
    // the connection mutex. The snapshot runs before any new memories are
    // linked, exactly when the lazy queries used to run, so a missing key
    // (entity without co-occurrences) matches the empty per-id result.
    let co_map = e.store.coentities_batch(
        &bank_entities
            .iter()
            .map(|x| x.id.clone())
            .collect::<Vec<_>>(),
        COENTITY_LIMIT,
    )?;
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
                        &co_map,
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
    let recent = e.store.recent_memories(&bank.id, 60)?;
    let existing_emb = e.store.all_embeddings(&bank.id)?;
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
        &embeddings,
        &existing_emb,
        &new_ids,
    ));
    edges.extend(build_causal_edges(&bank.id, &stored, &facts));

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

/// Semantic edges: connect each new fact to its most similar existing
/// memories (cosine >= `semantic_edge_threshold`), keeping the top
/// `semantic_edge_cap` per fact. `existing` is the bank's pre-existing
/// embeddings; anything in `new_ids` is filtered out as a self-match.
fn build_semantic_edges(
    cfg: &EngineConfig,
    bank_id: &str,
    stored: &[Memory],
    embeddings: &[Vec<f32>],
    existing: &[(MemoryId, MemoryKind, Vec<f32>)],
    new_ids: &HashSet<String>,
) -> Vec<Edge> {
    let mut edges = vec![];
    for (i, a) in stored.iter().enumerate() {
        let mut best: Vec<(f32, String)> = existing
            .iter()
            .filter_map(|(id, _, v)| {
                if new_ids.contains(id) {
                    None
                } else {
                    let c = cosine(&embeddings[i], v);
                    (c >= cfg.semantic_edge_threshold).then(|| (c, id.clone()))
                }
            })
            .collect();
        best.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.1.cmp(&y.1))
        });
        best.truncate(cfg.semantic_edge_cap);
        for (w, id) in best {
            edges.push(Edge {
                bank_id: bank_id.to_string(),
                a_id: a.id.clone(),
                b_id: id,
                kind: EdgeKind::Semantic,
                label: None,
                weight: w,
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
    co_map: &HashMap<String, Vec<String>>,
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
            let co = co_map.get(&c.id).map(Vec::as_slice).unwrap_or(&[]);
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
        let embeddings = vec![vec![1.0, 0.0]];
        let existing = vec![
            ("x1".to_string(), MemoryKind::World, vec![1.0, 0.0]), // cos 1.0
            ("x2".to_string(), MemoryKind::World, vec![0.0, 1.0]), // cos 0.0
            ("x3".to_string(), MemoryKind::World, vec![0.99, 0.1]), // cos ~0.995
            ("a".to_string(), MemoryKind::World, vec![1.0, 0.0]),  // new id -> excluded
        ];
        let new_ids: HashSet<String> = stored.iter().map(|m| m.id.clone()).collect();
        let edges = build_semantic_edges(&cfg, "b", &stored, &embeddings, &existing, &new_ids);
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
        let edges = build_semantic_edges(&cfg, "b", &stored, &embeddings, &existing, &new_ids);
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
