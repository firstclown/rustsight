use crate::engine::{with_retry, Engine};
use crate::prompts::consolidation::{
    build_consolidation_prompt, build_opinion_reinforcement_prompt, parse_observations,
    parse_reinforcement,
};
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::{string_similarity, Error, Result};
use futures::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Boxed per-entity consolidation future; boxing lets a pass buffer a
/// bounded number of entities concurrently (mirroring retain's chunk
/// extraction, CR-19).
type EntityFut<'a> = Pin<Box<dyn Future<Output = EntityStep> + Send + 'a>>;

/// What happened to one dirty entity inside a pass.
enum EntityStep {
    /// Observations were written and embedded; the entity's dirty row was
    /// cleared.
    Done {
        entity_id: String,
        observations: usize,
    },
    /// Fewer than two facts and no existing observations — nothing to
    /// synthesize; the dirty row was cleared (new facts re-dirty it).
    Ineligible,
    /// The soft pass deadline elapsed before this entity started; the
    /// dirty row survives and the next pass picks the entity up.
    Deadline,
    /// Synthesis failed for this entity (LLM error after retries, malformed
    /// response, store error); the dirty row survives for the next pass.
    Failed { entity_id: String, error: Error },
}

pub(super) async fn consolidate_once(
    e: &Engine,
    bank_id: &str,
    max_entities: usize,
) -> Result<usize> {
    if e.store.get_bank(bank_id)?.is_none() {
        return Err(Error::NotFound(format!("bank {bank_id}")));
    }
    let dirty = e.store.list_dirty_entities(bank_id, max_entities.max(1))?;
    if dirty.is_empty() {
        return Ok(0);
    }
    // Soft deadline: no NEW entity work starts once the pass budget
    // elapses; the caller (background worker or eval loop) immediately
    // starts a fresh pass, so a slow-but-progressing queue keeps making
    // durable progress instead of being killed mid-flight. The hard wall
    // in `Engine::consolidate_once` still aborts a pass whose in-flight
    // provider call truly wedges.
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(e.config.consolidate_timeout_secs.max(1));
    // One opinion snapshot for the whole pass (CR-6): `list_opinions` reads
    // the entire bank, so re-reading it per dirty entity made consolidation
    // quadratic in bank size. Reinforcement runs sequentially after the
    // synthesis fan-out below and writes updates back into the snapshot so
    // later entities still see fresh confidence/timestamps, exactly as the
    // per-entity re-fetch did.
    let mut opinions = e.store.list_opinions(bank_id)?;

    let mut futs: Vec<EntityFut<'_>> = Vec::with_capacity(dirty.len());
    for eid in &dirty {
        futs.push(consolidate_entity(e, bank_id, eid, deadline));
    }

    let mut updated = 0usize;
    let mut failed = 0usize;
    let mut deadline_skips = 0usize;
    let mut done: Vec<String> = vec![];
    // `buffered` (not `buffer_unordered`): per-entity results stay in dirty
    // order, which keeps scripted-provider tests deterministic at any
    // concurrency and gives the reinforcement loop a stable sequence.
    let mut steps = futures::stream::iter(futs).buffered(e.config.consolidate_concurrency.max(1));
    while let Some(step) = steps.next().await {
        match step {
            EntityStep::Done {
                entity_id,
                observations,
            } => {
                updated += observations;
                done.push(entity_id);
            }
            EntityStep::Ineligible => {}
            EntityStep::Deadline => deadline_skips += 1,
            EntityStep::Failed { entity_id, error } => {
                failed += 1;
                eprintln!("rustsight: consolidating {entity_id} in bank {bank_id} failed: {error}");
            }
        }
    }
    if failed > 0 {
        eprintln!(
            "rustsight: consolidation pass for bank {bank_id} left {failed} entities dirty \
             (reported above); they retry on the next pass"
        );
    }
    if deadline_skips > 0 {
        eprintln!(
            "rustsight: consolidation pass for bank {bank_id} hit its deadline with {deadline_skips} \
             entities left queued; they retry on the next pass"
        );
    }

    // ---- Reinforcement: sequential over the entities this pass completed,
    // sharing the snapshot taken above (see CR-6 note). ----
    if !opinions.is_empty() {
        for eid in &done {
            let matching: Vec<usize> = opinions
                .iter()
                .enumerate()
                .filter(|(_, o)| o.entity_ids.iter().any(|id| id == eid))
                .map(|(i, _)| i)
                .collect();
            if matching.is_empty() {
                continue;
            }
            let facts = match e.store.facts_for_entity(eid, 60) {
                Ok(f) => f,
                Err(err) => {
                    eprintln!("rustsight: reading facts for {eid} in bank {bank_id} failed: {err}");
                    continue;
                }
            };
            for i in matching {
                let op = opinions[i].clone();
                let new_facts: Vec<Memory> = facts
                    .iter()
                    .filter(|f| f.mentioned_at_ms > op.updated_at_ms)
                    .cloned()
                    .collect();
                if new_facts.is_empty() {
                    continue;
                }
                let now = e.clock.now_ms();
                let (system, user) =
                    build_opinion_reinforcement_prompt(&op.text, op.confidence, &new_facts);
                let request = LlmRequest {
                    system: Some(system),
                    messages: vec![ChatMessage::user(user)],
                    json_mode: true,
                    temperature: Some(0.0),
                    ..Default::default()
                };
                let llm = Arc::clone(&e.llm);
                let resp = match with_retry(e, || llm.complete(&request)).await {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!(
                            "rustsight: reinforcing opinion for {eid} in bank {bank_id} failed: {err}"
                        );
                        continue;
                    }
                };
                let verdict = match parse_reinforcement(&resp.content.unwrap_or_default()) {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!(
                            "rustsight: reinforcing opinion for {eid} in bank {bank_id} failed: {err}"
                        );
                        continue;
                    }
                };
                if let Some((text, confidence)) = verdict {
                    let mut upd = op.clone();
                    upd.history.push(ConfidencePoint {
                        confidence: op.confidence,
                        at_ms: now,
                        reason: "reinforced by consolidation".into(),
                    });
                    upd.text = text;
                    upd.confidence = confidence;
                    upd.updated_at_ms = now;
                    e.store.upsert_opinion(&upd)?;
                    opinions[i] = upd;
                }
            }
        }
    }

    Ok(updated)
}

/// Consolidate one dirty entity: synthesize (or refine) its observations,
/// embed them, then clear the entity's dirty row. Every failure isolates to
/// this entity — its dirty row survives and the next pass retries it — so
/// one malformed provider response can no longer discard the rest of a
/// pass's work.
fn consolidate_entity<'a>(
    e: &'a Engine,
    bank_id: &'a str,
    eid: &'a str,
    deadline: tokio::time::Instant,
) -> EntityFut<'a> {
    Box::pin(async move {
        if tokio::time::Instant::now() >= deadline {
            return EntityStep::Deadline;
        }
        let entity = match e.store.get_entity(eid) {
            Ok(Some(x)) => x,
            // Entity vanished (nothing references it anymore): nothing to
            // synthesize, clear the stale dirty row.
            Ok(None) => return clear_ineligible(e, bank_id, eid),
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        let facts = match e.store.facts_for_entity(eid, 60) {
            Ok(f) => f,
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        let existing = match e.store.observations_for_entity(eid) {
            Ok(f) => f,
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        if facts.len() < 2 && existing.is_empty() {
            return clear_ineligible(e, bank_id, eid);
        }
        let (system, user) = build_consolidation_prompt(&entity.name, &facts, &existing);
        let request = LlmRequest {
            system: Some(system),
            messages: vec![ChatMessage::user(user)],
            json_mode: true,
            temperature: Some(0.0),
            ..Default::default()
        };
        let llm = Arc::clone(&e.llm);
        let resp = match with_retry(e, || llm.complete(&request)).await {
            Ok(r) => r,
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        let parsed = match parse_observations(&resp.content.unwrap_or_default()) {
            Ok(p) => p,
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        let now = e.clock.now_ms();
        // (observation id, text) pairs awaiting the embedding backfill
        // below. Tracking ids (not just texts) means each observation gets
        // its own embedding even when two observations share identical text.
        let mut obs_pending: Vec<(String, String)> = vec![];
        let mut wrote = 0usize;
        for p in parsed {
            let mut evidence: Vec<Evidence> = vec![];
            for (idx, quote) in &p.evidence {
                if let Some(f) = facts.get(*idx) {
                    let quote = if quote.trim().is_empty() {
                        f.text.chars().take(160).collect()
                    } else {
                        quote.clone()
                    };
                    if !evidence.iter().any(|ev| ev.memory_id == f.id) {
                        evidence.push(Evidence {
                            memory_id: f.id.clone(),
                            quote,
                        });
                    }
                }
            }
            if evidence.is_empty() {
                if let Some(f) = facts.first() {
                    evidence.push(Evidence {
                        memory_id: f.id.clone(),
                        quote: f.text.chars().take(160).collect(),
                    });
                }
            }
            if evidence.is_empty() {
                continue;
            }
            let proof_count = evidence.len();
            let target = existing
                .iter()
                .find(|o| string_similarity(&p.text, &o.text) >= 0.85);
            match target {
                Some(o) => {
                    let mut upd = o.clone();
                    let old_text = upd.text.clone();
                    upd.text = p.text.clone();
                    for ev in evidence {
                        if !upd.evidence.iter().any(|x| x.memory_id == ev.memory_id) {
                            upd.evidence.push(ev);
                        }
                    }
                    upd.proof_count = upd.evidence.len();
                    upd.updated_at_ms = now;
                    if old_text != upd.text {
                        upd.history.push(old_text);
                    }
                    if upd.embedding.is_none() {
                        obs_pending.push((upd.id.clone(), upd.text.clone()));
                    }
                    if let Err(error) = e.store.upsert_observation(&upd) {
                        return EntityStep::Failed {
                            entity_id: eid.to_string(),
                            error,
                        };
                    }
                    wrote += 1;
                }
                None => {
                    let o = Observation {
                        id: uuid::Uuid::new_v4().to_string(),
                        bank_id: bank_id.to_string(),
                        entity_id: eid.to_string(),
                        text: p.text.clone(),
                        evidence,
                        proof_count,
                        history: vec![],
                        updated_at_ms: now,
                        created_at_ms: now,
                        embedding: None,
                    };
                    obs_pending.push((o.id.clone(), o.text.clone()));
                    if let Err(error) = e.store.upsert_observation(&o) {
                        return EntityStep::Failed {
                            entity_id: eid.to_string(),
                            error,
                        };
                    }
                    wrote += 1;
                }
            }
        }

        if !obs_pending.is_empty() {
            let embed = Arc::clone(&e.embed);
            let texts: Vec<String> = obs_pending.iter().map(|(_, t)| t.clone()).collect();
            match with_retry(e, || embed.embed(texts.clone())).await {
                Ok(embeddings) => {
                    for ((id, _), emb) in obs_pending.into_iter().zip(embeddings) {
                        if let Err(error) = e.store.set_observation_embedding(&id, Some(emb)) {
                            return EntityStep::Failed {
                                entity_id: eid.to_string(),
                                error,
                            };
                        }
                    }
                }
                Err(error) => {
                    // Observations are stored but unembeddable for now; the
                    // None-embedding check re-queues them on the next pass.
                    return EntityStep::Failed {
                        entity_id: eid.to_string(),
                        error,
                    };
                }
            }
        }

        // Success: clear just this entity's dirty row. Completed entities
        // survive a later crash or hard-wall abort of the pass.
        if let Err(error) = e.store.clear_dirty_entities(bank_id, &[eid.to_string()]) {
            return EntityStep::Failed {
                entity_id: eid.to_string(),
                error,
            };
        }
        EntityStep::Done {
            entity_id: eid.to_string(),
            observations: wrote,
        }
    })
}

/// Clear a dirty row for an entity with nothing to synthesize.
fn clear_ineligible(e: &Engine, bank_id: &str, eid: &str) -> EntityStep {
    match e.store.clear_dirty_entities(bank_id, &[eid.to_string()]) {
        Ok(()) => EntityStep::Ineligible,
        Err(error) => EntityStep::Failed {
            entity_id: eid.to_string(),
            error,
        },
    }
}
