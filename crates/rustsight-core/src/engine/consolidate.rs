use crate::engine::{with_retry, Engine};
use crate::prompts::consolidation::{
    build_entity_consolidation_prompt, build_opinion_reinforcement_prompt,
    parse_consolidation_plan, parse_reinforcement, ConsolidationPlan,
};
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::{Error, Result};
use futures::StreamExt;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Boxed per-entity consolidation future; boxing lets a pass buffer a
/// bounded number of entities concurrently (mirroring retain's chunk
/// extraction, CR-19).
type EntityFut<'a> = Pin<Box<dyn Future<Output = EntityStep> + Send + 'a>>;

/// Observation rows a plan (or a whole pass) changed — the
/// `consolidation.completed` event counts. Evidence merged into an existing
/// observation (a dupe-guarded create, a delete's survivor) counts as an
/// update.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PlanCounts {
    pub created: usize,
    pub updated: usize,
    pub deleted: usize,
}

impl PlanCounts {
    fn add(&mut self, other: PlanCounts) {
        self.created += other.created;
        self.updated += other.updated;
        self.deleted += other.deleted;
    }

    /// Observation rows written (creates + updates) — what a pass reports.
    fn written(&self) -> usize {
        self.created + self.updated
    }
}

/// What one pass did.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PassOutcome {
    /// Whether the pass found queued work (an empty queue is not a pass
    /// worth reporting).
    pub ran: bool,
    /// Observation rows written, the public `consolidate_once` count.
    pub written: usize,
    pub counts: PlanCounts,
}

/// What happened to one dirty entity inside a pass.
enum EntityStep {
    /// Observations were written and embedded; the entity's dirty row was
    /// cleared.
    Done {
        entity_id: String,
        counts: PlanCounts,
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
) -> Result<PassOutcome> {
    if e.store.get_bank(bank_id)?.is_none() {
        return Err(Error::NotFound(format!("bank {bank_id}")));
    }
    let dirty = e.store.list_dirty_entities(bank_id, max_entities.max(1))?;
    if dirty.is_empty() {
        return Ok(PassOutcome::default());
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

    let mut counts = PlanCounts::default();
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
                counts: entity_counts,
            } => {
                counts.add(entity_counts);
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
                let llm = e.llm_for_consolidate();
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

    Ok(PassOutcome {
        ran: true,
        written: counts.written(),
        counts,
    })
}

/// Consolidate one dirty entity: synthesize (or refine) its observations,
/// embed them, then clear the entity's dirty row. Every failure isolates to
/// this entity — its dirty row survives and the next pass retries it — so
/// one malformed provider response can no longer discard the rest of a
/// pass's work.
///
/// Entities with more facts than `consolidate_max_facts_per_entity` page
/// through them in newest-first windows, each window seeing the previous
/// windows' observations — no fact is silently dropped for being old (the
/// 2026-11-12 finding: the fixed 60-fact cap starved big entities).
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
        let facts = match e.store.facts_for_entity(eid, i64::MAX as usize) {
            Ok(f) => f,
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        let mut existing = match e.store.observations_for_entity(eid) {
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
        let mission = match e.store.get_bank(bank_id) {
            Ok(bank) => bank.and_then(|b| e.scoped_mission(&b, "observations_mission")),
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        // Knobs resolve global → tenant → bank (RS-10f).
        let resolved = match e.resolved_config(bank_id) {
            Ok(r) => r,
            Err(error) => {
                return EntityStep::Failed {
                    entity_id: eid.to_string(),
                    error,
                }
            }
        };
        let language = resolved.str("output_language");
        let dupe_cosine = resolved
            .f32("consolidate_dupe_cosine")
            .unwrap_or(e.config.consolidate_dupe_cosine);
        let now = e.clock.now_ms();
        let llm = e.llm_for_consolidate();
        let max_facts = resolved
            .usize("consolidate_max_facts_per_entity")
            .unwrap_or(e.config.consolidate_max_facts_per_entity)
            .max(1);
        let mut counts = PlanCounts::default();
        // (observation id, text) pairs needing (re-)embedding, across all
        // of this entity's windows — one batched embed at the end.
        let mut obs_pending: Vec<(String, String)> = vec![];
        for window in facts.chunks(max_facts) {
            if tokio::time::Instant::now() >= deadline {
                return EntityStep::Deadline;
            }
            let (system, user) = build_entity_consolidation_prompt(
                &entity.name,
                window,
                &existing,
                mission.as_deref(),
                language,
            );
            let request = LlmRequest {
                system: Some(system),
                messages: vec![ChatMessage::user(user)],
                json_mode: true,
                temperature: Some(0.0),
                ..Default::default()
            };
            let resp = match with_retry(e, || llm.complete(&request)).await {
                Ok(r) => r,
                Err(error) => {
                    return EntityStep::Failed {
                        entity_id: eid.to_string(),
                        error,
                    }
                }
            };
            let plan = match parse_consolidation_plan(&resp.content.unwrap_or_default()) {
                Ok(p) => p,
                Err(error) => {
                    return EntityStep::Failed {
                        entity_id: eid.to_string(),
                        error,
                    }
                }
            };
            match apply_consolidation_plan(e, bank_id, eid, window, plan, now, dupe_cosine).await {
                Ok((window_counts, pending)) => {
                    counts.add(window_counts);
                    obs_pending.extend(pending);
                    // Later windows see earlier windows' observations so the
                    // model UPDATEs/merges instead of re-CREATEing (the
                    // dupe guards back it up).
                    existing = e.store.observations_for_entity(eid).unwrap_or_default();
                }
                Err(error) => {
                    return EntityStep::Failed {
                        entity_id: eid.to_string(),
                        error,
                    }
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
            counts,
        }
    })
}

/// Resolve a plan entry's evidence to `Evidence` pairs: exact fact ids
/// (upstream `source_fact_ids`) and legacy fact indexes both map against
/// the batch's facts; duplicates by memory id collapse. Entries citing no
/// resolvable fact backfill from the batch's first fact (weak models still
/// produce attributable observations).
fn resolve_evidence(
    facts: &[Memory],
    source_fact_ids: &[String],
    legacy: &[(usize, String)],
) -> Vec<Evidence> {
    let quote_for = |fact: &Memory, quote: &str| -> Evidence {
        Evidence {
            memory_id: fact.id.clone(),
            quote: if quote.trim().is_empty() {
                fact.text.chars().take(160).collect()
            } else {
                quote.to_string()
            },
        }
    };
    let mut evidence: Vec<Evidence> = vec![];
    for id in source_fact_ids {
        if let Some(fact) = facts.iter().find(|f| f.id == *id) {
            let ev = quote_for(fact, "");
            if !evidence.iter().any(|e| e.memory_id == ev.memory_id) {
                evidence.push(ev);
            }
        }
    }
    for (idx, quote) in legacy {
        if let Some(fact) = facts.get(*idx) {
            let ev = quote_for(fact, quote);
            if !evidence.iter().any(|e| e.memory_id == ev.memory_id) {
                evidence.push(ev);
            }
        }
    }
    if evidence.is_empty() {
        if let Some(fact) = facts.first() {
            evidence.push(quote_for(fact, ""));
        }
    }
    evidence
}

/// Whitespace-collapsed lowercase text — the key of the bank-level exact
/// duplicate guard. Conservative on purpose: near-duplicates are the
/// model's job (UPDATE over CREATE), exact repeats are the engine's.
fn norm_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Apply the parsed creates/updates/deletes plan for one entity.
///
/// Ordering matters: updates land first (a delete's `merged_into` target
/// must carry its new text before evidence migrates into it), then deletes
/// (merge evidence into the survivor, then remove the row), then creates
/// (the duplicate guards see the post-delete state, and same-batch creates
/// that repeat an earlier create's text merge into it). Returns how many
/// observations were written plus the (id, text) pairs the caller must
/// (re-)embed — one batched embed per entity, not per plan.
async fn apply_consolidation_plan(
    e: &Engine,
    bank_id: &str,
    eid: &str,
    facts: &[Memory],
    plan: ConsolidationPlan,
    now: i64,
    dupe_cosine: f32,
) -> Result<(PlanCounts, Vec<(String, String)>)> {
    // id -> observation, for update/delete/merge target lookup.
    let mut by_id: HashMap<String, Observation> = e
        .store
        .observations_for_bank(bank_id)
        .unwrap_or_default()
        .into_iter()
        .map(|o| (o.id.clone(), o))
        .collect();
    // normalized text -> id, the cross-entity exact-dupe guard.
    let mut by_text: HashMap<String, String> = by_id
        .iter()
        .map(|(id, o)| (norm_text(&o.text), id.clone()))
        .collect();

    let mut counts = PlanCounts::default();
    // (observation id, text) pairs needing (re-)embedding.
    let mut obs_pending: Vec<(String, String)> = vec![];

    // ---- updates ----
    for u in &plan.updates {
        let Some(target) = by_id.get(&u.observation_id).cloned() else {
            // The model named an observation we cannot see (hallucinated id
            // or already deleted): the create pass below keeps the text.
            continue;
        };
        let evidence = resolve_evidence(facts, &u.source_fact_ids, &u.legacy_evidence);
        let mut upd = target.clone();
        let old_text = upd.text.clone();
        upd.text = u.text.clone();
        for ev in evidence {
            if !upd.evidence.iter().any(|x| x.memory_id == ev.memory_id) {
                upd.evidence.push(ev);
            }
        }
        upd.proof_count = upd.evidence.len();
        upd.updated_at_ms = now;
        if old_text != upd.text {
            upd.history.push(old_text);
            obs_pending.push((upd.id.clone(), upd.text.clone()));
        } else if upd.embedding.is_none() {
            obs_pending.push((upd.id.clone(), upd.text.clone()));
        }
        e.store.upsert_observation(&upd)?;
        by_text.insert(norm_text(&upd.text), upd.id.clone());
        by_id.insert(upd.id.clone(), upd);
        counts.updated += 1;
    }

    // ---- deletes (after updates: merged_into targets carry new text) ----
    for d in &plan.deletes {
        let Some(victim) = by_id.get(&d.observation_id).cloned() else {
            continue;
        };
        if let Some(target_id) = &d.merged_into {
            if let Some(survivor) = by_id.get_mut(target_id) {
                for ev in &victim.evidence {
                    if !survivor
                        .evidence
                        .iter()
                        .any(|x| x.memory_id == ev.memory_id)
                    {
                        survivor.evidence.push(ev.clone());
                    }
                }
                survivor.proof_count = survivor.evidence.len();
                survivor.updated_at_ms = now;
                e.store.upsert_observation(survivor)?;
                counts.updated += 1;
            }
        }
        e.store.delete_observation(bank_id, &victim.id)?;
        counts.deleted += 1;
        by_text.remove(&norm_text(&victim.text));
        by_id.remove(&victim.id);
    }

    // ---- creates (dupe guards see post-delete state) ----
    // Embed every proposed create once (batched) for the cosine near-dupe
    // guard: paraphrase siblings the exact-text guard cannot catch were the
    // 2026-11-12 runs' observation explosion.
    let create_texts: Vec<String> = plan.creates.iter().map(|c| c.text.clone()).collect();
    let mut create_embs_iter: Vec<Option<Vec<f32>>> = vec![None; plan.creates.len()];
    if !create_texts.is_empty() {
        let embed = Arc::clone(&e.embed);
        let threshold = dupe_cosine;
        let wants_embeddings = plan
            .creates
            .iter()
            .any(|c| !by_text.contains_key(&norm_text(&c.text)));
        if wants_embeddings && threshold < 1.0 {
            if let Ok(embeddings) = with_retry(e, || embed.embed(create_texts.clone())).await {
                create_embs_iter = embeddings.into_iter().map(Some).collect();
            }
        }
    }
    for (c, new_emb) in plan.creates.iter().zip(create_embs_iter) {
        let evidence = resolve_evidence(facts, &c.source_fact_ids, &c.legacy_evidence);
        if evidence.is_empty() {
            continue;
        }
        // Guard 1: exact text (free).
        let norm = norm_text(&c.text);
        let mut dupe_id = by_text.get(&norm).cloned();
        // Guard 2: cosine near-dupe against every embedded observation.
        if dupe_id.is_none() {
            if let Some(emb) = new_emb.as_ref() {
                let mut best: Option<(f32, &String)> = None;
                for (id, o) in &by_id {
                    if let Some(existing_emb) = o.embedding.as_ref() {
                        let score = crate::cosine(emb, existing_emb);
                        if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                            best = Some((score, id));
                        }
                    }
                }
                if let Some((score, id)) = best {
                    if score >= dupe_cosine {
                        dupe_id = Some(id.clone());
                    }
                }
            }
        }
        if let Some(existing_id) = dupe_id {
            // Duplicate (same entity, cross-entity, paraphrase, or earlier
            // in this same batch): merge the evidence instead of writing a
            // sibling.
            if let Some(obs) = by_id.get_mut(&existing_id) {
                for ev in evidence {
                    if !obs.evidence.iter().any(|x| x.memory_id == ev.memory_id) {
                        obs.evidence.push(ev);
                    }
                }
                obs.proof_count = obs.evidence.len();
                obs.updated_at_ms = now;
                if obs.embedding.is_none() {
                    if let Some(emb) = new_emb {
                        obs.embedding = Some(emb);
                    } else {
                        obs_pending.push((obs.id.clone(), obs.text.clone()));
                    }
                }
                e.store.upsert_observation(obs)?;
                counts.updated += 1;
            }
            continue;
        }
        let proof_count = evidence.len();
        let o = Observation {
            id: uuid::Uuid::new_v4().to_string(),
            bank_id: bank_id.to_string(),
            entity_id: eid.to_string(),
            text: c.text.clone(),
            evidence,
            proof_count,
            history: vec![],
            updated_at_ms: now,
            created_at_ms: now,
            // The create was already embedded for the guard — reuse it.
            embedding: new_emb,
        };
        let id = o.id.clone();
        if o.embedding.is_none() {
            obs_pending.push((id.clone(), o.text.clone()));
        }
        e.store.upsert_observation(&o)?;
        by_text.insert(norm, id.clone());
        by_id.insert(id, o);
        counts.created += 1;
    }

    Ok((counts, obs_pending))
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
