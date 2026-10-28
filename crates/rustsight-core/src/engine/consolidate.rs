use crate::engine::Engine;
use crate::prompts::consolidation::{
    build_consolidation_prompt, build_opinion_reinforcement_prompt, parse_observations,
    parse_reinforcement,
};
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::{string_similarity, Error, Result};

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
    let now = e.clock.now_ms();
    let mut updated = 0usize;
    // (observation id, text) pairs awaiting the embedding backfill below.
    // Tracking ids (not just texts) means each observation gets its own
    // embedding even when two observations share identical text.
    let mut obs_pending: Vec<(String, String)> = vec![];
    // One opinion snapshot for the whole pass (CR-6): `list_opinions` reads
    // the entire bank, so re-reading it per dirty entity made consolidation
    // quadratic in bank size. Reinforced opinions are written back into the
    // snapshot so later entities in this pass still see fresh
    // confidence/timestamps, exactly as the per-entity re-fetch did.
    let mut opinions = e.store.list_opinions(bank_id)?;

    for eid in &dirty {
        let entity = match e.store.get_entity(eid)? {
            Some(x) => x,
            None => continue,
        };
        let facts = e.store.facts_for_entity(eid, 60)?;
        let existing = e.store.observations_for_entity(eid)?;
        if facts.len() < 2 && existing.is_empty() {
            continue;
        }
        let (system, user) = build_consolidation_prompt(&entity.name, &facts, &existing);
        let resp = e
            .llm
            .complete(&LlmRequest {
                system: Some(system),
                messages: vec![ChatMessage::user(user)],
                json_mode: true,
                temperature: Some(0.0),
                ..Default::default()
            })
            .await?;
        let parsed = parse_observations(&resp.content.unwrap_or_default())?;
        // A parse failure above aborts the whole pass via `?` — intentional:
        // dirty rows are only cleared after a successful pass, so the next
        // worker tick re-discovers and retries this entity.
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
                    e.store.upsert_observation(&upd)?;
                    updated += 1;
                }
                None => {
                    let o = Observation {
                        id: uuid::Uuid::new_v4().to_string(),
                        bank_id: bank_id.to_string(),
                        entity_id: eid.clone(),
                        text: p.text.clone(),
                        evidence,
                        proof_count,
                        history: vec![],
                        updated_at_ms: now,
                        created_at_ms: now,
                        embedding: None,
                    };
                    obs_pending.push((o.id.clone(), o.text.clone()));
                    e.store.upsert_observation(&o)?;
                    updated += 1;
                }
            }
        }

        let matching: Vec<usize> = opinions
            .iter()
            .enumerate()
            .filter(|(_, o)| o.entity_ids.iter().any(|id| id == eid))
            .map(|(i, _)| i)
            .collect();
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
            let (system, user) =
                build_opinion_reinforcement_prompt(&op.text, op.confidence, &new_facts);
            let resp = e
                .llm
                .complete(&LlmRequest {
                    system: Some(system),
                    messages: vec![ChatMessage::user(user)],
                    json_mode: true,
                    temperature: Some(0.0),
                    ..Default::default()
                })
                .await?;
            if let Some((text, confidence)) =
                parse_reinforcement(&resp.content.unwrap_or_default())?
            {
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

    if !obs_pending.is_empty() {
        let texts: Vec<String> = obs_pending.iter().map(|(_, t)| t.clone()).collect();
        let embeddings = e.embed.embed(texts).await?;
        for ((id, _), emb) in obs_pending.into_iter().zip(embeddings) {
            e.store.set_observation_embedding(&id, Some(emb))?;
        }
    }

    e.store.clear_dirty_entities(bank_id, &dirty)?;
    Ok(updated)
}
