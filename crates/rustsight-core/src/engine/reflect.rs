use crate::engine::Engine;
use crate::entity::{combined_score, normalize_name, string_similarity};
use crate::prompts::reflect::{
    build_opinion_extraction_prompt, build_reflect_system, parse_opinions, reflect_tools,
};
use crate::traits::{ChatMessage, LlmRequest, LlmResponse, Role, ToolCall};
use crate::types::*;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};

pub(super) async fn reflect(e: &Engine, bank_id: &str, query: &str) -> Result<ReflectResponse> {
    if query.trim().is_empty() {
        return Err(Error::BadInput("query is empty".into()));
    }
    let bank = e.store.get_bank(bank_id)?.unwrap_or(Bank {
        id: bank_id.to_string(),
        mission: None,
        directives: None,
        background: None,
        disposition: Disposition::default(),
    });
    let system = build_reflect_system(&bank);
    let mut messages: Vec<ChatMessage> = vec![ChatMessage::user(query.to_string())];
    let mut retrieved: HashSet<String> = HashSet::new();
    let mut answer: Option<String> = None;
    let mut citations: Vec<String> = vec![];
    let mut rounds = 0usize;

    while rounds < e.config.reflect_max_rounds {
        rounds += 1;
        let resp = e
            .llm
            .complete(&LlmRequest {
                system: Some(system.clone()),
                messages: messages.clone(),
                tools: reflect_tools(),
                temperature: Some(0.2),
                ..Default::default()
            })
            .await?;

        if let Some(call) = resp.tool_calls.first().cloned() {
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: String::new(),
                tool_call_id: Some(call.id.clone()),
                tool_calls: vec![call.clone()],
            });
            match call.name.as_str() {
                "recall" => {
                    let q = call
                        .arguments
                        .get("query")
                        .and_then(|v| v.as_str())
                        .unwrap_or(query)
                        .to_string();
                    let budget = call
                        .arguments
                        .get("token_budget")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(2000) as usize;
                    let r = e
                        .recall(RecallRequest {
                            bank_id: bank_id.to_string(),
                            query: q,
                            token_budget: budget.min(6000),
                            top_k_per_arm: 10,
                            kinds: None,
                        })
                        .await?;
                    let mut text = String::new();
                    for m in &r.memories {
                        retrieved.insert(m.id.clone());
                        text.push_str(&format!(
                            "[{}] {}: {}\n",
                            m.kind.as_str(),
                            m.id,
                            m.text
                        ));
                    }
                    if text.is_empty() {
                        text = "(no memories found)".to_string();
                    }
                    messages.push(ChatMessage::tool_result(call.id.clone(), text));
                }
                "read_observation" => {
                    let name = call
                        .arguments
                        .get("entity")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let target = normalize_name(&name);
                    let ents = e.store.entities_for_bank(bank_id)?;
                    let text = ents
                        .iter()
                        .find(|en| normalize_name(&en.name) == target)
                        .map(|en| {
                            let obs = e.store.observations_for_entity(&en.id).unwrap_or_default();
                            if obs.is_empty() {
                                format!("(no observations for {})", en.name)
                            } else {
                                obs.iter()
                                    .map(|o| {
                                        format!(
                                            "[observation] {}: {} (evidence: {})",
                                            o.id,
                                            o.text,
                                            o.proof_count
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }
                        })
                        .unwrap_or_else(|| format!("(unknown entity {name})"));
                    for o in e.store.observations_for_bank(bank_id)? {
                        retrieved.insert(o.id.clone());
                    }
                    messages.push(ChatMessage::tool_result(call.id.clone(), text));
                }
                "finish" => {
                    answer = Some(
                        call.arguments
                            .get("answer")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    );
                    let cited = call
                        .arguments
                        .get("citations")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect::<Vec<String>>()
                        })
                        .unwrap_or_default();
                    let mut valid: Vec<String> = cited
                        .into_iter()
                        .filter(|id| retrieved.contains(id))
                        .collect();
                    valid.sort();
                    valid.dedup();
                    citations = valid;
                    messages.push(ChatMessage::tool_result(
                        call.id.clone(),
                        "answer recorded".to_string(),
                    ));
                    break;
                }
                other => {
                    messages.push(ChatMessage::tool_result(
                        call.id.clone(),
                        format!("unknown tool {other}"),
                    ));
                }
            }
        } else if let Some(c) = resp.content {
            answer = Some(c);
            break;
        } else {
            return Err(Error::Llm("reflect loop got an empty response".into()));
        }
    }

    let answer = answer.ok_or_else(|| Error::Llm("reflect produced no answer".into()))?;
    let retrieved_count = retrieved.len();
    let opinions_formed = form_opinions(e, bank_id, query, &answer).await?;

    Ok(ReflectResponse {
        answer,
        citations,
        rounds,
        retrieved: retrieved_count,
        opinions_formed,
    })
}

async fn form_opinions(e: &Engine, bank_id: &str, question: &str, answer: &str) -> Result<usize> {
    let (system, user) = build_opinion_extraction_prompt(question, answer);
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
    let parsed = parse_opinions(&resp.content.unwrap_or_default())?;
    if parsed.is_empty() {
        return Ok(0);
    }
    let existing = e.store.list_opinions(bank_id)?;
    let bank_entities = e.store.entities_for_bank(bank_id)?;
    let mut resolved_cache: HashMap<String, Option<Entity>> = HashMap::new();
    let now = e.clock.now_ms();
    let mut formed = 0usize;
    let mut new_opinions: Vec<(String, f32, Vec<String>)> = vec![];
    for op in parsed {
        let entity_ids = resolve_opinion_entities(
            e,
            bank_id,
            &op.entities,
            &bank_entities,
            &mut resolved_cache,
        )?;
        let similar = existing
            .iter()
            .find(|o| string_similarity(&op.text, &o.text) >= 0.85);
        match similar {
            Some(o) => {
                let mut updated = o.clone();
                updated.history.push(ConfidencePoint {
                    confidence: o.confidence,
                    at_ms: now,
                    reason: "reinforced by reflect".into(),
                });
                updated.confidence = ((o.confidence + op.confidence) / 2.0).clamp(0.0, 1.0);
                updated.updated_at_ms = now;
                for id in entity_ids {
                    if !updated.entity_ids.contains(&id) {
                        updated.entity_ids.push(id);
                    }
                }
                e.store.upsert_opinion(&updated)?;
            }
            None => {
                formed += 1;
                new_opinions.push((op.text, op.confidence, entity_ids));
            }
        }
    }
    if !new_opinions.is_empty() {
        let texts: Vec<String> = new_opinions.iter().map(|(t, _, _)| t.clone()).collect();
        let embeddings = e.embed.embed(texts).await?;
        for ((text, confidence, entity_ids), emb) in new_opinions.into_iter().zip(embeddings) {
            let o = Opinion {
                id: uuid::Uuid::new_v4().to_string(),
                bank_id: bank_id.to_string(),
                text,
                confidence: confidence.clamp(0.0, 1.0),
                evidence: vec![],
                history: vec![],
                entity_ids,
                updated_at_ms: now,
                created_at_ms: now,
                embedding: Some(emb),
            };
            e.store.upsert_opinion(&o)?;
        }
    }
    Ok(formed)
}

/// Resolve LLM-tagged entity mentions in an opinion against the bank's
/// canonical entities: exact alias match first, then fuzzy match above the
/// resolver's link threshold. Unmatched mentions are dropped (never linked,
/// never created) so opinions without known entities stay unlinked and are
/// skipped by consolidation-time reinforcement.
fn resolve_opinion_entities(
    e: &Engine,
    bank_id: &str,
    mentions: &[ExtractedEntity],
    bank_entities: &[Entity],
    cache: &mut HashMap<String, Option<Entity>>,
) -> Result<Vec<String>> {
    let now = e.clock.now_ms();
    let mut ids: Vec<String> = vec![];
    for m in mentions {
        let norm = normalize_name(&m.name);
        if norm.is_empty() {
            continue;
        }
        let hit = if let Some(h) = cache.get(&norm) {
            h.clone()
        } else {
            let h = match e.store.find_by_alias(bank_id, &norm)? {
                Some(x) => Some(x),
                None => fuzzy_match_entity(e, bank_id, &norm, &m.name, bank_entities, now)?,
            };
            cache.insert(norm, h.clone());
            h
        };
        if let Some(ent) = hit {
            if !ids.contains(&ent.id) {
                ids.push(ent.id);
            }
        }
    }
    Ok(ids)
}

fn fuzzy_match_entity(
    e: &Engine,
    bank_id: &str,
    norm: &str,
    display: &str,
    candidates: &[Entity],
    now: i64,
) -> Result<Option<Entity>> {
    let cfg = &e.config.entity;
    let mut best: Option<(f32, &Entity)> = None;
    for c in candidates {
        let s = string_similarity(display, &c.name);
        let t = 1.0
            - (((now - c.last_mentioned_ms).abs() as f32) / (90.0 * 86_400_000.0)).min(1.0);
        let score = combined_score(s, 0.0, t, cfg);
        if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
            best = Some((score, c));
        }
    }
    if let Some((score, ent)) = best {
        if score >= cfg.link_threshold {
            e.store.add_alias(bank_id, norm, &ent.id)?;
            return Ok(Some(ent.clone()));
        }
    }
    Ok(None)
}

#[allow(dead_code)]
fn demo_tool_call(id: &str, name: &str, args: serde_json::Value) -> LlmResponse {
    LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn demo_tool_call_shape() {
        let r = demo_tool_call("1", "finish", json!({"answer": "hi"}));
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "finish");
    }
}
