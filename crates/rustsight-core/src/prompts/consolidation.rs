use crate::types::{Memory, Observation};
use crate::{Error, Result};
use serde::Deserialize;

const CONSOLIDATION_SYSTEM: &str = r#"You are an OBJECTIVE OBSERVER SYNTHESIS ENGINE. You consolidate accumulated facts about a single entity into observations: concise, factual, preference-neutral summaries. You never inject opinions or behavioral influence.

RULES:
1. REFINE, DO NOT OVERWRITE: if existing observations already capture a fact set accurately, return them (possibly merged), rather than rewriting from scratch. When evidence changes a belief, the new observation text replaces the old one and the change is visible.
2. EVIDENCE: every observation must cite its supporting facts with exact quotes (use the fact's own words, trimmed to the relevant fragment).
3. DEDUPLICATE: overlapping facts merge into one observation instead of piling up as repeats.
4. Only use information present in the provided facts. Do not speculate.
5. Preserve the language of the source facts.

OUTPUT: a single JSON object, no prose:
{"observations":[{"text":"...","evidence":[{"fact_index":0,"quote":"..."}]}]}"#;

pub fn build_consolidation_prompt(
    entity_name: &str,
    facts: &[Memory],
    existing: &[Observation],
) -> (String, String) {
    let mut user = format!("ENTITY: {entity_name}\n\nFACTS:\n");
    for (i, f) in facts.iter().enumerate() {
        user.push_str(&format!("[{i}] (id: {}) {}\n", f.id, f.text));
    }
    if existing.is_empty() {
        user.push_str("\nEXISTING OBSERVATIONS: none\n");
    } else {
        user.push_str("\nEXISTING OBSERVATIONS:\n");
        for o in existing {
            user.push_str(&format!("- {}\n", o.text));
        }
    }
    user.push_str("\nGenerate the consolidated observations as JSON.");
    (CONSOLIDATION_SYSTEM.to_string(), user)
}

#[derive(Debug, Clone)]
pub struct ParsedObservation {
    pub text: String,
    pub evidence: Vec<(usize, String)>,
}

#[derive(Debug, Deserialize)]
struct RawEvidence {
    #[serde(default)]
    fact_index: Option<usize>,
    #[serde(default)]
    quote: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawObservation {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    evidence: Option<Vec<RawEvidence>>,
}

/// One element of the model's "observations" array. Most models emit
/// objects; some (seen from gpt-oss-20b via OpenRouter in the first
/// fullstack smoke run) interleave bare strings, which used to fail the
/// whole response with "invalid type: string, expected struct
/// RawObservation" and cost a consolidation pass. Strings parse as
/// observations with no evidence — the engine backfills evidence from
/// the entity's first fact when an observation cites none.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawObservationItem {
    Obj(RawObservation),
    Text(String),
}

#[derive(Debug, Deserialize)]
struct RawRoot {
    #[serde(default)]
    observations: Option<Vec<RawObservationItem>>,
}

fn json_slice(raw: &str) -> &str {
    match (raw.find('{'), raw.rfind('}')) {
        (Some(s), Some(e)) if e > s => &raw[s..=e],
        _ => raw,
    }
}

pub fn parse_observations(raw: &str) -> Result<Vec<ParsedObservation>> {
    let root: RawRoot = serde_json::from_str(json_slice(raw))
        .map_err(|e| Error::Parse(format!("consolidation response: {e}")))?;
    Ok(root
        .observations
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            let (text, evidence) = match item {
                RawObservationItem::Obj(o) => (
                    o.text,
                    o.evidence
                        .unwrap_or_default()
                        .into_iter()
                        .filter_map(|ev| {
                            let idx = ev.fact_index?;
                            Some((idx, ev.quote.unwrap_or_default()))
                        })
                        .collect(),
                ),
                RawObservationItem::Text(text) => (Some(text), vec![]),
            };
            let text = text?.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(ParsedObservation { text, evidence })
        })
        .collect())
}

const REINFORCEMENT_SYSTEM: &str = r#"You are an OPINION REINFORCEMENT ENGINE. Given an existing opinion held by an agent (with its current confidence) and new supporting or contradicting facts, decide whether the opinion should change.

RULES:
1. Supporting evidence may raise confidence; contradicting evidence may lower it or qualify the opinion. Small evidence makes small moves.
2. Keep the opinion written in the first person, as the agent itself would state it.
3. If the new facts are irrelevant or change nothing, return changed=false unchanged.

OUTPUT: a single JSON object: {"changed":true|false,"text":"...","confidence":0.0-1.0}"#;

pub fn build_opinion_reinforcement_prompt(
    opinion_text: &str,
    confidence: f32,
    new_facts: &[Memory],
) -> (String, String) {
    let mut user =
        format!("CURRENT OPINION (confidence {confidence:.2}):\n{opinion_text}\n\nNEW FACTS:\n");
    if new_facts.is_empty() {
        user.push_str("(none)\n");
    }
    for f in new_facts {
        user.push_str(&format!("- {}\n", f.text));
    }
    user.push_str("\nDecide whether the opinion changes. Respond with JSON only.");
    (REINFORCEMENT_SYSTEM.to_string(), user)
}

#[derive(Debug, Deserialize)]
struct RawReinforcement {
    #[serde(default)]
    changed: Option<bool>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

pub fn parse_reinforcement(raw: &str) -> Result<Option<(String, f32)>> {
    let r: RawReinforcement = serde_json::from_str(json_slice(raw))
        .map_err(|e| Error::Parse(format!("reinforcement response: {e}")))?;
    if !r.changed.unwrap_or(false) {
        return Ok(None);
    }
    let text = match r.text {
        Some(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => return Ok(None),
    };
    let confidence = r.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
    Ok(Some((text, confidence)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_obs_basic() {
        let raw = r#"{"observations":[{"text":"Bob works at Acme.","evidence":[{"fact_index":0,"quote":"Bob joined Acme"},{"fact_index":2}]},{"text":""}]}"#;
        let obs = parse_observations(raw).unwrap();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].evidence.len(), 2);
        assert_eq!(obs[0].evidence[1].1, "");
    }

    /// Regression: gpt-oss-20b returned a bare string inside
    /// "observations" and the whole pass failed with "invalid type:
    /// string, expected struct RawObservation". Strings must parse as
    /// observations with empty evidence, not kill the response.
    #[test]
    fn parse_obs_accepts_bare_string_elements() {
        let raw = r#"{"observations":["Alice works at Google.",{"text":"Bob joined Acme.","evidence":[{"fact_index":1,"quote":"Bob joined Acme"}]},"  "]}"#;
        let obs = parse_observations(raw).unwrap();
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].text, "Alice works at Google.");
        assert!(obs[0].evidence.is_empty());
        assert_eq!(obs[1].evidence, vec![(1, "Bob joined Acme".to_string())]);
    }

    /// A bare-string-only response parses too (no objects at all).
    #[test]
    fn parse_obs_accepts_all_string_observations() {
        let obs = parse_observations(r#"{"observations":["Only text."]}"#).unwrap();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].text, "Only text.");
        assert!(obs[0].evidence.is_empty());
    }

    #[test]
    fn parse_reinforce_changed_and_not() {
        let raw = r#"{"changed":true,"text":"I now think Alice is meticulous.","confidence":0.8}"#;
        let (t, c) = parse_reinforcement(raw).unwrap().unwrap();
        assert_eq!(t, "I now think Alice is meticulous.");
        assert!((c - 0.8).abs() < 1e-6);
        assert!(parse_reinforcement(r#"{"changed":false}"#)
            .unwrap()
            .is_none());
    }
}
