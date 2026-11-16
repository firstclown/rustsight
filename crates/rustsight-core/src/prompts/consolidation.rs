use crate::prompts::prompt_utils::iso_date_utc;
use crate::prompts::upstream::consolidation::{
    build_consolidation_input, build_consolidation_system_prompt,
};
use crate::types::{Memory, Observation};
use crate::{Error, Result};
use serde::Deserialize;
use serde_json::json;

/// Upstream response machine (v1.1.0 consolidation/prompts.py): the model
/// returns `{"creates": [...], "updates": [...], "deletes": [...]}` against
/// the existing observations it was shown. The legacy rustsight shape
/// (`{"observations": [...]}`) still parses — each element becomes a CREATE
/// whose evidence rides `legacy_evidence` (fact indexes, resolved by the
/// engine against the batch's fact ids), mirroring how `parse_extraction`
/// adopted the upstream extraction schema without breaking scripted
/// providers.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationPlan {
    pub creates: Vec<PlannedCreate>,
    pub updates: Vec<PlannedUpdate>,
    pub deletes: Vec<PlannedDelete>,
}

#[derive(Debug, Clone)]
pub struct PlannedCreate {
    pub text: String,
    /// Exact fact UUIDs from the input (upstream `source_fact_ids`).
    pub source_fact_ids: Vec<String>,
    /// Legacy-shape evidence: `(fact_index, quote)` pairs resolved against
    /// the batch's fact list by the engine.
    pub legacy_evidence: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub struct PlannedUpdate {
    pub observation_id: String,
    pub text: String,
    pub source_fact_ids: Vec<String>,
    pub legacy_evidence: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub struct PlannedDelete {
    pub observation_id: String,
    /// When the delete is a merge, the surviving observation's id.
    pub merged_into: Option<String>,
}

/// One new-fact line in upstream's input format:
/// `[uuid] fact text (occurred_start=..., occurred_end=..., mentioned_at=...)`
/// — temporal fields omitted when unknown.
fn fact_line(f: &Memory) -> String {
    let mut temporal = vec![];
    if let Some(s) = f.occurred_start_ms {
        temporal.push(format!("occurred_start={}", iso_date_utc(s)));
    }
    if let Some(t) = f.occurred_end_ms {
        temporal.push(format!("occurred_end={}", iso_date_utc(t)));
    }
    temporal.push(format!("mentioned_at={}", iso_date_utc(f.mentioned_at_ms)));
    format!("[{}] {} ({})", f.id, f.text, temporal.join(", "))
}

pub fn render_facts_text(facts: &[Memory]) -> String {
    facts.iter().map(fact_line).collect::<Vec<_>>().join("\n")
}

/// An existing observation rendered in upstream's input format. The
/// `occurred_*`/`mentioned_at` fields are derived from the observation's
/// evidence facts when they are in the current batch (upstream defines them
/// as earliest start / latest end / latest mention across source facts);
/// `updated_at_ms` stands in when the evidence facts are not at hand.
fn observation_json(o: &Observation, facts: &[Memory]) -> serde_json::Value {
    let by_id = |id: &str| facts.iter().find(|f| f.id == id);
    let starts: Vec<i64> = o
        .evidence
        .iter()
        .filter_map(|ev| by_id(&ev.memory_id).and_then(|f| f.occurred_start_ms))
        .collect();
    let ends: Vec<i64> = o
        .evidence
        .iter()
        .filter_map(|ev| by_id(&ev.memory_id).and_then(|f| f.occurred_end_ms))
        .collect();
    let mentioned = o
        .evidence
        .iter()
        .filter_map(|ev| by_id(&ev.memory_id).map(|f| f.mentioned_at_ms))
        .max()
        .unwrap_or(o.updated_at_ms);
    json!({
        "id": o.id,
        "text": o.text,
        "proof_count": o.proof_count,
        "occurred_start": starts.iter().min().map(|ms| iso_date_utc(*ms)),
        "occurred_end": ends.iter().max().map(|ms| iso_date_utc(*ms)),
        "mentioned_at": iso_date_utc(mentioned),
    })
}

pub fn render_observations_text(existing: &[Observation], facts: &[Memory]) -> String {
    if existing.is_empty() {
        return "(none)".to_string();
    }
    serde_json::to_string(
        &existing
            .iter()
            .map(|o| observation_json(o, facts))
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".to_string())
}

/// The consolidation prompt pair the engine sends: the byte-parity upstream
/// system prompt + user input, with one declared engine divergence —
/// rustsight consolidates per dirty entity (upstream pools bank-wide), so
/// the user message is prefixed with an `ENTITY:` scope line.
pub fn build_entity_consolidation_prompt(
    entity_name: &str,
    facts: &[Memory],
    existing: &[Observation],
    observations_mission: Option<&str>,
) -> (String, String) {
    let system = build_consolidation_system_prompt(None);
    let user = format!(
        "ENTITY: {entity_name}\n\n{}",
        build_consolidation_input(
            &render_facts_text(facts),
            &render_observations_text(existing, facts),
            observations_mission,
            None,
        )
    );
    (system, user)
}

#[derive(Debug, Clone)]
pub struct ParsedObservation {
    pub text: String,
    pub evidence: Vec<(usize, String)>,
}

#[derive(Debug, Deserialize, Clone)]
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

#[derive(Debug, Deserialize)]
struct RawPlanCreate {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    source_fact_ids: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct RawPlanUpdate {
    #[serde(default)]
    observation_id: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    source_fact_ids: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct RawPlanDelete {
    #[serde(default)]
    observation_id: Option<String>,
    #[serde(default)]
    merged_into: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPlan {
    #[serde(default)]
    creates: Option<Vec<RawPlanCreate>>,
    #[serde(default)]
    updates: Option<Vec<RawPlanUpdate>>,
    #[serde(default)]
    deletes: Option<Vec<RawPlanDelete>>,
    #[serde(default)]
    observations: Option<Vec<RawObservationItem>>,
}

fn legacy_evidence(o: &RawObservation) -> Vec<(usize, String)> {
    o.evidence
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|ev| {
            let idx = ev.fact_index?;
            Some((idx, ev.quote.unwrap_or_default()))
        })
        .collect()
}

/// Parse a consolidation response in either the upstream
/// creates/updates/deletes shape or the legacy `{"observations": [...]}`
/// shape. Entries with empty/missing required fields (create text, update
/// id/text, delete id) are dropped rather than failing the whole response —
/// one malformed entry must not discard its neighbors' work.
pub fn parse_consolidation_plan(raw: &str) -> Result<ConsolidationPlan> {
    let plan: RawPlan = serde_json::from_str(json_slice(raw))
        .map_err(|e| Error::Parse(format!("consolidation response: {e}")))?;
    let mut out = ConsolidationPlan::default();
    if plan.creates.is_some() || plan.updates.is_some() || plan.deletes.is_some() {
        for c in plan.creates.unwrap_or_default() {
            let Some(text) = c.text.map(|t| t.trim().to_string()) else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            out.creates.push(PlannedCreate {
                text,
                source_fact_ids: c.source_fact_ids.unwrap_or_default(),
                legacy_evidence: vec![],
            });
        }
        for u in plan.updates.unwrap_or_default() {
            let (Some(observation_id), Some(text)) = (
                u.observation_id.map(|s| s.trim().to_string()),
                u.text.map(|t| t.trim().to_string()),
            ) else {
                continue;
            };
            if observation_id.is_empty() || text.is_empty() {
                continue;
            }
            out.updates.push(PlannedUpdate {
                observation_id,
                text,
                source_fact_ids: u.source_fact_ids.unwrap_or_default(),
                legacy_evidence: vec![],
            });
        }
        for d in plan.deletes.unwrap_or_default() {
            let Some(observation_id) = d.observation_id.map(|s| s.trim().to_string()) else {
                continue;
            };
            if observation_id.is_empty() {
                continue;
            }
            out.deletes.push(PlannedDelete {
                observation_id,
                merged_into: d
                    .merged_into
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            });
        }
        return Ok(out);
    }
    // Legacy shape: every observation is a create against the batch's facts.
    for item in plan.observations.unwrap_or_default() {
        let (text, legacy_evidence) = match item {
            RawObservationItem::Obj(o) => {
                let legacy = legacy_evidence(&o);
                (o.text, legacy)
            }
            RawObservationItem::Text(text) => (Some(text), vec![]),
        };
        let text = text.map(|t| t.trim().to_string()).unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        out.creates.push(PlannedCreate {
            text,
            source_fact_ids: vec![],
            legacy_evidence,
        });
    }
    Ok(out)
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

    #[test]
    fn parse_plan_upstream_shape() {
        let raw = r#"{"creates":[{"text":"Alice works long hours.","source_fact_ids":["f2"],"reason":"new facet"}],
            "updates":[{"observation_id":"1111","text":"Alice owned a Civic; sold it.","source_fact_ids":["f1"],"reason":"state change"}],
            "deletes":[{"observation_id":"2222","merged_into":"1111","reason":"restates 1111"}]}"#;
        let plan = parse_consolidation_plan(raw).unwrap();
        assert_eq!(plan.creates.len(), 1);
        assert_eq!(plan.creates[0].source_fact_ids, vec!["f2".to_string()]);
        assert_eq!(plan.updates.len(), 1);
        assert_eq!(plan.updates[0].observation_id, "1111");
        assert_eq!(plan.deletes.len(), 1);
        assert_eq!(plan.deletes[0].merged_into.as_deref(), Some("1111"));
    }

    /// Entries missing required fields are dropped, not fatal — one bad
    /// entry must not discard its neighbors' work.
    #[test]
    fn parse_plan_drops_malformed_entries() {
        let raw = r#"{"creates":[{"source_fact_ids":["f1"]},{"text":"ok","source_fact_ids":["f2"]}],
            "updates":[{"text":"no id"}],
            "deletes":[{}]}"#;
        let plan = parse_consolidation_plan(raw).unwrap();
        assert_eq!(plan.creates.len(), 1);
        assert_eq!(plan.creates[0].text, "ok");
        assert!(plan.updates.is_empty());
        assert!(plan.deletes.is_empty());
    }

    /// Legacy `{"observations": [...]}` responses (scripted tests, older
    /// providers) become creates with index-based evidence the engine
    /// resolves against the batch.
    #[test]
    fn parse_plan_legacy_shape_becomes_creates() {
        let raw = r#"{"observations":["Alice works at Google.",{"text":"Bob joined Acme.","evidence":[{"fact_index":1,"quote":"Bob joined Acme"}]}]}"#;
        let plan = parse_consolidation_plan(raw).unwrap();
        assert!(plan.updates.is_empty() && plan.deletes.is_empty());
        assert_eq!(plan.creates.len(), 2);
        assert!(plan.creates[0].legacy_evidence.is_empty());
        assert_eq!(
            plan.creates[1].legacy_evidence,
            vec![(1, "Bob joined Acme".to_string())]
        );
    }

    #[test]
    fn parse_plan_empty_noop() {
        let plan = parse_consolidation_plan(r#"{"creates":[],"updates":[],"deletes":[]}"#).unwrap();
        assert!(plan.creates.is_empty() && plan.updates.is_empty() && plan.deletes.is_empty());
    }

    #[test]
    fn facts_text_matches_upstream_line_format() {
        let facts = vec![Memory {
            id: "a1".into(),
            bank_id: "b".into(),
            kind: crate::types::MemoryKind::World,
            text: "Alice joined Google.".into(),
            confidence: None,
            occurred_start_ms: Some(1_767_225_600_000), // 2026-01-01
            occurred_end_ms: None,
            mentioned_at_ms: 1_767_314_600_000, // 2026-01-02
            source: None,
            context: None,
            document_id: None,
            tags: vec![],
            metadata: None,
        }];
        let text = render_facts_text(&facts);
        assert_eq!(
            text,
            "[a1] Alice joined Google. (occurred_start=2026-01-01, mentioned_at=2026-01-02)"
        );
    }

    #[test]
    fn observations_text_renders_json_array() {
        let obs = Observation {
            id: "o1".into(),
            bank_id: "b".into(),
            entity_id: "e1".into(),
            text: "Alice works at Google.".into(),
            evidence: vec![],
            proof_count: 3,
            history: vec![],
            updated_at_ms: 1_767_225_600_000,
            created_at_ms: 1_767_225_600_000,
            embedding: None,
        };
        let text = render_observations_text(&[obs], &[]);
        assert!(text.contains(r#""id":"o1""#), "{text}");
        assert!(text.contains(r#""proof_count":3"#), "{text}");
        assert!(text.contains("2026-01-01"), "{text}");
        assert_eq!(render_observations_text(&[], &[]), "(none)");
    }

    #[test]
    fn entity_prompt_carries_upstream_system_and_entity_scope() {
        let (system, user) = build_entity_consolidation_prompt("Alice", &[], &[], None);
        assert!(
            system.contains("You are a memory consolidation system"),
            "upstream system prompt: {system}"
        );
        assert!(system.contains("PREFER UPDATE OVER CREATE"));
        assert!(user.starts_with("ENTITY: Alice\n\n## MISSION"), "{user}");
        assert!(user.contains("### New facts"), "{user}");
        assert!(user.contains("### Existing observations"), "{user}");
    }
}
