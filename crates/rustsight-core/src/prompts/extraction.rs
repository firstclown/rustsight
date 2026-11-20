//! Engine-facing retain extraction: config, message assembly (backed by the
//! upstream-parity builders in [`crate::prompts::upstream`]), and lenient
//! response parsing.
//!
//! The system prompt and user message are upstream's (byte-parity pinned in
//! `tests/upstream_prompt_parity.rs`) plus one documented divergence: a JSON
//! output-envelope section appended for providers without structured-output
//! schema support — see `prompts/upstream/DIVERGENCES.md`.

use crate::prompts::upstream::extraction::{
    build_chunk_user_message, build_extraction_system_prompt, infer_temporal_date_ms,
    retain_mission_preamble, ChunkUserMessage, ExtractionMode, LabelGroup,
    UpstreamExtractionConfig,
};
use crate::types::{CausalRef, EntityType, ExtractedEntity, ExtractedFact, MemoryKind};
use crate::{Error, Result};
use chrono::{DateTime, NaiveDate, Utc};
use serde_json::Value;

/// Retain extraction configuration, upstream-shaped
/// (`retain_extraction_mode` and friends). See
/// [`crate::prompts::upstream::extraction`] for the faithful builder.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub mode: ExtractionMode,
    pub extract_causal_links: bool,
    /// Bank-owner custom instructions for custom mode.
    pub custom_instructions: Option<String>,
    /// Forces all extracted output into one language.
    pub output_language: Option<String>,
    /// Controlled-vocabulary classification labels.
    pub entity_labels: Option<Vec<LabelGroup>>,
    /// Whether regular named entities are extracted alongside labels.
    pub entities_allow_free_form: bool,
    /// Swaps the "N/A" placeholder instruction for "null".
    pub optional_fact_dimensions: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            mode: ExtractionMode::Concise,
            extract_causal_links: false,
            custom_instructions: None,
            output_language: None,
            entity_labels: None,
            entities_allow_free_form: true,
            optional_fact_dimensions: false,
        }
    }
}

impl ExtractionConfig {
    fn to_upstream(&self) -> UpstreamExtractionConfig {
        UpstreamExtractionConfig {
            mode: self.mode,
            extract_causal_links: self.extract_causal_links,
            custom_instructions: self.custom_instructions.clone(),
            llm_output_language: self.output_language.clone(),
            entity_labels: self.entity_labels.clone(),
            entities_allow_free_form: self.entities_allow_free_form,
            optional_fact_dimensions: self.optional_fact_dimensions,
        }
    }
}

/// The JSON envelope divergence: upstream hands the model a structured-output
/// schema for the response; rustsight's OpenAI-compat provider offers JSON
/// mode only, so the envelope is stated in the prompt.
const OUTPUT_ENVELOPE: &str = "\n\nOUTPUT: respond with a single JSON object, no prose, in this exact shape: {\"facts\":[{\"what\":\"...\",\"when\":\"...\",\"where\":\"...\",\"who\":\"...\",\"why\":\"...\",\"fact_kind\":\"event\"|\"conversation\",\"occurred_start\":\"2026-03-14\"|null,\"occurred_end\":\"2026-03-20\"|null,\"fact_type\":\"world\"|\"assistant\",\"entities\":[\"...\"],\"causal_relations\":[{\"target_index\":0,\"relation_type\":\"caused_by\"}]}]}";

/// Added when the bank configures entity labels: without a response schema
/// the model has to be told where the classification goes.
const LABELS_ENVELOPE: &str = " Every fact must also carry a \"labels\" object holding the ENTITY LABELS fields described above.";

/// Build the (system, user) extraction messages for one retain chunk from
/// the upstream-parity builders.
#[allow(clippy::too_many_arguments)]
pub fn build_extraction_messages(
    cfg: &ExtractionConfig,
    bank_mission: Option<&str>,
    chunk: &str,
    chunk_index: usize,
    total_chunks: usize,
    event_date: Option<DateTime<Utc>>,
    context: Option<&str>,
) -> (String, String) {
    let mut system = build_extraction_system_prompt(&cfg.to_upstream()) + OUTPUT_ENVELOPE;
    if cfg.entity_labels.as_ref().is_some_and(|l| !l.is_empty()) {
        system.push_str(LABELS_ENVELOPE);
    }
    let user = build_chunk_user_message(&ChunkUserMessage {
        chunk,
        chunk_index,
        total_chunks,
        event_date,
        context: context.unwrap_or(""),
        metadata: &[],
        agent_name: None,
        mission_preamble: &retain_mission_preamble(bank_mission),
    });
    (system, user)
}

pub fn parse_dt_ms(s: &str) -> Option<i64> {
    let s = s.trim().trim_matches('"').trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp_millis());
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return d
            .and_hms_opt(0, 0, 0)
            .map(|t| t.and_utc().timestamp_millis());
    }
    // Date+time without a zone offset (upstream models sometimes emit this):
    // treat as UTC.
    if let Ok(t) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(t.and_utc().timestamp_millis());
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m") {
        return d
            .and_hms_opt(0, 0, 0)
            .map(|t| t.and_utc().timestamp_millis());
    }
    if let Ok(y) = s.parse::<i32>() {
        if (1000..=3000).contains(&y) {
            return NaiveDate::from_ymd_opt(y, 1, 1)
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|t| t.and_utc().timestamp_millis());
        }
    }
    None
}

/// Slice the outermost JSON value out of a raw LLM response. A top-level
/// array (bare fact list) wins over a `{...}` span: for `{"facts":[...]}`
/// both yield the same fact array, while `[...]` responses only parse via
/// the array slice (the `{...}` span would grab an inner fact object).
fn json_slice(raw: &str) -> &str {
    for (open, close) in [('[', ']'), ('{', '}')] {
        if let (Some(s), Some(e)) = (raw.find(open), raw.rfind(close)) {
            if e > s {
                return &raw[s..=e];
            }
        }
    }
    raw
}

/// Upstream `get_value`: the value if non-empty and not "N/A", else None.
fn get_value<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    let v = obj.get(key)?;
    let empty = match v {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    };
    if empty {
        return None;
    }
    if let Value::String(s) = v {
        if s.eq_ignore_ascii_case("n/a") {
            return None;
        }
    }
    Some(v)
}

fn value_as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Parse an extraction response. Accepts both upstream's schema (`what` /
/// `when` / `where` / `who` / `why`, `fact_type` world|assistant, entities as
/// plain strings, `causal_relations[].target_index`) and rustsight's legacy
/// schema (`text`, entities as `{name, etype}` objects, `causal[].fact_index`).
pub fn parse_extraction(raw: &str) -> Result<Vec<ExtractedFact>> {
    parse_extraction_with_reference(raw, None)
}

/// [`parse_extraction`] with the retain event date available for
/// upstream's relative-date inference on undated event facts
/// (`_infer_temporal_date`).
pub fn parse_extraction_with_reference(
    raw: &str,
    event_date: Option<DateTime<Utc>>,
) -> Result<Vec<ExtractedFact>> {
    let body = json_slice(raw);
    let parsed: Value = serde_json::from_str(body)
        .map_err(|e| Error::Parse(format!("extraction response: {e}")))?;
    let facts_arr: Vec<&Value> = match &parsed {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => match parsed.get("facts") {
            Some(Value::Array(items)) => items.iter().collect(),
            _ => vec![],
        },
        _ => vec![],
    };

    let mut out = vec![];
    for f in facts_arr {
        let Value::Object(_) = f else { continue };
        // NEW FORMAT: what (fallbacks: factual_core, legacy text)
        let mut what = get_value(f, "what").and_then(value_as_string);
        if what.is_none() {
            what = get_value(f, "factual_core").and_then(value_as_string);
        }
        if what.is_none() {
            what = get_value(f, "text").and_then(value_as_string);
        }
        let Some(what) = what.filter(|w| !w.trim().is_empty()) else {
            continue;
        };
        let when = get_value(f, "when").and_then(value_as_string);
        let who = get_value(f, "who").and_then(value_as_string);
        let why = get_value(f, "why").and_then(value_as_string);

        // fact_type — "assistant" maps to experience, everything else is
        // world. Legacy "experience" is accepted directly. If unexpected,
        // fall back to fact_kind before defaulting to world.
        let raw_fact_type = f.get("fact_type").and_then(value_as_string);
        let fact_type = match raw_fact_type.as_deref().map(str::to_lowercase).as_deref() {
            Some("assistant") | Some("experience") => MemoryKind::Experience,
            Some("world") => MemoryKind::World,
            _ => {
                let raw_fact_kind = f.get("fact_kind").and_then(value_as_string);
                if raw_fact_kind.as_deref() == Some("assistant") {
                    MemoryKind::Experience
                } else {
                    MemoryKind::World
                }
            }
        };

        let fact_kind = match f.get("fact_kind").and_then(value_as_string).as_deref() {
            Some("event") | Some("other") => "event",
            _ => "conversation",
        };

        // Build combined fact text: what | When: X | Involving: Y | why
        let mut combined_parts = vec![what.clone()];
        if let Some(when) = &when {
            combined_parts.push(format!("When: {when}"));
        }
        if let Some(who) = &who {
            combined_parts.push(format!("Involving: {who}"));
        }
        if let Some(why) = &why {
            combined_parts.push(why.clone());
        }
        let combined_text = combined_parts.join(" | ");

        // Temporal fields: explicit values are honored regardless of kind
        // (legacy responses carry them without fact_kind); the relative-date
        // inference only applies to events, as upstream.
        let occurred_start_ms = get_value(f, "occurred_start")
            .and_then(value_as_string)
            .and_then(|s| parse_dt_ms(&s))
            .or_else(|| {
                (fact_kind == "event")
                    .then(|| infer_temporal_date_ms(&combined_text, event_date))
                    .flatten()
            });
        let occurred_end_ms = get_value(f, "occurred_end")
            .and_then(value_as_string)
            .and_then(|s| parse_dt_ms(&s))
            .or(occurred_start_ms); // point events: end = start

        // Entities are plain strings upstream; object forms ({"text": ...}
        // upstream-legacy, {"name", "etype"} rustsight-legacy) are unwrapped.
        let entities = f
            .get("entities")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|item| match item {
                Value::String(s) => {
                    let name = s.trim().to_string();
                    (!name.is_empty()).then_some(ExtractedEntity {
                        etype: EntityType::Other,
                        name,
                    })
                }
                Value::Object(o) => {
                    let text = o
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| o.get("name").and_then(Value::as_str).map(str::to_string))?;
                    let name = text.trim().to_string();
                    if name.is_empty() {
                        return None;
                    }
                    Some(ExtractedEntity {
                        etype: EntityType::parse(
                            o.get("etype").and_then(Value::as_str).unwrap_or("other"),
                        ),
                        name,
                    })
                }
                _ => None,
            })
            .collect();

        // Causal relations: upstream target_index/relation_type plus the
        // legacy fact_index/relation shape.
        let mut causal: Vec<CausalRef> = vec![];
        if let Some(rels) = f.get("causal_relations").and_then(|v| v.as_array()) {
            for rel in rels {
                let Some(obj) = rel.as_object() else { continue };
                let Some(target) = obj.get("target_index").and_then(Value::as_i64) else {
                    continue;
                };
                if target < 0 {
                    continue;
                }
                if obj.get("relation_type").and_then(Value::as_str) == Some("caused_by") {
                    causal.push(CausalRef {
                        fact_index: target as usize,
                        relation: "caused_by".to_string(),
                    });
                }
            }
        }
        if let Some(rels) = f.get("causal").and_then(|v| v.as_array()) {
            for rel in rels {
                let Some(obj) = rel.as_object() else { continue };
                let Some(idx) = obj.get("fact_index").and_then(Value::as_i64) else {
                    continue;
                };
                let Some(rel_type) = obj.get("relation").and_then(Value::as_str) else {
                    continue;
                };
                let rel_type = rel_type.to_lowercase();
                if idx < 0
                    || !matches!(
                        rel_type.as_str(),
                        "causes" | "caused_by" | "enables" | "prevents"
                    )
                {
                    continue;
                }
                causal.push(CausalRef {
                    fact_index: idx as usize,
                    relation: rel_type,
                });
            }
        }

        out.push(ExtractedFact {
            text: combined_text,
            fact_type,
            entities,
            occurred_start_ms,
            occurred_end_ms,
            causal,
            chunk_index: 0,
            labels: f.get("labels").filter(|l| l.is_object()).cloned(),
            tags: vec![],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_shape_parses() {
        let raw = r#"{"facts":[{
            "what": "Alice joined Google.",
            "when": "Tuesday, October 27, 2026",
            "where": "N/A",
            "who": "Alice",
            "why": "career change",
            "fact_kind": "event",
            "occurred_start": "2026-10-27",
            "occurred_end": null,
            "fact_type": "world",
            "entities": ["Alice", "Google"],
            "causal_relations": [{"target_index": 0, "relation_type": "caused_by"}]
        }]}"#;
        let facts = parse_extraction(raw).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(
            facts[0].text,
            "Alice joined Google. | When: Tuesday, October 27, 2026 | Involving: Alice | career change"
        );
        assert_eq!(facts[0].fact_type, MemoryKind::World);
        assert_eq!(facts[0].entities.len(), 2);
        assert_eq!(facts[0].entities[0].name, "Alice");
        assert!(facts[0].occurred_start_ms.is_some());
        // point event: end defaults to start
        assert_eq!(facts[0].occurred_start_ms, facts[0].occurred_end_ms);
        assert_eq!(facts[0].causal[0].relation, "caused_by");
    }

    #[test]
    fn assistant_maps_to_experience() {
        let raw = r#"{"facts":[{"what":"I recommended Python.","fact_type":"assistant","entities":["user"]}]}"#;
        let facts = parse_extraction(raw).unwrap();
        assert_eq!(facts[0].fact_type, MemoryKind::Experience);
    }

    #[test]
    fn legacy_shape_still_parses() {
        let raw = r#"```json
        {"facts":[{"text":"Alice works at Google.","fact_type":"world","entities":[{"name":"Alice","etype":"PERSON"},{"name":"Google","etype":"organization"}],"occurred_start":"2026-03-14","occurred_end":null,"causal":[{"fact_index":1,"relation":"causes"}]},{"text":"I recommended Python.","fact_type":"experience"},{"text":"","fact_type":"world"}]}
        ```"#;
        let facts = parse_extraction(raw).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].text, "Alice works at Google.");
        assert_eq!(facts[0].fact_type, MemoryKind::World);
        assert_eq!(facts[0].entities.len(), 2);
        assert_eq!(facts[0].entities[0].etype, EntityType::Person);
        assert!(facts[0].occurred_start_ms.is_some());
        assert_eq!(facts[0].causal.len(), 1);
        assert_eq!(facts[1].fact_type, MemoryKind::Experience);
    }

    #[test]
    fn conversation_facts_drop_placeholder_when() {
        let raw = r#"{"facts":[{"what":"Alice likes coffee.","when":"N/A","who":"N/A","why":"N/A","fact_type":"world","entities":["Alice"]}]}"#;
        let facts = parse_extraction(raw).unwrap();
        assert_eq!(facts[0].text, "Alice likes coffee.");
        assert!(facts[0].occurred_start_ms.is_none());
    }

    #[test]
    fn relative_date_inference_on_events() {
        let raw = r#"{"facts":[{"what":"Went to the pottery workshop yesterday.","fact_kind":"event","fact_type":"world","entities":[]}]}"#;
        let event_date = NaiveDate::from_ymd_opt(2026, 10, 27)
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|t| t.and_utc());
        let facts = parse_extraction_with_reference(raw, event_date).unwrap();
        assert_eq!(
            facts[0].occurred_start_ms,
            event_date
                .map(|d| d.timestamp_millis())
                .map(|m| m - 86_400_000)
        );
    }

    #[test]
    fn top_level_array_parses() {
        let raw = r#"[{"what":"A fact.","fact_type":"world","entities":[]}]"#;
        let facts = parse_extraction(raw).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].text, "A fact.");
    }

    #[test]
    fn envelope_addendum_appended_and_mission_rides_user_message() {
        let cfg = ExtractionConfig {
            mode: ExtractionMode::Custom,
            custom_instructions: Some("Focus on compliance {strictly}.".into()),
            output_language: Some("German".into()),
            ..Default::default()
        };
        let (system, user) = build_extraction_messages(
            &cfg,
            Some(r#"{"focus": "compliance"}"#),
            "Alice joined Google.",
            0,
            2,
            None,
            Some("career"),
        );
        // Custom instructions keep their braces doubled in the final prompt
        // (upstream escapes for str.format; the substituted value is inserted
        // verbatim — verified against upstream output).
        assert!(system.contains("Focus on compliance {{strictly}}."));
        assert!(system.contains("Respond exclusively in German"));
        assert!(system.contains("OUTPUT: respond with a single JSON object"));
        assert!(user.contains(r#"{"focus": "compliance"}"#));
        assert!(user.contains("FOCUS — What to retain for this bank"));
        assert!(user.contains("Chunk: 1/2"));
        assert!(user.contains("Context: career"));
        assert!(user.contains("Content:\nAlice joined Google."));
    }

    #[test]
    fn parse_dt_variants() {
        assert!(parse_dt_ms("2026-10-27").is_some());
        assert!(parse_dt_ms("2026-10-27T10:00:00Z").is_some());
        assert!(parse_dt_ms("2026-10-27T10:00:00").is_some());
        assert!(parse_dt_ms("2026").is_some());
        assert!(parse_dt_ms("null").is_none());
        assert!(parse_dt_ms("").is_none());
    }
}
