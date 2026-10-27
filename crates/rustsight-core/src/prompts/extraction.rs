use crate::types::{CausalRef, EntityType, ExtractedEntity, ExtractedFact, MemoryKind};
use crate::{Error, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;

#[derive(Debug, Clone, Default)]
pub struct ExtractionConfig {
    pub labels: Option<Vec<String>>,
    pub custom_instructions: Option<String>,
    pub output_language: Option<String>,
}

const EXTRACTION_SYSTEM_BASE: &str = r#"You are a FACT EXTRACTION ENGINE for a long-term agent memory system. You convert conversational transcripts and documents into narrative facts for permanent storage.

RULES:
1. NARRATIVE FACTS: each fact must be self-contained and understandable without the surrounding conversation. Include what, when, where, who, and why. Include EVERY detail mentioned - more detail is ALWAYS better than less. Preserve emotions, preferences, motivations, significance, and nuance.
2. WHEN: always capture temporal information. Put absolute dates or date ranges in occurred_start / occurred_end using ISO 8601 (e.g. "2026-03-14"). Resolve relative expressions ("last week", "in March") against the CURRENT TIME provided below. Include weekday names in the fact text when the day is known.
3. COREFERENCE RESOLUTION (CRITICAL): resolve pronouns and aliases to their full referents. "my roommate Emily" and "She" are the same entity "Emily" - never create duplicate entities for the same referent.
4. FACT TYPES: "world" for objective facts about the world, the user, or other people. "experience" for actions the assistant/agent itself performed - write these in FIRST PERSON ("I recommended Python to Bob") and ALWAYS include what the user asked that triggered the action.
5. ENTITIES: list every person, organization, location, product, or concept mentioned, with etype one of: person, organization, location, product, concept, other. Use canonical surface names ("Emily", not "my roommate").
6. CAUSAL: if one fact causes, enables, or prevents another fact in the same input, add a causal entry referencing the target fact by its 0-based index with relation one of: causes, caused_by, enables, prevents.
7. Do NOT invent facts. Skip greetings, pleasantries, and meta-talk. Keep each fact's original language - do not translate.

OUTPUT: respond with a single JSON object, no prose, in this exact shape:
{"facts":[{"text":"...","fact_type":"world"|"experience","entities":[{"name":"...","etype":"person"}],"occurred_start":"2026-03-14"|null,"occurred_end":"2026-03-20"|null,"causal":[{"fact_index":1,"relation":"causes"}]}]}"#;

pub fn build_extraction_prompt(
    cfg: &ExtractionConfig,
    content: &str,
    context: Option<&str>,
    now_ms: i64,
) -> (String, String) {
    let mut system = EXTRACTION_SYSTEM_BASE.to_string();
    if let Some(labels) = &cfg.labels {
        if !labels.is_empty() {
            system.push_str(&format!(
                "\n\nLABEL FILTER: only extract facts that fall under these labels: {}. Discard everything else.",
                labels.join(", ")
            ));
        }
    }
    if let Some(custom) = &cfg.custom_instructions {
        system.push_str(&format!(
            "\n\nADDITIONAL INSTRUCTIONS FROM THE BANK OWNER:\n{}",
            custom
        ));
    }
    match &cfg.output_language {
        Some(lang) => system.push_str(&format!(
            "\n\nLANGUAGE: write all extracted facts in {}.",
            lang
        )),
        None => system.push_str(
            "\n\nLANGUAGE: preserve the original input language of each fact; do not translate.",
        ),
    }

    let now = DateTime::from_timestamp_millis(now_ms)
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
    let mut user = String::new();
    if let Some(ctx) = context {
        user.push_str(&format!("CONTEXT: {}\n\n", ctx));
    }
    user.push_str(&format!(
        "CURRENT TIME: {} ({})\n\n",
        now.to_rfc3339(),
        now.format("%A")
    ));
    user.push_str("CONTENT:\n");
    user.push_str(content);
    user.push_str("\n\nRespond with the JSON object only.");
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
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|t| t.and_utc().timestamp_millis());
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m") {
        return d.and_hms_opt(0, 0, 0).map(|t| t.and_utc().timestamp_millis());
    }
    if let Ok(y) = s.parse::<i32>() {
        if (1000..=3000).contains(&y) {
            return chrono::NaiveDate::from_ymd_opt(y, 1, 1)
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|t| t.and_utc().timestamp_millis());
        }
    }
    None
}

fn json_slice(raw: &str) -> &str {
    match (raw.find('{'), raw.rfind('}')) {
        (Some(s), Some(e)) if e > s => &raw[s..=e],
        _ => raw,
    }
}

#[derive(Debug, Deserialize)]
struct RawEntity {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    etype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawCausal {
    #[serde(default)]
    fact_index: Option<usize>,
    #[serde(default)]
    relation: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawFact {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    fact_type: Option<String>,
    #[serde(default)]
    entities: Option<Vec<RawEntity>>,
    #[serde(default)]
    occurred_start: Option<String>,
    #[serde(default)]
    occurred_end: Option<String>,
    #[serde(default)]
    causal: Option<Vec<RawCausal>>,
}

#[derive(Debug, Deserialize)]
struct RawRoot {
    #[serde(default)]
    facts: Option<Vec<RawFact>>,
}

pub fn parse_extraction(raw: &str) -> Result<Vec<ExtractedFact>> {
    let body = json_slice(raw);
    let root: RawRoot = serde_json::from_str(body)
        .map_err(|e| Error::Parse(format!("extraction response: {e}")))?;
    let mut out = vec![];
    for f in root.facts.unwrap_or_default() {
        let text = match f.text {
            Some(t) if !t.trim().is_empty() => t.trim().to_string(),
            _ => continue,
        };
        let fact_type = match f
            .fact_type
            .as_deref()
            .map(|s| s.trim().to_lowercase())
            .as_deref()
        {
            Some("experience") => MemoryKind::Experience,
            _ => MemoryKind::World,
        };
        let entities = f
            .entities
            .unwrap_or_default()
            .into_iter()
            .filter_map(|e| {
                let name = e.name?.trim().to_string();
                if name.is_empty() {
                    return None;
                }
                Some(ExtractedEntity {
                    etype: EntityType::parse(e.etype.as_deref().unwrap_or("other")),
                    name,
                })
            })
            .collect();
        let causal = f
            .causal
            .unwrap_or_default()
            .into_iter()
            .filter_map(|c| {
                let idx = c.fact_index?;
                let rel = c.relation?.to_lowercase();
                if !matches!(
                    rel.as_str(),
                    "causes" | "caused_by" | "enables" | "prevents"
                ) {
                    return None;
                }
                Some(CausalRef {
                    fact_index: idx,
                    relation: rel,
                })
            })
            .collect();
        out.push(ExtractedFact {
            text,
            fact_type,
            entities,
            occurred_start_ms: f.occurred_start.as_deref().and_then(parse_dt_ms),
            occurred_end_ms: f.occurred_end.as_deref().and_then(parse_dt_ms),
            causal,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_composition() {
        let cfg = ExtractionConfig {
            labels: Some(vec!["work".into()]),
            custom_instructions: Some("Prefer technical detail.".into()),
            output_language: Some("German".into()),
        };
        let (system, user) = build_extraction_prompt(&cfg, "Alice joined Google.", Some("career"), 0);
        assert!(system.contains("LABEL FILTER"));
        assert!(system.contains("ADDITIONAL INSTRUCTIONS"));
        assert!(system.contains("German"));
        assert!(user.contains("CONTEXT: career"));
        assert!(user.contains("CONTENT:\nAlice joined Google."));
    }

    #[test]
    fn parse_full() {
        let raw = r#"```json
        {"facts":[{"text":"Alice works at Google.","fact_type":"world","entities":[{"name":"Alice","etype":"PERSON"},{"name":"Google","etype":"organization"}],"occurred_start":"2026-03-14","occurred_end":null,"causal":[{"fact_index":1,"relation":"causes"}]},{"text":"I recommended Python.","fact_type":"experience"},{"text":"","fact_type":"world"}]}
        ```"#;
        let facts = parse_extraction(raw).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].fact_type, MemoryKind::World);
        assert_eq!(facts[0].entities.len(), 2);
        assert_eq!(facts[0].entities[0].etype, EntityType::Person);
        assert!(facts[0].occurred_start_ms.is_some());
        assert_eq!(facts[0].causal.len(), 1);
        assert_eq!(facts[1].fact_type, MemoryKind::Experience);
    }

    #[test]
    fn parse_dt_variants() {
        assert!(parse_dt_ms("2026-10-27").is_some());
        assert!(parse_dt_ms("2026-10-27T10:00:00Z").is_some());
        assert!(parse_dt_ms("2026").is_some());
        assert!(parse_dt_ms("null").is_none());
        assert!(parse_dt_ms("").is_none());
    }
}
