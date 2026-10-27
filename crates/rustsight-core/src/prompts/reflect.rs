use crate::traits::ToolDef;
use crate::types::{Bank, Disposition};
use crate::{Error, Result};
use serde::Deserialize;
use serde_json::json;

pub fn verbalize_disposition(d: &Disposition) -> String {
    let level = |n: u8, low: &str, high: &str| match n {
        1..=2 => format!("low ({n}/5): {low}"),
        3 => format!("moderate ({n}/5)"),
        4..=5 => format!("high ({n}/5): {high}"),
        _ => format!("({n}/5)"),
    };
    format!(
        "DISPOSITION (how you reason and speak):\n\
         - Skepticism: {} — weigh evidence cautiously and question weak claims.\n\
         - Literalism: {} — prefer precise, plain readings over figurative interpretation.\n\
         - Empathy: {} — attend to the emotional context behind what was said.\n\
         - Bias strength: {:.2} — how strongly your existing opinions and bank identity should color new answers (0 = neutral, 1 = strong prior).",
        level(d.skepticism, "accept stated claims at face value", "demand corroboration before believing"),
        level(d.literalism, "read between the lines", "take statements exactly as written"),
        level(d.empathy, "focus on facts over feelings", "actively acknowledge feelings"),
        d.bias_strength,
    )
}

pub fn build_reflect_system(bank: &Bank) -> String {
    let mut s = String::from(
        "You are the REFLECTIVE REASONING LOOP of a long-lived agent with persistent memory. \
         You answer questions on behalf of the agent by gathering evidence from its memory bank, \
         then reasoning over what you found.\n\n",
    );
    if let Some(m) = &bank.mission {
        s.push_str(&format!("MISSION: {m}\n"));
    }
    if let Some(b) = &bank.background {
        s.push_str(&format!("BACKGROUND: {b}\n"));
    }
    if let Some(d) = &bank.directives {
        s.push_str(&format!("DIRECTIVES (hard rules you must never break): {d}\n"));
    }
    s.push('\n');
    s.push_str(&verbalize_disposition(&bank.disposition));
    s.push_str("\n\nTOOL PROTOCOL:\n\
        - Call the `recall` tool to search memory. You may call it multiple times with different queries.\n\
        - Call `read_observation` to read the consolidated profile of a specific entity; prefer observations over raw facts when both exist.\n\
        - SOURCE PRIORITY: observations (distilled beliefs) first, then world/experience facts. Read the distilled answer before the transcript.\n\
        - You MUST retrieve something with a tool before answering. Never fabricate memory.\n\
        - When done, call `finish` with your answer. You may only cite memory ids that tools actually returned to you in this conversation. If the bank does not contain the answer, say so plainly instead of guessing.\n");
    s
}

pub fn reflect_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "recall".into(),
            description: "Search the memory bank and get relevant memories within a token budget.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "token_budget": {"type": "integer", "description": "Approximate token budget for returned context", "default": 2000}
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "read_observation".into(),
            description: "Read the consolidated observation profile for a named entity.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name, e.g. a person's name"}
                },
                "required": ["entity"]
            }),
        },
        ToolDef {
            name: "finish".into(),
            description: "Deliver the final answer. Cite only memory ids that were returned by tools.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}, "description": "Memory ids actually used"}
                },
                "required": ["answer"]
            }),
        },
    ]
}

pub fn build_opinion_extraction_prompt(question: &str, answer: &str) -> (String, String) {
    let system = "You are an OPINION EXTRACTION ENGINE. From an agent's just-given answer, extract any NEW opinions or perspectives the agent expressed, rewritten in the FIRST PERSON as if the agent itself is stating them.\n\n\
        RULES:\n\
        1. An opinion is a judgment, viewpoint, or conclusion that goes beyond stating facts.\n\
        2. Always start with first-person phrases: \"I think...\", \"I believe...\", \"In my view...\", \"Previously I thought... but now...\". Never write third person.\n\
        3. Include the reasoning naturally within the statement.\n\
        4. Do NOT extract statements like \"I don't have enough information\" or \"the facts don't contain X\" — only actual opinions about substantive topics.\n\
        5. Assign a confidence score between 0.0 and 1.0.\n\n\
        OUTPUT: a single JSON object: {\"opinions\":[{\"text\":\"I think ...\",\"confidence\":0.8}]} — return an empty list if there are no opinions.";
    let user = format!(
        "ORIGINAL QUESTION: {question}\n\nANSWER PROVIDED:\n{answer}\n\nExtract new opinions as JSON."
    );
    (system.to_string(), user)
}

#[derive(Debug, Deserialize)]
struct RawOpinion {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawRoot {
    #[serde(default)]
    opinions: Option<Vec<RawOpinion>>,
}

pub fn parse_opinions(raw: &str) -> Result<Vec<(String, f32)>> {
    let body = match (raw.find('{'), raw.rfind('}')) {
        (Some(s), Some(e)) if e > s => &raw[s..=e],
        _ => raw,
    };
    let root: RawRoot = serde_json::from_str(body)
        .map_err(|e| Error::Parse(format!("opinion extraction response: {e}")))?;
    Ok(root
        .opinions
        .unwrap_or_default()
        .into_iter()
        .filter_map(|o| {
            let text = o.text?.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some((text, o.confidence.unwrap_or(0.6).clamp(0.0, 1.0)))
        })
        .collect())
}

pub fn build_mental_model_prompt(question: &str, memories_block: &str) -> (String, String) {
    let system = "You are a MENTAL MODEL WRITER. You maintain a standing answer page for a question the bank's owner asks often. \
        Write a clear, well-organized answer to the question using ONLY the provided memories. \
        If the memories are insufficient, write what is known and mark the gaps. Plain prose or markdown. No preamble.";
    let user = format!(
        "QUESTION: {question}\n\nMEMORIES:\n{memories_block}\n\nWrite the standing answer page."
    );
    (system.to_string(), user)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disposition_text() {
        let d = Disposition {
            skepticism: 5,
            literalism: 2,
            empathy: 3,
            bias_strength: 0.7,
        };
        let s = verbalize_disposition(&d);
        assert!(s.contains("high (5/5)"));
        assert!(s.contains("low (2/5)"));
        assert!(s.contains("moderate (3/5)"));
    }

    #[test]
    fn reflect_system_includes_bank() {
        let bank = Bank {
            id: "b".into(),
            mission: Some("Research assistant".into()),
            directives: Some("Never recommend stocks.".into()),
            background: Some("ML focus".into()),
            disposition: Disposition::default(),
        };
        let s = build_reflect_system(&bank);
        assert!(s.contains("MISSION: Research assistant"));
        assert!(s.contains("DIRECTIVES"));
        assert!(s.contains("finish"));
    }

    #[test]
    fn parse_opinions_filters_empty() {
        let raw = r#"{"opinions":[{"text":"I think Alice is reliable.","confidence":1.7},{"text":"  "},{"text":"I believe X"}]}"#;
        let ops = parse_opinions(raw).unwrap();
        assert_eq!(ops.len(), 2);
        assert!((ops[0].1 - 1.0).abs() < 1e-6);
        assert!((ops[1].1 - 0.6).abs() < 1e-6);
    }
}
