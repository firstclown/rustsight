use crate::prompts::upstream::reflect::Directive;
use crate::traits::ToolDef;
use crate::types::{Bank, Disposition, EntityType, ExtractedEntity};
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

pub fn build_reflect_system(bank: &Bank, directives: &[Directive]) -> String {
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
    let rules: Vec<String> = directives
        .iter()
        .filter(|d| !d.content.trim().is_empty())
        .map(|d| {
            if d.name.is_empty() {
                d.content.clone()
            } else {
                format!("**{}**: {}", d.name, d.content)
            }
        })
        .collect();
    match rules.as_slice() {
        [] => {}
        // One rule keeps the single-line form banks have always had.
        [only] => s.push_str(&format!(
            "DIRECTIVES (hard rules you must never break): {only}\n"
        )),
        many => {
            s.push_str("DIRECTIVES (hard rules you must never break):\n");
            for rule in many {
                s.push_str(&format!("- {rule}\n"));
            }
        }
    }
    s.push('\n');
    s.push_str(&verbalize_disposition(&bank.disposition));
    s.push_str("\n\nTOOL PROTOCOL:\n\
         - Call `search_observations` first: it returns consolidated knowledge (observations) — \
         the distilled answer, with freshness (mentioned_at) and proof counts. If observations \
         exist but do not STATE the answer, or are stale, descend to raw facts.\n\
         - Call `recall` for raw facts (world/experience memories) when observations are missing, \
         stale, or lack the specific detail asked for. You MUST run recall before reporting that \
         the bank holds nothing about something.\n\
         - Call `read_observation` to read every observation for one named entity.\n\
         - Tool results are ordered by semantic relevance, NOT by time — the position of an entry \
         says nothing about when it was retained. For temporal reasoning read the per-entry \
         mentioned_at / occurred dates.\n\
         - You MUST retrieve something with a tool before answering. Never fabricate memory.\n\
         - When done, call `finish` with your answer. The answer is the full response, not a summary of it: \
         carry over every relevant fact, date and number from the tool results, at the same depth you would \
         write for a reader who cannot see them. Connect related memories to form a complete picture — if you \
         have related information, use it to give the best possible answer.\n\
         - Citations: you may only cite memory ids that tools actually returned to you in this conversation, \
         and they go in the `citations` array — NEVER cite memory ids or UUIDs in the answer text itself. \
         If the bank does not contain the answer, say so plainly instead of guessing.\n\n");
    // Upstream `build_system_prompt_for_tools` Query Strategy section,
    // adapted to this engine's tool names: one literal echo of the question
    // finds one facet of a multi-entity question; gold's answer breadth
    // comes from decomposed searches.
    s.push_str(
        "## Query Strategy\n\
         search_observations() and recall() use semantic search. NEVER just echo the user's \
         question - decompose it into targeted searches:\n\n\
         BAD: User asks 'recurring lesson themes between students' -> \
         search_observations('recurring lesson themes between students')\n\
         GOOD: Break it down into component searches:\n\
         \x20 1. search_observations('lessons') - find all lesson-related observations\n\
         \x20 2. search_observations('teaching sessions') - alternative phrasing\n\
         \x20 3. recall('student progress') - find student-related raw facts\n\n\
         Think: What ENTITIES and CONCEPTS does this question involve? Search for each separately.\n\n",
    );
    // Upstream's MODERATE research-depth wording, verbatim (names no tools).
    s.push_str(
        "## RESEARCH DEPTH: MODERATE (Balanced)\n\
         - Balance thoroughness with efficiency\n\
         - Check multiple sources when the question warrants it\n\
         - Verify stale data if it's central to the answer\n\
         - A result that is merely on the same topic does not answer the question: when it does \
         not cover it, go on to the next level\n\
         - Don't over-explore, but ensure reasonable coverage\n\n",
    );
    // Engine-side evidence discipline (extends upstream's grounding
    // boundary): the parity runs' contradictions were all synthesis-level
    // over-inference — attendance/preferences inferred from related notes,
    // mentioned-in-period read as happened-in-period, a yes/no answered
    // against the final milestone only. The participation line encodes the
    // bank owner's adjudicated semantics (2026-11-12): activities recorded
    // as done together — co-writing, pair work — ARE meetings; planning,
    // scheduling, or intending is not; and liking a specific meal is a
    // record about that meal, never a standing preference.
    s.push_str(
        "## Evidence Discipline\n\
         - A record is evidence only of what it states. Do NOT infer that someone liked, \
         owned, or used something merely because they ate it, went there, or bought it — and \
         a record of enjoying one meal says only that that meal was enjoyed, not a food \
         preference. Activities recorded as done together with someone (co-writing, pair \
         work) DO count as meeting them; planning, scheduling, or intending something with \
         them does not.\n\
         - `mentioned_at` is when a memory was written down; `occurred` dates are when it \
         happened. For \"in <period>\" questions select on occurred dates: an event merely \
         mentioned in the period did not necessarily happen in it.\n\
         - Prefer stating what IS recorded over inferring. \"The memories do not record X\" \
         followed by the nearest recorded facts is a complete, correct answer.\n\
         - For before/after-a-date questions, weigh every recorded milestone of the subject — \
         an earlier partial ship, visit, or change counts — not only the final one.\n\n",
    );
    s
}

/// Upstream `_DONE_SUFFICIENCY_GATE` (tools_schema.py), adapted from `done`
/// to this engine's `finish` tool: the stopping condition the model weighs
/// at the moment it picks finish over another search.
pub const FINISH_SUFFICIENCY_GATE: &str = " You have NOT gathered enough while the levels you \
     searched do not STATE the answer: a result that only shares the question's topic is not an \
     answer, so search the next level (search_observations, then recall) instead of answering \
     around it. Never call finish to report that nothing is known unless recall has already run \
     on the question's key terms.";

pub fn reflect_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "search_observations".into(),
            description: "Search the consolidated knowledge base (observations): distilled \
                beliefs synthesized from raw facts, with freshness and proof counts. Start \
                here; descend to `recall` when observations do not state the answer."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    // Upstream search tool default: 5000 result tokens.
                    "token_budget": {"type": "integer", "description": "Approximate token budget for returned context. Use higher values for broader searches.", "default": 5000}
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "recall".into(),
            description: "Search raw memory facts (world/experience memories — the source \
                observations are built from) within a token budget."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    // Upstream recall tool default: 2048 result tokens.
                    "token_budget": {"type": "integer", "description": "Optional limit on result size. Use higher values for broader searches.", "default": 2048}
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "read_observation".into(),
            description: "Read every consolidated observation for one named entity.".into(),
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
            description: format!(
                "Deliver the final answer. Cite only memory ids that were returned by tools.{FINISH_SUFFICIENCY_GATE}"
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your response as well-formatted markdown; use headers, lists, and bold where they help. \
                            This is the ANSWER, not a summary of it: carry over every relevant fact, date and number \
                            from the tool results, at the same depth you would write for a reader who cannot see them. \
                            NEVER include memory IDs, UUIDs, or citation markers in this text — put ids only in the citations array."
                    },
                    "citations": {"type": "array", "items": {"type": "string"}, "description": "Memory ids actually used"},
                    // Upstream `_DONE_ANSWER_FOUND_PROPERTY` (#4567): a false with
                    // no recall yet is refused by the engine and re-asked.
                    "answer_found": {"type": "boolean", "description": "true if what you retrieved states the answer to the question. false if you are reporting that nothing (or not enough) is known about it."}
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
        5. Assign a confidence score between 0.0 and 1.0.\n\
        6. ENTITIES: list the canonical surface names of every person, organization, location, product, or concept the opinion is about, with etype one of: person, organization, location, product, concept, other. Use an empty list if the opinion is not about any specific entity.\n\n\
        OUTPUT: a single JSON object: {\"opinions\":[{\"text\":\"I think ...\",\"confidence\":0.8,\"entities\":[{\"name\":\"...\",\"etype\":\"person\"}]}]} — return an empty list if there are no opinions.";
    let user = format!(
        "ORIGINAL QUESTION: {question}\n\nANSWER PROVIDED:\n{answer}\n\nExtract new opinions as JSON."
    );
    (system.to_string(), user)
}

#[derive(Debug, Clone)]
pub struct ParsedOpinion {
    pub text: String,
    pub confidence: f32,
    pub entities: Vec<ExtractedEntity>,
}

#[derive(Debug, Deserialize)]
struct RawOpinionEntity {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    etype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawOpinion {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    entities: Option<Vec<RawOpinionEntity>>,
}

#[derive(Debug, Deserialize)]
struct RawRoot {
    #[serde(default)]
    opinions: Option<Vec<RawOpinion>>,
}

pub fn parse_opinions(raw: &str) -> Result<Vec<ParsedOpinion>> {
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
            let entities = o
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
            Some(ParsedOpinion {
                text,
                confidence: o.confidence.unwrap_or(0.6).clamp(0.0, 1.0),
                entities,
            })
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

    fn bank_directives(bank: &Bank) -> Vec<Directive> {
        bank.directives
            .iter()
            .map(|d| Directive::new("", d))
            .collect()
    }

    #[test]
    fn several_directives_render_as_a_named_list() {
        let bank = Bank {
            id: "b".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        };
        let one = build_reflect_system(&bank, &[Directive::new("", "Never recommend stocks.")]);
        assert!(
            one.contains("DIRECTIVES (hard rules you must never break): Never recommend stocks.\n")
        );
        let many = build_reflect_system(
            &bank,
            &[
                Directive::new("Formal", "Answer formally."),
                Directive::new("", "Cite sources."),
            ],
        );
        assert!(many.contains(
            "DIRECTIVES (hard rules you must never break):\n- **Formal**: Answer formally.\n- Cite sources.\n"
        ));
    }

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
        let s = build_reflect_system(&bank, &bank_directives(&bank));
        assert!(s.contains("MISSION: Research assistant"));
        assert!(s.contains("DIRECTIVES"));
        assert!(s.contains("finish"));
    }

    /// The tool ladder mirrors upstream's leveling: observations first
    /// (`search_observations`), raw facts behind `recall`, `finish` last.
    #[test]
    fn reflect_tools_expose_search_ladder() {
        let tools = reflect_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "search_observations",
                "recall",
                "read_observation",
                "finish"
            ]
        );
        let recall = tools.iter().find(|t| t.name == "recall").unwrap();
        assert!(recall.description.contains("raw"));
        let search = tools
            .iter()
            .find(|t| t.name == "search_observations")
            .unwrap();
        assert!(search.description.contains("consolidated"));
    }

    /// The system prompt carries the observations-first ladder and the
    /// temporal-ordering warning (upstream's two strongest reflect
    /// behaviors).
    #[test]
    fn reflect_system_carries_ladder_and_temporal_note() {
        let s = build_reflect_system(
            &Bank {
                id: "b".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            },
            &[],
        );
        assert!(s.contains("search_observations"));
        assert!(s.contains("ordered by semantic relevance, NOT by time"));
        assert!(s.contains("mentioned_at"));
    }

    #[test]
    fn reflect_system_carries_answer_completeness_guidance() {
        // Upstream parity-run finding (2026-11-11): without the answer-not-a-
        // summary wording, models answer the literal question in one line and
        // the judge scores adjacent retrieved facts as omitted information.
        let s = build_reflect_system(
            &Bank {
                id: "b".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            },
            &[],
        );
        assert!(s.contains("not a summary of it"));
        assert!(s.contains("carry over every relevant fact, date and number"));
        assert!(s.contains("NEVER cite memory ids or UUIDs in the answer text"));
    }

    #[test]
    fn finish_tool_answer_field_guides_depth_and_bans_inline_ids() {
        let tools = reflect_tools();
        let finish = tools.iter().find(|t| t.name == "finish").unwrap();
        let answer = &finish.parameters["properties"]["answer"];
        let desc = answer["description"].as_str().unwrap();
        assert!(desc.contains("well-formatted markdown"));
        assert!(desc.contains("not a summary of it"));
        assert!(desc.contains("NEVER include memory IDs"));
    }

    /// Upstream's Query Strategy and MODERATE research-depth sections drive
    /// gold's evidence breadth (decomposed searches, not one literal echo) —
    /// pin their presence.
    #[test]
    fn reflect_system_carries_query_strategy_and_depth() {
        let s = build_reflect_system(
            &Bank {
                id: "b".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            },
            &[],
        );
        assert!(s.contains("## Query Strategy"));
        assert!(s.contains("NEVER just echo the user's question"));
        assert!(s.contains("Search for each separately"));
        assert!(s.contains("## RESEARCH DEPTH: MODERATE (Balanced)"));
    }

    /// The engine-side evidence-discipline section (RS-20): the parity
    /// runs' contradictions were inference — preferences from outings,
    /// mentioned-as-occurred, final-milestone yes/nos. The participation
    /// line encodes the bank owner's adjudicated semantics: together
    /// activities are meetings, planning is not, one liked meal is not a
    /// preference.
    #[test]
    fn reflect_system_carries_evidence_discipline() {
        let s = build_reflect_system(
            &Bank {
                id: "b".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            },
            &[],
        );
        assert!(s.contains("## Evidence Discipline"));
        assert!(s.contains("DO count as meeting them"));
        assert!(s.contains("not a food"));
        assert!(s.contains("mentioned"));
        assert!(s.contains("every recorded milestone"));
    }

    /// Tool budget defaults mirror upstream's schemas: search 5000, recall
    /// 2048 (the 2026-11-11 rerun gathered one literal 2000-token search).
    #[test]
    fn reflect_tool_budget_defaults_match_upstream() {
        let tools = reflect_tools();
        let search = tools
            .iter()
            .find(|t| t.name == "search_observations")
            .unwrap();
        assert_eq!(
            search.parameters["properties"]["token_budget"]["default"],
            5000
        );
        let recall = tools.iter().find(|t| t.name == "recall").unwrap();
        assert_eq!(
            recall.parameters["properties"]["token_budget"]["default"],
            2048
        );
    }

    /// The finish tool carries upstream's #4567 machinery: the sufficiency
    /// gate in the description and the `answer_found` verdict the engine
    /// refuses a premature negative on.
    #[test]
    fn finish_tool_carries_sufficiency_gate_and_answer_found() {
        let tools = reflect_tools();
        let finish = tools.iter().find(|t| t.name == "finish").unwrap();
        assert!(finish.description.contains("You have NOT gathered enough"));
        assert!(finish.description.contains(
            "Never call finish to report that nothing is known unless recall has already run"
        ));
        let answer_found = &finish.parameters["properties"]["answer_found"];
        assert_eq!(answer_found["type"], "boolean");
        assert!(answer_found["description"]
            .as_str()
            .unwrap()
            .contains("nothing (or not enough) is known"));
    }

    #[test]
    fn parse_opinions_filters_empty() {
        let raw = r#"{"opinions":[{"text":"I think Alice is reliable.","confidence":1.7},{"text":"  "},{"text":"I believe X"}]}"#;
        let ops = parse_opinions(raw).unwrap();
        assert_eq!(ops.len(), 2);
        assert!((ops[0].confidence - 1.0).abs() < 1e-6);
        assert!((ops[1].confidence - 0.6).abs() < 1e-6);
        assert!(ops.iter().all(|o| o.entities.is_empty()));
    }

    #[test]
    fn parse_opinions_with_entities() {
        let raw = r#"{"opinions":[{"text":"I think Alice is reliable.","confidence":0.8,"entities":[{"name":"Alice","etype":"PERSON"},{"name":"  "},{"etype":"person"}]},{"text":"I believe X","entities":null}]}"#;
        let ops = parse_opinions(raw).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].entities.len(), 1);
        assert_eq!(ops[0].entities[0].name, "Alice");
        assert_eq!(ops[0].entities[0].etype, EntityType::Person);
        assert!(ops[1].entities.is_empty());
    }

    #[test]
    fn opinion_extraction_prompt_asks_for_entities() {
        let (system, _) = build_opinion_extraction_prompt("q", "a");
        assert!(system.contains("ENTITIES"));
        assert!(system.contains("entities"));
    }
}
