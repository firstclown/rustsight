//! Byte-for-byte parity suite: the ported upstream prompt builders vs golden
//! fixtures generated from the upstream Python itself.
//!
//! Fixtures live in `tests/fixtures/upstream_prompts/<case>.txt` and are
//! produced by `scripts/prompt-goldens` from the vendored sources under
//! `crates/rustsight-core/prompts/upstream/<tag>/`. Every case asserts BOTH:
//!
//! 1. exact equality with the upstream-generated fixture (the parity gate —
//!    fails loudly if either side drifts), and
//! 2. an `insta` snapshot (reviewable diffs; refresh deliberately after a
//!    `scripts/prompt-sync` + `scripts/prompt-goldens` round with
//!    `INSTA_FORCE_UPDATE=1 cargo test -p rustsight-core`, then review).

use rustsight_core::prompts::upstream::consolidation::{
    build_consolidation_input, build_consolidation_system_prompt, build_mission_section,
};
use rustsight_core::prompts::upstream::extraction::{
    build_chunk_user_message, build_extraction_system_prompt, retain_mission_preamble,
    ChunkUserMessage, ExtractionMode, LabelGroup, LabelGroupKind, LabelValue, MapField,
    MapFieldKind, UpstreamExtractionConfig,
};
use rustsight_core::prompts::upstream::reflect::{
    build_agent_user_prompt, build_final_system_prompt, build_system_prompt_for_tools, Directive,
    DispositionTraits, ReflectBankProfile, ToolsPromptOptions,
};

/// Frozen "now" — the same value upstream's own reflect prompt tests
/// monkeypatch (`test_reflect_prompt_builder.py`).
const FROZEN_NOW: &str = "2026-08-09 14:32 UTC";

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/upstream_prompts/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
        // Strip the two-line provenance header scripts/prompt-goldens writes.
        .splitn(3, '\n')
        .nth(2)
        .unwrap()
        .to_string()
}

fn assert_parity(case: &str, output: &str) {
    let golden = fixture(case);
    assert!(
        output == golden,
        "parity failure for '{case}': Rust builder output differs from upstream fixture.\n\
         Run `scripts/prompt-sync && scripts/prompt-goldens` to refresh goldens first; if they\n\
         are current, the Rust port in src/prompts/upstream/ has drifted. First difference:\n{}",
        first_diff(&golden, output)
    );
    let mut settings = insta::Settings::clone_current();
    settings.set_snapshot_suffix(case);
    settings.bind(|| insta::assert_snapshot!(output));
}

fn first_diff(a: &str, b: &str) -> String {
    for (i, (ca, cb)) in a.chars().zip(b.chars()).enumerate() {
        if ca != cb {
            let ctx = 60;
            let lo = i.saturating_sub(ctx);
            return format!(
                "at char {i}:\n  upstream: ...{}...\n  rust:     ...{}...",
                a.get(lo..(i + ctx).min(a.len())).unwrap_or(a),
                b.get(lo..(i + ctx).min(b.len())).unwrap_or(b),
            );
        }
    }
    if a.len() != b.len() {
        return format!("lengths differ: upstream {} vs rust {}", a.len(), b.len());
    }
    "identical".to_string()
}

// --- shared configs (mirror scripts/prompt-goldens::generate_cases) ---

fn extraction_cfg() -> UpstreamExtractionConfig {
    UpstreamExtractionConfig::baseline()
}

fn label_groups() -> Vec<LabelGroup> {
    let pedagogy = LabelGroup {
        key: "pedagogy".into(),
        description: "Teaching approach the fact demonstrates".into(),
        kind: LabelGroupKind::Value,
        values: vec![
            LabelValue::new("scaffolding", "Support scaled to the learner"),
            LabelValue::new("modeling", ""),
        ],
        fields: vec![],
    };
    let interest = LabelGroup {
        key: "interest".into(),
        description: "Engagement signals".into(),
        kind: LabelGroupKind::MultiValues,
        values: vec![
            LabelValue::new("active", ""),
            LabelValue::new("passive", ""),
        ],
        fields: vec![],
    };
    let notes = LabelGroup {
        key: "notes".into(),
        description: "Free-text teaching notes".into(),
        kind: LabelGroupKind::Text,
        values: vec![],
        fields: vec![],
    };
    let people = LabelGroup {
        key: "people".into(),
        description: "Structured entities".into(),
        kind: LabelGroupKind::Map,
        values: vec![],
        fields: vec![
            ("student".into(), MapField::text("Student name")),
            (
                "grade".into(),
                MapField {
                    kind: MapFieldKind::Value,
                    description: "Grade level".into(),
                    values: vec![LabelValue::new("freshman", "")],
                    fields: vec![],
                },
            ),
            (
                "guardian".into(),
                MapField {
                    kind: MapFieldKind::Map,
                    description: "Guardian details".into(),
                    values: vec![],
                    fields: vec![
                        ("name".into(), MapField::text("")),
                        (
                            "relation".into(),
                            MapField {
                                kind: MapFieldKind::Value,
                                description: "".into(),
                                values: vec![LabelValue::new("parent", "")],
                                fields: vec![],
                            },
                        ),
                    ],
                },
            ),
        ],
    };
    vec![pedagogy, interest, notes, people]
}

const BANK_NAME: &str = "TestBank";

fn bank<'a>(name: &'a str, mission: &'a str) -> ReflectBankProfile<'a> {
    ReflectBankProfile {
        name: Some(name),
        mission: Some(mission),
        disposition: None,
    }
}

fn tools_opts(bank_profile: ReflectBankProfile<'_>) -> ToolsPromptOptions<'_> {
    ToolsPromptOptions {
        bank: bank_profile,
        context: None,
        directives: &[],
        has_mental_models: false,
        include_observations: true,
        budget: None,
        answer_as_document: false,
        llm_output_language: None,
        answer_ready: false,
        current_datetime: FROZEN_NOW,
    }
}

// --- extraction ---

#[test]
fn extraction_prompts_match_upstream() {
    let cases: Vec<(&str, String)> = vec![
        (
            "extraction_default",
            build_extraction_system_prompt(&extraction_cfg()),
        ),
        (
            "extraction_causal",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                extract_causal_links: true,
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_verbose",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                mode: ExtractionMode::Verbose,
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_custom",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                mode: ExtractionMode::Custom,
                custom_instructions: Some(
                    "Output as {key: value} pairs. Prefer technical detail.".into(),
                ),
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_custom_no_instructions",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                mode: ExtractionMode::Custom,
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_language",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                llm_output_language: Some("Spanish".into()),
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_labels",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                entity_labels: Some(label_groups()),
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_labels_only",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                entity_labels: Some(label_groups()),
                entities_allow_free_form: false,
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_optional_dimensions",
            build_extraction_system_prompt(&UpstreamExtractionConfig {
                optional_fact_dimensions: true,
                ..extraction_cfg()
            }),
        ),
        (
            "extraction_user_message",
            build_chunk_user_message(&ChunkUserMessage {
                chunk: "Alice joined Google. She works on search infrastructure.",
                chunk_index: 1,
                total_chunks: 3,
                event_date: chrono::NaiveDate::from_ymd_opt(2026, 10, 27)
                    .and_then(|d| d.and_hms_opt(0, 0, 0))
                    .map(|t| t.and_utc()),
                context: "career conversation",
                metadata: &[("source", "transcript"), ("room", "hallway")],
                agent_name: Some("Rustsight"),
                mission_preamble: &retain_mission_preamble(Some(r#"{"focus": "compliance"}"#)),
            }),
        ),
        (
            "extraction_user_message_plain",
            build_chunk_user_message(&ChunkUserMessage {
                chunk: "Hello there.",
                chunk_index: 0,
                total_chunks: 1,
                event_date: None,
                context: "",
                metadata: &[],
                agent_name: None,
                mission_preamble: "",
            }),
        ),
    ];
    assert_eq!(cases.len(), 11);
    for (name, output) in cases {
        assert_parity(name, &output);
    }
}

// --- consolidation ---

#[test]
fn consolidation_prompts_match_upstream() {
    let cases: Vec<(&str, String)> = vec![
        (
            "consolidation_system_default",
            build_consolidation_system_prompt(None),
        ),
        (
            "consolidation_system_language",
            build_consolidation_system_prompt(Some("Spanish")),
        ),
        ("consolidation_mission_default", build_mission_section(None)),
        (
            "consolidation_mission_custom",
            build_mission_section(Some(r#"{"dedup": true}"#)),
        ),
        (
            "consolidation_input",
            build_consolidation_input(
                "<facts>",
                r#"<observations>{"id": "o1"}</observations>"#,
                Some(r#"{"dedup": true}"#),
                Some("Use shape {limit, used}"),
            ),
        ),
    ];
    assert_eq!(cases.len(), 5);
    for (name, output) in cases {
        assert_parity(name, &output);
    }
}

// --- reflect: tools system prompt branch matrix ---

#[test]
fn reflect_tools_prompts_match_upstream() {
    let no_competitors = vec![Directive::new(
        "No Competitors",
        "Never mention competitor names.",
    )];
    let cases: Vec<(&str, String)> = vec![
        (
            "reflect_tools_mm_and_obs",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                has_mental_models: true,
                include_observations: true,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_mm_only",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                has_mental_models: true,
                include_observations: false,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_obs_only",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: true,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_recall_only",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_budget_low",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                budget: Some("low"),
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_budget_mid",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                budget: Some("mid"),
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_budget_high",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                budget: Some("high"),
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_mission",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                ..tools_opts(bank(BANK_NAME, "Track customer feedback"))
            }),
        ),
        (
            "reflect_tools_directives",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                directives: &no_competitors,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_disposition",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                bank: ReflectBankProfile {
                    name: Some(BANK_NAME),
                    mission: Some(""),
                    disposition: Some(DispositionTraits {
                        skepticism: Some(5),
                        ..Default::default()
                    }),
                },
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_disposition_neutral",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                bank: ReflectBankProfile {
                    name: Some(BANK_NAME),
                    mission: Some(""),
                    disposition: Some(DispositionTraits {
                        skepticism: Some(3),
                        literalism: Some(2),
                        empathy: Some(4),
                    }),
                },
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_context",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                context: Some("Focus on Q3 metrics."),
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_language",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: true,
                llm_output_language: Some("Spanish"),
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_answer_ready",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                has_mental_models: true,
                include_observations: true,
                answer_ready: true,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
        (
            "reflect_tools_answer_as_document",
            build_system_prompt_for_tools(&ToolsPromptOptions {
                include_observations: false,
                answer_as_document: true,
                ..tools_opts(bank(BANK_NAME, ""))
            }),
        ),
    ];
    assert_eq!(cases.len(), 15);
    for (name, output) in cases {
        assert_parity(name, &output);
    }
}

// --- reflect: final system prompt + agent user prompt ---

#[test]
fn reflect_final_prompts_match_upstream() {
    let language_policy = vec![Directive::new(
        "Language Policy",
        "ALWAYS respond in French language. Never respond in English.",
    )];
    let cases: Vec<(&str, String)> = vec![
        (
            "reflect_final_default",
            build_final_system_prompt(None, None, &[], FROZEN_NOW),
        ),
        (
            "reflect_final_mission",
            build_final_system_prompt(Some("Track {entity} changes"), None, &[], FROZEN_NOW),
        ),
        (
            "reflect_final_language_directives",
            build_final_system_prompt(None, Some("Spanish"), &language_policy, FROZEN_NOW),
        ),
        (
            "reflect_agent_user_prompt",
            build_agent_user_prompt(
                "Who owns billing?",
                Some("Spanish"),
                Some("  Chat decisions outrank commits.  "),
            ),
        ),
    ];
    assert_eq!(cases.len(), 4);
    for (name, output) in cases {
        assert_parity(name, &output);
    }
}
