//! Faithful port of upstream's consolidation prompt builders.
//!
//! Source: `prompts/upstream/v1.1.0/engine/consolidation/prompts.py`
//! (`build_consolidation_system_prompt`, `build_mission_section`,
//! `build_consolidation_input`), MIT, Copyright 2025 Vectorize AI, Inc.
//!
//! Byte-for-byte output parity is pinned by `tests/upstream_prompt_parity.rs`.
//! Note: these builders produce upstream's prompts; the rustsight
//! consolidation *pipeline* does not yet consume the creates/updates/deletes
//! response machine they describe — see `prompts/upstream/DIVERGENCES.md`.

use crate::prompts::prompt_utils::{
    default_language_section, escape_for_prompt, output_language_directive,
};
use crate::prompts::upstream::python_format;
use crate::prompts::upstream::texts;

/// Upstream `build_consolidation_system_prompt`.
///
/// Bank-agnostic, cacheable system instruction for batch consolidation.
/// Holds only what is constant across banks: processing rules, input format,
/// decision guide, and output format. The bank's MISSION is deliberately NOT
/// here — baking it in would make the prefix bank-specific. The mission, the
/// per-batch INPUT, and any capacity constraint all ride in the user message
/// (see [`build_consolidation_input`]).
///
/// `llm_output_language` picks between two mutually exclusive language rules:
/// unset keeps each observation in the language of its own source facts (the
/// default), set forces every observation into that one configured language.
pub fn build_consolidation_system_prompt(llm_output_language: Option<&str>) -> String {
    let language_section = default_language_section(
        texts::CONSOLIDATION_DEFAULT_LANGUAGE_RULE,
        llm_output_language,
    );
    // No {facts_text}/{observations_text} placeholders here — the only braces
    // are the doubled {{ }} in the OUTPUT examples, which Python's .format()
    // unescapes (the dumped constant is already unescaped).
    format!(
        "You are a memory consolidation system. Synthesize new facts into \
         observations, merging with existing observations when appropriate.\n\n\
         {}\n\n{}{}\n\n{}\n\n{}\n\n{}{}",
        texts::CONSOLIDATION_MISSION_PRIORITY_NOTE,
        language_section,
        texts::CONSOLIDATION_PROCESSING_RULES,
        texts::CONSOLIDATION_INPUT_FORMAT_NOTE,
        texts::CONSOLIDATION_DECISION_GUIDE,
        texts::CONSOLIDATION_OUTPUT_SECTION,
        output_language_directive(llm_output_language),
    )
}

/// Upstream `build_mission_section`: the MISSION section, heading included,
/// as the user message writes it.
pub fn build_mission_section(observations_mission: Option<&str>) -> String {
    let mission = observations_mission.filter(|m| !m.is_empty());
    format!(
        "## MISSION\n\n{}",
        escape_for_prompt(mission.unwrap_or(texts::CONSOLIDATION_DEFAULT_MISSION))
    )
}

/// Upstream `build_consolidation_input`: per-batch user message — MISSION +
/// INPUT data + any capacity constraint.
///
/// The MISSION lives here (not in the cached system prefix) so the prefix
/// stays bank-agnostic. The capacity note also lives here since it varies as
/// observation slots fill.
pub fn build_consolidation_input(
    facts_text: &str,
    observations_text: &str,
    observations_mission: Option<&str>,
    observation_capacity_note: Option<&str>,
) -> String {
    let mission_section = format!("{}\n\n", build_mission_section(observations_mission));
    let capacity_section = match observation_capacity_note.filter(|n| !n.is_empty()) {
        Some(note) => format!("## CAPACITY CONSTRAINT\n\n{}\n\n", escape_for_prompt(note)),
        None => String::new(),
    };
    // Upstream concatenates the escaped sections into one template and runs
    // .format(facts_text=..., observations_text=...), which both substitutes
    // the placeholders and unescapes the doubled braces — python_format
    // reproduces that single pass.
    python_format(
        &format!(
            "{mission_section}{capacity_section}{}",
            texts::CONSOLIDATION_SPLIT_INPUT_SECTION
        ),
        &[
            ("facts_text", facts_text),
            ("observations_text", observations_text),
        ],
    )
}
