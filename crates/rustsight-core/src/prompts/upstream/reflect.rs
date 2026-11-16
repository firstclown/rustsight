//! Faithful port of upstream's reflect prompt builders.
//!
//! Source: `prompts/upstream/v1.1.0/engine/reflect/prompts.py`
//! (`build_system_prompt_for_tools`, `build_agent_user_prompt`,
//! `build_final_system_prompt`, `build_directives_section`,
//! `build_directives_reminder`, `bank_name_line`, `bank_disposition_line`)
//! plus `search/think_utils.py` (`describe_trait_level`,
//! `build_disposition_description`), MIT, Copyright 2025 Vectorize AI, Inc.
//!
//! Byte-for-byte output parity is pinned by `tests/upstream_prompt_parity.rs`.
//! The rustsight reflect loop exposes a smaller tool set than upstream's, so
//! the engine does not yet send this prompt as-is — see
//! `prompts/upstream/DIVERGENCES.md`.

use crate::prompts::prompt_utils::{
    default_language_section, escape_for_prompt, output_language_directive,
};
use crate::prompts::upstream::python_format;
use crate::prompts::upstream::texts;

/// Trait value used for a trait the bank does not set, matching the neutral
/// default the disposition model itself documents.
const NEUTRAL_TRAIT: i32 = 3;

const TRAITS: [&str; 3] = ["skepticism", "literalism", "empathy"];

/// A directive mental model: a hard rule injected into reflect prompts.
#[derive(Debug, Clone)]
pub struct Directive {
    pub name: String,
    pub content: String,
}

impl Directive {
    pub fn new(name: &str, content: &str) -> Self {
        Self {
            name: name.to_string(),
            content: content.to_string(),
        }
    }
}

/// Disposition traits on a bank profile. `None` marks a trait the bank did
/// not set (neutral); upstream models this as a missing dict key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DispositionTraits {
    pub skepticism: Option<i32>,
    pub literalism: Option<i32>,
    pub empathy: Option<i32>,
}

impl DispositionTraits {
    fn get(&self, trait_name: &str) -> Option<i32> {
        match trait_name {
            "skepticism" => self.skepticism,
            "literalism" => self.literalism,
            "empathy" => self.empathy,
            _ => None,
        }
    }

    fn value_or_neutral(&self, trait_name: &str) -> i32 {
        self.get(trait_name).unwrap_or(NEUTRAL_TRAIT)
    }
}

/// The bank profile as the reflect prompt builders see it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReflectBankProfile<'a> {
    /// Defaults to "Assistant" when None, mirroring upstream's dict lookup.
    pub name: Option<&'a str>,
    /// Empty/None renders the default role.
    pub mission: Option<&'a str>,
    pub disposition: Option<DispositionTraits>,
}

impl ReflectBankProfile<'_> {
    fn mission_str(&self) -> &str {
        self.mission.unwrap_or("")
    }
}

/// Options for [`build_system_prompt_for_tools`].
#[derive(Debug, Clone, Copy)]
pub struct ToolsPromptOptions<'a> {
    pub bank: ReflectBankProfile<'a>,
    pub context: Option<&'a str>,
    pub directives: &'a [Directive],
    /// Whether the bank has any mental models (skips the level if not).
    pub has_mental_models: bool,
    /// Whether search_observations is in the tool list.
    pub include_observations: bool,
    /// Search depth budget — "low", "mid", or "high"; anything else is a
    /// no-op.
    pub budget: Option<&'a str>,
    /// Whether done() takes a structured document instead of markdown.
    pub answer_as_document: bool,
    /// Configured output language; drops the default language rule (the
    /// directive itself goes on the user message,
    /// [`build_agent_user_prompt`]).
    pub llm_output_language: Option<&'a str>,
    /// Whether the loop uses answer_ready() before done().
    pub answer_ready: bool,
    /// The rendered "now" (e.g. "2026-08-09 14:32 UTC"); injected so tests
    /// are deterministic. Upstream reads the clock (minute precision).
    pub current_datetime: &'a str,
}

impl Default for ToolsPromptOptions<'_> {
    fn default() -> Self {
        Self {
            bank: ReflectBankProfile::default(),
            context: None,
            directives: &[],
            has_mental_models: false,
            include_observations: true,
            budget: None,
            answer_as_document: false,
            llm_output_language: None,
            answer_ready: false,
            current_datetime: "",
        }
    }
}

fn current_datetime_section(now: &str) -> String {
    format!("## Current Date and Time\nThe current date and time is {now}.")
}

/// Upstream `_extract_directive_rules`.
fn extract_directive_rules(directives: &[Directive]) -> Vec<String> {
    let mut rules = vec![];
    for directive in directives {
        let name = directive.name.as_str();
        let content = directive.content.as_str();
        if !content.is_empty() {
            if !name.is_empty() {
                rules.push(format!("**{name}**: {content}"));
            } else {
                rules.push(content.to_string());
            }
        }
    }
    rules
}

/// Upstream `build_directives_section`.
pub fn build_directives_section(directives: &[Directive]) -> String {
    if directives.is_empty() {
        return String::new();
    }
    let rules = extract_directive_rules(directives);
    if rules.is_empty() {
        return String::new();
    }

    let mut parts = vec![
        "## DIRECTIVES (MANDATORY)".to_string(),
        "These are hard rules you MUST follow in ALL responses:".to_string(),
        String::new(),
    ];
    for rule in &rules {
        parts.push(format!("- {rule}"));
    }
    parts.extend([
        String::new(),
        "NEVER violate these directives, even if other context suggests otherwise.".to_string(),
        "IMPORTANT: Do NOT explain or justify how you handled directives in your answer. Just follow them silently.".to_string(),
        String::new(),
    ]);
    parts.join("\n")
}

/// Upstream `build_directives_reminder`.
pub fn build_directives_reminder(directives: &[Directive]) -> String {
    if directives.is_empty() {
        return String::new();
    }
    let rules = extract_directive_rules(directives);
    if rules.is_empty() {
        return String::new();
    }

    let mut parts = vec![
        String::new(),
        "## REMINDER: MANDATORY DIRECTIVES".to_string(),
        "Before responding, ensure your answer complies with ALL of these directives:".to_string(),
        String::new(),
    ];
    for (i, rule) in rules.iter().enumerate() {
        parts.push(format!("{}. {rule}", i + 1));
    }
    parts.push(String::new());
    parts.push("Your response will be REJECTED if it violates any directive above.".to_string());
    parts.push(
        "Do NOT include any commentary about how you handled directives - just follow them."
            .to_string(),
    );
    parts.join("\n")
}

/// Upstream `bank_name_line`.
pub fn bank_name_line(bank_profile: &ReflectBankProfile) -> String {
    format!(
        "## Memory Bank: {}",
        bank_profile.name.unwrap_or("Assistant")
    )
}

/// Upstream `describe_trait_level` (think_utils).
fn describe_trait_level(value: i32) -> &'static str {
    match value {
        1 => "very low",
        2 => "low",
        4 => "high",
        5 => "very high",
        _ => "moderate",
    }
}

/// Upstream `build_disposition_description` (think_utils). Unset traits fall
/// back to neutral.
pub fn build_disposition_description(d: &DispositionTraits) -> String {
    let skepticism_desc = |v: i32| {
        match v {
        1 => "You are very trusting and tend to take information at face value.",
        2 => "You tend to trust information but may question obvious inconsistencies.",
        3 => "You have a balanced approach to information, neither too trusting nor too skeptical.",
        4 => "You are somewhat skeptical and often question the reliability of information.",
        5 => "You are highly skeptical and critically examine all information for accuracy and hidden motives.",
        _ => "You have a balanced approach to information, neither too trusting nor too skeptical.",
    }
    };
    let literalism_desc = |v: i32| {
        match v {
        1 => "You interpret information very flexibly, reading between the lines and inferring intent.",
        2 => "You tend to consider context and implied meaning alongside literal statements.",
        3 => "You balance literal interpretation with contextual understanding.",
        4 => "You prefer to interpret information more literally and precisely.",
        5 => "You interpret information very literally and focus on exact wording and commitments.",
        _ => "You balance literal interpretation with contextual understanding.",
    }
    };
    let empathy_desc = |v: i32| {
        match v {
        1 => "You focus primarily on facts and data, setting aside emotional context.",
        2 => "You consider facts first but acknowledge emotional factors exist.",
        3 => "You balance factual analysis with emotional understanding.",
        4 => "You give significant weight to emotional context and human factors.",
        5 => "You strongly consider the emotional state and circumstances of others when forming memories.",
        _ => "You balance factual analysis with emotional understanding.",
    }
    };
    format!(
        "Your disposition traits:\n- Skepticism ({}): {}\n- Literalism ({}): {}\n- Empathy ({}): {}",
        describe_trait_level(d.value_or_neutral("skepticism")),
        skepticism_desc(d.value_or_neutral("skepticism")),
        describe_trait_level(d.value_or_neutral("literalism")),
        literalism_desc(d.value_or_neutral("literalism")),
        describe_trait_level(d.value_or_neutral("empathy")),
        empathy_desc(d.value_or_neutral("empathy")),
    )
}

/// Upstream `bank_disposition_line`: the bank's disposition traits as the
/// prompt writes them, or "" if it has none.
///
/// An all-neutral disposition is what a bank that never touched the traits
/// reports, so it keeps the exact prompt it had before this block existed.
/// Otherwise the numbers are spelled out — the numbers alone are not an
/// instruction for weaker models.
pub fn bank_disposition_line(bank_profile: &ReflectBankProfile) -> String {
    let Some(disposition) = bank_profile.disposition else {
        return String::new();
    };
    let traits: Vec<String> = TRAITS
        .iter()
        .filter_map(|t| disposition.get(t).map(|v| format!("{t}={v}")))
        .collect();
    if traits.is_empty() {
        return String::new();
    }

    if TRAITS
        .iter()
        .all(|t| disposition.value_or_neutral(t) == NEUTRAL_TRAIT)
    {
        return format!("Disposition: {}", traits.join(", "));
    }
    format!(
        "Disposition: {}\n{}",
        traits.join(", "),
        build_disposition_description(&disposition)
    )
}

/// Python `str.title()` for the level names in the Search Plan ("RAW FACTS"
/// → "Raw Facts"). Word boundaries on non-alphanumerics, like Python.
fn python_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_is_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_is_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_is_alpha = true;
        } else {
            out.push(c);
            prev_is_alpha = false;
        }
    }
    out
}

/// Upstream `build_system_prompt_for_tools`: the system prompt for the
/// tool-calling reflect agent.
///
/// The retrieval-strategy and workflow sections are built to match the tools
/// actually exposed to the LLM — mentioning a tool the agent has disabled
/// causes weaker LLMs to either hallucinate the call or give up (upstream
/// #1724).
pub fn build_system_prompt_for_tools(o: &ToolsPromptOptions) -> String {
    let mission = o.bank.mission_str();

    let mut parts: Vec<String> = vec![];

    // Anti-hallucination rule at the very top
    parts.extend([
        "CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.".to_string(),
        String::new(),
    ]);

    // Inject directives after anti-hallucination rule
    if !o.directives.is_empty() {
        parts.push(build_directives_section(o.directives));
    }

    parts.extend([
        if mission.is_empty() {
            texts::REFLECT_DEFAULT_ROLE.to_string()
        } else {
            mission.trim().to_string()
        },
        String::new(),
        "Answer the user's question by reasoning over retrieved memories.".to_string(),
        String::new(),
    ]);

    // Mutually exclusive with the configured output language — the rule is
    // dropped outright rather than left to out-rank the directive (#3776).
    let tools_language_section =
        default_language_section(texts::REFLECT_TOOLS_LANGUAGE_RULE, o.llm_output_language);
    if !tools_language_section.is_empty() {
        parts.push(tools_language_section.trim_end_matches('\n').to_string());
        parts.push(String::new());
    }

    parts.extend(
        [
            "## CRITICAL RULES",
            "- ONLY use information from tool results - no external knowledge or guessing",
            "- You SHOULD synthesize, infer, and reason from the retrieved memories",
            "- You MUST search before saying you don't have information",
            "",
            "## How to Reason",
            "- If memories mention someone did an activity, you can infer they likely enjoyed it",
            "- Synthesize a coherent narrative from related memories",
            "- Be a thoughtful interpreter, not just a literal repeater",
            "- When the exact answer isn't stated, use what IS stated to give a best-effort answer AND surface any uncertainty — never invent confidence the data doesn't support.",
            "",
            texts::REFLECT_GROUNDING_BOUNDARY,
            "",
            "## Temporal Reasoning",
            "Every memory and observation carries temporal fields in the JSON tool result:",
            "- `mentioned_at` — when the user retained the fact (always set).",
            "- `occurred_start` / `occurred_end` — when the underlying event happened (optional, set for dated events).",
            "",
            "When facts about the SAME facet conflict — counts, statuses, ownership, location, presence, etc. — the fact with the LATEST `mentioned_at` is authoritative. Later statements SUPERSEDE earlier ones. Do NOT average, sum, or favor an explicitly-dated fact over a more recent one. The one exception is a DECISION, below.",
            "",
            "A DECISION is different from a state. When the record holds a decision with its rationale (what was decided and why) and a later record changes the same thing WITHOUT stating a reason that overturns that decision (a refactor, a 'simplification', a cleanup, a performance change), the later record does NOT supersede the decision: report the decision as the standing one, and report the later change as a departure from it. Only a later record that itself decides, with its own rationale, replaces an earlier decision. Example: 'Invoices are due net-30; finance chose it for cash flow' (March) and 'cleanup: simplify billing config, terms = net-45' (May) → the rule is net-30; the May cleanup is a change that departed from it, NOT the rule.",
            "",
            "Example: three count facts come back from recall:",
            "  - 'Team has 2 engineers' (mentioned_at=T1)",
            "  - 'Team now has 1 engineer' (mentioned_at=T2, occurred_start=2026-05-25)",
            "  - 'Team has 5 engineers' (mentioned_at=T3)",
            "with T1 < T2 < T3. The current size is 5, not 1. Then apply later events (e.g. someone leaving after T3) on top of that.",
            "",
            "For reconstructing a TIMELINE of events, order by `occurred_start` / `occurred_end` (when things happened), not `mentioned_at` (when they were retained).",
            "",
            "## Conflicts and Ambiguity",
            "Not every retrieval converges on a single answer. Distinguish two cases:",
            "",
            "- RESOLVABLE conflict — the temporal rule above (latest `mentioned_at` wins) cleanly picks a winner. Apply it and move on.",
            "- UNRESOLVABLE ambiguity — the data is internally inconsistent in a way the temporal rule does NOT settle. Examples: a recent aggregate (count, total) is incompatible with the individual entities you can enumerate; two equally-recent facts disagree and no later fact resolves them; events are described but their relative order is unclear; the user's own statements contradict each other and nothing later reconciles them.",
            "",
            "When the data is genuinely ambiguous: SAY SO in your answer. Name the conflicting facts. Explain why they can't be reconciled. Give a range or a best-effort interpretation with explicit uncertainty (e.g. 'between X and Y, depending on [unresolved condition]'; or 'the most recent statement says A, but B was stated earlier and the gap isn't accounted for in any later fact').",
            "",
            "An honest 'the data is inconsistent about X' beats a confident wrong answer. Do NOT pick a value arbitrarily, average conflicting values, or smooth over gaps in confident prose. Acknowledging ambiguity is a successful answer, not a failure mode.",
            "",
            "## Showing Your Reasoning",
            "For any answer that resolves a conflict between facts, applies events on top of a count or status, or settles an ambiguity — show your work in the answer text so a reader can audit it.",
            "",
            "Walk through these steps explicitly:",
            "1. **List the relevant facts in `mentioned_at` order (oldest → newest)**, each with the value it asserts. Use a short bulleted list.",
            "2. **Identify the authoritative fact** under the temporal rule (latest `mentioned_at` for the contested facet). Write its date down.",
            "3. **List candidate events to apply on top** — anything that changes the count, status, or state being asked about. Write each event's date down next to it.",
            "4. **Sanity-check each candidate event against the authoritative date** — for EVERY event from step 3, write a one-line check in the form `<event> (<event_date>) vs authoritative (<authoritative_date>) → BEFORE/AFTER → KEEP/DROP`. If the event is BEFORE or EQUAL to the authoritative date, DROP it: it is already reflected in the authoritative fact, and applying it again is double-counting. This is the single most common mistake — do not skip this step even if you feel confident.",
            "5. **Show the arithmetic or derivation explicitly** using only the KEEP events from step 4 — e.g. 'authoritative count = 5 (at 2025-02-12); kept events: Shadow died (2025-03-12, AFTER); 5 − 1 = 4'.",
            "6. If step 2 or 3 cannot be done cleanly (no clear winner, overlapping timestamps, unclear event order), STOP and surface this as an UNRESOLVABLE ambiguity per the section above — do not fabricate a derivation.",
            "",
            "For simple factual lookups that don't involve conflict or arithmetic, you can answer directly without this scaffolding.",
            "",
            "## HIERARCHICAL RETRIEVAL STRATEGY",
            "",
        ]
        .into_iter()
        .map(String::from),
    );

    // Assemble the retrieval-level blocks for whatever tools are exposed.
    // MM and Observations bodies are unconditional; recall's fallback wording
    // adapts to which upstream tools precede it.
    let mut levels: Vec<(&str, Vec<String>)> = vec![];
    if o.has_mental_models {
        levels.push((
            "MENTAL MODELS (search_mental_models)",
            [
                "- User-curated summaries about specific topics",
                "- HIGHEST quality - manually created and maintained",
                "- Search returns the best match in full and a SNIPPET of the others; call read_mental_models on any id whose snippet looks like it answers the question, and read it before answering from it",
                "- A FRESH mental model may fully answer the question — but only if it actually STATES the answer. One that shares the question's topic without stating the answer (e.g. it describes a process, or names the thing without its status) has not answered it: go to the next level",
                "- Check `is_stale` field - if stale, also verify with lower levels",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        ));
    }
    if o.include_observations {
        levels.push((
            "OBSERVATIONS (search_observations)",
            [
                "- Auto-consolidated knowledge from memories",
                "- Check `is_stale` field - if stale, ALSO use recall() to verify",
                "- Good for understanding patterns and summaries",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        ));
    }
    let mut recall_body = vec!["- Individual memories (world facts and experiences)".to_string()];
    if o.has_mental_models && o.include_observations {
        recall_body.extend(
            [
                "- Use when: no mental models/observations exist, they're stale, or you need specific details",
                "- MANDATORY: If search_mental_models and search_observations return 0 results, OR return results that do not STATE the answer, you MUST call recall() before giving up. Never report that the bank holds nothing about something until recall() has run with the question's key terms (an issue key, name or identifier) verbatim",
                "- This is the source of truth that other levels are built from",
                "",
                "**Tool result ordering:** `recall()` and `search_observations()` return their `memories` / `observations` arrays sorted by SEMANTIC RELEVANCE to the query, NOT by time. The POSITION of an entry tells you nothing about when it was retained. For any temporal reasoning — recency, supersession, applying events on top of a state — IGNORE the position and read the per-entry `mentioned_at` field (and `occurred_start` / `occurred_end` for events).",
                "",
            ]
            .into_iter()
            .map(String::from),
        );
    } else if o.has_mental_models {
        recall_body.extend(
            [
                "- Use when: no mental model exists, it's stale, or you need specific details",
                "- MANDATORY: If search_mental_models returns 0 results, OR returns a model that does not STATE the answer, you MUST call recall() before giving up. Never report that the bank holds nothing about something until recall() has run with the question's key terms verbatim",
                "- This is the source of truth that mental models are built from",
            ]
            .into_iter()
            .map(String::from),
        );
    } else if o.include_observations {
        recall_body.extend(
            [
                "- Use when: no observations exist, they're stale, or you need specific details",
                "- MANDATORY: If search_observations returns 0 results or count=0, OR returns observations that do not STATE the answer, you MUST call recall() before giving up. Never report that the bank holds nothing about something until recall() has run with the question's key terms verbatim",
                "- This is the source of truth that observations are built from",
                "",
                "**Tool result ordering:** `recall()` and `search_observations()` return their `memories` / `observations` arrays sorted by SEMANTIC RELEVANCE to the query, NOT by time. The POSITION of an entry tells you nothing about when it was retained. For any temporal reasoning — recency, supersession, applying events on top of a state — IGNORE the position and read the per-entry `mentioned_at` field (and `occurred_start` / `occurred_end` for events).",
                "",
            ]
            .into_iter()
            .map(String::from),
        );
    } else {
        recall_body.extend(
            [
                "- MANDATORY: Call recall() to gather facts before giving up",
                "- This is the source of truth.",
            ]
            .into_iter()
            .map(String::from),
        );
    }
    levels.push(("RAW FACTS (recall) - Ground Truth", recall_body));

    // Position-dependent suffix for upstream tools; recall already carries
    // its fixed "- Ground Truth" suffix in the header text.
    let mut suffixes = vec![String::new(); levels.len()];
    if levels.len() >= 2 {
        suffixes[0] = " - Try First".to_string();
    }
    if levels.len() == 3 {
        suffixes[1] = " - Second Priority".to_string();
    }

    if levels.len() == 1 {
        parts.push("You have access to ONE level of knowledge:".to_string());
    } else {
        let word = if levels.len() == 2 { "TWO" } else { "THREE" };
        parts.push(format!(
            "You have access to {word} levels of knowledge. Use them in this order:"
        ));
    }
    parts.push(String::new());
    for (idx, ((header, body), suffix)) in levels.iter().zip(&suffixes).enumerate() {
        parts.push(format!("### {}. {header}{suffix}", idx + 1));
        parts.extend(body.iter().cloned());
        parts.push(String::new());
    }

    // Stating the ladder here, rather than only forcing it turn by turn.
    if levels.len() > 1 {
        let names: Vec<String> = levels
            .iter()
            .map(|(header, _)| {
                python_title(
                    header
                        .split(" (")
                        .next()
                        .unwrap_or(header)
                        .split(" - ")
                        .next()
                        .unwrap_or(header),
                )
            })
            .collect();
        parts.extend([
            "## Search Plan".to_string(),
            format!(
                "Work down the levels in order ({}) before you answer:",
                names.join(" → ")
            ),
            "- Search a level before deciding it has nothing; a level you did not search is not evidence of absence.".to_string(),
            // Named from the levels actually offered.
            format!(
                "- Stop descending as soon as what you have answers the question — fresh {} often do.",
                names[0]
            ),
            "- Go deeper when the level above is stale, thin, or silent on what was asked.".to_string(),
            if o.answer_ready {
                "- Call `answer_ready` once you have the evidence; you are then asked for the answer. Do not write the answer before that.".to_string()
            } else {
                "- Call `done` with the answer once you have the evidence. Do not write the answer as plain text.".to_string()
            },
            String::new(),
        ]);
    }

    parts.extend(
        [
            "## Query Strategy",
            "recall() uses semantic search. NEVER just echo the user's question - decompose it into targeted searches:",
            "",
            "BAD: User asks 'recurring lesson themes between students' → recall('recurring lesson themes between students')",
            "GOOD: Break it down into component searches:",
            "  1. recall('lessons') - find all lesson-related memories",
            "  2. recall('teaching sessions') - alternative phrasing",
            "  3. recall('student progress') - find student-related memories",
            "",
            "Think: What ENTITIES and CONCEPTS does this question involve? Search for each separately.",
            "",
        ]
        .into_iter()
        .map(String::from),
    );

    // Budget guidance bounds how LONG the answer is and how many searches get
    // spent on it — never whether the lower retrieval layers are consulted.
    if let Some(budget) = o.budget {
        match budget.to_lowercase().as_str() {
            "low" => parts.extend(
                [
                    "## RESEARCH DEPTH: SHALLOW (Quick Response)",
                    "- Keep the ANSWER short: a quick overview, not exhaustive detail. Depth is what you cut, not coverage",
                    "- Spend few searches, but make them count: vary the query instead of repeating one that already ran",
                    "- A mental model or observation that ANSWERS the question is enough to stop; one that is merely on the same topic is not",
                    "- If what you found does not cover the question, go on to the next level rather than answering from it",
                    "",
                ]
                .into_iter()
                .map(String::from),
            ),
            "mid" => parts.extend(
                [
                    "## RESEARCH DEPTH: MODERATE (Balanced)",
                    "- Balance thoroughness with efficiency",
                    "- Check multiple sources when the question warrants it",
                    "- Verify stale data if it's central to the answer",
                    "- A result that is merely on the same topic does not answer the question: when it does not cover it, go on to the next level",
                    "- Don't over-explore, but ensure reasonable coverage",
                    "",
                ]
                .into_iter()
                .map(String::from),
            ),
            "high" => parts.extend(
                [
                    "## RESEARCH DEPTH: DEEP (Thorough Exploration)",
                    "- Explore comprehensively before answering",
                    "- Search across all available knowledge levels",
                    "- Use multiple query variations to ensure coverage",
                    "- Verify information across different retrieval levels",
                    "- Use expand() to get full context on important memories",
                    "- Take time to synthesize a complete, well-researched answer",
                    "",
                ]
                .into_iter()
                .map(String::from),
            ),
            _ => {}
        }
    }

    parts.push("## Workflow".to_string());

    let mut steps: Vec<String> = vec![];
    if o.has_mental_models {
        steps.push(
            "First, try search_mental_models() - check if a curated summary exists".to_string(),
        );
    }
    if o.include_observations {
        if o.has_mental_models {
            steps.push(
                "If there is no mental model, it's stale, OR it does not state the answer, try search_observations() for consolidated knowledge".to_string(),
            );
        } else {
            steps.push(
                "First, try search_observations() - check for consolidated knowledge".to_string(),
            );
        }
    }
    // Recall step phrasing varies with whichever upstream tool(s) precede it.
    if o.include_observations {
        steps.push(
            if o.has_mental_models {
                "If the levels above are stale, do not state the answer, OR you need specific details, use recall() for raw facts. Reporting that nothing is known requires recall() first"
            } else {
                "If search_observations returns 0 results, is stale, OR does not state the answer, you MUST call recall() for raw facts"
            }
            .to_string(),
        );
    } else if o.has_mental_models {
        steps.push(
            "If there is no mental model, it's stale, OR it does not state the answer, use recall() for raw facts. Reporting that nothing is known requires recall() first".to_string(),
        );
    } else {
        steps.push("Call recall() to gather raw facts".to_string());
    }
    steps.push("Use expand() if you need more context on specific memories".to_string());
    steps.push(
        if o.answer_ready {
            "When ready, call answer_ready(); you are then asked for the answer through done()"
        } else {
            "When ready, call done() with your answer and supporting IDs"
        }
        .to_string(),
    );
    steps
        .iter()
        .enumerate()
        .for_each(|(idx, step)| parts.push(format!("{}. {step}", idx + 1)));

    let non_conversational_rule = "- CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer further assistance, or suggest next steps. Your answer must be complete and self-contained. The user cannot reply.";
    if o.answer_as_document {
        parts.extend(
            [
                String::new(),
                "## Output Format: Structured Document".to_string(),
                "Call done() with a 'document' field. Do NOT write a markdown document — state its structure and the markdown is generated from it.".to_string(),
                "- 'document' holds a 'sections' array — one entry per section, in order. Shape: {\"sections\": [{\"heading\": \"Overview\", \"level\": 2, \"blocks\": [\"First paragraph.\", \"- a list item\\n- another\"]}]}".to_string(),
                "- Even a one-section document uses the 'sections' array; never emit a bare section".to_string(),
                "- Each section carries its heading text (no '#') and a level; the heading is NOT a block".to_string(),
                "- 'blocks' holds the section's content, ONE block per paragraph, list, table or code fence".to_string(),
                "- Never put two paragraphs in one block, and never put a heading inside a block".to_string(),
                "- Inside a block, write list items and table rows on their own lines as usual".to_string(),
                "- The structure is how the answer is laid out, NOT a reason to say less: give each block the same specifics, numbers, names and examples you would write in prose".to_string(),
                "- Prefer a section with several blocks over one block that summarises them away".to_string(),
                format!("- {}", texts::REFLECT_NO_IDS_IN_TEXT_RULE),
                non_conversational_rule.to_string(),
            ],
        );
    } else {
        parts.extend(
            [
                String::new(),
                "## Output Format: Well-Formatted Markdown Answer".to_string(),
                "Call done() with a well-formatted markdown 'answer' field.".to_string(),
                "- USE markdown formatting for structure (headers, lists, bold, italic, code blocks, tables, etc.)".to_string(),
                "- CRITICAL: Add blank lines before and after block elements (tables, code blocks, lists)".to_string(),
                "- Format for clarity and readability with proper spacing and hierarchy".to_string(),
                "- NEVER include memory IDs, UUIDs, or 'Memory references' in the answer text".to_string(),
                "- Put IDs ONLY in the memory_ids/mental_model_ids/observation_ids arrays, not in the answer".to_string(),
                non_conversational_rule.to_string(),
            ],
        );
    }

    // Volatile "now" reference goes here — after all the static instructions
    // and right before the bank-specific/custom data.
    parts.push(String::new());
    parts.push(current_datetime_section(o.current_datetime));

    parts.push(String::new());
    parts.push(bank_name_line(&o.bank));

    if !mission.is_empty() {
        parts.push(format!("Mission: {mission}"));
    }

    let disposition_line = bank_disposition_line(&o.bank);
    if !disposition_line.is_empty() {
        parts.push(disposition_line);
    }

    if let Some(context) = o.context.filter(|c| !c.is_empty()) {
        parts.push(format!("\n## Additional Context\n{context}"));
    }

    // Add directive reminder at the END for recency effect
    if !o.directives.is_empty() {
        parts.push(build_directives_reminder(o.directives));
    }

    parts.join("\n")
}

/// Upstream `build_agent_user_prompt`: the reasoning loop's opening user
/// message — the question, the mission, closed by the language directive.
///
/// The mission is restated here (not only in the system prompt) because the
/// question arrives after the system prompt and out-ranks it.
pub fn build_agent_user_prompt(
    query: &str,
    llm_output_language: Option<&str>,
    mission: Option<&str>,
) -> String {
    let mission_section = match mission.filter(|m| !m.trim().is_empty()) {
        Some(mission) => format!(
            "\n\n## Bank mission (it governs what you search and how you answer)\n{}",
            mission.trim()
        ),
        None => String::new(),
    };
    format!(
        "{query}{mission_section}{}",
        output_language_directive(llm_output_language)
    )
}

/// Upstream `build_final_system_prompt`: the final synthesis system prompt,
/// using mission as role when set.
///
/// Directives are re-injected here (they live in the agent/reasoning prompt,
/// but the final answer is a separate call) so output-constraining rules are
/// honoured by the model that actually writes the answer.
///
/// Note (faithful to upstream): the mission is brace-escaped for Python's
/// `str.format` and the escaped form survives into this prompt — a mission
/// containing `{entity}` renders as `{{entity}}`. Upstream's own test only
/// passes because `{{entity}}` contains `{entity}` as a substring.
pub fn build_final_system_prompt(
    mission: Option<&str>,
    llm_output_language: Option<&str>,
    directives: &[Directive],
    current_datetime: &str,
) -> String {
    let role_section = match mission.filter(|m| !m.is_empty()) {
        Some(mission) => escape_for_prompt(mission.trim()),
        None => texts::REFLECT_DEFAULT_FINAL_ROLE.to_string(),
    };

    let mut parts: Vec<String> = vec![if directives.is_empty() {
        String::new()
    } else {
        build_directives_section(directives)
    }];
    parts.push(python_format(
        texts::REFLECT_FINAL_SYSTEM_BASE,
        &[
            ("role_section", &role_section),
            ("grounding_boundary", texts::REFLECT_GROUNDING_BOUNDARY),
            ("no_ids_rule", texts::REFLECT_NO_IDS_IN_TEXT_RULE),
        ],
    ));
    parts.push(default_language_section(
        texts::REFLECT_FINAL_LANGUAGE_RULE,
        llm_output_language,
    ));
    parts.push(if directives.is_empty() {
        String::new()
    } else {
        build_directives_reminder(directives)
    });
    // Volatile "now" reference last, so the static/per-bank instructions
    // above remain a cacheable prefix.
    parts.push(current_datetime_section(current_datetime));

    parts
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// Final-synthesis machine (upstream reflect/prompts.py, hand-ported).
//
// These builders ARE vendored upstream (build_final_prompt,
// build_done_request_prompt, build_chunk_claims_prompt, build_reduce_prompt,
// split_context_history, _render_history_block, _bank_identity_section,
// _length_directive, CLAIMS_SYSTEM_PROMPT) but are not yet covered by the
// prompt-goldens generator — the texts below are copied verbatim from
// `prompts/upstream/v1.1.0/engine/reflect/prompts.py` and unit-tested for
// shape, not golden-pinned. Extending `scripts/prompt-goldens` to emit
// fixtures for them is the follow-up that makes them parity-tested.
// ---------------------------------------------------------------------------

/// Upstream `_FINAL_PROMPT_CONTEXT_FRACTION`: the share of the context
/// budget the retrieved-data section may consume.
const FINAL_PROMPT_CONTEXT_FRACTION: f64 = 0.8;
/// Upstream `_MIN_SPLIT_CHUNK_TOKENS`: floor for per-chunk packing budgets.
const MIN_SPLIT_CHUNK_TOKENS: usize = 1024;
/// Upstream `_MAX_SHARED_SIBLING_FRACTION`: a non-splittable sibling key of
/// an over-budget block may take at most this share of a chunk budget.
const MAX_SHARED_SIBLING_FRACTION: f64 = 0.5;
/// Upstream `_SPLITTABLE_RESULT_KEYS`: output keys an over-budget block is
/// split on, in priority order.
const SPLITTABLE_RESULT_KEYS: [&str; 3] = ["observations", "memories", "results"];

/// Upstream `CLAIMS_SYSTEM_PROMPT`: the intermediate (map) system prompt of
/// split synthesis — compress one chunk into dated, cited claims.
pub const CLAIMS_SYSTEM_PROMPT: &str = r#"You extract evidence from retrieved memory data. You MUST ONLY use information from the provided data. NEVER make up names, people, events, or entities.

Output a markdown bulleted list of factual claims relevant to the question. For EVERY claim:
- state the fact in one sentence, in the same language as the question;
- append its provenance in parentheses, exactly: (mentioned_at: <ISO date or unknown>; occurred: <ISO date/range or unknown>; memory_ids: <comma-separated ids>)

Rules:
- Be exhaustive over RELEVANT evidence; skip clearly irrelevant entries.
- Do NOT synthesize, conclude, resolve conflicts, or answer the question — report conflicting claims as separate bullets with their dates; a later pass reconciles them.
- Copy memory ids exactly as they appear in the data.
- If nothing in the data is relevant, output exactly: (no relevant evidence)"#;

/// One tool result as the synthesis prompts see it: the tool's name and its
/// serialized output (a JSON object — e.g. `{"memories": [...]}` from
/// recall, `{"observations": [...]}` from search_observations).
#[derive(Debug, Clone)]
pub struct ContextHistoryEntry {
    pub tool: String,
    pub output: serde_json::Value,
}

impl ContextHistoryEntry {
    pub fn new(tool: &str, output: serde_json::Value) -> Self {
        Self {
            tool: tool.to_string(),
            output,
        }
    }
}

/// Upstream `_render_history_block`: one entry as a fenced JSON block.
pub fn render_history_block(entry: &ContextHistoryEntry) -> String {
    let output_str =
        serde_json::to_string(&entry.output).unwrap_or_else(|_| entry.output.to_string());
    format!("\n### From {}:\n```json\n{}\n```", entry.tool, output_str)
}

/// Upstream `_length_directive`: soft visible-length directive, or `None`.
fn length_directive(max_tokens: Option<usize>) -> Option<String> {
    let max_tokens = max_tokens?;
    Some(format!(
        "\n## Length\nAim for a complete, self-contained answer of approximately {max_tokens} \
         tokens. Finishing cleanly matters more than length: end on a complete sentence and \
         NEVER stop mid-word, mid-list, or mid-code-fence. If you near the budget, wrap up \
         gracefully rather than cutting off."
    ))
}

/// Upstream `_cut_entry_to_budget`: token-bound one indivisible over-budget
/// entry by cutting its serialized text (proportional shrink, wrapped back
/// into a `{"truncated": true, "content": ...}` output).
fn cut_entry_to_budget(
    entry: &ContextHistoryEntry,
    token_budget: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> ContextHistoryEntry {
    let mut output_str = serde_json::to_string(&entry.output).unwrap_or_default();
    let mut cut = ContextHistoryEntry {
        tool: entry.tool.clone(),
        output: serde_json::json!({"truncated": true, "content": output_str.clone()}),
    };
    let mut tokens = count_tokens(&render_history_block(&cut));
    while !output_str.is_empty() && tokens > token_budget {
        // Proportional shrink with a safety margin; always makes progress.
        let keep = (output_str.len().saturating_sub(1)).min(
            ((output_str.len() as f64) * (token_budget as f64) / (tokens as f64) * 0.95) as usize,
        );
        let keep = keep.max(1);
        output_str.truncate(keep);
        cut.output = serde_json::json!({"truncated": true, "content": output_str.clone()});
        tokens = count_tokens(&render_history_block(&cut));
    }
    cut
}

/// Upstream `split_context_history`: partition tool-result history into
/// chunks that each fit the per-chunk budget (`max_context_tokens * 0.8`,
/// floored at `MIN_SPLIT_CHUNK_TOKENS`).
///
/// Greedy chronological packing; an over-budget block splits on its
/// `observations`/`memories`/`results` list so evidence is split across
/// chunks rather than dropped; only an indivisible over-budget entry gets
/// token-cut. Every original result entry appears in exactly one chunk.
pub fn split_context_history(
    context_history: &[ContextHistoryEntry],
    max_context_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Vec<Vec<ContextHistoryEntry>> {
    let budget = MIN_SPLIT_CHUNK_TOKENS
        .max((max_context_tokens as f64 * FINAL_PROMPT_CONTEXT_FRACTION) as usize);
    let mut packer = HistoryPacker::new(budget, count_tokens);

    for entry in context_history {
        let tokens = packer.count(entry);
        if tokens <= budget {
            packer.append_block(entry.clone(), tokens);
            continue;
        }

        // Over-budget block: split it on result-entry boundaries.
        let output = &entry.output;
        let split_key = SPLITTABLE_RESULT_KEYS.iter().find_map(|k| {
            output
                .get(*k)
                .and_then(|v| v.as_array())
                .filter(|list| !list.is_empty())
                .map(|_| *k)
        });
        let Some(split_key) = split_key else {
            let cut = cut_entry_to_budget(entry, budget, count_tokens);
            let cut_tokens = packer.count(&cut);
            packer.append_block(cut, cut_tokens);
            continue;
        };
        let items = output
            .get(split_key)
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        // A big non-splittable sibling key is packed on its own instead of
        // being copied into every piece (upstream #4495).
        let allowance = (budget as f64 * MAX_SHARED_SIBLING_FRACTION) as usize;
        let empty_map = serde_json::Map::new();
        let obj = output.as_object().unwrap_or(&empty_map);
        let big: Vec<String> = obj
            .keys()
            .filter(|k| {
                k.as_str() != split_key && {
                    let sibling = serde_json::json!({ (**k).clone(): obj[*k].clone() });
                    count_tokens(&render_history_block(&ContextHistoryEntry {
                        tool: entry.tool.clone(),
                        output: sibling,
                    })) > allowance
                }
            })
            .cloned()
            .collect();
        let shared: serde_json::Map<String, serde_json::Value> = obj
            .iter()
            .filter(|(k, _)| k.as_str() != split_key && !big.iter().any(|b| b == *k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        packer.pack_items(entry, &shared, split_key, &items);
        for k in big {
            match &obj[&k] {
                serde_json::Value::Object(map) => {
                    let pairs: Vec<serde_json::Value> = map
                        .iter()
                        .map(|(ik, iv)| serde_json::json!([ik, iv]))
                        .collect();
                    packer.pack_items(entry, &shared, &k, &pairs);
                }
                serde_json::Value::Array(list) => {
                    packer.pack_items(entry, &shared, &k, list);
                }
                other => {
                    let mut o = shared.clone();
                    o.insert(k.clone(), other.clone());
                    let candidate = ContextHistoryEntry {
                        tool: entry.tool.clone(),
                        output: serde_json::Value::Object(o),
                    };
                    let cut = cut_entry_to_budget(&candidate, budget, count_tokens);
                    let cut_tokens = packer.count(&cut);
                    packer.append_block(cut, cut_tokens);
                }
            }
        }
    }
    packer.close_current();
    packer.chunks
}

/// Chunk-packing state for [`split_context_history`]: the greedy
/// chronological packer (current chunk, its token load, the finished
/// chunks), mirroring upstream's closures over local state.
struct HistoryPacker<'a> {
    budget: usize,
    count_tokens: &'a dyn Fn(&str) -> usize,
    chunks: Vec<Vec<ContextHistoryEntry>>,
    current: Vec<ContextHistoryEntry>,
    current_tokens: usize,
}

impl<'a> HistoryPacker<'a> {
    fn new(budget: usize, count_tokens: &'a dyn Fn(&str) -> usize) -> Self {
        Self {
            budget,
            count_tokens,
            chunks: vec![],
            current: vec![],
            current_tokens: 0,
        }
    }

    fn close_current(&mut self) {
        if !self.current.is_empty() {
            self.chunks.push(std::mem::take(&mut self.current));
            self.current_tokens = 0;
        }
    }

    fn append_block(&mut self, block: ContextHistoryEntry, tokens: usize) {
        if !self.current.is_empty() && self.current_tokens + tokens > self.budget {
            self.close_current();
        }
        self.current.push(block);
        self.current_tokens += tokens;
    }

    fn count(&self, entry: &ContextHistoryEntry) -> usize {
        (self.count_tokens)(&render_history_block(entry))
    }

    /// Upstream `_pack_items`: pack a list under `output[key]` into
    /// in-budget blocks, each piece carrying the entry's other keys.
    fn pack_items(
        &mut self,
        entry: &ContextHistoryEntry,
        shared: &serde_json::Map<String, serde_json::Value>,
        key: &str,
        items: &[serde_json::Value],
    ) {
        let with_items = |piece: &[serde_json::Value]| -> ContextHistoryEntry {
            let mut output = serde_json::Map::clone(shared);
            output.insert(key.to_string(), serde_json::Value::Array(piece.to_vec()));
            ContextHistoryEntry {
                tool: entry.tool.clone(),
                output: serde_json::Value::Object(output),
            }
        };
        let mut piece: Vec<serde_json::Value> = vec![];
        for item in items {
            let mut candidate = with_items(&{
                let mut next = piece.clone();
                next.push(item.clone());
                next
            });
            let candidate_tokens = self.count(&candidate);
            if !piece.is_empty() && candidate_tokens > self.budget {
                let partial = with_items(&piece);
                let partial_tokens = self.count(&partial);
                self.append_block(partial, partial_tokens);
                piece.clear();
                candidate = with_items(std::slice::from_ref(item));
            }
            let single_tokens = self.count(&candidate);
            if piece.is_empty() && single_tokens > self.budget {
                let cut = cut_entry_to_budget(&candidate, self.budget, self.count_tokens);
                let cut_tokens = self.count(&cut);
                self.append_block(cut, cut_tokens);
            } else {
                piece.push(item.clone());
            }
        }
        if !piece.is_empty() {
            let partial = with_items(&piece);
            let partial_tokens = self.count(&partial);
            self.append_block(partial, partial_tokens);
        }
    }
}

/// Upstream `_bank_identity_section`: the shared bank-identity head of a
/// synthesis prompt (name, mission, disposition traits, extra context).
fn bank_identity_section(
    bank_profile: &ReflectBankProfile<'_>,
    additional_context: Option<&str>,
) -> Vec<String> {
    let name = bank_profile.name.unwrap_or("Assistant");
    let mut parts = vec![format!("## Memory Bank Context\nName: {name}")];
    if let Some(mission) = bank_profile.mission.filter(|m| !m.is_empty()) {
        parts.push(format!("Mission: {mission}"));
    }
    if let Some(traits) = bank_profile.disposition {
        let mut rendered = vec![];
        if let Some(s) = traits.skepticism {
            rendered.push(format!("skepticism={s}"));
        }
        if let Some(l) = traits.literalism {
            rendered.push(format!("literalism={l}"));
        }
        if let Some(e) = traits.empathy {
            rendered.push(format!("empathy={e}"));
        }
        if !rendered.is_empty() {
            parts.push(format!("Disposition: {}", rendered.join(", ")));
        }
    }
    if let Some(context) = additional_context.filter(|c| !c.is_empty()) {
        parts.push(format!("\n## Additional Context\n{context}"));
    }
    parts
}

/// Upstream `build_final_prompt`: the user message of the single-shot final
/// synthesis call — bank identity, as much retrieved data as fits (newest
/// entries preferred), the question, the instructions.
#[allow(clippy::too_many_arguments)]
pub fn build_final_prompt(
    query: &str,
    context_history: &[ContextHistoryEntry],
    bank_profile: &ReflectBankProfile<'_>,
    additional_context: Option<&str>,
    max_context_tokens: usize,
    max_tokens: Option<usize>,
    llm_output_language: Option<&str>,
    count_tokens: &dyn Fn(&str) -> usize,
) -> String {
    let mut parts = bank_identity_section(bank_profile, additional_context);

    if !context_history.is_empty() {
        parts.push("\n## Retrieved Data (synthesize and reason from this data)".to_string());
        let mut token_budget = (max_context_tokens as f64 * FINAL_PROMPT_CONTEXT_FRACTION) as usize;
        // Render entries newest-first, then reverse so the prompt reads
        // chronologically.
        let mut rendered: Vec<String> = vec![];
        let mut truncated = false;
        for entry in context_history.iter().rev() {
            let block = render_history_block(entry);
            let block_tokens = count_tokens(&block);
            if block_tokens > token_budget {
                truncated = true;
                break;
            }
            rendered.push(block);
            token_budget = token_budget.saturating_sub(block_tokens);
        }
        rendered.reverse();
        parts.extend(rendered);
        if truncated {
            parts.push(
                "\n*Note: Some earlier tool results were omitted to stay within the context \
                 window.*"
                    .to_string(),
            );
        }
    } else {
        parts.push("\n## Retrieved Data\nNo data was retrieved.".to_string());
    }

    parts.push(format!("\n## Question\n{query}"));
    parts.push(format!(
        "\n## Instructions\n{}",
        texts::REFLECT_FINAL_INSTRUCTIONS
    ));
    if let Some(directive) = length_directive(max_tokens) {
        parts.push(directive);
    }
    format!(
        "{}{}",
        parts.join("\n"),
        output_language_directive(llm_output_language)
    )
}

/// Upstream `build_done_request_prompt`: the closing user turn that asks a
/// loop that has not finished for its `done` call — stop retrieving, the
/// question, the instructions and the length target.
pub fn build_done_request_prompt(
    query: &str,
    max_tokens: Option<usize>,
    llm_output_language: Option<&str>,
) -> String {
    let mut parts = vec![
        "## Answer now".to_string(),
        "Stop retrieving. Call the `done` tool with your final answer, built from the tool \
         results above."
            .to_string(),
        "This is the ANSWER, not a summary of it: carry over every relevant fact, date and \
         number from the tool results, at the same depth you would write for a reader who \
         cannot see them."
            .to_string(),
        format!("\n## Question\n{query}"),
        format!("\n## Instructions\n{}", texts::REFLECT_FINAL_INSTRUCTIONS),
    ];
    if let Some(directive) = length_directive(max_tokens) {
        parts.push(directive);
    }
    format!(
        "{}{}",
        parts.join("\n"),
        output_language_directive(llm_output_language)
    )
}

/// Upstream `build_chunk_claims_prompt`: the user prompt for one
/// intermediate (map) call of split synthesis.
pub fn build_chunk_claims_prompt(query: &str, chunk: &[ContextHistoryEntry]) -> String {
    let mut parts = vec!["## Retrieved Data (extract relevant claims from this data)".to_string()];
    for entry in chunk {
        parts.push(render_history_block(entry));
    }
    parts.push(format!("\n## Question\n{query}"));
    parts.push(
        "\n## Instructions\nList every claim in the retrieved data relevant to the question, \
         one bullet per claim, each with its (mentioned_at: ...; occurred: ...; memory_ids: \
         ...) provenance. Do not answer the question."
            .to_string(),
    );
    parts.join("\n")
}

/// Upstream `build_reduce_prompt`: the final prompt that synthesizes the
/// answer from per-chunk claims, applying latest-statement-wins
/// supersession across sections.
pub fn build_reduce_prompt(
    query: &str,
    claim_sections: &[String],
    bank_profile: &ReflectBankProfile<'_>,
    additional_context: Option<&str>,
    max_tokens: Option<usize>,
    llm_output_language: Option<&str>,
) -> String {
    let mut parts = bank_identity_section(bank_profile, additional_context);
    parts.push(
        "\n## Retrieved Evidence (synthesize and reason from these claims)\nThe retrieved \
         data was processed in parallel passes; each section below holds one pass's extracted \
         claims with provenance dates and memory ids. Treat the sections as ONE evidence \
         pool: related and conflicting claims may appear in different sections."
            .to_string(),
    );
    for (i, section) in claim_sections.iter().enumerate() {
        parts.push(format!("\n### Evidence pass {}:\n{}", i + 1, section));
    }
    parts.push(format!("\n## Question\n{query}"));
    parts.push(format!(
        "\n## Instructions\nWhen claims about the same fact conflict, the claim with the \
         LATEST mentioned_at date is authoritative — later statements supersede earlier \
         ones, regardless of which section they appear in. If equally-recent claims \
         disagree and nothing resolves them, say so explicitly rather than picking \
         one.\n\n{}",
        texts::REFLECT_FINAL_INSTRUCTIONS
    ));
    if let Some(directive) = length_directive(max_tokens) {
        parts.push(directive);
    }
    format!(
        "{}{}",
        parts.join("\n"),
        output_language_directive(llm_output_language)
    )
}
