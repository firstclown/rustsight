//! Faithful port of upstream's retain fact-extraction prompt builders.
//!
//! Source: `prompts/upstream/v1.1.0/engine/retain/fact_extraction.py`
//! (`_build_extraction_prompt_and_schema`, `build_chunk_prompt_parts`,
//! `_retain_mission_preamble`, `_build_user_message`,
//! `_build_labels_prompt_section`, `build_free_form_entities_instruction`,
//! `_append_map_fields_prompt`, `_numbering_phrase`, `_infer_temporal_date`,
//! `_sanitize_text`), MIT, Copyright 2025 Vectorize AI, Inc.
//!
//! Byte-for-byte output parity with the upstream builders is pinned by
//! `tests/upstream_prompt_parity.rs`. Intentional differences are recorded in
//! `prompts/upstream/DIVERGENCES.md`.

use crate::prompts::prompt_utils::{
    default_language_section, escape_for_prompt, output_language_directive,
};
use crate::prompts::upstream::texts;
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use std::borrow::Cow;

/// Retain extraction mode; upstream `retain_extraction_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtractionMode {
    /// Default; also what custom mode falls back to with no instructions.
    #[default]
    Concise,
    Verbose,
    /// Preserves original text; metadata-only extraction.
    Verbatim,
    Custom,
}

/// A single allowed value for a label group (upstream `LabelValue`).
#[derive(Debug, Clone)]
pub struct LabelValue {
    pub value: String,
    pub description: String,
}

impl LabelValue {
    pub fn new(value: &str, description: &str) -> Self {
        Self {
            value: value.to_string(),
            description: description.to_string(),
        }
    }
}

/// Field shape inside a map group (upstream `MapField.type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapFieldKind {
    Text,
    MultiText,
    Value,
    MultiValues,
    Map,
}

/// A field within a map-type entity label group (upstream `MapField`).
/// Supports recursion via `kind: Map`.
#[derive(Debug, Clone)]
pub struct MapField {
    pub kind: MapFieldKind,
    pub description: String,
    pub values: Vec<LabelValue>,
    /// Insertion-ordered, mirroring Python's dict for prompt order.
    pub fields: Vec<(String, MapField)>,
}

impl MapField {
    pub fn text(description: &str) -> Self {
        Self {
            kind: MapFieldKind::Text,
            description: description.to_string(),
            values: vec![],
            fields: vec![],
        }
    }
}

/// Label group shape (upstream `LabelGroup.type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelGroupKind {
    Value,
    MultiValues,
    Text,
    MultiText,
    Map,
}

/// A label group (dimension) with its type and allowed values (upstream
/// `LabelGroup`).
#[derive(Debug, Clone)]
pub struct LabelGroup {
    pub key: String,
    pub description: String,
    pub kind: LabelGroupKind,
    pub values: Vec<LabelValue>,
    pub fields: Vec<(String, MapField)>,
}

impl LabelGroup {
    pub fn value(key: &str, description: &str, values: Vec<LabelValue>) -> Self {
        Self {
            key: key.to_string(),
            description: description.to_string(),
            kind: LabelGroupKind::Value,
            values,
            fields: vec![],
        }
    }
}

/// The upstream extraction-config fields that shape the *prompt text*.
/// (Schema-only flags such as `llm_supports_string_pattern` are absent: they
/// change the response schema upstream, never the prompt.)
#[derive(Debug, Clone, Default)]
pub struct UpstreamExtractionConfig {
    pub mode: ExtractionMode,
    pub extract_causal_links: bool,
    pub custom_instructions: Option<String>,
    pub llm_output_language: Option<String>,
    pub entity_labels: Option<Vec<LabelGroup>>,
    pub entities_allow_free_form: bool,
    pub optional_fact_dimensions: bool,
}

impl UpstreamExtractionConfig {
    /// Concise mode, no labels, no language, no causal links — the shape
    /// upstream's own baseline test uses.
    pub fn baseline() -> Self {
        Self {
            mode: ExtractionMode::Concise,
            extract_causal_links: false,
            custom_instructions: None,
            llm_output_language: None,
            entity_labels: None,
            entities_allow_free_form: true,
            optional_fact_dimensions: false,
        }
    }
}

/// Upstream `_build_extraction_prompt_and_schema` — prompt side only.
///
/// The bank's retain mission is deliberately NOT baked into this system
/// prompt upstream (it would make the prompt bank-specific and defeat prompt
/// caching); it rides in the per-request user message via
/// [`retain_mission_preamble`]. The `{retain_mission_section}` placeholder is
/// kept (the templates still reference it) but always empty here.
pub fn build_extraction_system_prompt(cfg: &UpstreamExtractionConfig) -> String {
    let language = cfg.llm_output_language.as_deref();
    let language_section =
        default_language_section(texts::EXTRACTION_DEFAULT_LANGUAGE_RULE, language);

    // Select base prompt based on extraction mode. Only the custom template
    // references {custom_instructions}; concise is the default and also what
    // custom mode falls back to with no instructions configured.
    let mut prompt = match (cfg.mode, cfg.custom_instructions.as_deref()) {
        (ExtractionMode::Custom, Some(instructions)) => render_template(
            texts::EXTRACTION_CUSTOM_TEMPLATE,
            &[
                ("language_section", language_section.as_str()),
                ("retain_mission_section", ""),
                // Upstream escapes then .format()s — a net no-op that leaves
                // the caller's braces single in the final output.
                ("custom_instructions", &escape_for_prompt(instructions)),
            ],
        ),
        (ExtractionMode::Verbose, _) => render_template(
            texts::EXTRACTION_VERBOSE_TEMPLATE,
            &[
                ("language_section", language_section.as_str()),
                ("retain_mission_section", ""),
            ],
        ),
        (ExtractionMode::Verbatim, _) => render_template(
            texts::EXTRACTION_VERBATIM_TEMPLATE,
            &[
                ("language_section", language_section.as_str()),
                ("retain_mission_section", ""),
            ],
        ),
        _ => render_template(
            texts::EXTRACTION_CONCISE_TEMPLATE,
            &[
                ("language_section", language_section.as_str()),
                ("retain_mission_section", ""),
            ],
        ),
    };

    // Causal relationships section if enabled; verbatim mode never uses
    // causal relations (no fact text to relate causally).
    if cfg.mode != ExtractionMode::Verbatim && cfg.extract_causal_links {
        prompt.push_str(texts::EXTRACTION_CAUSAL_SECTION);
    }

    // Let a fact leave the four descriptive dimensions empty instead of
    // filling them with "N/A". The placeholder instruction is mechanically
    // swapped across the whole prompt (custom instructions included) — that
    // is upstream's intended reading of the flag.
    if cfg.optional_fact_dimensions {
        prompt = format!(
            "{}{}",
            null_instead_of_na(&prompt),
            texts::EXTRACTION_OPTIONAL_DIMENSIONS_SECTION
        );
    }

    // Entity labels section if configured.
    if let Some(labels) = &cfg.entity_labels {
        let section = build_labels_prompt_section(Some(labels), cfg.entities_allow_free_form);
        prompt.push_str(&section);
    }

    prompt.push_str(&output_language_directive(language));
    prompt
}

/// Swap the `"N/A"` placeholder instruction for `null`, changing nothing else
/// (upstream `_null_instead_of_na` — deliberately mechanical).
fn null_instead_of_na(text: &str) -> String {
    text.replace("'N/A'", "null").replace("\"N/A\"", "null")
}

/// Single-pass `{name}` substitution over a template. Unlike chained
/// `str::replace`, substituted values can never themselves be re-substituted
/// (Python `str.format` semantics for known keys).
pub(crate) fn render_template(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    'outer: while let Some(start) = rest.find('{') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            out.push_str(rest);
            break;
        };
        let name = &after[..end];
        for (key, value) in vars {
            if *key == name {
                out.push_str(&rest[..start]);
                out.push_str(value);
                rest = &after[end + 1..];
                continue 'outer;
            }
        }
        // Unknown name: emit the brace verbatim (the caller scoped the vars).
        out.push_str(&rest[..start + 1]);
        rest = &rest[start + 1..];
    }
    out.push_str(rest);
    out
}

/// Upstream `build_free_form_entities_instruction`: the one line
/// `entities_allow_free_form` decides, inside the entity-labels section.
pub fn build_free_form_entities_instruction(free_form_entities: bool) -> String {
    if free_form_entities {
        "Classify each fact using the structured 'labels' field below. \
         Continue extracting regular named entities in the 'entities' field."
            .to_string()
    } else {
        "Classify each fact using the structured 'labels' field below. \
         Do NOT add regular named entities — labels-only mode."
            .to_string()
    }
}

/// Upstream `_append_map_fields_prompt`: recursively append map field
/// descriptions to the prompt lines.
fn append_map_fields_prompt(fields: &[(String, MapField)], lines: &mut Vec<String>, indent: usize) {
    let pad = " ".repeat(indent);
    for (field_name, map_field) in fields {
        let field_desc = if map_field.description.is_empty() {
            String::new()
        } else {
            format!(": {}", map_field.description)
        };
        match map_field.kind {
            MapFieldKind::Map if !map_field.fields.is_empty() => {
                lines.push(format!("{pad}• {field_name} (object){field_desc}"));
                append_map_fields_prompt(&map_field.fields, lines, indent + 4);
            }
            MapFieldKind::MultiText => {
                lines.push(format!(
                    "{pad}• {field_name} (list of free text, [] if none){field_desc}"
                ));
            }
            MapFieldKind::MultiValues => {
                let vals: Vec<&str> = map_field
                    .values
                    .iter()
                    .filter(|v| !v.value.is_empty())
                    .map(|v| v.value.as_str())
                    .collect();
                let type_hint = if vals.is_empty() {
                    "multi-values".to_string()
                } else {
                    format!("multi-values: {}", vals.join(", "))
                };
                lines.push(format!("{pad}• {field_name} ({type_hint}){field_desc}"));
            }
            MapFieldKind::Value if !map_field.values.is_empty() => {
                let vals: Vec<&str> = map_field
                    .values
                    .iter()
                    .filter(|v| !v.value.is_empty())
                    .map(|v| v.value.as_str())
                    .collect();
                lines.push(format!(
                    "{pad}• {field_name} (one of: {}){field_desc}",
                    vals.join(", ")
                ));
            }
            _ => {
                lines.push(format!("{pad}• {field_name} (text){field_desc}"));
            }
        }
    }
}

/// Upstream `_build_labels_prompt_section`: the entity labels classification
/// section appended to the extraction prompt. `None` or an empty attribute
/// list yields "".
pub fn build_labels_prompt_section(
    labels_cfg: Option<&[LabelGroup]>,
    free_form_entities: bool,
) -> String {
    let Some(attributes) = labels_cfg.filter(|a| !a.is_empty()) else {
        return String::new();
    };

    let entities_instruction = build_free_form_entities_instruction(free_form_entities);

    let mut lines: Vec<String> = vec![
        "\n\n══════════════════════════════════════════════════════════════════════════".into(),
        "ENTITY LABELS - CLASSIFICATION ATTRIBUTES".into(),
        "══════════════════════════════════════════════════════════════════════════".into(),
        String::new(),
        entities_instruction,
        String::new(),
        "For each fact, fill the 'labels' object. Each field is a label group:".into(),
        String::new(),
    ];

    let mut has_classification_attrs = false;
    let mut has_map_attrs = false;

    for attr in attributes {
        if attr.kind == LabelGroupKind::Map {
            has_map_attrs = true;
            continue;
        }
        has_classification_attrs = true;
        match attr.kind {
            LabelGroupKind::Text => {
                // Free-text: no predefined values — LLM writes any relevant string or null
                lines.push(format!(
                    "- {} (free text or null): {}",
                    attr.key, attr.description
                ));
            }
            LabelGroupKind::MultiText => {
                // Open vocabulary: LLM writes as many strings as the content warrants
                lines.push(format!(
                    "- {} (list of free text, empty list if none): {}",
                    attr.key, attr.description
                ));
            }
            _ => {
                let mode = if attr.kind == LabelGroupKind::MultiValues {
                    "multi-value (list)"
                } else {
                    "single value or null"
                };
                lines.push(format!("- {} ({}): {}", attr.key, mode, attr.description));
                for v in &attr.values {
                    let desc = if v.description.is_empty() {
                        String::new()
                    } else {
                        format!(" — {}", v.description)
                    };
                    lines.push(format!("    • \"{}\"{desc}", v.value));
                }
            }
        }
        lines.push(String::new());
    }

    if has_classification_attrs {
        lines.push(
            "Only assign labels when clearly applicable. Leave null/empty if the fact does not match."
                .into(),
        );
        lines.push(String::new());
    }

    if has_map_attrs {
        lines.push(String::new());
        lines.push(
            "══════════════════════════════════════════════════════════════════════════".into(),
        );
        lines.push("STRUCTURED ENTITY TYPES".into());
        lines.push(
            "══════════════════════════════════════════════════════════════════════════".into(),
        );
        lines.push(String::new());
        lines.push("For each fact, extract structured entities into the corresponding list field in 'labels'.".into());
        lines.push("Each structured entity type has defined fields. Return a list of objects, one per entity found.".into());
        lines.push(String::new());
        for attr in attributes {
            if attr.kind != LabelGroupKind::Map || attr.fields.is_empty() {
                continue;
            }
            let desc = if attr.description.is_empty() {
                String::new()
            } else {
                format!(": {}", attr.description)
            };
            lines.push(format!("- {}{desc}", attr.key));
            append_map_fields_prompt(&attr.fields, &mut lines, 4);
            lines.push(String::new());
        }
        lines.push(
            "Only extract structured entities when clearly present in the text. Leave the list empty if none found."
                .into(),
        );
    }

    lines.join("\n")
}

/// Upstream `_retain_mission_preamble`: the bank's retain mission, formatted
/// for the per-request user message. "" when unset.
///
/// Used verbatim (not passed through `str.format` upstream), so braces are
/// safe unescaped.
pub fn retain_mission_preamble(retain_mission: Option<&str>) -> String {
    let Some(retain_mission) = retain_mission.filter(|m| !m.is_empty()) else {
        return String::new();
    };
    format!(
        "══════════════════════════════════════════════════════════════════════════\n\
         FOCUS — What to retain for this bank (takes priority over the general guidelines)\n\
         ══════════════════════════════════════════════════════════════════════════\n\n\
         {retain_mission}\n\n"
    )
}

/// Upstream `_numbering_phrase`: "1" for one, "1 to N" beyond.
fn numbering_phrase(count: usize) -> String {
    if count <= 1 {
        "1".to_string()
    } else {
        format!("1 to {count}")
    }
}

fn is_unsafe_char(c: char) -> bool {
    matches!(c as u32, 0x00..=0x08 | 0x0b | 0x0c | 0x0e..=0x1f | 0x7f)
}

/// Upstream `_sanitize_text`: remove characters that break downstream
/// systems — control chars (keeping \t \n \r). Surrogates cannot exist in a
/// Rust `String`, so they need no handling here.
pub(crate) fn sanitize_text(text: &str) -> Cow<'_, str> {
    if text.chars().any(is_unsafe_char) {
        Cow::Owned(text.chars().filter(|c| !is_unsafe_char(*c)).collect())
    } else {
        Cow::Borrowed(text)
    }
}

/// Attachment placeholders (`⟦hs-att:<hex>⟧`) as counted by upstream's
/// `attachment_content.iter_placeholder_ids`. rustsight chunks never carry
/// them today; the count keeps the ported section reachable.
fn attachment_placeholder_count(text: &str) -> usize {
    text.matches("⟦hs-att:").count()
}

/// Arguments for [`build_chunk_user_message`].
pub struct ChunkUserMessage<'a> {
    pub chunk: &'a str,
    pub chunk_index: usize,
    pub total_chunks: usize,
    /// When the conversation/document occurred; None renders "Unknown".
    pub event_date: Option<DateTime<Utc>>,
    /// Empty renders "none".
    pub context: &'a str,
    /// Insertion-ordered key/value pairs shown under "Metadata:".
    pub metadata: &'a [(&'a str, &'a str)],
    /// Names the first-person narrator when the chunk is an agent transcript.
    pub agent_name: Option<&'a str>,
    /// Prepend from [`retain_mission_preamble`]; "" for none.
    pub mission_preamble: &'a str,
}

/// Upstream `_build_user_message`: the per-chunk extraction user message.
pub fn build_chunk_user_message(args: &ChunkUserMessage) -> String {
    let sanitized_chunk = sanitize_text(args.chunk);
    let sanitized_context = if args.context.is_empty() {
        "none".to_string()
    } else {
        sanitize_text(args.context).into_owned()
    };

    // Python: f"{event_date.strftime('%A, %B %d, %Y')} ({event_date.isoformat()})"
    let event_date_str = match args.event_date {
        Some(dt) => {
            let dt = dt.with_timezone(&Utc);
            let iso = dt.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, false);
            format!("{} ({iso})", dt.format("%A, %B %d, %Y"))
        }
        None => "Unknown".to_string(),
    };

    let metadata_section = if args.metadata.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = args
            .metadata
            .iter()
            .map(|(k, v)| format!("  {k}: {v}"))
            .collect();
        format!("\nMetadata:\n{}", lines.join("\n"))
    };

    let mut narrator_section = String::new();
    if let Some(agent_name) = args.agent_name {
        narrator_section = format!(
            "\nNarrator: {agent_name} (the AI agent whose memory this is). By default, \
             first-person statements like \"I did X\" are {agent_name}'s own actions → classify as \
             \"assistant\"."
        );
        // Only defer to the Context when one was actually provided — otherwise
        // this clause points at a "Context: none" line and just adds noise.
        if !args.context.is_empty() {
            narrator_section.push_str(
                " BUT the Context above takes precedence: if it identifies a different \
                 first-person speaker (e.g. a user or customer in a transcript), attribute those \
                 statements to that speaker and classify them as \"world\", not \"assistant\".",
            );
        }
    }

    let attachment_count = attachment_placeholder_count(&sanitized_chunk);
    let attachment_section = if attachment_count > 0 {
        let mut s = String::from(
            "\n\nATTACHMENTS: this chunk contains one or more attachments (images, PDFs, other \
             files), each shown inline at the exact position it occupies in the source. Read them. \
             Facts stated only in an attachment — a button's label, a value in a table, the boxes \
             of a diagram, text on a page — are as extractable as facts stated in the prose, and \
             are often the point of the document. Attribute each attachment to the sentences \
             around it: an image that follows \"click the button shown:\" is that button.\n",
        );
        s.push_str(
            "When an attachment carries structured data — a chart, a table, a form, a \
             spreadsheet — the data IS the document: give each row, bar, slice or labelled \
             value its own fact, carrying its label, its exact figure, AND how it is drawn — \
             its colour, and where it sits in the order (leftmost, third bar, bottom row). \
             Someone reading the chart asks about \"the green bar\" or \"the bottom row\", never \
             about the internal label, so a fact without those is unreachable no matter how \
             accurate it is. Do not summarize the series: a summary keeps the two or three \
             values it happens to name and silently discards every other one, which is the \
             bulk of what the attachment was showing. Record what the whole thing is as well \
             — its title, its units, the period it covers, how many items it plots, and what \
             each colour in its legend stands for.\n",
        );
        s.push_str(&format!(
            "They are numbered {} in the order they appear above. For each fact, set \
             'from_attachments' to the number(s) of the attachments it came from, and leave it \
             empty when the fact is stated in the text. This is what lets a reader see the \
             picture a fact came from, so be accurate: list an attachment only when the fact \
             could not be stated without looking at it.",
            numbering_phrase(attachment_count)
        ));
        s
    } else {
        String::new()
    };

    format!(
        "{mission}Extract facts from the following chunk.\n\n\
         Chunk: {chunk_index}/{total_chunks}\n\
         Event Date: {event_date}\n\
         Context: {context}{metadata}{narrator}{attachments}\n\n\
         Content:\n{chunk}",
        mission = args.mission_preamble,
        chunk_index = args.chunk_index + 1,
        total_chunks = args.total_chunks,
        event_date = event_date_str,
        context = sanitized_context,
        metadata = metadata_section,
        narrator = narrator_section,
        attachments = attachment_section,
        chunk = sanitized_chunk,
    )
}

/// Upstream `_infer_temporal_date`: infer a temporal date from fact text when
/// the LLM didn't provide `occurred_start`. Returns ms-epoch at midnight.
pub(crate) fn infer_temporal_date_ms(
    fact_text: &str,
    event_date: Option<DateTime<Utc>>,
) -> Option<i64> {
    let event_date = event_date?;
    let fact_lower = fact_text.to_lowercase();

    // Relative time expressions mapped to day offsets (upstream's
    // \b-anchored patterns; `tonigh?t` matches both spellings).
    const PATTERNS: &[(&str, i64)] = &[
        ("last night", -1),
        ("yesterday", -1),
        ("today", 0),
        ("this morning", 0),
        ("this afternoon", 0),
        ("this evening", 0),
        ("tonight", 0),
        ("tonigth", 0),
        ("tomorrow", 1),
        ("last week", -7),
        ("this week", 0),
        ("next week", 7),
        ("last month", -30),
        ("this month", 0),
        ("next month", 30),
    ];

    for (pattern, offset_days) in PATTERNS {
        if contains_word(&fact_lower, pattern) {
            let target = event_date + Duration::days(*offset_days);
            let naive = NaiveDate::from_ymd_opt(target.year(), target.month(), target.day())?
                .and_hms_opt(0, 0, 0)?;
            return Some(naive.and_utc().timestamp_millis());
        }
    }
    None
}

/// `\b<phrase>\b`-style containment for lowercase, space-delimited phrases.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut search_from = 0;
    while let Some(pos) = haystack[search_from..].find(needle) {
        let start = search_from + pos;
        let end = start + needle.len();
        let boundary_before = haystack[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word(c));
        let boundary_after = haystack[end..].chars().next().is_none_or(|c| !is_word(c));
        if boundary_before && boundary_after {
            return true;
        }
        search_from = start + 1;
    }
    false
}
