//! Shared utilities for prompt assembly.
//!
//! Faithful port of upstream `hindsight_api/engine/prompt_utils.py`
//! (vendored at `prompts/upstream/v1.1.0/engine/prompt_utils.py`, MIT,
//! Copyright 2025 Vectorize AI, Inc.). Byte-for-byte parity is pinned by the
//! unit tests below and by the builder-level parity suite in
//! `tests/upstream_prompt_parity.rs`.

/// Max characters of a memory's `context` shown in internal prompts
/// (consolidation, reflect tool results, think). A caller can store a huge
/// context on every fact; repeated per memory it would bloat the prompt.
pub const PROMPT_CONTEXT_MAX_CHARS: usize = 200;

/// Double any lone `{` / `}` so the text survives Python-side `str.format`
/// untouched.
///
/// Prompt templates upstream are passed through `str.format` to substitute
/// real placeholders like `{facts_text}`. Any literal braces in
/// caller-supplied text — e.g. a bank mission that contains JSON examples —
/// would otherwise be interpreted as format keys.
///
/// Idempotent: text that already contains escaped `{{` / `}}` pairs is
/// left as-is. Only lone braces (not adjacent to another brace of the same
/// kind) are doubled.
///
/// The Rust builders never run Python's `str.format`, so escaping is a no-op
/// *for us* — but the escaped text is what upstream's builders embed in their
/// final output, so the helper exists to reproduce that output exactly.
pub fn escape_for_prompt(text: &str) -> String {
    // Python: _LONE_OPEN_BRACE = (?<!\{)\{(?!\{), _LONE_CLOSE_BRACE = (?<!\})\}(?!\})
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(i + 1).copied();
        match c {
            '{' if prev != Some('{') && next != Some('{') => out.push_str("{{"),
            '}' if prev != Some('}') && next != Some('}') => out.push_str("}}"),
            _ => out.push(c),
        }
    }
    out
}

/// Cut a memory's context to [`PROMPTS_CONTEXT_MAX_CHARS`][PROMPT_CONTEXT_MAX_CHARS],
/// adding `...` when cut.
pub fn truncate_context_for_prompt(context: Option<&str>) -> Option<String> {
    if let Some(context) = context {
        if context.chars().count() > PROMPT_CONTEXT_MAX_CHARS {
            let cut: String = context.chars().take(PROMPT_CONTEXT_MAX_CHARS).collect();
            return Some(format!("{cut}..."));
        }
    }
    context.map(|c| c.to_string())
}

/// Return an LLM directive forcing all output into `language`.
///
/// Used by retain (fact extraction), consolidation (observations), and reflect
/// (response synthesis) so a configured output language applies uniformly
/// across every LLM-generated artifact. Returns an empty string when
/// `language` is unset so the calling prompt stays unchanged.
pub fn output_language_directive(language: Option<&str>) -> String {
    let Some(language) = language.filter(|l| !l.is_empty()) else {
        return String::new();
    };
    format!(
        "\n\nIMPORTANT: Respond exclusively in {language}. \
         Translate any source content into {language}. \
         All output text — including fact text, observations, entity names, \
         and the final response — must be in {language}."
    )
}

/// The preserve-the-source-language rule, unless `language` overrides it.
///
/// The other half of [`output_language_directive`], and its mutual exclusion.
/// Each pipeline runs an all-English prompt, so with no configured language a
/// multilingual model drifts to English and needs telling to keep the input's
/// language. That rule flatly contradicts "translate everything into X", and
/// the model resolves the contradiction in favour of the source-language rule
/// — it is phrased more forcefully and comes first — silently no-opping the
/// setting (upstream #3776). So an explicit language drops the rule outright
/// rather than arguing with it.
///
/// `default_rule` is the pipeline's own wording (retain, consolidation and
/// reflect each say it differently); only the selection is shared. Returns the
/// rule followed by a blank line, ready to prepend to a prompt body, or `""`
/// when `language` is set.
pub fn default_language_section(default_rule: &str, language: Option<&str>) -> String {
    if language.filter(|l| !l.is_empty()).is_some() {
        String::new()
    } else {
        format!("{default_rule}\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from upstream tests/test_prompt_brace_escape.py::TestEscapeForPrompt
    // and tests/test_prompt_context_truncation.py::test_truncate_context_for_prompt.

    #[test]
    fn lone_braces_doubled() {
        assert_eq!(escape_for_prompt("{x}"), "{{x}}");
    }

    #[test]
    fn already_escaped_left_alone() {
        assert_eq!(escape_for_prompt("{{x}}"), "{{x}}");
    }

    #[test]
    fn idempotent() {
        let once = escape_for_prompt("{\"dedup\": true}");
        let twice = escape_for_prompt(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn plain_text_unchanged() {
        assert_eq!(escape_for_prompt("no braces here"), "no braces here");
    }

    #[test]
    fn mixed_lone_and_escaped() {
        assert_eq!(escape_for_prompt("{x} and {{y}}"), "{{x}} and {{y}}");
    }

    #[test]
    fn edge_cases_do_not_crash() {
        for text in ["{single}", "}}weird{{", "trailing {", "leading }", ""] {
            // Must terminate and stay balanced for the Python-side .format().
            let escaped = escape_for_prompt(text);
            let opens = escaped.matches('{').count();
            let closes = escaped.matches('}').count();
            assert_eq!(opens % 2, 0, "{text:?} -> {escaped:?}");
            assert_eq!(closes % 2, 0, "{text:?} -> {escaped:?}");
        }
    }

    #[test]
    fn truncate_context() {
        let long = "x".repeat(PROMPT_CONTEXT_MAX_CHARS + 50);
        let cut = format!("{}...", "x".repeat(PROMPT_CONTEXT_MAX_CHARS));
        assert_eq!(
            truncate_context_for_prompt(Some(&long)).as_deref(),
            Some(cut.as_str())
        );
        assert_eq!(
            truncate_context_for_prompt(Some(&"x".repeat(PROMPT_CONTEXT_MAX_CHARS))).as_deref(),
            Some("x".repeat(PROMPT_CONTEXT_MAX_CHARS).as_str())
        );
        assert_eq!(
            truncate_context_for_prompt(Some("chat")).as_deref(),
            Some("chat")
        );
        assert_eq!(truncate_context_for_prompt(None), None);
    }

    #[test]
    fn language_directive_shapes() {
        assert_eq!(output_language_directive(None), "");
        assert_eq!(output_language_directive(Some("")), "");
        assert!(output_language_directive(Some("Spanish"))
            .starts_with("\n\nIMPORTANT: Respond exclusively in Spanish."));
        assert_eq!(default_language_section("RULE", None), "RULE\n\n");
        assert_eq!(default_language_section("RULE", Some("Spanish")), "");
    }
}
