pub mod auth;
pub mod compat;
pub mod files;
pub mod http;
pub mod mcp;
pub mod webhooks;

use std::time::Duration;

pub const DEFAULT_CONSOLIDATE_INTERVAL_MS: u64 = 5_000;

/// Background consolidation cadence from `RUSTSIGHT_CONSOLIDATE_INTERVAL_MS`
/// (default 5s). Invalid values fall back to the default with a warning.
pub fn consolidate_interval_from_env() -> Duration {
    let raw = std::env::var("RUSTSIGHT_CONSOLIDATE_INTERVAL_MS").unwrap_or_default();
    if raw.trim().is_empty() {
        return Duration::from_millis(DEFAULT_CONSOLIDATE_INTERVAL_MS);
    }
    match raw.trim().parse::<u64>() {
        Ok(ms) => Duration::from_millis(ms.max(1)),
        Err(_) => {
            eprintln!(
                "rustsight: invalid RUSTSIGHT_CONSOLIDATE_INTERVAL_MS {raw:?}, using {DEFAULT_CONSOLIDATE_INTERVAL_MS}ms"
            );
            Duration::from_millis(DEFAULT_CONSOLIDATE_INTERVAL_MS)
        }
    }
}

/// `RUSTSIGHT_LLM_OUTPUT_LANGUAGE` (or `HINDSIGHT_API_LLM_OUTPUT_LANGUAGE`):
/// the language facts, observations and reflect answers are written in.
/// Unset keeps the source language.
pub fn llm_output_language_from_env() -> Option<String> {
    [
        "RUSTSIGHT_LLM_OUTPUT_LANGUAGE",
        "HINDSIGHT_API_LLM_OUTPUT_LANGUAGE",
    ]
    .into_iter()
    .find_map(|name| std::env::var(name).ok())
    .map(|v| v.trim().to_string())
    .filter(|v| !v.is_empty())
}

/// The bank template applied to every new bank, from
/// `RUSTSIGHT_DEFAULT_BANK_TEMPLATE` (or `HINDSIGHT_API_DEFAULT_BANK_TEMPLATE`):
/// inline JSON when the value starts with `{`, else a path to a JSON file.
/// Unset is `Ok(None)`; an unreadable or invalid template is an error, so a
/// server never starts with a template it would silently skip.
pub fn default_bank_template_from_env(
) -> Result<Option<rustsight_core::templates::Manifest>, String> {
    let (name, raw) = match [
        "RUSTSIGHT_DEFAULT_BANK_TEMPLATE",
        "HINDSIGHT_API_DEFAULT_BANK_TEMPLATE",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(|v| (name, v))
    }) {
        Some(found) => found,
        None => return Ok(None),
    };
    parse_default_template(name, &raw).map(Some)
}

fn parse_default_template(
    name: &str,
    raw: &str,
) -> Result<rustsight_core::templates::Manifest, String> {
    let text = if raw.trim_start().starts_with('{') {
        raw.to_string()
    } else {
        std::fs::read_to_string(raw.trim()).map_err(|e| format!("{name}: reading {raw}: {e}"))?
    };
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{name}: invalid JSON: {e}"))?;
    rustsight_core::templates::parse_manifest(&value).map_err(|e| format!("{name}: {e}"))
}

#[cfg(test)]
mod template_env_tests {
    use super::parse_default_template;

    #[test]
    fn default_template_reads_inline_json_or_a_file() {
        let inline = r#"{"version": "1", "directives": [{"name": "n", "content": "c"}]}"#;
        let m = parse_default_template("T", inline).unwrap();
        assert_eq!(m.directives.unwrap()[0].name, "n");
        let path = std::env::temp_dir().join(format!("rs-template-{}.json", uuid::Uuid::new_v4()));
        std::fs::write(&path, inline).unwrap();
        assert!(parse_default_template("T", path.to_str().unwrap()).is_ok());
        std::fs::remove_file(&path).ok();
        let err = parse_default_template("T", r#"{"version": "9"}"#).unwrap_err();
        assert!(
            err.starts_with("T: invalid input: Template schema validation failed"),
            "{err}"
        );
        assert!(parse_default_template("T", "/nonexistent/t.json")
            .unwrap_err()
            .contains("reading"));
    }
}
