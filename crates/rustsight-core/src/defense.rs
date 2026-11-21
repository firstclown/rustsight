//! Memory Defense: secret/PII screening of retained content (RS-10a).
//!
//! Port of upstream hindsight-api 1.2.1 `extensions/memory_defense.py`
//! (`_REDACTION_PATTERNS`, `apply_redaction`, `parse_policy`) and the
//! default regex extension's `screen` (`extensions/builtin/
//! memory_defense_regex.py`), MIT, Copyright 2025 Vectorize AI, Inc.
//!
//! The pattern catalog is verbatim; `fancy-regex` because it needs
//! look-around (`regex` has none). Behavior is pinned against upstream's
//! own `apply_redaction` output by `tests/memory_defense_parity.rs`.

use crate::{Error, Result};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;

/// What a policy rule does when its detector matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DefenseAction {
    Allow,
    Redact,
    Block,
}

impl DefenseAction {
    pub fn as_str(self) -> &'static str {
        match self {
            DefenseAction::Allow => "allow",
            DefenseAction::Redact => "redact",
            DefenseAction::Block => "block",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRule {
    /// Detector name. Only `sensitive_data` is acted on here; other names
    /// are accepted no-ops (upstream leaves them to other extensions).
    pub on: String,
    pub action: DefenseAction,
}

/// A bank's `memory_defense` setting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DefensePolicy {
    pub enabled: bool,
    pub rules: Vec<PolicyRule>,
}

/// The detector the regex screener implements.
pub const SENSITIVE_DATA: &str = "sensitive_data";

/// One matched value, fingerprinted — never the raw secret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefenseHit {
    pub detector: String,
    pub preview: String,
}

/// Outcome of screening one retain item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefenseDecision {
    pub action: DefenseAction,
    pub detector: Option<String>,
    pub message: String,
    /// The scrubbed content, for `redact` decisions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_content: Option<String>,
    pub matched_types: Vec<String>,
    pub hits: Vec<DefenseHit>,
}

impl DefenseDecision {
    fn allow() -> Self {
        Self {
            action: DefenseAction::Allow,
            detector: None,
            message: String::new(),
            redacted_content: None,
            matched_types: vec![],
            hits: vec![],
        }
    }
}

/// What screening did to a retained item, as reported on the retain
/// response (no redacted text, no raw values — previews only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefenseReport {
    pub action: DefenseAction,
    pub detector: Option<String>,
    pub message: String,
    pub matched_types: Vec<String>,
    pub hits: Vec<DefenseHit>,
}

impl From<&DefenseDecision> for DefenseReport {
    fn from(d: &DefenseDecision) -> Self {
        Self {
            action: d.action,
            detector: d.detector.clone(),
            message: d.message.clone(),
            matched_types: d.matched_types.clone(),
            hits: d.hits.clone(),
        }
    }
}

/// `apply_redaction`'s result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionResult {
    pub content: String,
    /// Pattern labels that matched, deduplicated, first-occurrence order.
    pub matched_types: Vec<String>,
    /// One entry per matched value.
    pub hits: Vec<DefenseHit>,
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Parse a raw `memory_defense` value (upstream `parse_policy`). `None` or
/// null is the disabled default; a rule without a non-empty `on` or with an
/// unknown action is an error (the HTTP layer turns it into a 4xx).
pub fn parse_policy(raw: Option<&Value>) -> Result<DefensePolicy> {
    let raw = match raw {
        None | Some(Value::Null) => return Ok(DefensePolicy::default()),
        Some(Value::Object(map)) => map,
        Some(_) => {
            return Err(Error::BadInput(
                "memory_defense must be an object".to_string(),
            ))
        }
    };
    let mut rules = vec![];
    let items = match raw.get("rules") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(items)) => items.clone(),
        Some(_) => {
            return Err(Error::BadInput(
                "memory_defense.rules must be a list".to_string(),
            ))
        }
    };
    for item in &items {
        let on = item.get("on");
        let on = match on.and_then(Value::as_str) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                return Err(Error::BadInput(format!(
                    "invalid on {}; must be a non-empty string",
                    py_repr(on)
                )))
            }
        };
        let action_raw = item.get("action");
        let action = match action_raw.and_then(Value::as_str) {
            Some("allow") => DefenseAction::Allow,
            Some("redact") => DefenseAction::Redact,
            Some("block") => DefenseAction::Block,
            _ => {
                return Err(Error::BadInput(format!(
                    "invalid action {}; must be one of ['allow', 'block', 'redact']",
                    py_repr(action_raw)
                )))
            }
        };
        rules.push(PolicyRule { on, action });
    }
    Ok(DefensePolicy {
        enabled: raw.get("enabled").is_some_and(truthy),
        rules,
    })
}

fn py_repr(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => format!("'{s}'"),
        Some(other) => other.to_string(),
    }
}

/// The default regex screener (upstream `MemoryDefenseRegexExtension.screen`).
pub fn screen(policy: &DefensePolicy, content: &str) -> DefenseDecision {
    if !policy.enabled {
        return DefenseDecision::allow();
    }
    let Some(rule) = policy.rules.iter().find(|r| r.on == SENSITIVE_DATA) else {
        return DefenseDecision::allow();
    };
    if rule.action == DefenseAction::Allow {
        return DefenseDecision::allow();
    }
    let result = apply_redaction(content);
    if result.matched_types.is_empty() {
        return DefenseDecision::allow();
    }
    DefenseDecision {
        action: rule.action,
        detector: Some(SENSITIVE_DATA.to_string()),
        message: format!(
            "Sensitive data pattern matched: {}",
            result.matched_types.join(", ")
        ),
        redacted_content: (rule.action == DefenseAction::Redact).then_some(result.content),
        matched_types: result.matched_types,
        hits: result.hits,
    }
}

// ── Pattern catalog ─────────────────────────────────────────────────────

// ASCII token boundaries: `\b` would treat CJK letters as word characters,
// so a secret butted up against Chinese text would not match.
const TS: &str = r"(?<![A-Za-z0-9_])";
const TE: &str = r"(?![A-Za-z0-9_])";

/// `(label, pattern, ascii_token_wrapped)`, in upstream order: more
/// specific patterns run first so broader ones don't consume them.
const PATTERNS: &[(&str, &str, bool)] = &[
    // --- AI / LLM providers ---
    ("anthropic_key", r"sk-ant-[A-Za-z0-9_-]{20,}", true),
    ("openai_project_key", r"sk-proj-[A-Za-z0-9_-]{48,}", true),
    ("openai_admin_key", r"sk-admin-[A-Za-z0-9_-]{40,}", true),
    ("openai_key", r"sk-[A-Za-z0-9_-]{20,}", true),
    ("google_api_key", r"AIza[0-9A-Za-z_-]{35}", true),
    ("google_oauth_token", r"ya29\.[0-9A-Za-z_-]{20,}", true),
    ("xai_key", r"xai-[A-Za-z0-9]{40,}", true),
    ("groq_key", r"gsk_[A-Za-z0-9]{20,}", true),
    (
        "hindsight_key",
        r"hsk_(?:sys_)?[0-9a-f]{32}(?:_[0-9a-f]{8,})?",
        true,
    ),
    ("huggingface_token", r"hf_[A-Za-z0-9]{30,}", true),
    ("replicate_token", r"r8_[A-Za-z0-9]{30,}", true),
    ("perplexity_key", r"pplx-[A-Za-z0-9]{40,}", true),
    ("databricks_token", r"dapi[A-Za-z0-9]{32}", true),
    // --- Cloud providers ---
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", true),
    ("aws_session_token", r"ASIA[0-9A-Z]{16}", true),
    (
        "aws_secret_key",
        r#"(?i)aws(.{0,20})?(secret|private)?[\s_-]?access[\s_-]?key[\s_-]?[:=][\s"']*([A-Za-z0-9/+=]{40})"#,
        false,
    ),
    ("digitalocean_token", r"dop_v1_[a-f0-9]{64}", true),
    // --- Source control & CI ---
    ("github_fg_pat", r"github_pat_[A-Za-z0-9_]{60,}", true),
    ("github_token", r"ghp_[A-Za-z0-9]{36}", true),
    ("github_app_token", r"ghs_[A-Za-z0-9]{36}", true),
    ("github_user_token", r"ghu_[A-Za-z0-9]{36}", true),
    ("github_refresh", r"ghr_[A-Za-z0-9]{36}", true),
    ("github_oauth", r"gho_[A-Za-z0-9]{36}", true),
    ("gitlab_pat", r"glpat-[A-Za-z0-9_-]{20,}", true),
    ("npm_token", r"npm_[A-Za-z0-9]{30,}", true),
    (
        "pypi_token",
        r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}",
        true,
    ),
    // --- Payment processors ---
    ("stripe_secret", r"sk_(?:live|test)_[A-Za-z0-9]{20,}", true),
    (
        "stripe_restricted",
        r"rk_(?:live|test)_[A-Za-z0-9]{20,}",
        true,
    ),
    ("square_token", r"sq0[a-z]{3}-[A-Za-z0-9_-]{22,}", true),
    (
        "braintree_token",
        r"access_token\$production\$[a-z0-9]{16}\$[a-f0-9]{32}",
        true,
    ),
    // --- Communication / email ---
    ("slack_token", r"xox[abpr]-[0-9A-Za-z-]{10,}", true),
    (
        "slack_webhook",
        r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]{8,}/B[A-Za-z0-9_]{8,}/[A-Za-z0-9_]{20,}",
        false,
    ),
    ("twilio_api_key", r"SK[0-9a-fA-F]{32}", true),
    ("twilio_account_sid", r"AC[0-9a-fA-F]{32}", true),
    (
        "sendgrid_key",
        r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        true,
    ),
    ("mailgun_key", r"key-[A-Za-z0-9]{32}", true),
    (
        "discord_bot",
        r"[MNO][A-Za-z0-9]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}",
        true,
    ),
    ("telegram_bot", r"[0-9]{8,10}:[A-Za-z0-9_-]{35}", true),
    // --- Commerce ---
    ("shopify_token", r"shpat_[a-fA-F0-9]{32}", true),
    // --- Database connection strings (creds embedded in URL) ---
    (
        "db_url_postgres",
        r"postgres(?:ql)?://[^\s:/@]+:[^\s/@]+@[^\s]+",
        false,
    ),
    ("db_url_mysql", r"mysql://[^\s:/@]+:[^\s/@]+@[^\s]+", false),
    (
        "db_url_mongodb",
        r"mongodb(?:\+srv)?://[^\s:/@]+:[^\s/@]+@[^\s]+",
        false,
    ),
    // --- Private keys & generic credentials ---
    (
        "private_key_pem",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY( BLOCK)?-----",
        false,
    ),
    (
        "jwt",
        r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        true,
    ),
    // --- PII (US-centric defaults) ---
    (
        "credit_card",
        r"(?<!\d)(?<!\d\.)(?:\d{4}[ -]?){3}\d{1,4}(?!\d)(?!\.\d)",
        true,
    ),
    ("ssn_us", r"\d{3}-\d{2}-\d{4}", true),
];

/// Number of patterns in the catalog (46 at 1.2.1).
pub fn pattern_count() -> usize {
    PATTERNS.len()
}

fn compiled() -> &'static [(&'static str, Regex)] {
    static C: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    C.get_or_init(|| {
        PATTERNS
            .iter()
            .map(|(label, body, wrap)| {
                let src = if *wrap {
                    format!("{TS}{body}{TE}")
                } else {
                    (*body).to_string()
                };
                (*label, Regex::new(&src).expect("defense pattern compiles"))
            })
            .collect()
    })
}

fn uuid_pattern() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(&format!(
            "{TS}[0-9a-fA-F]{{8}}-(?:[0-9a-fA-F]{{4}}-){{3}}[0-9a-fA-F]{{12}}{TE}"
        ))
        .expect("uuid pattern compiles")
    })
}

/// A length-aware preview that identifies a matched value without leaking
/// it (upstream `_fingerprint_value`; lengths are in code points).
pub fn fingerprint_value(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let n = chars.len();
    let edge = |k: usize| -> (String, String) {
        (chars[..k].iter().collect(), chars[n - k..].iter().collect())
    };
    if n < 6 {
        "[redacted]".to_string()
    } else if n <= 15 {
        let (a, b) = edge(2);
        format!("{a}...{b}")
    } else {
        let (a, b) = edge(4);
        format!("{a}...{b}")
    }
}

/// Code points of every Unicode `Nd` zero digit (generated from Python's
/// `unicodedata`; each decimal block is ten consecutive code points).
const ND_ZEROS: &[u32] = &[
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x10D40, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0,
    0x11650, 0x116C0, 0x116D0, 0x116DA, 0x11730, 0x118E0, 0x11950, 0x11BF0, 0x11C50, 0x11D50,
    0x11DA0, 0x11F50, 0x16130, 0x16A60, 0x16AC0, 0x16B50, 0x16D70, 0x1CCF0, 0x1D7CE, 0x1D7D8,
    0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E5F1, 0x1E950, 0x1FBF0,
];

/// Decimal value of a Unicode digit (Python `int(char)`).
fn digit_value(c: char) -> Option<u32> {
    let cp = c as u32;
    let idx = ND_ZEROS.partition_point(|&z| z <= cp);
    let zero = *ND_ZEROS.get(idx.checked_sub(1)?)?;
    (cp - zero < 10).then_some(cp - zero)
}

/// Luhn check of a card candidate (upstream `_is_luhn_card`).
fn is_luhn_card(value: &str) -> bool {
    let digits: Vec<u32> = value
        .chars()
        .filter(|c| *c != ' ' && *c != '-')
        .filter_map(digit_value)
        .collect();
    if digits.is_empty() || digits.iter().all(|d| *d == digits[0]) {
        return false;
    }
    let mut checksum = 0;
    for (i, d) in digits.iter().rev().enumerate() {
        let mut d = *d;
        if i % 2 == 1 {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        checksum += d;
    }
    checksum % 10 == 0
}

/// Byte offset `n` code points before/after `from` (clamped).
fn char_offset_back(s: &str, from: usize, n: usize) -> usize {
    if n == 0 {
        return from;
    }
    s[..from]
        .char_indices()
        .rev()
        .nth(n - 1)
        .map_or(0, |(i, _)| i)
}

fn char_offset_fwd(s: &str, from: usize, n: usize) -> usize {
    s[from..]
        .char_indices()
        .nth(n)
        .map_or(s.len(), |(i, _)| from + i)
}

fn next_char_boundary(s: &str, at: usize) -> usize {
    s[at..]
        .chars()
        .next()
        .map_or(s.len(), |c| at + c.len_utf8())
}

/// Whether a UUID overlaps `[start, end)` — searched only in the bounded
/// neighborhood upstream inspects (35 code points back, 36 forward).
fn overlaps_uuid(content: &str, start: usize, end: usize) -> bool {
    let lo = char_offset_back(content, start, 35);
    let hi = char_offset_fwd(content, end, 36);
    let window = &content[..hi];
    let mut pos = lo;
    while pos <= window.len() {
        let Ok(Some(m)) = uuid_pattern().find_from_pos(window, pos) else {
            return false;
        };
        if m.start() < end && m.end() > start {
            return true;
        }
        pos = m.end();
    }
    false
}

fn redact_cards(re: &Regex, content: &str, out: &mut RedactionResult) -> String {
    let mut parts = String::with_capacity(content.len());
    let mut search_start = 0;
    let mut copied_until = 0;
    while search_start <= content.len() {
        let Ok(Some(m)) = re.find_from_pos(content, search_start) else {
            break;
        };
        // A rejected candidate must not consume a later valid start:
        // search again from its next character.
        search_start = next_char_boundary(content, m.start());
        let value = m.as_str();
        if !is_luhn_card(value) || overlaps_uuid(content, m.start(), m.end()) {
            if m.start() >= content.len() {
                break;
            }
            continue;
        }
        if !out.matched_types.iter().any(|t| t == "credit_card") {
            out.matched_types.push("credit_card".into());
        }
        out.hits.push(DefenseHit {
            detector: "credit_card".into(),
            preview: fingerprint_value(value),
        });
        parts.push_str(&content[copied_until..m.start()]);
        parts.push_str("[REDACTED:credit_card]");
        copied_until = m.end();
        search_start = m.end();
    }
    parts.push_str(&content[copied_until..]);
    parts
}

/// Scrub every catalog pattern from `content` with `[REDACTED:<label>]`
/// markers (upstream `apply_redaction`). Patterns apply in order, each to
/// the previous one's output.
pub fn apply_redaction(content: &str) -> RedactionResult {
    let mut out = RedactionResult {
        content: String::new(),
        matched_types: vec![],
        hits: vec![],
    };
    let mut content = content.to_string();
    for (label, re) in compiled() {
        if *label == "credit_card" {
            content = redact_cards(re, &content, &mut out);
            continue;
        }
        let groups = re.captures_len() - 1;
        let mut replaced = String::with_capacity(content.len());
        let mut last = 0;
        let mut any = false;
        for caps in re.captures_iter(&content).flatten() {
            any = true;
            let whole = caps.get(0).expect("group 0");
            // Python `findall`: no groups → the match; one group → that
            // group; several → upstream picks the longest non-empty one.
            let raw = match groups {
                0 => whole.as_str().to_string(),
                1 => caps.get(1).map_or("", |m| m.as_str()).to_string(),
                _ => (1..=groups)
                    .filter_map(|i| caps.get(i).map(|m| m.as_str()))
                    .filter(|s| !s.is_empty())
                    .fold("", |best, s| {
                        if s.chars().count() > best.chars().count() {
                            s
                        } else {
                            best
                        }
                    })
                    .to_string(),
            };
            if !raw.is_empty() {
                out.hits.push(DefenseHit {
                    detector: (*label).to_string(),
                    preview: fingerprint_value(&raw),
                });
            }
            replaced.push_str(&content[last..whole.start()]);
            replaced.push_str(&format!("[REDACTED:{label}]"));
            last = whole.end();
        }
        if !any {
            continue;
        }
        if !out.matched_types.iter().any(|t| t == label) {
            out.matched_types.push((*label).to_string());
        }
        replaced.push_str(&content[last..]);
        content = replaced;
    }
    out.content = content;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn catalog_has_upstream_46_patterns_and_compiles() {
        assert_eq!(pattern_count(), 46);
        assert_eq!(compiled().len(), 46);
    }

    #[test]
    fn redacts_tokens_with_labels() {
        let token = format!("ghp_{}", "A".repeat(36));
        let r = apply_redaction(&format!("my token is {token} ok"));
        assert_eq!(r.content, "my token is [REDACTED:github_token] ok");
        assert_eq!(r.matched_types, vec!["github_token"]);
        assert_eq!(r.hits[0].preview, "ghp_...AAAA");
    }

    #[test]
    fn matches_secrets_inside_cjk_prose_but_not_partial_ascii_tokens() {
        let key = format!("sk_test_{}", "A".repeat(24));
        let r = apply_redaction(&format!("凭证为{key}请保密"));
        assert_eq!(r.content, "凭证为[REDACTED:stripe_secret]请保密");
        let r = apply_redaction(&format!("x{key}"));
        assert!(r.matched_types.is_empty(), "{:?}", r.matched_types);
    }

    #[test]
    fn cards_need_luhn_and_are_not_uuids_or_timestamps() {
        let r = apply_redaction("card 4111 1111 1111 1111 exp");
        assert_eq!(r.content, "card [REDACTED:credit_card] exp");
        // Same digit count, fails Luhn: a timestamp-ish run.
        assert!(apply_redaction("ts 2026 1121 0930 1234")
            .matched_types
            .is_empty());
        // Repeated-digit placeholder passes Luhn but is rejected.
        assert!(apply_redaction("0000 0000 0000 0000")
            .matched_types
            .is_empty());
        // A rejected candidate does not consume the valid one after it.
        let r = apply_redaction("2026 4111 1111 1111 1111");
        assert_eq!(r.content, "2026 [REDACTED:credit_card]");
        // Full-width digits count, like Python's int().
        let r = apply_redaction("４１１１ １１１１ １１１１ １１１１");
        assert_eq!(r.matched_types, vec!["credit_card"]);
    }

    #[test]
    fn fingerprints_are_length_aware() {
        assert_eq!(fingerprint_value("abc"), "[redacted]");
        assert_eq!(fingerprint_value("123-45-6789"), "12...89");
        assert_eq!(
            fingerprint_value(&format!("AKIA{}", "B".repeat(16))),
            "AKIA...BBBB"
        );
    }

    #[test]
    fn policy_parsing_follows_upstream() {
        assert_eq!(parse_policy(None).unwrap(), DefensePolicy::default());
        let p = parse_policy(Some(&json!({
            "enabled": true,
            "rules": [{"on": "sensitive_data", "action": "block"}, {"on": "cloud_only", "action": "redact"}]
        })))
        .unwrap();
        assert!(p.enabled);
        assert_eq!(p.rules.len(), 2);
        let err = parse_policy(Some(&json!({"rules": [{"on": "", "action": "block"}]})));
        assert!(err.unwrap_err().to_string().contains("invalid on ''"));
        let err = parse_policy(Some(&json!({"rules": [{"on": "x", "action": "nuke"}]})));
        assert!(err
            .unwrap_err()
            .to_string()
            .contains("invalid action 'nuke'"));
    }

    #[test]
    fn screen_honors_enabled_rule_and_action() {
        let secret = format!("AKIA{}", "Z".repeat(16));
        let text = format!("key {secret}");
        let policy = |action: &str, enabled: bool| {
            parse_policy(Some(&json!({
                "enabled": enabled, "rules": [{"on": "sensitive_data", "action": action}]
            })))
            .unwrap()
        };
        assert_eq!(
            screen(&policy("redact", false), &text).action,
            DefenseAction::Allow
        );
        assert_eq!(
            screen(&policy("allow", true), &text).action,
            DefenseAction::Allow
        );
        let d = screen(&policy("redact", true), &text);
        assert_eq!(d.action, DefenseAction::Redact);
        assert_eq!(
            d.redacted_content.as_deref(),
            Some("key [REDACTED:aws_access_key]")
        );
        assert_eq!(d.message, "Sensitive data pattern matched: aws_access_key");
        let d = screen(&policy("block", true), &text);
        assert_eq!(d.action, DefenseAction::Block);
        assert!(d.redacted_content.is_none());
        assert_eq!(
            screen(&policy("block", true), "nothing here").action,
            DefenseAction::Allow
        );
        // A policy whose only rule names another detector does nothing.
        let other = parse_policy(Some(
            &json!({"enabled": true, "rules": [{"on": "prompt_injection", "action": "block"}]}),
        ))
        .unwrap();
        assert_eq!(screen(&other, &text).action, DefenseAction::Allow);
    }

    #[test]
    fn digit_values_cover_unicode_blocks() {
        assert_eq!(digit_value('7'), Some(7));
        assert_eq!(digit_value('٣'), Some(3));
        assert_eq!(digit_value('\u{1D7D9}'), Some(1));
        assert_eq!(digit_value('a'), None);
    }
}
