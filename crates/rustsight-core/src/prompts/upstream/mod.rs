//! Faithful ports of upstream prompt builders, held to byte-for-byte output
//! parity with the vendored sources under `prompts/upstream/v1.1.0/`.
//!
//! Parity is pinned by `tests/upstream_prompt_parity.rs` (golden fixtures
//! generated from the upstream Python itself). Intentional differences are
//! recorded in `prompts/upstream/DIVERGENCES.md`.

pub mod consolidation;
pub mod extraction;
pub mod reflect;
pub(crate) mod texts;

/// Python `str.format`-faithful template rendering for known keys: `{{` and
/// `}}` collapse to literal braces, `{known}` substitutes verbatim (values
/// are never re-scanned, matching Python's single-pass semantics).
pub(crate) fn python_format(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'{' if bytes.get(i + 1) == Some(&b'{') => {
                out.push('{');
                i += 2;
            }
            b'{' => {
                let rest = &template[i + 1..];
                let Some(end) = rest.find('}') else {
                    // Python raises ValueError on an unclosed field; our
                    // templates are fixed constants, so treat as literal.
                    out.push_str(&template[i..]);
                    break;
                };
                let name = &rest[..end];
                match vars.iter().find(|(k, _)| *k == name) {
                    Some((_, value)) => out.push_str(value),
                    // Python raises KeyError on unknown fields; defensive
                    // literal passthrough (unreachable for our templates).
                    None => out.push_str(&template[i..i + end + 2]),
                }
                i += end + 2;
            }
            b'}' if bytes.get(i + 1) == Some(&b'}') => {
                out.push('}');
                i += 2;
            }
            _ => {
                let ch_len = template[i..].chars().next().map_or(1, char::len_utf8);
                out.push_str(&template[i..i + ch_len]);
                i += ch_len;
            }
        }
    }
    out
}
