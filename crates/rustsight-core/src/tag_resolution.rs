//! Fuzzy tag matching (upstream `engine/search/tag_resolution.py`).
//!
//! A tag-group leaf with `resolve: "fuzzy"` treats its tags as tokens: each
//! is matched against the bank's tag vocabulary and the leaf is rewritten
//! into ordinary exact groups, so nothing downstream knows fuzzy matching
//! exists. A token that matches nothing stays as itself, which keeps the
//! leaf unsatisfiable instead of turning it into "no filter".

use crate::trigram::{trigram_set, trigram_set_similarity};
use crate::types::{TagGroup, TagsMatch};
use std::collections::HashMap;

/// Trigram similarity at or above which a token resolves to a tag.
pub const MIN_SIMILARITY: f32 = 0.45;
/// Distinct tags a bank may hold before fuzzy matching refuses to run:
/// matching a truncated vocabulary would silently change results.
pub const MAX_VOCABULARY: usize = 5000;
/// Ceiling on the branches one `exact` leaf may expand into.
pub const MAX_EXACT_COMBINATIONS: usize = 32;

fn normalize(value: &str) -> String {
    value.trim().to_lowercase()
}

/// `("name", "k8s")` for `name:k8s`; `("", tag)` without a namespace.
/// Similarity is measured on the value alone: a shared namespace would
/// otherwise count as agreement between unrelated short tags.
fn split_namespace(tag: &str) -> (&str, &str) {
    tag.split_once(':').unwrap_or(("", tag))
}

/// The bank's distinct tags, most used first (ties alphabetical), from the
/// id → tags index.
pub fn tag_vocabulary(index: &HashMap<String, Vec<String>>) -> Result<Vec<String>, String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for tag in index.values().flatten() {
        *counts.entry(tag).or_default() += 1;
    }
    if counts.len() > MAX_VOCABULARY {
        return Err(format!(
            "Bank has {} distinct tags, above the {MAX_VOCABULARY} that fuzzy tag matching can \
             resolve against. Use resolve='exact'.",
            counts.len()
        ));
    }
    let mut tags: Vec<(&str, usize)> = counts.into_iter().collect();
    tags.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    Ok(tags.into_iter().map(|(t, _)| t.to_string()).collect())
}

/// The tags `token` should match. A case-insensitive exact hit wins outright;
/// otherwise every same-namespace tag at or above [`MIN_SIMILARITY`], most
/// similar first; otherwise the token itself.
pub fn expand_token(token: &str, vocabulary: &[String]) -> Vec<String> {
    let normalized = normalize(token);
    let exact: Vec<String> = vocabulary
        .iter()
        .filter(|tag| normalize(tag) == normalized)
        .cloned()
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    let (namespace, value) = split_namespace(&normalized);
    let token_trigrams = trigram_set(value);
    let mut scored: Vec<(f32, &String)> = vocabulary
        .iter()
        .filter_map(|tag| {
            let lowered = normalize(tag);
            let (tag_namespace, tag_value) = split_namespace(&lowered);
            (tag_namespace == namespace).then(|| {
                let score = trigram_set_similarity(&token_trigrams, &trigram_set(tag_value));
                (score, tag)
            })
        })
        .filter(|(score, _)| *score >= MIN_SIMILARITY)
        .collect();
    // Stable: ties keep vocabulary order.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if scored.is_empty() {
        vec![token.to_string()]
    } else {
        scored.into_iter().map(|(_, tag)| tag.clone()).collect()
    }
}

fn leaf(tags: Vec<String>, mode: TagsMatch) -> TagGroup {
    TagGroup::Leaf { tags, mode }
}

/// Every way to pick one tag per token, skipping picks that reuse a tag.
fn exact_combinations(expanded: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut combos: Vec<Vec<String>> = vec![vec![]];
    for options in expanded {
        let mut next = vec![];
        for combo in &combos {
            for option in options {
                if !combo.contains(option) {
                    let mut extended = combo.clone();
                    extended.push(option.clone());
                    next.push(extended);
                }
            }
        }
        combos = next;
    }
    let mut unique: Vec<Vec<String>> = vec![];
    for combo in combos {
        if !unique.contains(&combo) {
            unique.push(combo);
        }
    }
    unique
}

/// Rewrite one fuzzy leaf into exact-only groups.
pub fn resolve_fuzzy_leaf(
    tags: Vec<String>,
    mode: TagsMatch,
    vocabulary: &[String],
) -> Result<TagGroup, String> {
    if tags.is_empty() {
        return Ok(leaf(tags, mode));
    }
    let expanded: Vec<Vec<String>> = tags
        .iter()
        .map(|token| expand_token(token, vocabulary))
        .collect();
    match mode {
        TagsMatch::Any | TagsMatch::AnyStrict => {
            let mut union: Vec<String> = vec![];
            for tag in expanded.into_iter().flatten() {
                if !union.contains(&tag) {
                    union.push(tag);
                }
            }
            Ok(leaf(union, mode))
        }
        TagsMatch::All | TagsMatch::AllStrict => {
            // One OR per token, AND-ed. Flattening would demand every
            // spelling of every token at once, which matches nothing.
            let inner = if mode == TagsMatch::All {
                TagsMatch::Any
            } else {
                TagsMatch::AnyStrict
            };
            let mut conjuncts: Vec<TagGroup> =
                expanded.into_iter().map(|t| leaf(t, inner)).collect();
            Ok(if conjuncts.len() == 1 {
                conjuncts.remove(0)
            } else {
                TagGroup::And(conjuncts)
            })
        }
        TagsMatch::Exact => {
            let branches = expanded
                .iter()
                .fold(1usize, |n, t| n.saturating_mul(t.len()));
            if branches > MAX_EXACT_COMBINATIONS {
                return Err(format!(
                    "Fuzzy resolution of an 'exact' tag filter expands to {branches} candidate \
                     scopes, above the ceiling of {MAX_EXACT_COMBINATIONS}. Narrow the tags, or \
                     use resolve='exact'."
                ));
            }
            let mut combos = exact_combinations(&expanded);
            Ok(match combos.len() {
                0 => leaf(tags, TagsMatch::Exact),
                1 => leaf(combos.remove(0), TagsMatch::Exact),
                _ => TagGroup::Or(
                    combos
                        .into_iter()
                        .map(|c| leaf(c, TagsMatch::Exact))
                        .collect(),
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(tags: &[&str]) -> Vec<String> {
        tags.iter().map(|t| t.to_string()).collect()
    }

    #[test]
    fn exact_hit_is_never_widened() {
        let v = vocab(&["name:typescript", "name:typescripts"]);
        assert_eq!(
            expand_token("NAME:TypeScript", &v),
            vec!["name:typescript".to_string()]
        );
    }

    #[test]
    fn misspelling_resolves_within_its_namespace_only() {
        let v = vocab(&["name:typescript", "scope:typescript", "name:mongo"]);
        assert_eq!(
            expand_token("name:typescropt", &v),
            vec!["name:typescript".to_string()]
        );
    }

    #[test]
    fn unmatched_token_keeps_the_filter_unsatisfiable() {
        let v = vocab(&["name:mongo"]);
        let group = resolve_fuzzy_leaf(vec!["name:zzzz".into()], TagsMatch::AnyStrict, &v).unwrap();
        assert!(!group.matches(&["name:mongo".to_string()]));
    }

    #[test]
    fn all_mode_needs_one_spelling_of_each_token() {
        let v = vocab(&["typescript", "typescripts", "mongoose"]);
        let group = resolve_fuzzy_leaf(
            vec!["typescrip".into(), "mongoos".into()],
            TagsMatch::AllStrict,
            &v,
        )
        .unwrap();
        assert!(group.matches(&["typescripts".to_string(), "mongoose".to_string()]));
        assert!(!group.matches(&["typescript".to_string(), "typescripts".to_string()]));
    }

    #[test]
    fn exact_mode_requires_distinct_tags_per_token() {
        let v = vocab(&["typescript"]);
        let group = resolve_fuzzy_leaf(
            vec!["typescrip".into(), "typescripts".into()],
            TagsMatch::Exact,
            &v,
        )
        .unwrap();
        assert!(!group.matches(&["typescript".to_string()]));
    }

    #[test]
    fn oversized_vocabulary_is_refused() {
        let index: HashMap<String, Vec<String>> = (0..=MAX_VOCABULARY)
            .map(|i| (format!("m{i}"), vec![format!("t{i}")]))
            .collect();
        assert!(tag_vocabulary(&index).is_err());
    }
}
