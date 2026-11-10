//! Entity-name machinery: normalization plus the upstream (v1.1.0)
//! fuzzy-resolution scoring ported from `hindsight_api/engine/memories/pg/
//! entity_resolver.py`. The scoring model, gates, and constants mirror
//! upstream exactly; see `DIVERGENCES.md` for the one structural
//! simplification (candidate generation scans the bank's entities rather
//! than a trigram index — upstream's own "full" strategy shape — with the
//! same `merge_min_similarity` gate applied afterwards, so the scored
//! candidate set is identical).

use crate::trigram::{trgm_words, trigram_set, trigram_set_similarity};
use std::collections::{HashMap, HashSet};

pub fn normalize_name(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

pub fn string_similarity(a: &str, b: &str) -> f32 {
    let na = normalize_name(a);
    let nb = normalize_name(b);
    if na.is_empty() || nb.is_empty() {
        return 0.0;
    }
    if na == nb {
        return 1.0;
    }
    let max = na.chars().count().max(nb.chars().count()) as f32;
    1.0 - levenshtein(&na, &nb) as f32 / max
}

/// Upstream `_normalize_entity_name`: collapse internal whitespace runs to a
/// single space and strip the ends. Case is deliberately untouched — the
/// registry matches on lowercased names, so normalizing case here would only
/// lose the display form.
pub fn collapse_ws(name: &str) -> String {
    name.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// SequenceMatcher ratio (Python difflib port)
// ---------------------------------------------------------------------------

/// Total length of difflib's matching blocks for `a` vs `b` — the `M` of
/// `SequenceMatcher(None, a, b).ratio() == 2.0 * M / (len(a) + len(b))`.
///
/// Port of difflib's `find_longest_match` + `get_matching_blocks`: longest
/// block per (sub)range, recursing on both sides, adjacent blocks merged.
/// autojunk never fires for our inputs (it only applies to `b` of length ≥
/// 200), and no junk characters are configured, so both are omitted.
#[allow(clippy::needless_range_loop)] // indexes are positions in difflib's algorithm
fn matching_total(a: &[char], b: &[char]) -> usize {
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, ch) in b.iter().enumerate() {
        b2j.entry(*ch).or_default().push(j);
    }
    let mut blocks: Vec<(usize, usize, usize)> = vec![];
    let mut queue: Vec<(usize, usize, usize, usize)> = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let mut besti = alo;
        let mut bestj = blo;
        let mut bestsize = 0usize;
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = b2j.get(&a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    // difflib does `j2len.get(j-1, 0)`; j == 0 simply misses.
                    let k = if j == 0 {
                        1
                    } else {
                        j2len.get(&(j - 1)).copied().unwrap_or(0) + 1
                    };
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        if bestsize > 0 {
            blocks.push((besti, bestj, bestsize));
            queue.push((alo, besti, blo, bestj));
            queue.push((besti + bestsize, ahi, bestj + bestsize, bhi));
        }
    }
    blocks.sort_unstable();

    // Merge adjacent blocks (difflib's non_adjacent pass); the total is the
    // same whether merged or summed, so just sum.
    blocks.iter().map(|(_, _, k)| k).sum()
}

/// `SequenceMatcher(None, a, b).ratio()` — Python difflib's similarity, the
/// name-similarity metric upstream's resolver weights at 0.5. Two empty
/// strings ratio to 1.0 (difflib's `_calculate_ratio` guard).
pub fn sequence_ratio(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * matching_total(&a, &b) as f32 / total as f32
}

// ---------------------------------------------------------------------------
// Word-level gates (upstream `_tokens_match`, `_numbers_in`,
// `_tokens_are_compatible`)
// ---------------------------------------------------------------------------

/// Sequence ratio at/above which two *words* count as the same word.
/// Calibrated upstream on "John Smith" vs "Jane Smith" (john/jane is 0.50)
/// against legitimate word-level differences below it (são/sao 0.67,
/// waler/wall 0.67, arbor/arbour 0.91). Abbreviations (corp/corporation,
/// 0.53) are admitted by the prefix rule instead, not by lowering this.
pub const MIN_TOKEN_SIMILARITY: f32 = 0.6;

/// Whether two words are plausibly the same word — equal, an abbreviation
/// of, or a near-miss.
pub fn tokens_match(a: &str, b: &str) -> bool {
    a == b || a.starts_with(b) || b.starts_with(a) || sequence_ratio(a, b) >= MIN_TOKEN_SIMILARITY
}

/// The numbers in a name ("UA0123" gives {"123"}, "4.0" gives {"4"},
/// "3.10" stays "3.10"). A number names one specific thing, so
/// "Room 101"/"Room 102" are two rooms even though 101/102 is 0.67 by
/// sequence ratio.
pub fn numbers_in(name: &str) -> HashSet<String> {
    let mut numbers = HashSet::new();
    let chars: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // digit run, optionally followed by ('.' + another digit run)
        let mut run: Vec<char> = vec![];
        while i < chars.len() && chars[i].is_ascii_digit() {
            run.push(chars[i]);
            i += 1;
        }
        let mut parts = vec![run.iter().collect::<String>()];
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            i += 1;
            let mut frac: Vec<char> = vec![];
            while i < chars.len() && chars[i].is_ascii_digit() {
                frac.push(chars[i]);
                i += 1;
            }
            parts.push(frac.iter().collect::<String>());
        }
        // strip leading zeros from the integer part; drop an all-zero
        // fractional part (4.0 → "4"), keep anything else verbatim (3.10)
        let int = parts[0].trim_start_matches('0');
        parts[0] = if int.is_empty() {
            "0".to_string()
        } else {
            int.to_string()
        };
        if parts.len() == 2 && parts[1].trim_matches('0').is_empty() {
            parts.pop();
        }
        numbers.insert(parts.join("."));
    }
    numbers
}

/// Whether two multi-word names agree word by word. Whole-name similarity
/// lets one long shared word drown out a completely different short one
/// ("John Smith"/"Jane Smith" is 0.80 by ratio), so every word of the
/// shorter name has to find a counterpart in the longer one. Single-word
/// names are exempt. When each name has a number the other lacks, they are
/// two things.
pub fn tokens_are_compatible(a: &str, b: &str) -> bool {
    let (na, nb) = (numbers_in(a), numbers_in(b));
    if !na.difference(&nb).collect::<HashSet<_>>().is_empty()
        && !nb.difference(&na).collect::<HashSet<_>>().is_empty()
    {
        return false;
    }
    let ta = trgm_words(a);
    let tb = trgm_words(b);
    if ta.len() < 2 && tb.len() < 2 {
        return true;
    }
    let (short, rest) = if ta.len() <= tb.len() {
        (&ta, &tb)
    } else {
        (&tb, &ta)
    };
    short
        .iter()
        .all(|w| rest.iter().any(|o| tokens_match(w, o)))
}

// ---------------------------------------------------------------------------
// Resolver scoring (upstream `_resolve_from_candidates` scoring loop)
// ---------------------------------------------------------------------------

/// Name-similarity weight (upstream: `score += name_similarity * 0.5`).
pub const NAME_SIMILARITY_WEIGHT: f32 = 0.5;
/// Co-occurring-entities weight (upstream: `score += (matched/|nearby|) * 0.3`).
pub const COOCCURRENCE_WEIGHT: f32 = 0.3;
/// Temporal-proximity weight (upstream: `score += temporal_score * 0.2`).
pub const RECENCY_WEIGHT: f32 = 0.2;
/// Recency only votes inside this window (upstream: `days_diff < 7`).
pub const RECENCY_WINDOW_DAYS: f64 = 7.0;
/// Trigram similarity of identical word sets (same words, case/punctuation/
/// decoration differing) — evidence strong enough to score 1.0 on its own.
pub const IDENTICAL_TRIGRAMS: f32 = 1.0;
/// The in-batch dedup pass is O(N^2) over the batch's new names; upstream
/// skips it past this cap with a warning rather than silently degrading.
pub const INTRABATCH_MAX_NAMES: usize = 250;

/// How much one shared co-occurrence partner is worth, damped by how common
/// it is: a partner seen with one other entity keeps its full vote, one seen
/// with 100 is worth 0.1, one seen with 1500 is worth 0.026 (upstream
/// `_cooccurrence_weight`, the #3751 hub protection).
pub fn cooccurrence_weight(degree: usize) -> f32 {
    1.0 / (degree.max(1) as f32).sqrt()
}

#[derive(Debug, Clone, Copy)]
pub struct ResolverConfig {
    /// Trigram-set similarity floor a candidate must clear before scoring
    /// (upstream `merge_min_similarity`). Keeps a name merely *considered*
    /// similar from merging on co-occurrence + recency alone.
    pub merge_min_similarity: f32,
    /// Trigram similarity at/above which two new names in one retain batch
    /// cluster into one entity (upstream `entity_intrabatch_merge_similarity`).
    pub intrabatch_merge_similarity: f32,
    /// Acceptance threshold on the combined score (upstream's unified 0.6,
    /// applied inclusively on a score rounded to 6 decimal places).
    pub link_threshold: f32,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            merge_min_similarity: 0.3,
            intrabatch_merge_similarity: 0.5,
            link_threshold: 0.6,
        }
    }
}

/// Score one candidate entity for one mention, upstream's loop body.
///
/// * `mention_lower` / `candidate_lower` — whitespace-collapsed, lowercased
///   display names (upstream compares `entity_text_lower` vs `canonical_lower`).
/// * `nearby` — lowercased names of the *same fact's* other entities
///   (upstream `nearby_entity_set`).
/// * `partner_names` — the candidate's co-occurring entity names,
///   lowercased (upstream `cooccurrence_map[candidate_id]`, missing = empty).
/// * `degrees` — lowercased entity name → count of that entity's distinct
///   co-occurring partners (upstream `cooccurrence_degrees`; unknown = 1).
/// * `event_ms` / `last_seen_ms` — the fact's effective date and the
///   candidate's last-seen time; recency only votes within 7 days.
///
/// Returns `None` when a gate rejects the candidate (trigram floor,
/// word-by-word compatibility), `Some(1.0)` on identical trigram sets,
/// else the additive 0.5/0.3/0.2 score (uncapped, unthresholded).
#[allow(clippy::too_many_arguments)]
pub fn score_candidate(
    mention_lower: &str,
    candidate_lower: &str,
    nearby: &HashSet<String>,
    partner_names: Option<&HashSet<String>>,
    degrees: &HashMap<String, usize>,
    event_ms: i64,
    last_seen_ms: i64,
    cfg: &ResolverConfig,
) -> Option<f32> {
    let mention_trigrams = trigram_set(mention_lower);
    let candidate_trigrams = trigram_set(candidate_lower);
    let t = trigram_set_similarity(&mention_trigrams, &candidate_trigrams);
    if t < cfg.merge_min_similarity {
        return None;
    }
    if !tokens_are_compatible(mention_lower, candidate_lower) {
        return None;
    }
    if t >= IDENTICAL_TRIGRAMS {
        // Identical trigram sets: the same words, differing only in case,
        // punctuation or decoration. Name evidence that strong stands on
        // its own.
        return Some(1.0);
    }

    let mut score = sequence_ratio(mention_lower, candidate_lower) * NAME_SIMILARITY_WEIGHT;

    // Co-occurring entities, each weighted by how selective it is.
    if !nearby.is_empty() {
        static EMPTY: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
        let empty = EMPTY.get_or_init(HashSet::new);
        let partners = partner_names.unwrap_or(empty);
        let matched_weight: f32 = nearby
            .intersection(partners)
            .map(|name| cooccurrence_weight(degrees.get(name.as_str()).copied().unwrap_or(1)))
            .sum();
        score += (matched_weight / nearby.len() as f32) * COOCCURRENCE_WEIGHT;
    }

    // Temporal proximity, only inside the 7-day window.
    let days = (event_ms - last_seen_ms).abs() as f64 / 86_400_000.0;
    if days < RECENCY_WINDOW_DAYS {
        score += ((1.0 - days / RECENCY_WINDOW_DAYS) as f32) * RECENCY_WEIGHT;
    }

    Some(score)
}

// ---------------------------------------------------------------------------
// In-batch clustering (upstream `_intrabatch_canonical_map` +
// `_find_intrabatch_similar_pairs` + `_cluster_new_entity_names`)
// ---------------------------------------------------------------------------

/// Cluster the names about to be created as new entities so same-batch
/// surface variants (case/emoji/suffix/typo of one name) collapse to a
/// single entity. Returns lowercased name → the cluster's canonical
/// original-case spelling; singletons map to themselves.
///
/// A similar pair must also agree word by word (`tokens_are_compatible`) —
/// trigram similarity alone lets "Dr John Richardson"/"Dr Jane Richardson"
/// (0.65) clear the 0.5 bar. Pairs are enumerated O(N^2); upstream's
/// prefix-filtering join is a documented-identical performance path for the
/// same pair set, and the same 250-name cap applies.
pub fn cluster_new_entity_names(
    names: &[(String, usize)],
    cfg: &ResolverConfig,
) -> HashMap<String, String> {
    // Representative spelling per lowercase name + mention counts.
    let mut rep_by_lower: HashMap<String, String> = HashMap::new();
    let mut count_by_lower: HashMap<String, usize> = HashMap::new();
    for (display, count) in names {
        let lower = display.to_lowercase();
        rep_by_lower
            .entry(lower.clone())
            .or_insert_with(|| display.clone());
        *count_by_lower.entry(lower).or_insert(0) += count;
    }
    if rep_by_lower.len() < 2 {
        return rep_by_lower;
    }
    if rep_by_lower.len() > INTRABATCH_MAX_NAMES {
        eprintln!(
            "rustsight: skipping in-batch entity dedup: {} unique new names exceeds the \
             {} cap (O(N^2) comparison); same-batch surface variants may not be merged",
            rep_by_lower.len(),
            INTRABATCH_MAX_NAMES
        );
        return rep_by_lower;
    }

    let lowers: Vec<String> = rep_by_lower.keys().cloned().collect();
    let trigrams_by_lower: HashMap<String, std::collections::HashSet<String>> =
        lowers.iter().map(|l| (l.clone(), trigram_set(l))).collect();
    let mut pairs: Vec<(String, String)> = vec![];
    for i in 0..lowers.len() {
        for j in (i + 1)..lowers.len() {
            let (a, b) = (&lowers[i], &lowers[j]);
            let sim = trigram_set_similarity(&trigrams_by_lower[a], &trigrams_by_lower[b]);
            if sim >= cfg.intrabatch_merge_similarity && tokens_are_compatible(a, b) {
                pairs.push((a.clone(), b.clone()));
            }
        }
    }

    // Union-find with path halving.
    let mut parent: HashMap<String, String> =
        lowers.iter().map(|l| (l.clone(), l.clone())).collect();
    fn find(parent: &mut HashMap<String, String>, mut x: String) -> String {
        while parent[&x] != x {
            let grandparent = parent[&parent[&x]].clone();
            parent.insert(x.clone(), grandparent.clone());
            x = grandparent;
        }
        x
    }
    for (a, b) in pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    let mut clusters: HashMap<String, Vec<String>> = HashMap::new();
    for lower in &lowers {
        let root = find(&mut parent, lower.clone());
        clusters.entry(root).or_default().push(lower.clone());
    }

    let mut canonical_by_member = HashMap::new();
    for members in clusters.values() {
        // Canonical = most-mentioned, then shortest, then lexicographically
        // smallest — prefers the plainest spelling in the cluster.
        let canonical_lower = members
            .iter()
            .min_by(|a, b| {
                count_by_lower[b.as_str()]
                    .cmp(&count_by_lower[a.as_str()])
                    .then_with(|| {
                        rep_by_lower[a.as_str()]
                            .len()
                            .cmp(&rep_by_lower[b.as_str()].len())
                    })
                    .then_with(|| a.cmp(b))
            })
            .unwrap();
        let canonical_name = rep_by_lower[canonical_lower].clone();
        for m in members {
            canonical_by_member.insert(m.clone(), canonical_name.clone());
        }
    }
    canonical_by_member
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_punctuation() {
        assert_eq!(normalize_name("  Alice ! Smith "), "alice smith");
        assert_eq!(normalize_name("ACME Corp."), "acme corp");
    }

    #[test]
    fn similarity_identical_and_far() {
        assert!((string_similarity("Alice", "alice") - 1.0).abs() < 1e-6);
        assert!(string_similarity("Alice", "Quantum Electrodynamics") < 0.4);
    }

    #[test]
    fn similarity_near_miss_high() {
        let s = string_similarity("Alice Smith", "Alise Smith");
        assert!(s > 0.7, "{s}");
    }

    // --- difflib parity, values generated with python3 difflib ---

    #[test]
    fn sequence_ratio_matches_python_difflib() {
        let cases: &[(&str, &str, f32)] = &[
            ("alice", "alice chen", 0.666667),
            ("john smith", "jane smith", 0.800000),
            ("tigran", "iran", 0.800000),
            ("bob", "robert", 0.444444),
            ("nick", "nicolas", 0.545455),
            ("dr chen", "alice chen", 0.588235),
            ("waler", "wall", 0.666667),
            ("corp", "corporation", 0.533333),
            ("sao", "são", 0.666667),
            ("arbor", "arbour", 0.909091),
            ("alice c", "alice chen", 0.823529),
            ("chen", "alice chen", 0.571429),
            ("ua123", "ua124", 0.800000),
            ("room 101", "room 102", 0.875000),
            ("q3 earnings", "q3 2024 earnings", 0.814815),
            ("acme corp", "acme corporation", 0.720000),
            ("dr waler", "dr wall", 0.800000),
            ("use:use-001", "use:use-002", 0.909091),
            ("topic empathy", "topic:empathy", 0.923077),
        ];
        for (a, b, want) in cases {
            let got = sequence_ratio(a, b);
            assert!(
                (got - want).abs() < 5e-7,
                "ratio({a:?}, {b:?}) = {got}, want {want}"
            );
        }
        assert!((sequence_ratio("alice", "alice") - 1.0).abs() < 1e-9);
        assert!((sequence_ratio("", "") - 1.0).abs() < 1e-9);
    }

    // --- word-level gates (upstream `_tokens_are_compatible` cases) ---

    #[test]
    fn token_compat_rejects_different_first_names() {
        assert!(!tokens_are_compatible("john smith", "jane smith"));
        assert!(!tokens_are_compatible(
            "dr john richardson",
            "dr jane richardson"
        ));
    }

    #[test]
    fn token_compat_admits_subset_and_near_miss() {
        assert!(tokens_are_compatible("alice", "alice chen"));
        assert!(tokens_are_compatible("q3 earnings", "q3 2024 earnings"));
        assert!(tokens_are_compatible("são paulo", "sao paulo"));
        assert!(tokens_are_compatible("acme corp", "acme corporation"));
    }

    #[test]
    fn token_compat_number_rule() {
        assert!(!tokens_are_compatible("room 101", "room 102"));
        assert!(!tokens_are_compatible("ua123", "ua124"));
        // subset numbers fall through to the word check
        assert!(tokens_are_compatible("python", "python 3"));
        assert!(tokens_are_compatible("boeing 737 max", "boeing 737 max 8"));
    }

    #[test]
    fn token_compat_single_word_exempt() {
        // "nick"/"nicolas" is 0.55 by ratio — no long shared word to hide
        // behind, so single-word names take the whole-name check only.
        assert!(tokens_are_compatible("nick", "nicolas"));
    }

    #[test]
    fn numbers_in_normalizes() {
        assert_eq!(
            numbers_in("UA0123"),
            ["123".to_string()].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            numbers_in("4.0"),
            ["4".to_string()].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            numbers_in("3.10"),
            ["3.10".to_string()].into_iter().collect::<HashSet<_>>()
        );
        assert!(numbers_in("no digits").is_empty());
    }

    // --- scoring composition (upstream weights) ---

    fn empty_cfg() -> ResolverConfig {
        ResolverConfig::default()
    }

    #[test]
    fn score_identical_trigrams_is_one() {
        let n = HashSet::new();
        let d = HashMap::new();
        assert_eq!(
            score_candidate("gpt 4", "gpt-4", &n, None, &d, 0, 0, &empty_cfg()),
            Some(1.0)
        );
    }

    #[test]
    fn score_gates_below_trigram_floor() {
        let n = HashSet::new();
        let d = HashMap::new();
        // "tigran"/"iran" trigram 0.20 < 0.3, even though sequence ratio 0.8
        assert_eq!(
            score_candidate("tigran", "iran", &n, None, &d, 0, 0, &empty_cfg()),
            None
        );
    }

    #[test]
    fn score_gates_incompatible_words() {
        let n = HashSet::new();
        let d = HashMap::new();
        assert_eq!(
            score_candidate("john smith", "jane smith", &n, None, &d, 0, 0, &empty_cfg()),
            None
        );
    }

    #[test]
    fn score_alice_vs_alice_chen_needs_corroboration() {
        let d = HashMap::new();
        let empty = HashSet::new();
        let far_past = -90 * 86_400_000i64;
        // No same-fact siblings, no recency: 0.5 * 0.6667 = 0.3333
        let got = score_candidate(
            "alice",
            "alice chen",
            &empty,
            None,
            &d,
            0,
            far_past,
            &empty_cfg(),
        );
        assert!((got.unwrap() - 0.333333).abs() < 5e-7, "{got:?}");
        // 1-day recency: + 0.2 * (1 - 1/7) = 0.1714
        let day = 86_400_000i64;
        let got = score_candidate(
            "alice",
            "alice chen",
            &empty,
            None,
            &d,
            day,
            0,
            &empty_cfg(),
        );
        assert!((got.unwrap() - 0.504762).abs() < 5e-7, "{got:?}");
        // 8 days out: no recency vote
        let got = score_candidate(
            "alice",
            "alice chen",
            &empty,
            None,
            &d,
            8 * day,
            0,
            &empty_cfg(),
        );
        assert!((got.unwrap() - 0.333333).abs() < 5e-7, "{got:?}");
    }

    #[test]
    fn score_cooccurrence_damped_by_degree() {
        // nearby {globex}; globex co-occurs with 3 distinct partners → weight
        // 1/sqrt(3); co = (0.5774/1) * 0.3 = 0.1732
        let mut nearby = HashSet::new();
        nearby.insert("globex".to_string());
        let mut partners = HashSet::new();
        partners.insert("globex".to_string());
        let mut degrees = HashMap::new();
        degrees.insert("globex".to_string(), 3usize);
        let got = score_candidate(
            "alice",
            "alice chen",
            &nearby,
            Some(&partners),
            &degrees,
            0,
            -90 * 86_400_000,
            &empty_cfg(),
        );
        let want = 0.5 * 0.6666667 + (1.0f32 / 3.0f32.sqrt()) * 0.3;
        assert!((got.unwrap() - want).abs() < 5e-7, "{got:?} vs {want}");
    }

    #[test]
    fn cooccurrence_weight_curve() {
        assert!((cooccurrence_weight(1) - 1.0).abs() < 1e-6);
        assert!((cooccurrence_weight(0) - 1.0).abs() < 1e-6);
        assert!((cooccurrence_weight(100) - 0.1).abs() < 1e-6);
        assert!((cooccurrence_weight(1500) - 0.0258).abs() < 1e-4);
    }

    // --- in-batch clustering ---

    #[test]
    fn cluster_folds_decoration_variants() {
        let names = vec![
            ("Wren 🕯️".to_string(), 1),
            ("Wren".to_string(), 3),
            ("Aster".to_string(), 1),
        ];
        let out = cluster_new_entity_names(&names, &empty_cfg());
        assert_eq!(out["wren 🕯️"], "Wren");
        assert_eq!(out["wren"], "Wren"); // most-mentioned spelling wins
        assert_eq!(out["aster"], "Aster"); // singleton maps to itself
    }

    #[test]
    fn cluster_prefers_plainest_spelling_on_tie() {
        let names = vec![
            ("Alice Smith".to_string(), 1),
            ("alice  smith".to_string(), 1),
        ];
        let out = cluster_new_entity_names(&names, &empty_cfg());
        let canonical = out["alice smith"].clone();
        assert_eq!(canonical, "Alice Smith");
    }

    #[test]
    fn cluster_keeps_distinct_names_apart() {
        // "Aster"/"Astrid" is 0.30 by trigram — below the 0.5 bar
        let names = vec![("Aster".to_string(), 1), ("Astrid".to_string(), 1)];
        let out = cluster_new_entity_names(&names, &empty_cfg());
        assert_eq!(out["aster"], "Aster");
        assert_eq!(out["astrid"], "Astrid");
    }

    #[test]
    fn cluster_word_check_blocks_shared_surname() {
        // "Dr John Richardson"/"Dr Jane Richardson" is 0.65 by trigram — over
        // the 0.5 bar — but must stay two people (word check).
        let names = vec![
            ("Dr John Richardson".to_string(), 1),
            ("Dr Jane Richardson".to_string(), 1),
        ];
        let out = cluster_new_entity_names(&names, &empty_cfg());
        assert_eq!(out["dr john richardson"], "Dr John Richardson");
        assert_eq!(out["dr jane richardson"], "Dr Jane Richardson");
    }

    #[test]
    fn typo_variants_cluster() {
        // "Merrivale"/"Merryvale" — the typo case the pass exists for
        let names = vec![("Merrivale".to_string(), 1), ("Merryvale".to_string(), 2)];
        let out = cluster_new_entity_names(&names, &empty_cfg());
        assert_eq!(out["merrivale"], "Merryvale");
        assert_eq!(out["merryvale"], "Merryvale");
    }

    #[test]
    fn collapse_ws_only_touches_whitespace() {
        assert_eq!(collapse_ws(" Alice\n\tChen "), "Alice Chen");
        assert_eq!(collapse_ws("ACME Corp."), "ACME Corp.");
    }
}
