//! pg_trgm-compatible trigram similarity, computed in memory.
//!
//! Port of upstream `hindsight_api/engine/trigram.py` (tag v1.1.0), verified
//! by upstream against PostgreSQL pg_trgm across emoji / accent / CJK /
//! hyphen / apostrophe cases (issue #3107) so the merge cutoffs calibrated on
//! pg_trgm transfer exactly. Shared by entity resolution only today, but one
//! notion of "similar name" governs everything that compares names.

use std::collections::HashSet;

/// A pg_trgm "word" is a maximal run of alphanumerics (Unicode letters and
/// digits; underscore is a separator in pg_trgm's tokenizer). Everything else
/// — spaces, punctuation, emoji — separates words, which is why decoration
/// variants like "Wren 🕯️" collapse to the same trigram set as "Wren".
fn is_trgm_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

/// Trigrams of `text` the way pg_trgm generates them: lowercase, split into
/// words, pad each word with two leading + one trailing blank, and take every
/// 3-character window.
pub fn trigram_set(text: &str) -> HashSet<String> {
    let mut trigrams = HashSet::new();
    let mut word = String::new();
    let flush = |word: &str, trigrams: &mut HashSet<String>| {
        if word.is_empty() {
            return;
        }
        let padded: Vec<char> = format!("  {word} ").chars().collect();
        for i in 0..padded.len().saturating_sub(2) {
            let g: String = padded[i..i + 3].iter().collect();
            trigrams.insert(g);
        }
    };
    for c in text.to_lowercase().chars() {
        if is_trgm_word_char(c) {
            word.push(c);
        } else {
            flush(&word, &mut trigrams);
            word.clear();
        }
    }
    flush(&word, &mut trigrams);
    trigrams
}

/// Jaccard index of two already-computed trigram sets.
pub fn trigram_set_similarity(ta: &HashSet<String>, tb: &HashSet<String>) -> f32 {
    let intersection = ta.intersection(tb).count();
    let union = ta.len() + tb.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// pg_trgm `similarity(a, b)` computed in memory — the Jaccard index of the
/// trigram sets.
pub fn trigram_similarity(a: &str, b: &str) -> f32 {
    trigram_set_similarity(&trigram_set(a), &trigram_set(b))
}

/// The words of a name the way the trigram tokenizer sees them (lowercased
/// alphanumeric runs) — shared by the word-by-word compatibility gate.
pub fn trgm_words(text: &str) -> Vec<String> {
    let mut words = vec![];
    let mut word = String::new();
    for c in text.to_lowercase().chars() {
        if is_trgm_word_char(c) {
            word.push(c);
        } else if !word.is_empty() {
            words.push(std::mem::take(&mut word));
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_words_differing_only_in_decoration() {
        // "Wren 🕯️" and "Wren" are the same words with different separators,
        // so their trigram sets — and their similarity — are identical.
        assert_eq!(trigram_set("Wren 🕯️"), trigram_set("Wren"));
        assert!((trigram_similarity("Wren 🕯️", "Wren") - 1.0).abs() < 1e-6);
        assert!((trigram_similarity("GPT-4", "GPT 4") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn alice_vs_alice_chen_similarity() {
        // trigram_set("alice") ⊂ trigram_set("alice chen"): 6 shared of 11
        // total ("alice" pads to 6 trigrams, "chen" to 5).
        let s = trigram_similarity("alice", "alice chen");
        assert!((s - 6.0 / 11.0).abs() < 1e-6, "{s}");
    }

    #[test]
    fn tigran_vs_iran_ranks_below_floor() {
        // Upstream's calibration case: a coincidental shared run inflates the
        // sequence ratio ("Tigran"/"Iran" is 0.80 by SequenceMatcher) but the
        // trigram metric that gates candidates sees them apart.
        let s = trigram_similarity("tigran", "iran");
        assert!(s < 0.3, "{s}");
    }

    #[test]
    fn john_smith_vs_jane_smith() {
        // Whole-name similarity lets one long shared word ("smith") drown out
        // a completely different short one ("john" vs "jane").
        let s = trigram_similarity("john smith", "jane smith");
        assert!(s < 0.5, "{s}");
    }

    #[test]
    fn empty_and_punctuation_only() {
        assert!(trigram_set("!!!").is_empty());
        assert_eq!(trigram_set(""), HashSet::new());
        assert_eq!(trigram_similarity("!!!", "??"), 0.0);
    }

    #[test]
    fn words_split_on_non_alphanumerics() {
        assert_eq!(trgm_words("Dr. Chen"), vec!["dr", "chen"]);
        assert_eq!(trgm_words("alice chen"), vec!["alice", "chen"]);
        assert_eq!(trgm_words("são-paulo"), vec!["são", "paulo"]);
    }

    #[test]
    fn single_letter_words_produce_two_trigrams() {
        // "  a " has 4 chars → windows [0..3] and [1..4].
        assert_eq!(trigram_set("a").len(), 2);
    }
}
