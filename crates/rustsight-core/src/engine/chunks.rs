use crate::entity::string_similarity;
use crate::traits::TokenEstimator;
use crate::types::{CausalRef, ExtractedFact};

/// Facts re-extracted from chunk overlap regions are considered duplicates
/// when their normalized text similarity reaches this threshold.
pub(super) const OVERLAP_DEDUP_THRESHOLD: f32 = 0.9;

/// Split `text` into chunks of at most `size_tokens` (per `est`), carrying an
/// `overlap_tokens` tail of each chunk into the next so facts spanning a cut
/// remain extractable. Content that fits in one chunk is returned whole.
pub(super) fn split_chunks(
    text: &str,
    size_tokens: usize,
    overlap_tokens: usize,
    est: &dyn TokenEstimator,
) -> Vec<String> {
    let size = size_tokens.max(1);
    let overlap = overlap_tokens.min(size.saturating_sub(1));
    let text = text.trim();
    if est.count(text) <= size {
        return vec![text.to_string()];
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let tok = |w: &str| est.count(w).max(1);
    let mut chunks: Vec<String> = vec![];
    let mut current: Vec<&str> = vec![];
    let mut current_tokens = 0usize;
    for w in words {
        let wt = tok(w);
        if current_tokens + wt > size && !current.is_empty() {
            chunks.push(current.join(" "));
            let mut tail: Vec<&str> = vec![];
            let mut tail_tokens = 0usize;
            for t in current.iter().rev() {
                let tt = tok(t);
                if tail_tokens + tt > overlap || tail.len() == current.len() {
                    break;
                }
                tail_tokens += tt;
                tail.push(t);
            }
            current = tail.into_iter().rev().collect();
            current_tokens = tail_tokens;
        }
        current.push(w);
        current_tokens += wt;
    }
    if !current.is_empty() {
        chunks.push(current.join(" "));
    }
    chunks
}

/// Split `text` into two halves at a word boundary near the token midpoint,
/// for the re-ask-at-half-size recovery when extraction output fails to
/// parse. Returns `None` when the text cannot be meaningfully split.
pub(super) fn split_half(text: &str, est: &dyn TokenEstimator) -> Option<(String, String)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 {
        return None;
    }
    let tokens: Vec<usize> = words.iter().map(|w| est.count(w).max(1)).collect();
    let total: usize = tokens.iter().sum();
    let mut acc = 0usize;
    let mut split_at = words.len() / 2;
    for (i, t) in tokens.iter().enumerate() {
        acc += *t;
        if acc * 2 >= total {
            split_at = i + 1;
            break;
        }
    }
    if split_at == 0 || split_at >= words.len() {
        return None;
    }
    Some((words[..split_at].join(" "), words[split_at..].join(" ")))
}

/// Merge per-chunk extraction results into one global fact list:
/// - chunk-local `causal[].fact_index` values are remapped to global indexes,
/// - overlap duplicates (similarity >= `dedup_threshold` against an already
///   merged fact) are dropped, and their causal refs are merged into the kept
///   fact so cross-chunk links survive,
/// - causal refs pointing outside their chunk's fact list are dropped.
pub(super) fn merge_chunk_facts(
    per_chunk: Vec<Vec<ExtractedFact>>,
    dedup_threshold: f32,
) -> Vec<ExtractedFact> {
    let mut global: Vec<ExtractedFact> = vec![];
    for local in per_chunk {
        if local.is_empty() {
            continue;
        }
        // Pass 1: assign each local fact a global index. New facts get
        // sequential slots after `global.len()`; duplicates map to the kept
        // fact's slot so causal refs pointing at them stay correct.
        let mut index_map: Vec<usize> = vec![0; local.len()];
        let mut is_new: Vec<bool> = vec![false; local.len()];
        let mut new_count = 0usize;
        for (i, fact) in local.iter().enumerate() {
            let kept = global
                .iter()
                .position(|g| string_similarity(&fact.text, &g.text) >= dedup_threshold)
                .or_else(|| {
                    (0..i).find(|&j| {
                        is_new[j]
                            && string_similarity(&fact.text, &local[j].text) >= dedup_threshold
                    })
                });
            match kept {
                Some(gi) => index_map[i] = gi,
                None => {
                    index_map[i] = global.len() + new_count;
                    is_new[i] = true;
                    new_count += 1;
                }
            }
        }
        let remap = |causal: &[CausalRef], index_map: &[usize]| -> Vec<CausalRef> {
            causal
                .iter()
                .filter_map(|c| {
                    let target = *index_map.get(c.fact_index)?;
                    Some(CausalRef {
                        fact_index: target,
                        relation: c.relation.clone(),
                    })
                })
                .collect()
        };
        // Pass 2: push new facts in local order (so assigned slots hold),
        // remapping their causal refs to global indexes.
        for (i, fact) in local.iter().enumerate() {
            if !is_new[i] {
                continue;
            }
            let mut fact = fact.clone();
            fact.causal = remap(&fact.causal, &index_map);
            global.push(fact);
        }
        // Pass 3: merge causal refs of dropped duplicates into their kept
        // facts so cross-chunk links declared on the overlap copy survive.
        for (i, fact) in local.iter().enumerate() {
            if is_new[i] {
                continue;
            }
            let gi = index_map[i];
            for c in remap(&fact.causal, &index_map) {
                if !global[gi]
                    .causal
                    .iter()
                    .any(|x| x.fact_index == c.fact_index && x.relation == c.relation)
                {
                    global[gi].causal.push(c);
                }
            }
        }
    }
    global
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntityType, ExtractedEntity, MemoryKind};

    struct WordEstimator;
    impl TokenEstimator for WordEstimator {
        fn count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    fn fact(text: &str, causal: Vec<(usize, &str)>) -> ExtractedFact {
        ExtractedFact {
            text: text.into(),
            fact_type: MemoryKind::World,
            entities: vec![ExtractedEntity {
                name: "X".into(),
                etype: EntityType::Other,
            }],
            occurred_start_ms: None,
            occurred_end_ms: None,
            causal: causal
                .into_iter()
                .map(|(fact_index, relation)| CausalRef {
                    fact_index,
                    relation: relation.into(),
                })
                .collect(),
            chunk_index: 0,
            labels: None,
            tags: vec![],
        }
    }

    #[test]
    fn small_content_is_a_single_chunk() {
        let chunks = split_chunks("a b c", 10, 2, &WordEstimator);
        assert_eq!(chunks, vec!["a b c".to_string()]);
    }

    #[test]
    fn splits_at_token_limit_with_overlap() {
        let text = "one two three four five six";
        let chunks = split_chunks(text, 4, 1, &WordEstimator);
        assert_eq!(
            chunks,
            vec![
                "one two three four".to_string(),
                "four five six".to_string()
            ]
        );
    }

    #[test]
    fn every_chunk_respects_the_token_limit() {
        let text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11";
        let chunks = split_chunks(text, 5, 2, &WordEstimator);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(WordEstimator.count(c) <= 5, "chunk too big: {c:?}");
        }
        assert!(chunks.last().unwrap().contains("w11"));
        for pair in chunks.windows(2) {
            // overlap: the next chunk repeats the previous chunk's tail words
            let prev_last = pair[0].split_whitespace().last().unwrap();
            assert!(
                pair[1].split_whitespace().any(|w| w == prev_last),
                "{:?} misses overlap with {:?}",
                pair[1],
                pair[0]
            );
        }
    }

    #[test]
    fn single_oversized_word_becomes_its_own_chunk() {
        struct CharEstimator;
        impl TokenEstimator for CharEstimator {
            fn count(&self, text: &str) -> usize {
                text.chars().count() / 4 + 1
            }
        }
        let chunks = split_chunks("aaaaaaaaaaaaaaaaaaaa b", 3, 1, &CharEstimator);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("aaaaaaaaaaaaaaaaaaaa"));
        assert_eq!(chunks[1], "b");
    }

    #[test]
    fn halves_split_near_the_middle() {
        let (a, b) = split_half("a b c d", &WordEstimator).unwrap();
        assert_eq!(a, "a b");
        assert_eq!(b, "c d");
    }

    #[test]
    fn single_word_cannot_be_halved() {
        assert!(split_half("solo", &WordEstimator).is_none());
    }

    #[test]
    fn merge_remaps_causal_indexes_across_chunks() {
        let per_chunk = vec![
            vec![
                fact("Alice joined Google.", vec![]),
                fact("Alice chose the SRE team.", vec![]),
            ],
            vec![
                fact("Alice moved to Zurich.", vec![]),
                fact("Alice moved for the team.", vec![(0, "causes")]),
            ],
        ];
        let merged = merge_chunk_facts(per_chunk, OVERLAP_DEDUP_THRESHOLD);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[3].causal.len(), 1);
        assert_eq!(merged[3].causal[0].fact_index, 2, "{merged:?}");
    }

    #[test]
    fn merge_drops_overlap_duplicates_and_keeps_links() {
        let per_chunk = vec![
            vec![
                fact("Bob joined Acme.", vec![]),
                fact("Bob works on Figma.", vec![]),
            ],
            vec![
                // overlap re-extraction of chunk 1's second fact
                fact("Bob works on Figma!", vec![(1, "enables")]),
                fact("Bob redesigned the Acme site.", vec![(0, "causes")]),
            ],
        ];
        let merged = merge_chunk_facts(per_chunk, OVERLAP_DEDUP_THRESHOLD);
        assert_eq!(merged.len(), 3, "{merged:?}");
        assert_eq!(merged[2].text, "Bob redesigned the Acme site.");
        // causal on the duplicate fact was merged into the kept fact
        assert!(
            merged[1]
                .causal
                .iter()
                .any(|c| c.fact_index == 2 && c.relation == "enables"),
            "{merged:?}"
        );
        // causal pointing at the duplicate remapped to the kept fact
        assert_eq!(merged[2].causal[0].fact_index, 1);
    }

    #[test]
    fn merge_dedupes_within_a_single_chunk() {
        let per_chunk = vec![vec![
            fact("Cara leads design.", vec![]),
            fact("Cara leads design.", vec![]),
        ]];
        let merged = merge_chunk_facts(per_chunk, OVERLAP_DEDUP_THRESHOLD);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_drops_out_of_range_causal_refs() {
        let per_chunk = vec![vec![fact("Dana moved to Austin.", vec![(7, "causes")])]];
        let merged = merge_chunk_facts(per_chunk, OVERLAP_DEDUP_THRESHOLD);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].causal.is_empty());
    }
}
