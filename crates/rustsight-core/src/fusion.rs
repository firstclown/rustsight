use crate::traits::TokenEstimator;
use crate::types::MemoryRef;
use std::collections::HashMap;

pub fn rrf_fuse(lists: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<&str, f32> = HashMap::new();
    for list in lists {
        for (i, id) in list.iter().enumerate() {
            *scores.entry(id.as_str()).or_insert(0.0) += 1.0 / (k + i as f32 + 1.0);
        }
    }
    let mut v: Vec<(String, f32)> = scores
        .into_iter()
        .map(|(s, f)| (s.to_string(), f))
        .collect();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    v
}

/// BM25 query terms: lowercased words, plus character bigrams for CJK runs
/// (indexed the same way, `lang::cjk_segment`). Capped at 12 terms, 24 when
/// the query has CJK text — a CJK sentence yields a bigram per character.
pub fn bm25_terms(query: &str) -> Vec<String> {
    let mut out = crate::lang::search_terms(query);
    out.truncate(if crate::lang::has_cjk(query) { 24 } else { 12 });
    out
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

pub fn apply_token_budget(
    items: Vec<MemoryRef>,
    budget: usize,
    est: &dyn TokenEstimator,
) -> Vec<MemoryRef> {
    let mut used = 0usize;
    let mut out = vec![];
    for it in items {
        let cost = est.count(&it.text) + 4;
        if used + cost > budget {
            continue;
        }
        used += cost;
        out.push(it);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: &str, text: &str) -> MemoryRef {
        MemoryRef {
            id: id.into(),
            kind: crate::types::MemoryKind::World,
            text: text.into(),
            score: 0.0,
            occurred_start_ms: None,
            occurred_end_ms: None,
            mentioned_at_ms: None,
            entity_ids: vec![],
            proof_count: None,
            context: None,
            document_id: None,
            tags: vec![],
            metadata: None,
        }
    }

    struct Div4;
    impl TokenEstimator for Div4 {
        fn count(&self, text: &str) -> usize {
            text.len() / 4 + 1
        }
    }

    #[test]
    fn rrf_prefers_multi_list_agreement() {
        let a = vec!["x".into(), "y".into()];
        let b = vec!["z".into(), "x".into()];
        let fused = rrf_fuse(&[a, b], 60.0);
        assert_eq!(fused[0].0, "x");
    }

    #[test]
    fn rrf_empty() {
        assert!(rrf_fuse(&[vec![], vec![]], 60.0).is_empty());
    }

    #[test]
    fn budget_cuts_greedily() {
        let items = vec![mem("a", &"x".repeat(400)), mem("b", &"y".repeat(20))];
        let out = apply_token_budget(items, 50, &Div4);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "b");
        let items2 = vec![mem("a", &"y".repeat(20)), mem("b", &"x".repeat(400))];
        let out2 = apply_token_budget(items2, 50, &Div4);
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].id, "a");
    }

    #[test]
    fn budget_skips_oversized_and_fills() {
        let items = vec![
            mem("a", &"x".repeat(400)),
            mem("b", &"y".repeat(20)),
            mem("c", &"z".repeat(400)),
            mem("d", &"w".repeat(20)),
        ];
        let out = apply_token_budget(items, 50, &Div4);
        let ids: Vec<&str> = out.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "d"]);
    }

    #[test]
    fn bm25_terms_filters_and_dedupes() {
        let t = bm25_terms("The the ALICE, alice-of-Google!");
        assert!(t.contains(&"alice".to_string()));
        assert!(t.contains(&"google".to_string()));
        assert_eq!(
            t.iter().filter(|x| x == &"the").count(),
            1,
            "duplicates must collapse: {t:?}"
        );
        assert!(t.len() <= 12);
    }

    #[test]
    fn cosine_sanity() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[], &[1.0]), 0.0);
    }
}
