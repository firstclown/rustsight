use crate::engine::Engine;
use crate::types::*;
use crate::{apply_token_budget, bm25_terms, cosine, parse_window, rrf_fuse, Error, Result};

pub(super) async fn recall(e: &Engine, req: RecallRequest) -> Result<RecallResponse> {
    let now = e.clock.now_ms();
    let window = parse_window(&req.query, now);
    let qemb = e
        .embed
        .embed(vec![req.query.clone()])
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Embed("no embedding returned".into()))?;

    let kinds = req.kinds.clone().unwrap_or_else(|| {
        vec![
            MemoryKind::World,
            MemoryKind::Experience,
            MemoryKind::Observation,
            MemoryKind::Opinion,
        ]
    });
    let k = req.top_k_per_arm.max(1);

    let mut sem: Vec<String> = vec![];
    {
        let all = e.store.all_embeddings(&req.bank_id)?;
        let mut scored: Vec<(f32, String)> = all
            .into_iter()
            .filter(|(_, kind, _)| kinds.contains(kind))
            .map(|(id, _, v)| (cosine(&qemb, &v), id))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        scored.truncate(k);
        sem = scored.into_iter().map(|(_, id)| id).collect();
    }

    let mut bm: Vec<String> = vec![];
    {
        let terms = bm25_terms(&req.query);
        if !terms.is_empty() {
            let hits = e.store.search_bm25(&req.bank_id, &terms, k * 2)?;
            let mut s: Vec<(f32, String)> = hits
                .into_iter()
                .filter(|(_, kind, _)| kinds.contains(kind))
                .map(|(id, _, sc)| (sc, id))
                .collect();
            s.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            s.truncate(k);
            bm = s.into_iter().map(|(_, id)| id).collect();
        }
    }

    let mut graph: Vec<String> = vec![];
    {
        let mut seeds: Vec<String> = sem.iter().chain(bm.iter()).cloned().collect();
        seeds.truncate(5);
        if !seeds.is_empty() {
            let n = e.store.neighbors(&req.bank_id, &seeds, 2, k)?;
            graph = n
                .into_iter()
                .filter(|id| !sem.contains(id) && !bm.contains(id))
                .collect();
        }
    }

    let mut temp_arm: Vec<String> = vec![];
    if let Some((s, t)) = window {
        let ms = e.store.memories_in_window(&req.bank_id, s, t, k * 3)?;
        temp_arm = ms
            .into_iter()
            .filter(|m| kinds.contains(&m.kind))
            .map(|m| m.id)
            .take(k)
            .collect();
    }

    let mut used: Vec<String> = vec![];
    if !sem.is_empty() {
        used.push("semantic".into());
    }
    if !bm.is_empty() {
        used.push("keyword".into());
    }
    if !graph.is_empty() {
        used.push("graph".into());
    }
    if !temp_arm.is_empty() {
        used.push("temporal".into());
    }

    let fused = rrf_fuse(&[sem, bm, graph, temp_arm], e.config.rrf_k);
    let ids: Vec<String> = fused.iter().map(|(id, _)| id.clone()).collect();
    if ids.is_empty() {
        return Ok(RecallResponse {
            memories: vec![],
            window,
            strategies_used: used,
        });
    }

    let mut refs = e.store.get_refs(&req.bank_id, &ids)?;
    for (r, (_, f)) in refs.iter_mut().zip(&fused) {
        r.score = *f;
    }

    let docs: Vec<String> = refs.iter().map(|r| r.text.clone()).collect();
    let rr = e
        .rerank
        .rerank(&req.query, &docs, refs.len().min(50))
        .await?;
    if !rr.is_empty() {
        let mut scored: Vec<(f32, MemoryRef)> = rr
            .into_iter()
            .filter_map(|(i, s)| {
                refs.get(i).cloned().map(|mut r| {
                    r.score = s;
                    (s, r)
                })
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.id.cmp(&b.1.id))
        });
        refs = scored.into_iter().map(|(_, r)| r).collect();
    }

    let memories = apply_token_budget(refs, req.token_budget.max(1), e.tokens.as_ref());
    Ok(RecallResponse {
        memories,
        window,
        strategies_used: used,
    })
}
