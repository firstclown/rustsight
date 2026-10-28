use crate::engine::{with_retry, Engine};
use crate::types::*;
use crate::{
    apply_token_budget, bm25_terms, cosine, parse_window, rrf_fuse, Error, MemoryStore, Result,
};
use std::sync::Arc;

pub(super) async fn recall(e: &Engine, req: RecallRequest) -> Result<RecallResponse> {
    let now = e.clock.now_ms();
    let window = parse_window(&req.query, now);
    let embed = Arc::clone(&e.embed);
    let qemb = with_retry(e, || embed.embed(vec![req.query.clone()]))
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

    // The four retrieval arms are independent SQLite scans. Run them
    // concurrently on the blocking pool (never inline on the async runtime),
    // honoring the one dependency: the graph arm seeds from the semantic +
    // BM25 results. Join those two first, then graph + temporal together.
    // Arm behavior (filters, truncation, `strategies_used` assembly) is
    // unchanged — this is purely an execution-order change.
    let (sem, bm) = tokio::join!(
        arm_semantic(e, &req.bank_id, &qemb, &kinds, k),
        arm_bm25(e, &req.bank_id, &req.query, &kinds, k),
    );
    let sem = sem?;
    let bm = bm?;
    let (graph, temp_arm) = tokio::join!(
        arm_graph(e, &req.bank_id, &sem, &bm, k),
        arm_temporal(e, &req.bank_id, window, &kinds, k),
    );
    let graph = graph?;
    let temp_arm = temp_arm?;

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

    let mut refs =
        blocking_store(&e.store, move |store| store.get_refs(&req.bank_id, &ids)).await?;
    for (r, (_, f)) in refs.iter_mut().zip(&fused) {
        r.score = *f;
    }

    let docs: Vec<String> = refs.iter().map(|r| r.text.clone()).collect();
    let rerank = Arc::clone(&e.rerank);
    let query = req.query.clone();
    let top_k = refs.len().min(50);
    let rr = with_retry(e, || rerank.rerank(&query, &docs, top_k)).await?;
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

/// Run one short synchronous store query on tokio's blocking pool so the
/// sync SQLite store never stalls the async runtime's workers.
async fn blocking_store<T, F>(store: &Arc<dyn MemoryStore>, f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce(&dyn MemoryStore) -> Result<T> + Send + 'static,
{
    let store = Arc::clone(store);
    match tokio::task::spawn_blocking(move || f(store.as_ref())).await {
        Ok(result) => result,
        Err(join_err) => Err(Error::Store(format!(
            "recall store task failed: {join_err}"
        ))),
    }
}

async fn arm_semantic(
    e: &Engine,
    bank_id: &str,
    qemb: &[f32],
    kinds: &[MemoryKind],
    k: usize,
) -> Result<Vec<String>> {
    let bank_id = bank_id.to_string();
    let qemb = qemb.to_vec();
    let kinds = kinds.to_vec();
    blocking_store(&e.store, move |store| {
        let all = store.all_embeddings(&bank_id)?;
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
        Ok(scored.into_iter().map(|(_, id)| id).collect())
    })
    .await
}

async fn arm_bm25(
    e: &Engine,
    bank_id: &str,
    query: &str,
    kinds: &[MemoryKind],
    k: usize,
) -> Result<Vec<String>> {
    let bank_id = bank_id.to_string();
    let query = query.to_string();
    let kinds = kinds.to_vec();
    blocking_store(&e.store, move |store| {
        let terms = bm25_terms(&query);
        if terms.is_empty() {
            return Ok(vec![]);
        }
        let hits = store.search_bm25(&bank_id, &terms, k * 2)?;
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
        Ok(s.into_iter().map(|(_, id)| id).collect())
    })
    .await
}

async fn arm_graph(
    e: &Engine,
    bank_id: &str,
    sem: &[String],
    bm: &[String],
    k: usize,
) -> Result<Vec<String>> {
    let mut seeds: Vec<String> = sem.iter().chain(bm.iter()).cloned().collect();
    seeds.truncate(5);
    if seeds.is_empty() {
        return Ok(vec![]);
    }
    let bank_id = bank_id.to_string();
    let sem = sem.to_vec();
    let bm = bm.to_vec();
    blocking_store(&e.store, move |store| {
        let n = store.neighbors(&bank_id, &seeds, 2, k)?;
        Ok(n.into_iter()
            .filter(|id| !sem.contains(id) && !bm.contains(id))
            .collect())
    })
    .await
}

async fn arm_temporal(
    e: &Engine,
    bank_id: &str,
    window: Option<(i64, i64)>,
    kinds: &[MemoryKind],
    k: usize,
) -> Result<Vec<String>> {
    let Some((s, t)) = window else {
        return Ok(vec![]);
    };
    let bank_id = bank_id.to_string();
    let kinds = kinds.to_vec();
    blocking_store(&e.store, move |store| {
        let ms = store.memories_in_window(&bank_id, s, t, k * 3)?;
        Ok(ms
            .into_iter()
            .filter(|m| kinds.contains(&m.kind))
            .map(|m| m.id)
            .take(k)
            .collect())
    })
    .await
}
