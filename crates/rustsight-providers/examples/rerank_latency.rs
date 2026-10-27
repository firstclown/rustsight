//! Measures how much latency the local ONNX cross-encoder reranker adds to
//! recall. Recall caps rerank input at `min(50, refs.len())` today, so the
//! 50-doc number is what a recall response pays; 100 docs shows headroom if
//! that cap grows (RS-4 tracking).
//!
//! Build & run:
//! ```text
//! cargo run -p rustsight-providers --features onnx-rerank --example rerank_latency
//! ```
//! Downloads the model on first run (cached under ~/.rustsight/models).

use rustsight_core::traits::RerankProvider;
use rustsight_providers::onnx::{OrtRerank, OrtRerankSpec};
use std::time::Instant;

const ROUNDS: usize = 20;

fn percentile(mut samples: Vec<f64>, p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (samples.len() - 1) as f64).round() as usize;
    samples[idx.min(samples.len() - 1)]
}

async fn measure(rerank: &OrtRerank, docs: &[String], query: &str) -> Vec<f64> {
    // warmup
    let _ = rerank.rerank(query, docs, docs.len()).await.unwrap();
    let mut times = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t = Instant::now();
        let out = rerank.rerank(query, docs, docs.len()).await.unwrap();
        let elapsed = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), docs.len());
        times.push(elapsed * 1000.0);
    }
    times
}

#[tokio::main]
async fn main() {
    let env = rustsight_providers::ProviderEnv::from_env().expect("bad provider config");
    let spec = OrtRerankSpec::from_provider_env(&env);
    eprintln!(
        "loading reranker (batch_size={}, device={:?}, model={})...",
        spec.batch_size, spec.device, spec.repo_file
    );
    let rerank = OrtRerank::load(spec).await.expect("model load failed");

    // ~40-word synthetic docs, vaguely memory-shaped
    let template = |i: usize| {
        format!(
            "Memory record {i}: the agent observed that session {i} covered project planning, \
             tooling choices, and a long discussion about release schedules. Participants \
             disagreed on deadlines but agreed to revisit the roadmap after the next milestone \
             review. Notes were stored with entity tags and temporal markers for later recall."
        )
    };
    let docs_100: Vec<String> = (0..100).map(template).collect();
    let query = "What did the team decide about the release schedule?";

    for &n in &[50usize, 100] {
        let docs: Vec<String> = docs_100[..n].to_vec();
        let times = measure(&rerank, &docs, query).await;
        println!(
            "{n:>3} docs: p50 {:>7.1} ms   p95 {:>7.1} ms   min {:>7.1} ms   max {:>7.1} ms",
            percentile(times.clone(), 50.0),
            percentile(times.clone(), 95.0),
            times.iter().cloned().fold(f64::INFINITY, f64::min),
            times.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }
}
