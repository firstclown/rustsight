//! Embedding rustsight in a Rust application.
//!
//! Offline (mock providers, no API key):
//!     cargo run -p rustsight-cli --example embed
//!
//! Against a real OpenAI-compatible provider:
//!     OPENAI_API_KEY=sk-... cargo run -p rustsight-cli --example embed
//!
//! The README's "Use it as a library" section mirrors this file.

use std::sync::Arc;
use std::time::Duration;

use rustsight_core::{
    EmbedProvider, Engine, LlmProvider, RecallRequest, RerankProvider, RetainRequest, SystemClock,
};
use rustsight_providers::{
    http_client, MockEmbed, MockLlm, MockRerank, NoopRerank, OpenAiCompat, OpenAiEmbed,
    SimpleTokenEstimator,
};
use rustsight_storage::SqliteStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Storage: one SQLite file (use `open_in_memory()` for tests).
    let clock = Arc::new(SystemClock);
    let db = std::env::temp_dir().join("rustsight-embed-example.db");
    let store = Arc::new(SqliteStore::open_with_clock(&db, clock.clone())?);

    // 2. Providers: any OpenAI-compatible chat + embeddings endpoint.
    let (llm, embed, rerank): (
        Arc<dyn LlmProvider>,
        Arc<dyn EmbedProvider>,
        Arc<dyn RerankProvider>,
    ) = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => {
            let client = http_client()?;
            let base_url = "https://api.openai.com/v1".to_string();
            (
                Arc::new(OpenAiCompat {
                    client: client.clone(),
                    base_url: base_url.clone(),
                    api_key: key.clone(),
                    model: "gpt-4o-mini".into(),
                }),
                Arc::new(OpenAiEmbed {
                    client,
                    base_url,
                    api_key: key,
                    model: "text-embedding-3-small".into(),
                    dim_cache: std::sync::Mutex::new(1536),
                }),
                // Swap in `ApiRerank` or the `onnx-rerank` feature's
                // `OrtRerank` for better recall quality.
                Arc::new(NoopRerank),
            )
        }
        _ => (
            Arc::new(MockLlm::demo()),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
        ),
    };

    // 3. The engine, plus the background consolidation worker.
    let engine = Arc::new(Engine::new(
        store,
        llm,
        embed,
        rerank,
        Arc::new(SimpleTokenEstimator),
        clock,
    ));
    let worker = engine.spawn_worker(Duration::from_secs(5), 50);

    // 4. Retain, recall, reflect.
    engine
        .retain(RetainRequest {
            bank_id: "demo".into(),
            content: "Alice joined Google last week. She works on search infrastructure.".into(),
            ..Default::default()
        })
        .await?;

    let recalled = engine
        .recall(RecallRequest {
            bank_id: "demo".into(),
            query: "Where does Alice work?".into(),
            token_budget: 1000,
            top_k_per_arm: 50,
            ..Default::default()
        })
        .await?;
    for m in &recalled.memories {
        println!("[{:?}] {:.3}  {}", m.kind, m.score, m.text);
    }

    let reflected = engine
        .reflect("demo", "What should I know about Alice?")
        .await?;
    println!("\n{}", reflected.answer);

    worker.abort();
    let _ = worker.await;
    Ok(())
}
