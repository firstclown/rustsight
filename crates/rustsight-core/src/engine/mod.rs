mod chunks;
mod consolidate;
mod mental;
mod recall;
mod reflect;
mod retain;

use crate::prompts::extraction::ExtractionConfig;
use crate::traits::{Clock, EmbedProvider, LlmProvider, RerankProvider, TokenEstimator};
use crate::types::*;
use crate::{Error, MemoryStore, ResolverConfig, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub rrf_k: f32,
    pub semantic_edge_threshold: f32,
    pub temporal_sigma_ms: f64,
    pub temporal_edge_min_weight: f32,
    pub entity: ResolverConfig,
    pub entity_backfill: usize,
    pub semantic_edge_cap: usize,
    pub extract: ExtractionConfig,
    pub reflect_max_rounds: usize,
    /// Target size of retain content chunks, measured with the configured
    /// `TokenEstimator`. Content at or below this size is extracted in one
    /// request.
    pub chunk_size_tokens: usize,
    /// Token overlap between consecutive retain chunks; the overlap usually
    /// re-extracts boundary facts, which are deduped on merge.
    pub chunk_overlap_tokens: usize,
    /// Total attempts for transient (HTTP 429/5xx) provider failures.
    pub retry_max_attempts: usize,
    /// Backoff base for retries; attempt N waits `base * 2^(N-1)` ms.
    pub retry_base_delay_ms: u64,
    /// Hard wall (seconds) for one consolidation pass. A stalled provider
    /// call inside a pass would otherwise hold the single-flight
    /// consolidation lock forever, wedging background consolidation for
    /// every bank. Timed-out passes abort with an error; dirty rows
    /// survive (cleared only after success) and are retried next tick.
    pub consolidate_timeout_secs: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            semantic_edge_threshold: 0.82,
            temporal_sigma_ms: 7.0 * 86_400_000.0,
            temporal_edge_min_weight: 0.25,
            entity: ResolverConfig::default(),
            entity_backfill: 20,
            semantic_edge_cap: 10,
            extract: ExtractionConfig::default(),
            reflect_max_rounds: 8,
            chunk_size_tokens: 3000,
            chunk_overlap_tokens: 200,
            retry_max_attempts: 3,
            retry_base_delay_ms: 400,
            consolidate_timeout_secs: 600,
        }
    }
}

pub struct Engine {
    pub store: Arc<dyn MemoryStore>,
    pub llm: Arc<dyn LlmProvider>,
    pub embed: Arc<dyn EmbedProvider>,
    pub rerank: Arc<dyn RerankProvider>,
    pub tokens: Arc<dyn TokenEstimator>,
    pub clock: Arc<dyn Clock>,
    pub config: EngineConfig,
    /// Wakes the background worker as soon as retain marks entities dirty.
    pub(crate) wakeup: Notify,
    /// Single-flight guard: serializes consolidation passes (worker vs
    /// explicit `consolidate_once` calls) so dirty entities are never
    /// double-processed.
    pub(crate) consolidate_lock: Mutex<()>,
}

impl Engine {
    pub fn new(
        store: Arc<dyn MemoryStore>,
        llm: Arc<dyn LlmProvider>,
        embed: Arc<dyn EmbedProvider>,
        rerank: Arc<dyn RerankProvider>,
        tokens: Arc<dyn TokenEstimator>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            store,
            llm,
            embed,
            rerank,
            tokens,
            clock,
            config: EngineConfig::default(),
            wakeup: Notify::new(),
            consolidate_lock: Mutex::new(()),
        }
    }

    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn retain(&self, req: RetainRequest) -> Result<RetainResponse> {
        retain::retain(self, req).await
    }

    pub async fn recall(&self, req: RecallRequest) -> Result<RecallResponse> {
        recall::recall(self, req).await
    }

    pub async fn reflect(&self, bank_id: &str, query: &str) -> Result<ReflectResponse> {
        reflect::reflect(self, bank_id, query).await
    }

    /// Consolidate up to `max_entities` dirty entities in one bank.
    ///
    /// Serialized by an engine-level single-flight guard: concurrent callers
    /// (background worker, HTTP endpoint, CLI) queue up instead of
    /// double-processing the same dirty entities. Dirty rows are only cleared
    /// at the end of a successful pass, so a caller that arrives after a
    /// completed pass simply finds an empty queue and returns `0`.
    ///
    /// The pass itself is bounded by `consolidate_timeout_secs`; a timed-out
    /// pass drops its in-flight work (dirty rows survive, retried next tick)
    /// so a wedged provider can never hold the single-flight lock forever.
    pub async fn consolidate_once(&self, bank_id: &str, max_entities: usize) -> Result<usize> {
        let _guard = self.consolidate_lock.lock().await;
        let pass = tokio::time::timeout(
            Duration::from_secs(self.config.consolidate_timeout_secs.max(1)),
            consolidate::consolidate_once(self, bank_id, max_entities),
        );
        match pass.await {
            Ok(result) => result,
            Err(_elapsed) => Err(Error::Llm(format!(
                "consolidation pass for bank {bank_id} timed out after {}s",
                self.config.consolidate_timeout_secs
            ))),
        }
    }

    /// Wake the background worker without blocking (used by `retain`).
    pub fn wake_worker(&self) {
        self.wakeup.notify_one();
    }

    /// Spawn the background consolidation worker on the current tokio
    /// runtime. Long-running entry points (`serve`, the MCP server, desktop
    /// embedders) should spawn it once and keep the returned handle; aborting
    /// the task stops the worker.
    ///
    /// The loop wakes either when `interval` elapses or when a `retain`
    /// notifies it (debounced by a short window so bursts of retains
    /// coalesce into one pass), then runs `consolidate_once` for every bank
    /// with a non-empty dirty queue. Errors are logged and retried on the
    /// next tick; the store lock is only held for short queries between LLM
    /// calls, so retain/recall are never blocked.
    pub fn spawn_worker(
        self: &Arc<Self>,
        interval: Duration,
        max_entities: usize,
    ) -> JoinHandle<()> {
        let engine = Arc::clone(self);
        tokio::spawn(async move {
            let debounce =
                (interval / 4).clamp(Duration::from_millis(25), Duration::from_millis(250));
            loop {
                tokio::select! {
                    _ = engine.wakeup.notified() => {}
                    _ = tokio::time::sleep(interval) => {}
                }
                // Debounce: let a burst of retains land before one pass.
                tokio::time::sleep(debounce).await;
                match engine.store.banks_with_dirty() {
                    Ok(banks) => {
                        for bank_id in banks {
                            if let Err(err) = engine.consolidate_once(&bank_id, max_entities).await
                            {
                                eprintln!(
                                    "rustsight worker: consolidating bank {bank_id} failed: {err}"
                                );
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("rustsight worker: listing dirty banks failed: {err}");
                    }
                }
            }
        })
    }

    pub async fn refresh_mental_model(
        &self,
        bank_id: &str,
        question: &str,
        path: Option<&str>,
    ) -> Result<MentalModel> {
        mental::refresh(self, bank_id, question, path).await
    }
}

/// Run a provider call, retrying transient failures (HTTP 429/5xx surfaced as
/// typed statuses) with exponential backoff: attempt N waits
/// `retry_base_delay_ms * 2^(N-1)` before firing, up to
/// `retry_max_attempts` total attempts. Non-transient errors surface
/// immediately after the first attempt.
async fn with_retry<T, F, Fut>(e: &Engine, op: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let max_attempts = e.config.retry_max_attempts.max(1);
    let mut attempt = 1;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(err) if err.is_transient() && attempt < max_attempts => {
                let shift = (attempt - 1).min(5) as u32;
                tokio::time::sleep(Duration::from_millis(
                    e.config.retry_base_delay_ms.saturating_mul(1 << shift),
                ))
                .await;
                attempt += 1;
            }
            Err(err) => return Err(err),
        }
    }
}
