mod chunks;
mod consolidate;
mod mental;
mod recall;
mod reflect;
mod retain;
mod scoring;

use crate::prompts::extraction::ExtractionConfig;
use crate::traits::{Clock, EmbedProvider, LlmProvider, RerankProvider, TokenEstimator};
use crate::types::*;
use crate::{Error, MemoryStore, ResolverConfig, Result};
pub use scoring::{Budget, BudgetFunction, RecallBudgetConfig, ScoringConfig};
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
    /// Recall-pool sizing (upstream `recall_budget_*`): the per-arm limit a
    /// reflect/recall tool call uses, derived from the budget level and —
    /// for the adaptive function — the caller's token budget. Defaults to
    /// upstream's fixed MID = 300.
    pub recall_budget: RecallBudgetConfig,
    /// Rerank combined-scoring knobs (upstream `apply_combined_scoring`
    /// defaults: recency/temporal ±10%, proof-count ±5%, linear recency
    /// decay over 365 days).
    pub scoring: ScoringConfig,
    /// Cap on the number of candidates handed to the reranker; 0 = no cap
    /// (upstream `reranker_max_candidates`, default 0 = unlimited).
    pub reranker_max_candidates: usize,
    /// Target size of retain content chunks, measured with the configured
    /// `TokenEstimator`. Content at or below this size is extracted in one
    /// request.
    pub chunk_size_tokens: usize,
    /// Token overlap between consecutive retain chunks; the overlap usually
    /// re-extracts boundary facts, which are deduped on merge.
    pub chunk_overlap_tokens: usize,
    /// Total attempts for transient provider failures (HTTP 429/5xx and
    /// transport-level drops/timeouts).
    pub retry_max_attempts: usize,
    /// Backoff base for retries; attempt N waits `base * 2^(N-1)` ms.
    pub retry_base_delay_ms: u64,
    /// Consolidation pass budget, in seconds. Doubles as (a) a soft
    /// deadline — once it elapses a pass stops *starting* new entities and
    /// returns the progress it made, so a slow-but-progressing queue keeps
    /// moving instead of being killed mid-flight — and (b) the base for the
    /// hard wall at twice the budget that aborts a pass whose in-flight
    /// provider call truly wedges (a wedged call would otherwise hold the
    /// single-flight consolidation lock forever). Per-entity dirty rows are
    /// cleared as each entity completes, so both early returns keep every
    /// finished entity's work.
    pub consolidate_timeout_secs: u64,
    /// How many dirty entities a consolidation pass synthesizes
    /// concurrently. Bounded on purpose: each entity is one LLM round
    /// trip, and an unbounded fan-out would trip provider rate limits.
    pub consolidate_concurrency: usize,
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
            recall_budget: RecallBudgetConfig::default(),
            scoring: ScoringConfig::default(),
            reranker_max_candidates: 0,
            chunk_size_tokens: 3000,
            chunk_overlap_tokens: 200,
            retry_max_attempts: 3,
            retry_base_delay_ms: 400,
            consolidate_timeout_secs: 600,
            consolidate_concurrency: 4,
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
    /// double-processing the same dirty entities. Dirty rows are cleared as
    /// each entity completes, so a caller that arrives after a completed
    /// pass simply finds an empty queue and returns `0`.
    ///
    /// The pass is bounded two ways (see `consolidate_timeout_secs`): a soft
    /// deadline stops starting new entities — the pass then returns the
    /// progress it made, and the caller starts a fresh pass for the rest —
    /// and a hard wall aborts a pass with a wedged in-flight provider call
    /// so it can never hold the single-flight lock forever. Both paths keep
    /// every completed entity's work; unfinished entities stay dirty and
    /// retry on the next pass.
    pub async fn consolidate_once(&self, bank_id: &str, max_entities: usize) -> Result<usize> {
        let _guard = self.consolidate_lock.lock().await;
        // The hard wall fires at twice the pass budget: entities stop
        // *starting* at the budget (see `consolidate_timeout_secs`), and the
        // extra budget of grace lets those in-flight calls land instead of
        // being cancelled mid-request. Only a truly wedged provider call —
        // one that ignores even that window — is cut off so it cannot hold
        // the single-flight lock forever.
        let wall = self
            .config
            .consolidate_timeout_secs
            .max(1)
            .saturating_mul(2);
        let pass = tokio::time::timeout(
            Duration::from_secs(wall),
            consolidate::consolidate_once(self, bank_id, max_entities),
        );
        match pass.await {
            Ok(result) => result,
            Err(_elapsed) => Err(Error::Llm(format!(
                "consolidation pass for bank {bank_id} timed out after {wall}s"
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

/// Run a provider call, retrying transient failures — HTTP 429/5xx (typed
/// statuses) and transport-level failures like a dropped connection or
/// timeout (`Error::LlmTransport`/`EmbedTransport`, where the call never
/// got a protocol answer and is safe to re-issue) — with exponential
/// backoff: attempt N waits `retry_base_delay_ms * 2^(N-1)` before firing,
/// up to `retry_max_attempts` total attempts. Non-transient errors surface
/// immediately after the first attempt.
pub(crate) async fn with_retry<T, F, Fut>(e: &Engine, op: F) -> Result<T>
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
