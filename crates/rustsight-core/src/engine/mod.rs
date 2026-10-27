mod consolidate;
mod mental;
mod recall;
mod reflect;
mod retain;

use crate::prompts::extraction::ExtractionConfig;
use crate::traits::{Clock, EmbedProvider, LlmProvider, RerankProvider, TokenEstimator};
use crate::types::*;
use crate::{MemoryStore, ResolverConfig, Result};
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
    pub async fn consolidate_once(&self, bank_id: &str, max_entities: usize) -> Result<usize> {
        let _guard = self.consolidate_lock.lock().await;
        consolidate::consolidate_once(self, bank_id, max_entities).await
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
