mod bank_gate;
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
use bank_gate::BankGates;
pub use bank_gate::BankGuard;
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
    /// Base iteration count the Hindsight-compatible API scales by the
    /// request's `budget` (upstream `reflect_max_iterations` = 10, with
    /// multipliers low 0.5 / mid 1 / high 2 — so the API default, `low`,
    /// runs 5). Only used when a request names a budget; the native
    /// `Engine::reflect` keeps `reflect_max_rounds`.
    pub reflect_budget_base_rounds: usize,
    /// Context budget (tokens) for reflect's final-synthesis call over the
    /// gathered tool-result history (upstream `max_context_tokens`, default
    /// 100k). Histories over the budget split into chunks, map to dated
    /// claims, then reduce — evidence is split, never dropped.
    pub reflect_max_context_tokens: usize,
    /// Transport-level output cap (tokens) for reflect's answer-path calls
    /// (closing ask + synthesis; upstream `reflect_max_completion_tokens`,
    /// decoupled from the loop turns). Without it, gateways allocate a
    /// huge default reasoning/output budget for thinking models — a
    /// 52k-token closing ask plus a 79k allocation blew the 131k window
    /// (2026-11-12 run, one scenario 500). `None` sends no cap.
    pub reflect_max_completion_tokens: Option<u64>,
    /// Visible answer-length target (tokens) for the closing ask and the
    /// synthesis prompts — upstream `reflect_async`'s `max_tokens`
    /// parameter, default 4096. It rides as the prompt's `## Length`
    /// directive ("a complete, self-contained answer of approximately N
    /// tokens"), which is why gold's answers are exhaustive while the
    /// same model wrote 41-char one-liners for us with it unset.
    /// `None` omits the directive.
    pub reflect_answer_max_tokens: Option<usize>,
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
    /// Output cap on each fact-extraction call (upstream
    /// `retain_max_completion_tokens`, default 64000). A reply cut off here
    /// fails to parse and takes the half-size re-ask path.
    pub retain_max_completion_tokens: Option<u64>,
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
    /// How many of a dirty entity's facts feed one consolidation call.
    /// Entities with more facts page through them in windows of this size
    /// (newest first), each window seeing the previous windows'
    /// observations — the old fixed cap of 60 silently dropped older facts
    /// of big entities, so they never reached an observation (2025-11:
    /// 159 facts, the "Joe" entity alone exceeded the cap).
    pub consolidate_max_facts_per_entity: usize,
    /// Cosine threshold for the bank-level near-duplicate observation
    /// guard at CREATE time: a proposed observation this similar to an
    /// existing one merges its evidence instead of writing a sibling
    /// (the 2026-11-12 runs: 5-8x upstream's observation count with
    /// paraphrase siblings the exact-text guard cannot catch).
    pub consolidate_dupe_cosine: f32,
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
            reflect_budget_base_rounds: 10,
            reflect_max_context_tokens: 100_000,
            reflect_max_completion_tokens: Some(8192),
            reflect_answer_max_tokens: Some(4096),
            recall_budget: RecallBudgetConfig::default(),
            scoring: ScoringConfig::default(),
            reranker_max_candidates: 0,
            chunk_size_tokens: 3000,
            chunk_overlap_tokens: 200,
            retain_max_completion_tokens: Some(64_000),
            retry_max_attempts: 3,
            retry_base_delay_ms: 400,
            consolidate_timeout_secs: 600,
            consolidate_concurrency: 4,
            consolidate_max_facts_per_entity: 200,
            consolidate_dupe_cosine: 0.9,
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
    /// Per-operation LLM overrides (upstream `HINDSIGHT_API_RETAIN_LLM_*` /
    /// `_REFLECT_LLM_*` / `_CONSOLIDATION_LLM_*`); `None` falls back to
    /// `llm`. Lets a deployment extract with a small model and answer with
    /// a large one.
    pub retain_llm: Option<Arc<dyn LlmProvider>>,
    pub reflect_llm: Option<Arc<dyn LlmProvider>>,
    pub consolidate_llm: Option<Arc<dyn LlmProvider>>,
    /// Wakes the background worker as soon as retain marks entities dirty.
    pub(crate) wakeup: Notify,
    /// Single-flight guard: serializes consolidation passes (worker vs
    /// explicit `consolidate_once` calls) so dirty entities are never
    /// double-processed.
    pub(crate) consolidate_lock: Mutex<()>,
    /// Keeps bank deletion from interleaving with work writing to the bank.
    pub(crate) bank_gates: Arc<BankGates>,
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
            retain_llm: None,
            reflect_llm: None,
            consolidate_llm: None,
            wakeup: Notify::new(),
            consolidate_lock: Mutex::new(()),
            bank_gates: Arc::default(),
        }
    }

    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_retain_llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.retain_llm = Some(llm);
        self
    }

    pub fn with_reflect_llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.reflect_llm = Some(llm);
        self
    }

    pub fn with_consolidate_llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.consolidate_llm = Some(llm);
        self
    }

    pub(crate) fn llm_for_retain(&self) -> Arc<dyn LlmProvider> {
        Arc::clone(self.retain_llm.as_ref().unwrap_or(&self.llm))
    }

    pub(crate) fn llm_for_reflect(&self) -> Arc<dyn LlmProvider> {
        Arc::clone(self.reflect_llm.as_ref().unwrap_or(&self.llm))
    }

    pub(crate) fn llm_for_consolidate(&self) -> Arc<dyn LlmProvider> {
        Arc::clone(self.consolidate_llm.as_ref().unwrap_or(&self.llm))
    }

    /// The mission text for a non-reflect pipeline (`retain_mission` or
    /// `observations_mission`). A bank configured the Hindsight way keeps
    /// its `mission` for reflect only (`mission_scope: "reflect"`), so the
    /// pipeline gets its own mission or none; a natively configured bank
    /// keeps sharing `mission` across pipelines.
    pub(crate) fn scoped_mission(&self, bank: &Bank, key: &str) -> Option<String> {
        let config = self
            .store
            .bank_config(&bank.id)
            .unwrap_or(serde_json::Value::Null);
        let own = config
            .get(key)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if own.is_some() {
            return own;
        }
        if config.get("mission_scope").and_then(|v| v.as_str()) == Some("reflect") {
            return None;
        }
        bank.mission.clone()
    }

    /// Delete a document and every memory extracted from it; entities whose
    /// facts or observations changed are re-queued for consolidation.
    /// Returns the number of memories deleted.
    pub fn forget_document(&self, bank_id: &str, document_id: &str) -> Result<usize> {
        let (deleted, affected) = self.store.delete_document_memories(bank_id, document_id)?;
        self.store.delete_document(bank_id, document_id)?;
        if !affected.is_empty() {
            self.store
                .mark_entities_dirty(bank_id, &affected, "document_deleted")?;
            self.wake_worker();
        }
        Ok(deleted)
    }

    /// Whether consolidation may build observations for this bank (on unless
    /// its config says otherwise). Retain keeps queueing dirty entities
    /// either way, so re-enabling picks up everything retained meanwhile.
    pub fn observations_enabled(&self, bank_id: &str) -> Result<bool> {
        Ok(self
            .store
            .bank_config(bank_id)?
            .get("enable_observations")
            .and_then(|v| v.as_bool())
            .unwrap_or(true))
    }

    /// Shared access to a bank for a multi-step write; see
    /// [`Engine::retain_guarded`].
    pub async fn bank_read(&self, bank_id: &str) -> BankGuard {
        self.bank_gates.read(bank_id).await
    }

    /// Delete a bank and everything in it. In-flight retains, consolidation
    /// and opinion formation for the bank are cancelled first.
    pub async fn delete_bank(&self, bank_id: &str) -> Result<BankDeletion> {
        let _exclusive = self.bank_gates.exclusive(bank_id).await;
        self.store.delete_bank(bank_id)
    }

    /// Reflect rounds for a Hindsight budget level (upstream:
    /// `max(1, int(reflect_max_iterations * multiplier))`).
    pub fn reflect_rounds_for_budget(&self, budget: Budget) -> usize {
        let base = self.config.reflect_budget_base_rounds;
        match budget {
            Budget::Low => (base / 2).max(1),
            Budget::Mid => base.max(1),
            Budget::High => (base * 2).max(1),
        }
    }

    pub async fn retain(&self, req: RetainRequest) -> Result<RetainResponse> {
        let guard = self.bank_read(&req.bank_id).await;
        self.retain_guarded(&guard, req).await
    }

    /// [`Engine::retain`] under a guard the caller already holds, so several
    /// retains and their bookkeeping stay one unit against a bank delete.
    pub async fn retain_guarded(
        &self,
        guard: &BankGuard,
        req: RetainRequest,
    ) -> Result<RetainResponse> {
        guard.until_deleted(retain::retain(self, req)).await
    }

    pub async fn recall(&self, req: RecallRequest) -> Result<RecallResponse> {
        recall::recall(self, req).await
    }

    pub async fn reflect(&self, bank_id: &str, query: &str) -> Result<ReflectResponse> {
        reflect::reflect(self, bank_id, query, &ReflectOptions::default()).await
    }

    /// Reflect with per-request options (budgeted rounds, tag scope, kind
    /// restriction, answer length, structured output).
    pub async fn reflect_with(
        &self,
        bank_id: &str,
        query: &str,
        options: &ReflectOptions,
    ) -> Result<ReflectResponse> {
        reflect::reflect(self, bank_id, query, options).await
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
        let bank = self.bank_read(bank_id).await;
        let pass = bank.until_deleted(self.consolidate_serialized(bank_id, max_entities));
        match pass.await {
            // The bank went away mid-pass: there is nothing left to do.
            Err(Error::NotFound(_)) if bank.is_deleting() => Ok(0),
            result => result,
        }
    }

    async fn consolidate_serialized(&self, bank_id: &str, max_entities: usize) -> Result<usize> {
        let _guard = self.consolidate_lock.lock().await;
        if !self.observations_enabled(bank_id)? {
            return Ok(0);
        }
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
        let guard = self.bank_read(bank_id).await;
        guard
            .until_deleted(mental::refresh(self, bank_id, question, path))
            .await
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
    with_retry_config(
        e.config.retry_max_attempts,
        e.config.retry_base_delay_ms,
        op,
    )
    .await
}

/// `with_retry` for detached tasks that cannot borrow the `Engine` (e.g.
/// the backgrounded opinion extraction): same transient-failure policy,
/// explicit retry knobs.
pub(crate) async fn with_retry_config<T, F, Fut>(
    max_attempts_cfg: usize,
    base_delay_ms: u64,
    op: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let max_attempts = max_attempts_cfg.max(1);
    let mut attempt = 1;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(err) if err.is_transient() && attempt < max_attempts => {
                let shift = (attempt - 1).min(5) as u32;
                tokio::time::sleep(Duration::from_millis(
                    base_delay_ms.saturating_mul(1 << shift),
                ))
                .await;
                attempt += 1;
            }
            Err(err) => return Err(err),
        }
    }
}
