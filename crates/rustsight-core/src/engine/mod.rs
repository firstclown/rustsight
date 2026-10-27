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

    pub async fn consolidate_once(&self, bank_id: &str, max_entities: usize) -> Result<usize> {
        consolidate::consolidate_once(self, bank_id, max_entities).await
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
