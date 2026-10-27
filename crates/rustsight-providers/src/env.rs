use crate::mock::{MockEmbed, MockLlm, MockRerank};
use crate::openai::{ApiRerank, NoopRerank, OpenAiCompat, OpenAiEmbed};
use rustsight_core::traits::{EmbedProvider, LlmProvider, RerankProvider};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ProviderEnv {
    pub llm_base_url: String,
    pub llm_api_key: String,
    pub llm_model: String,
    pub embed_base_url: String,
    pub embed_api_key: String,
    pub embed_model: String,
    pub rerank_url: Option<String>,
    pub rerank_api_key: String,
    pub rerank_model: String,
}

fn var(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn var_opt(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

impl ProviderEnv {
    pub fn from_env() -> Self {
        let llm_base_url = var("RUSTSIGHT_LLM_BASE_URL", "https://api.openai.com/v1");
        let llm_api_key = var("RUSTSIGHT_LLM_API_KEY", "");
        let llm_model = var("RUSTSIGHT_LLM_MODEL", "gpt-4o-mini");
        let embed_base_url = var("RUSTSIGHT_EMBED_BASE_URL", &llm_base_url);
        let embed_api_key = var("RUSTSIGHT_EMBED_API_KEY", &llm_api_key);
        let embed_model = var("RUSTSIGHT_EMBED_MODEL", "text-embedding-3-small");
        let rerank_url = var_opt("RUSTSIGHT_RERANK_URL");
        let rerank_api_key = var("RUSTSIGHT_RERANK_API_KEY", &llm_api_key);
        let rerank_model = var("RUSTSIGHT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3");
        Self {
            llm_base_url,
            llm_api_key,
            llm_model,
            embed_base_url,
            embed_api_key,
            embed_model,
            rerank_url,
            rerank_api_key,
            rerank_model,
        }
    }
}

pub type Providers = (
    Arc<dyn LlmProvider>,
    Arc<dyn EmbedProvider>,
    Arc<dyn RerankProvider>,
);

pub fn build_providers(env: &ProviderEnv) -> Providers {
    let llm = Arc::new(OpenAiCompat {
        client: reqwest::Client::new(),
        base_url: env.llm_base_url.clone(),
        api_key: env.llm_api_key.clone(),
        model: env.llm_model.clone(),
    });
    let embed = Arc::new(OpenAiEmbed {
        client: reqwest::Client::new(),
        base_url: env.embed_base_url.clone(),
        api_key: env.embed_api_key.clone(),
        model: env.embed_model.clone(),
        dim_cache: std::sync::Mutex::new(1536),
    });
    let rerank: Arc<dyn RerankProvider> = match &env.rerank_url {
        Some(url) => Arc::new(ApiRerank {
            client: reqwest::Client::new(),
            url: url.clone(),
            api_key: env.rerank_api_key.clone(),
            model: env.rerank_model.clone(),
        }),
        None => Arc::new(NoopRerank),
    };
    (llm, embed, rerank)
}

pub fn build_mock_providers() -> Providers {
    (
        Arc::new(MockLlm::demo()),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
    )
}
