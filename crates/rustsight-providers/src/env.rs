use crate::mock::{MockEmbed, MockLlm, MockRerank};
use crate::openai::{ApiRerank, NoopRerank, OpenAiCompat, OpenAiEmbed};
use rustsight_core::traits::{EmbedProvider, LlmProvider, RerankProvider};
use std::sync::Arc;
use std::time::Duration;

/// Shared HTTP client for all provider calls. Bounded connect + total
/// timeouts so a stalled provider connection can never hang retain/recall
/// — or hold the single-flight consolidation lock — forever. One client
/// across LLM/embed/rerank also gives a shared connection pool.
///
/// The total timeout covers connect through the last body byte: a slow
/// provider that streams a long completion past the wall dies as a
/// transient transport error (retried by the engine). Defaults to 300s
/// because large extraction calls on cheap providers routinely exceed two
/// minutes end-to-end; override with `RUSTSIGHT_HTTP_TIMEOUT_SECS`
/// (clamped to 5–3600, invalid values fall back with a warning).
pub fn http_client() -> Result<reqwest::Client, String> {
    let timeout_secs = var_num("RUSTSIGHT_HTTP_TIMEOUT_SECS", 300, 5, 3600);
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(timeout_secs as u64))
        .build()
        .map_err(|e| format!("http client: {e}"))
}

/// Default HuggingFace repo for CPU runs of the local reranker: an int8
/// dynamic-quant export of bge-reranker-v2-m3 in a single ~544 MB
/// self-contained file. Measured fastest of the available int8 exports on
/// CPU (roughly 3x faster than the onnx-community int8 export on a 16-core
/// x86-64). Override with `RUSTSIGHT_RERANK_MODEL_REPO`.
pub const DEFAULT_RERANK_REPO: &str = "rafa754/bge-reranker-v2-m3";
/// Model file inside [`DEFAULT_RERANK_REPO`] for CPU runs. Its
/// `MatMulInteger` ops are CPU-only, so it is never paired with the CUDA
/// execution provider.
pub const DEFAULT_RERANK_REPO_FILE: &str = "onnx/model_quantized.onnx";
/// Default repo for CUDA runs: the canonical transformers.js/onnx-community
/// export, the only one shipping an fp16 variant, which hits GPU tensor
/// cores (~6x the CPU int8 throughput on a laptop RTX 3050 Ti).
pub const DEFAULT_RERANK_REPO_CUDA: &str = "onnx-community/bge-reranker-v2-m3-ONNX";
/// Model file inside [`DEFAULT_RERANK_REPO_CUDA`] for CUDA runs: ~1.1 GB
/// fp16. Selected automatically whenever the reranker runs on CUDA and
/// neither `RUSTSIGHT_RERANK_MODEL_REPO` nor `RUSTSIGHT_RERANK_MODEL_FILE`
/// was set explicitly.
pub const DEFAULT_RERANK_REPO_FILE_CUDA: &str = "onnx/model_fp16.onnx";

/// Compute device for the reranker cross-encoder, parsed from
/// `RUSTSIGHT_RERANK_DEVICE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankDevice {
    /// Use CUDA when the binary was built with `onnx-rerank-cuda` and the
    /// runtime (driver + CUDA libs) is available; otherwise CPU.
    #[default]
    Auto,
    /// Force the CPU execution provider.
    Cpu,
    /// Force CUDA; fails loudly when unavailable instead of falling back.
    Cuda,
}

impl RerankDevice {
    /// Parses `RUSTSIGHT_RERANK_DEVICE`; unknown values are rejected so a
    /// typo can't silently pin the reranker to the wrong device.
    pub fn parse(value: &str) -> std::result::Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(RerankDevice::Auto),
            "cpu" => Ok(RerankDevice::Cpu),
            "cuda" | "gpu" => Ok(RerankDevice::Cuda),
            other => Err(format!(
                "invalid RUSTSIGHT_RERANK_DEVICE value {other:?}: expected auto, cpu, or cuda"
            )),
        }
    }
}

/// How recall reranking is provided, parsed from `RUSTSIGHT_RERANK`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankMode {
    /// Unset: URL wins, then a local model path, else `NoopRerank` with a
    /// loud accuracy warning.
    #[default]
    Auto,
    /// Local ONNX cross-encoder (`OrtRerank`, requires the `onnx-rerank`
    /// build feature).
    Local,
    /// Remote `/rerank` endpoint (requires `RUSTSIGHT_RERANK_URL`).
    Api,
    /// Explicit opt-out: `NoopRerank`, no warning.
    Off,
}

impl RerankMode {
    /// Parses `RUSTSIGHT_RERANK`; unknown values are rejected (a typo that
    /// silently degraded reranking would be worse than failing to start).
    fn parse(value: Option<String>) -> Result<Self, String> {
        let normalized = value
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_ascii_lowercase);
        match normalized.as_deref() {
            None | Some("auto") => Ok(RerankMode::Auto),
            Some("local") | Some("onnx") => Ok(RerankMode::Local),
            Some("api") | Some("remote") => Ok(RerankMode::Api),
            Some("off") | Some("none") | Some("noop") => Ok(RerankMode::Off),
            Some(other) => Err(format!(
                "invalid RUSTSIGHT_RERANK value {other:?}: expected local, api, off, or auto"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProviderEnv {
    pub llm_base_url: String,
    pub llm_api_key: String,
    pub llm_model: String,
    pub embed_base_url: String,
    pub embed_api_key: String,
    pub embed_model: String,
    pub rerank_mode: RerankMode,
    pub rerank_url: Option<String>,
    pub rerank_api_key: String,
    pub rerank_model: String,
    /// `RUSTSIGHT_RERANK_MODEL_PATH`: pre-placed `.onnx` file (or directory
    /// with `model.onnx` + `tokenizer.json`); skips the first-run download.
    pub rerank_model_path: Option<String>,
    /// `RUSTSIGHT_RERANK_MODEL_REPO`: HuggingFace repo for first-run model
    /// download.
    pub rerank_model_repo: String,
    /// `RUSTSIGHT_RERANK_MODEL_FILE`: model file within that repo.
    pub rerank_model_file: String,
    /// `RUSTSIGHT_RERANK_MODEL_SHA256`: optional hex sha256 pin for the
    /// downloaded model file; a mismatching download is refused.
    pub rerank_model_sha256: Option<String>,
    /// `RUSTSIGHT_MODELS_DIR`: download cache root (default
    /// `~/.rustsight/models`).
    pub rerank_models_dir: Option<String>,
    /// `RUSTSIGHT_RERANK_BATCH`: pairs scored per ONNX run (default 16).
    pub rerank_batch_size: usize,
    /// `RUSTSIGHT_RERANK_MAX_LEN`: max tokens per query/doc pair (default
    /// 512).
    pub rerank_max_len: usize,
    /// `RUSTSIGHT_RERANK_DEVICE`: `auto`, `cpu`, or `cuda` (default auto —
    /// CUDA when built with `onnx-rerank-cuda` and available, else CPU).
    pub rerank_device: String,
}

fn var(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn var_opt(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn var_num(name: &str, default: usize, min: usize, max: usize) -> usize {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => v
            .trim()
            .parse::<usize>()
            .map(|n| n.clamp(min, max))
            .unwrap_or_else(|_| {
                eprintln!("rustsight: invalid {name}={v:?}, using default {default}");
                default
            }),
        _ => default,
    }
}

/// Reads an optional sha256 pin env var, rejecting malformed values at
/// startup — a typo would otherwise only surface as a mismatch after the
/// multi-hundred-MB download finished.
fn validated_sha256(name: &str) -> Result<Option<String>, String> {
    parse_sha256(var_opt(name).as_deref(), name)
}

fn parse_sha256(value: Option<&str>, name: &str) -> Result<Option<String>, String> {
    // Mirror `var_opt`: an empty value means "unset", same as absent.
    let Some(raw) = value.filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    let hex = raw.trim();
    if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(Some(hex.to_ascii_lowercase()))
    } else {
        Err(format!(
            "invalid {name}={raw:?}: expected 64 hex characters (sha256)"
        ))
    }
}

impl ProviderEnv {
    pub fn from_env() -> Result<Self, String> {
        let llm_base_url = var("RUSTSIGHT_LLM_BASE_URL", "https://api.openai.com/v1");
        let llm_api_key = var("RUSTSIGHT_LLM_API_KEY", "");
        let llm_model = var("RUSTSIGHT_LLM_MODEL", "gpt-4o-mini");
        let embed_base_url = var("RUSTSIGHT_EMBED_BASE_URL", &llm_base_url);
        let embed_api_key = var("RUSTSIGHT_EMBED_API_KEY", &llm_api_key);
        let embed_model = var("RUSTSIGHT_EMBED_MODEL", "text-embedding-3-small");
        let rerank_mode = RerankMode::parse(var_opt("RUSTSIGHT_RERANK"))?;
        let rerank_url = var_opt("RUSTSIGHT_RERANK_URL");
        let rerank_api_key = var("RUSTSIGHT_RERANK_API_KEY", &llm_api_key);
        let rerank_model = var("RUSTSIGHT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3");
        let rerank_model_path = var_opt("RUSTSIGHT_RERANK_MODEL_PATH");
        let rerank_model_repo = var("RUSTSIGHT_RERANK_MODEL_REPO", DEFAULT_RERANK_REPO);
        let rerank_model_file = var("RUSTSIGHT_RERANK_MODEL_FILE", DEFAULT_RERANK_REPO_FILE);
        let rerank_model_sha256 = validated_sha256("RUSTSIGHT_RERANK_MODEL_SHA256")?;
        let rerank_models_dir = var_opt("RUSTSIGHT_MODELS_DIR");
        let rerank_batch_size = var_num("RUSTSIGHT_RERANK_BATCH", 16, 1, 128);
        let rerank_max_len = var_num("RUSTSIGHT_RERANK_MAX_LEN", 512, 32, 8192);
        let rerank_device = var("RUSTSIGHT_RERANK_DEVICE", "auto");
        RerankDevice::parse(&rerank_device)?;
        Ok(Self {
            llm_base_url,
            llm_api_key,
            llm_model,
            embed_base_url,
            embed_api_key,
            embed_model,
            rerank_mode,
            rerank_url,
            rerank_api_key,
            rerank_model,
            rerank_model_path,
            rerank_model_repo,
            rerank_model_file,
            rerank_model_sha256,
            rerank_models_dir,
            rerank_batch_size,
            rerank_max_len,
            rerank_device,
        })
    }
}

pub type Providers = (
    Arc<dyn LlmProvider>,
    Arc<dyn EmbedProvider>,
    Arc<dyn RerankProvider>,
);

/// Builds the production provider stack. Falls back to `NoopRerank` (raw RRF
/// recall order) only when no reranker is configured, and warns loudly then:
/// the paper attributes a significant share of its benchmark lead to
/// cross-encoder reranking (RS-4).
pub async fn build_providers(env: &ProviderEnv) -> Result<Providers, String> {
    let client = http_client()?;
    let llm = Arc::new(OpenAiCompat {
        client: client.clone(),
        base_url: env.llm_base_url.clone(),
        api_key: env.llm_api_key.clone(),
        model: env.llm_model.clone(),
    });
    let embed = Arc::new(OpenAiEmbed {
        client: client.clone(),
        base_url: env.embed_base_url.clone(),
        api_key: env.embed_api_key.clone(),
        model: env.embed_model.clone(),
        dim_cache: std::sync::Mutex::new(1536),
    });
    let rerank: Arc<dyn RerankProvider> = match env.rerank_mode {
        RerankMode::Off => Arc::new(NoopRerank),
        RerankMode::Api => api_rerank(env, &client)?,
        RerankMode::Local => local_rerank(env).await?,
        RerankMode::Auto => match (&env.rerank_url, &env.rerank_model_path) {
            (Some(_), _) => api_rerank(env, &client)?,
            (None, Some(_)) => local_rerank(env).await?,
            (None, None) => {
                warn_no_reranker();
                Arc::new(NoopRerank)
            }
        },
    };
    Ok((llm, embed, rerank))
}

fn api_rerank(
    env: &ProviderEnv,
    client: &reqwest::Client,
) -> Result<Arc<dyn RerankProvider>, String> {
    let url = env.rerank_url.clone().ok_or(
        "RUSTSIGHT_RERANK=api but RUSTSIGHT_RERANK_URL is not set (an HTTP /rerank \
         endpoint, Jina/Cohere-shaped)",
    )?;
    Ok(Arc::new(ApiRerank {
        client: client.clone(),
        url,
        api_key: env.rerank_api_key.clone(),
        model: env.rerank_model.clone(),
    }))
}

#[cfg(feature = "onnx-rerank")]
async fn local_rerank(env: &ProviderEnv) -> Result<Arc<dyn RerankProvider>, String> {
    use crate::onnx::{OrtRerank, OrtRerankSpec};
    let rerank = OrtRerank::load(OrtRerankSpec::from_provider_env(env)).await?;
    Ok(Arc::new(rerank))
}

#[cfg(not(feature = "onnx-rerank"))]
async fn local_rerank(_env: &ProviderEnv) -> Result<Arc<dyn RerankProvider>, String> {
    Err(
        "a local ONNX reranker was requested (RUSTSIGHT_RERANK=local or \
         RUSTSIGHT_RERANK_MODEL_PATH) but this binary was built without the \
         `onnx-rerank` feature; rebuild with `--features onnx-rerank`, or use \
         RUSTSIGHT_RERANK_URL instead"
            .to_string(),
    )
}

fn warn_no_reranker() {
    eprintln!(
        "rustsight: WARNING — no reranker configured; recall order falls back to raw \
         reciprocal-rank fusion.\n  The Hindsight paper attributes a significant share of \
         its benchmark lead to cross-encoder reranking, so\n  answer quality will suffer on \
         large memory banks. Fix one of these ways:\n    - run a build with the \
         `onnx-rerank` feature and set RUSTSIGHT_RERANK=local (bge-reranker-v2-m3, in-process, no API cost)\n    - set RUSTSIGHT_RERANK_URL to a /rerank endpoint (Jina/Cohere style)\n    - set RUSTSIGHT_RERANK=off to accept raw RRF ordering and silence this warning"
    );
}

pub fn build_mock_providers() -> Providers {
    (
        Arc::new(MockLlm::demo()),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_mode_parses_known_values() {
        assert_eq!(RerankMode::parse(None), Ok(RerankMode::Auto));
        assert_eq!(RerankMode::parse(Some(String::new())), Ok(RerankMode::Auto));
        assert_eq!(RerankMode::parse(Some("auto".into())), Ok(RerankMode::Auto));
        assert_eq!(
            RerankMode::parse(Some("local".into())),
            Ok(RerankMode::Local)
        );
        assert_eq!(
            RerankMode::parse(Some(" LOCAL ".into())),
            Ok(RerankMode::Local)
        );
        assert_eq!(RerankMode::parse(Some("api".into())), Ok(RerankMode::Api));
        assert_eq!(RerankMode::parse(Some("off".into())), Ok(RerankMode::Off));
        assert_eq!(RerankMode::parse(Some("none".into())), Ok(RerankMode::Off));
        assert!(RerankMode::parse(Some("locall".into())).is_err());
    }

    #[test]
    fn rerank_device_parses_known_values() {
        assert_eq!(RerankDevice::parse(""), Ok(RerankDevice::Auto));
        assert_eq!(RerankDevice::parse("auto"), Ok(RerankDevice::Auto));
        assert_eq!(RerankDevice::parse(" CPU "), Ok(RerankDevice::Cpu));
        assert_eq!(RerankDevice::parse("cuda"), Ok(RerankDevice::Cuda));
        assert_eq!(RerankDevice::parse("gpu"), Ok(RerankDevice::Cuda));
        assert!(RerankDevice::parse("cudaa").is_err());
        assert!(RerankDevice::parse("metal").is_err());
    }

    #[test]
    fn sha256_pin_is_validated_and_normalized() {
        assert_eq!(parse_sha256(None, "X").unwrap(), None);
        assert_eq!(parse_sha256(Some(""), "X").unwrap(), None);
        let pin = format!(" {} ", "aB".repeat(32));
        assert_eq!(
            parse_sha256(Some(&pin), "X").unwrap(),
            Some("ab".repeat(32))
        );
        // wrong length, non-hex characters, and empty-after-trim all fail
        assert!(parse_sha256(Some("a".repeat(63)).as_deref(), "X").is_err());
        assert!(parse_sha256(Some("z".repeat(64)).as_deref(), "X").is_err());
        assert!(parse_sha256(Some("nothex"), "X").is_err());
    }
}
