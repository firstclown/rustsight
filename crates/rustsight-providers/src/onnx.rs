//! Local ONNX cross-encoder reranker (RS-4).
//!
//! Implements [`RerankProvider`] with a bge-reranker-v2-m3 cross-encoder
//! exported to ONNX, running in-process through the `ort` crate. Only
//! compiled with the `onnx-rerank` cargo feature — the ONNX runtime adds
//! ~50-100 MB of binaries, so it is opt-in (see the README for build and
//! configuration instructions).
//!
//! Model acquisition:
//! - by default the model and tokenizer are downloaded from HuggingFace on
//!   first use into the models cache (`~/.rustsight/models/` unless
//!   `RUSTSIGHT_MODELS_DIR` is set);
//! - `RUSTSIGHT_RERANK_MODEL_PATH` points at a pre-placed `.onnx` file (or a
//!   directory containing `model.onnx` + `tokenizer.json`) for offline hosts;
//!   no download is attempted then.
//!
//! Scoring: query/doc pairs are tokenized as
//! `<s> query </s></s> doc </s>` (special tokens applied by the tokenizer),
//! padded per batch, and scored by the cross-encoder in batches of
//! `batch_size` (default 16; 16-64 is the sweet spot — naive per-doc session
//! runs are 10-50x slower). The sigmoid of the output logit is reported as
//! the relevance score, so scores are comparable across calls.

use crate::env::{RerankDevice, DEFAULT_RERANK_REPO_CUDA, DEFAULT_RERANK_REPO_FILE_CUDA};
use async_trait::async_trait;
use rustsight_core::traits::RerankProvider;
use rustsight_core::{Error, Result};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

/// How to locate the ONNX model + tokenizer for [`OrtRerank`].
#[derive(Debug, Clone)]
pub struct OrtRerankSpec {
    /// Pre-placed model file, or a directory holding `model.onnx` and
    /// `tokenizer.json`. When set, no download is attempted.
    pub local_path: Option<PathBuf>,
    /// HuggingFace repo id used for first-run download.
    pub repo: String,
    /// Model file within `repo`.
    pub repo_file: String,
    /// Download cache root; defaults to `~/.rustsight/models`.
    pub cache_dir: Option<PathBuf>,
    /// Query/doc pairs scored per ONNX session run (16-64 recommended).
    pub batch_size: usize,
    /// Maximum token length per query/doc pair; longer docs are truncated
    /// (longest-first), matching upstream reranker behavior.
    pub max_len: usize,
    /// Which execution provider to run the cross-encoder on.
    pub device: RerankDevice,
}

impl OrtRerankSpec {
    /// Builds a spec from the rerank-related `RUSTSIGHT_*` variables in
    /// [`crate::env::ProviderEnv`]; the single source of truth shared by
    /// `build_providers` and tooling.
    pub fn from_provider_env(env: &crate::env::ProviderEnv) -> Self {
        Self {
            local_path: env.rerank_model_path.clone().map(PathBuf::from),
            repo: env.rerank_model_repo.clone(),
            repo_file: env.rerank_model_file.clone(),
            cache_dir: env.rerank_models_dir.clone().map(PathBuf::from),
            batch_size: env.rerank_batch_size,
            max_len: env.rerank_max_len,
            device: crate::env::RerankDevice::parse(&env.rerank_device).unwrap_or_default(),
        }
    }
}

impl Default for OrtRerankSpec {
    fn default() -> Self {
        Self {
            local_path: None,
            repo: crate::env::DEFAULT_RERANK_REPO.to_string(),
            repo_file: crate::env::DEFAULT_RERANK_REPO_FILE.to_string(),
            cache_dir: None,
            batch_size: 16,
            max_len: 512,
            device: RerankDevice::Auto,
        }
    }
}

/// The ONNX runtime allows one environment per process and `ort` errors on a
/// second `init().build()`, so every [`OrtRerank`] in the process shares this
/// one. Sessions keep the environment alive themselves; the static exists to
/// survive repeated loads (tests, re-configuration).
static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

fn ort_env() -> std::result::Result<&'static ort::environment::Environment, String> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .build()
        .map_err(|e| format!("failed to initialize ONNX runtime: {e}"))?;
    let _ = ORT_ENV.set(env);
    ORT_ENV
        .get()
        .ok_or_else(|| "failed to initialize ONNX runtime".to_string())
}

pub struct OrtRerank {
    session: Mutex<ort::session::Session>,
    tokenizer: Mutex<Tokenizer>,
    input_ids_name: String,
    attention_name: String,
    token_type_name: Option<String>,
    pad_id: u32,
    batch_size: usize,
}

impl OrtRerank {
    /// Resolve (downloading if necessary) and load the cross-encoder. Does
    /// blocking file IO and may download hundreds of MB on first run; call
    /// once at startup, not on a hot path.
    ///
    /// Device policy: CPU runs use the int8-quantized default model (its
    /// `MatMulInteger` ops have no CUDA kernels, so registering the CUDA EP
    /// for it only adds device-copy ping-pong — measured slower than pure
    /// CPU); CUDA runs use the fp16 default. An explicit
    /// `RUSTSIGHT_RERANK_MODEL_FILE` or a pre-placed local model is always
    /// respected as-is. `auto` picks CUDA when this binary was built with
    /// `onnx-rerank-cuda` and the CUDA runtime stack resolves (probed by
    /// dlopen, because ONNX Runtime silently falls back to CPU when
    /// provider libraries are missing), otherwise CPU.
    pub async fn load(spec: OrtRerankSpec) -> std::result::Result<Self, String> {
        let env = ort_env()?;
        let mut spec = spec;

        let want_cuda = match spec.device {
            RerankDevice::Cpu => false,
            RerankDevice::Cuda => {
                ensure_cuda_feature()?;
                if !cuda_stack_available() {
                    return Err(CUDA_MISSING_MSG.to_string());
                }
                true
            }
            RerankDevice::Auto => cuda_stack_available(),
        };

        let target_file = cuda_paired_repo_file(&spec.repo_file);
        let target_repo = cuda_paired_repo(&spec.repo);
        let upgraded_file = want_cuda
            && spec.local_path.is_none()
            && (target_file != spec.repo_file || target_repo != spec.repo);
        if upgraded_file {
            spec.repo_file = target_file.to_string();
            spec.repo = target_repo.to_string();
        }

        let (model_path, tokenizer_path) = resolve_model_files(&spec).await?;
        let tokenizer = load_tokenizer(&tokenizer_path, spec.max_len)?;

        if !want_cuda {
            let session = session_on(env, &model_path, EpPreference::Cpu)?;
            return Ok(Self::from_parts(session, tokenizer, spec.batch_size));
        }
        match session_on(env, &model_path, EpPreference::CudaRequired) {
            Ok(session) => Ok(Self::from_parts(session, tokenizer, spec.batch_size)),
            Err(err) if spec.device == RerankDevice::Auto => {
                // Probe passed but the session refused (broken driver etc.).
                // Retry on CPU; if we swapped the default file to fp16, put
                // the CPU-appropriate int8 default back first.
                eprintln!("rustsight: CUDA reranker unavailable ({err}); falling back to CPU");
                if upgraded_file {
                    spec.repo_file = crate::env::DEFAULT_RERANK_REPO_FILE.to_string();
                    spec.repo = crate::env::DEFAULT_RERANK_REPO.to_string();
                    let (model_path, tokenizer_path) = resolve_model_files(&spec).await?;
                    let tokenizer = load_tokenizer(&tokenizer_path, spec.max_len)?;
                    let session = session_on(env, &model_path, EpPreference::Cpu)?;
                    return Ok(Self::from_parts(session, tokenizer, spec.batch_size));
                }
                let session = session_on(env, &model_path, EpPreference::Cpu)?;
                Ok(Self::from_parts(session, tokenizer, spec.batch_size))
            }
            Err(err) => Err(err),
        }
    }

    fn from_parts(session: ort::session::Session, tokenizer: Tokenizer, batch_size: usize) -> Self {
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let by_name = |n: &str| input_names.iter().position(|x| x == n);
        let input_ids_name = by_name("input_ids")
            .map(|i| input_names[i].clone())
            .or_else(|| input_names.first().cloned())
            .unwrap_or_default();
        let attention_name = by_name("attention_mask")
            .map(|i| input_names[i].clone())
            .or_else(|| input_names.get(1).cloned())
            .unwrap_or_else(|| input_ids_name.clone());
        let token_type_name = by_name("token_type_ids").map(|i| input_names[i].clone());
        let pad_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(1);
        Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
            input_ids_name,
            attention_name,
            token_type_name,
            pad_id,
            batch_size: batch_size.max(1),
        }
    }

    fn tokenize_and_score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
        let tokenizer = self
            .tokenizer
            .lock()
            .map_err(|_| Error::Rerank("tokenizer lock poisoned".to_string()))?;
        let mut scores = Vec::with_capacity(docs.len());
        for chunk in docs.chunks(self.batch_size) {
            let pairs: Vec<(String, String)> = chunk
                .iter()
                .map(|d| (query.to_string(), d.clone()))
                .collect();
            let encodings = tokenizer
                .encode_batch(pairs, true)
                .map_err(|e| Error::Rerank(format!("tokenization failed: {e}")))?;
            let rows: Vec<&[u32]> = encodings.iter().map(|e| e.get_ids()).collect();
            scores.extend(self.score_rows(&rows).map_err(Error::Rerank)?);
        }
        Ok(scores)
    }

    fn score_rows(&self, rows: &[&[u32]]) -> std::result::Result<Vec<f32>, String> {
        let n = rows.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let len = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
        let (ids, mask) = padded_batch(rows, len, self.pad_id);
        let shape = vec![n as i64, len as i64];

        let mut session = self
            .session
            .lock()
            .map_err(|_| "rerank session lock poisoned".to_string())?;
        let outputs = if let Some(tt_name) = &self.token_type_name {
            let tt = vec![0i64; n * len];
            session
                .run(ort::inputs![
                    self.input_ids_name.as_str() => tensor(&shape, ids)?,
                    self.attention_name.as_str() => tensor(&shape, mask)?,
                    tt_name.as_str() => tensor(&shape, tt)?,
                ])
                .map_err(|e| format!("reranker inference failed: {e}"))?
        } else {
            session
                .run(ort::inputs![
                    self.input_ids_name.as_str() => tensor(&shape, ids)?,
                    self.attention_name.as_str() => tensor(&shape, mask)?,
                ])
                .map_err(|e| format!("reranker inference failed: {e}"))?
        };
        let (_shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("failed to read reranker output: {e}"))?;
        Ok(logits.iter().map(|&l| sigmoid(l)).collect())
    }
}

fn tensor(shape: &[i64], data: Vec<i64>) -> std::result::Result<ort::value::Tensor<i64>, String> {
    ort::value::Tensor::from_array((shape.to_vec(), data))
        .map_err(|e| format!("tensor build failed: {e}"))
}

/// The model file actually used when the reranker runs on CUDA: the int8
/// default is swapped for the fp16 export (its `MatMulInteger` ops have no
/// CUDA kernels); any explicit file choice passes through untouched.
pub(crate) fn cuda_paired_repo_file(repo_file: &str) -> &str {
    if repo_file == crate::env::DEFAULT_RERANK_REPO_FILE {
        DEFAULT_RERANK_REPO_FILE_CUDA
    } else {
        repo_file
    }
}

/// The repo actually used when the reranker runs on CUDA: the CPU default
/// repo has no fp16 export, so CUDA runs point at the onnx-community repo
/// that does; any explicit repo choice passes through untouched.
pub(crate) fn cuda_paired_repo(repo: &str) -> &str {
    if repo == crate::env::DEFAULT_RERANK_REPO {
        DEFAULT_RERANK_REPO_CUDA
    } else {
        repo
    }
}

const CUDA_MISSING_MSG: &str = "RUSTSIGHT_RERANK_DEVICE=cuda but no usable CUDA stack was \
     found: need the NVIDIA driver plus CUDA 13 runtime libraries (libcudart.so.13, \
     libcublas.so.13, libcublasLt.so.13, libcurand.so.10 — system install, CUDA toolkit, \
     or LD_LIBRARY_PATH); see the reranking section of the README";

fn ensure_cuda_feature() -> std::result::Result<(), String> {
    #[cfg(feature = "onnx-rerank-cuda")]
    {
        Ok(())
    }
    #[cfg(not(feature = "onnx-rerank-cuda"))]
    {
        Err(
            "RUSTSIGHT_RERANK_DEVICE=cuda but this binary was built without the \
             `onnx-rerank-cuda` feature; rebuild with `--features onnx-rerank-cuda` \
             or use device=auto/cpu"
                .to_string(),
        )
    }
}

/// Whether the CUDA runtime stack required by the ONNX Runtime CUDA provider
/// resolves in this process. Probed by dlopen because ONNX Runtime itself
/// silently falls back to CPU when provider libraries are missing — without
/// this check, `device=auto` would quietly claim CUDA while running on CPU.
#[cfg(feature = "onnx-rerank-cuda")]
fn cuda_stack_available() -> bool {
    let libs: &[&str] = if cfg!(target_os = "windows") {
        &[
            "nvcuda.dll",
            "cudart64_13.dll",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "curand64_10.dll",
        ]
    } else {
        &[
            "libcuda.so.1",
            "libcudart.so.13",
            "libcublas.so.13",
            "libcublasLt.so.13",
            "libcurand.so.10",
        ]
    };
    libs.iter()
        .all(|name| unsafe { libloading::Library::new(name) }.is_ok())
}

#[cfg(not(feature = "onnx-rerank-cuda"))]
fn cuda_stack_available() -> bool {
    false
}

fn load_tokenizer(path: &Path, max_len: usize) -> std::result::Result<Tokenizer, String> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|e| format!("failed to load tokenizer {}: {e}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_len,
            strategy: TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| format!("failed to configure truncation: {e}"))?;
    Ok(tokenizer)
}

/// Which execution providers a session should be created with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpPreference {
    /// CPU only, never register an accelerator EP.
    Cpu,
    /// Register CUDA (the model file must be CUDA-compatible — see
    /// [`cuda_paired_repo_file`]).
    CudaRequired,
}

fn session_on(
    env: &ort::environment::Environment,
    model_path: &Path,
    pref: EpPreference,
) -> std::result::Result<ort::session::Session, String> {
    let build =
        || ort::session::Session::builder(env).map_err(|e| format!("session builder failed: {e}"));
    #[cfg(feature = "onnx-rerank-cuda")]
    {
        if pref == EpPreference::CudaRequired {
            let session = build()?
                .with_execution_providers([ort::ep::cuda::CUDA::default().build()])
                .map_err(|e| format!("CUDA execution provider registration failed: {e}"))?
                .commit_from_file(model_path)
                .map_err(|e| {
                    format!(
                        "failed to load reranker model {} on CUDA: {e}",
                        model_path.display()
                    )
                })?;
            eprintln!("rustsight: reranker running on CUDA");
            return Ok(session);
        }
    }
    #[cfg(not(feature = "onnx-rerank-cuda"))]
    {
        if pref == EpPreference::CudaRequired {
            ensure_cuda_feature()?;
        }
    }
    let session = build()?.commit_from_file(model_path).map_err(|e| {
        format!(
            "failed to load reranker model {}: {e}",
            model_path.display()
        )
    })?;
    eprintln!("rustsight: reranker running on CPU");
    Ok(session)
}

#[async_trait]
impl RerankProvider for OrtRerank {
    async fn rerank(
        &self,
        query: &str,
        docs: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let scores = self.tokenize_and_score(query, docs)?;
        Ok(select_top_k(scores, top_k))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sort scores descending (ties broken by lower index) and keep `top_k`.
pub(crate) fn select_top_k(scores: Vec<f32>, top_k: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(top_k);
    ranked
}

/// Right-pad a batch of token-id rows to `len`, producing flattened input
/// ids and an attention mask (1 for real tokens, 0 for padding).
pub(crate) fn padded_batch(rows: &[&[u32]], len: usize, pad_id: u32) -> (Vec<i64>, Vec<i64>) {
    let mut ids = Vec::with_capacity(rows.len() * len);
    let mut mask = Vec::with_capacity(rows.len() * len);
    for row in rows {
        let l = row.len();
        ids.extend(row.iter().map(|&v| v as i64));
        mask.extend(std::iter::repeat_n(1i64, l));
        ids.extend(std::iter::repeat_n(pad_id as i64, len - l));
        mask.extend(std::iter::repeat_n(0i64, len - l));
    }
    (ids, mask)
}

async fn resolve_model_files(
    spec: &OrtRerankSpec,
) -> std::result::Result<(PathBuf, PathBuf), String> {
    if let Some(local) = &spec.local_path {
        let (m, t) = if local.is_file() {
            (
                local.clone(),
                local
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("tokenizer.json"),
            )
        } else if local.is_dir() {
            (local.join("model.onnx"), local.join("tokenizer.json"))
        } else {
            return Err(format!(
                "RUSTSIGHT_RERANK_MODEL_PATH does not exist: {}",
                local.display()
            ));
        };
        if !m.is_file() {
            return Err(format!(
                "reranker model file not found at {} (point RUSTSIGHT_RERANK_MODEL_PATH at a model.onnx file or its directory)",
                m.display()
            ));
        }
        if !t.is_file() {
            return Err(format!(
                "tokenizer file not found next to the model at {} (expected tokenizer.json)",
                t.display()
            ));
        }
        return Ok((m, t));
    }

    let cache_root = match &spec.cache_dir {
        Some(d) => d.clone(),
        None => default_models_dir()?,
    };
    let dir = cache_root.join(spec.repo.replace('/', "--"));
    let model = dir.join("model.onnx");
    let tokenizer = dir.join("tokenizer.json");
    let data = dir.join("model.onnx_data");
    // fp32 exports keep their weights in a sibling `<repo_file>_data` blob
    // referenced by the bare name `model.onnx_data`; quantized exports are
    // self-contained. Probe for the sibling so both work.
    let data_repo_file = format!("{}_data", spec.repo_file);
    let token = hf_token();
    let external_data = hf_file_exists(&spec.repo, &data_repo_file, token.as_deref()).await?;
    if model.is_file() && tokenizer.is_file() && (!external_data || data.is_file()) {
        return Ok((model, tokenizer));
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create models cache {}: {e}", dir.display()))?;
    download_hf_file(&spec.repo, "tokenizer.json", &tokenizer, token.as_deref()).await?;
    download_hf_file(&spec.repo, &spec.repo_file, &model, token.as_deref()).await?;
    if external_data {
        download_hf_file(&spec.repo, &data_repo_file, &data, token.as_deref()).await?;
    }
    Ok((model, tokenizer))
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
}

async fn hf_file_exists(
    repo: &str,
    file: &str,
    token: Option<&str>,
) -> std::result::Result<bool, String> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let client = reqwest::Client::new();
    let mut req = client.head(&url);
    if let Some(token) = token {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("HEAD {url} failed: {e}"))?;
    Ok(resp.status().is_success())
}

fn default_models_dir() -> std::result::Result<PathBuf, String> {
    let home = std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .ok_or_else(|| {
            "HOME is not set; set RUSTSIGHT_MODELS_DIR to a writable directory".to_string()
        })?;
    Ok(PathBuf::from(home).join(".rustsight").join("models"))
}

/// Stream `file` from HuggingFace `repo` into `dest` (via a `.part` temp
/// file, renamed into place on success so a partial download never shadows a
/// good model).
async fn download_hf_file(
    repo: &str,
    file: &str,
    dest: &Path,
    hf_token: Option<&str>,
) -> std::result::Result<(), String> {
    if dest.is_file() {
        return Ok(());
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    eprintln!("rustsight: downloading {url} -> {}", dest.display());
    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if let Some(token) = hf_token {
        req = req.bearer_auth(token);
    }
    let mut resp = req
        .send()
        .await
        .map_err(|e| format!("download failed for {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download failed for {url}: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let tmp = dest.with_extension("part");
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&tmp).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?,
    );
    let mut done: u64 = 0;
    let mut next_progress: u64 = 64 * 1024 * 1024;
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("download failed for {url}: {e}"))?
    {
        out.write_all(&chunk)
            .map_err(|e| format!("write to {} failed: {e}", tmp.display()))?;
        done += chunk.len() as u64;
        if done >= next_progress {
            let progress = if total > 0 {
                format!("{}/{} MB", done / 1048576, total / 1048576)
            } else {
                format!("{} MB", done / 1048576)
            };
            eprintln!("rustsight:   {progress}");
            next_progress += 64 * 1024 * 1024;
        }
    }
    out.flush()
        .map_err(|e| format!("write to {} failed: {e}", tmp.display()))?;
    drop(out);
    std::fs::rename(&tmp, dest)
        .map_err(|e| format!("rename {} -> {} failed: {e}", tmp.display(), dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_maps_logits_to_scores() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(8.0) > 0.999);
        assert!(sigmoid(-8.0) < 0.001);
        assert!(sigmoid(2.0) > sigmoid(1.0));
    }

    #[test]
    fn select_top_k_orders_and_truncates() {
        let scores = vec![0.1, 0.9, 0.5, 0.9];
        let top = select_top_k(scores, 2);
        assert_eq!(top.len(), 2);
        // ties broken by lower index
        assert_eq!(top[0], (1, 0.9));
        assert_eq!(top[1], (3, 0.9));
        assert!(select_top_k(vec![], 5).is_empty());
        // top_k larger than input returns everything
        assert_eq!(select_top_k(vec![0.3, 0.7], 10).len(), 2);
    }

    #[test]
    fn padded_batch_pads_right_with_mask() {
        let rows: Vec<&[u32]> = vec![&[1, 2, 3], &[4, 5]];
        let (ids, mask) = padded_batch(&rows, 3, 9);
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 9]);
        assert_eq!(mask, vec![1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn cuda_pairing_swaps_only_the_cpu_defaults() {
        assert_eq!(
            cuda_paired_repo_file(crate::env::DEFAULT_RERANK_REPO_FILE),
            DEFAULT_RERANK_REPO_FILE_CUDA
        );
        assert_eq!(
            cuda_paired_repo(crate::env::DEFAULT_RERANK_REPO),
            DEFAULT_RERANK_REPO_CUDA
        );
        // explicit user choices pass through untouched
        assert_eq!(
            cuda_paired_repo_file("onnx/model_fp16.onnx"),
            "onnx/model_fp16.onnx"
        );
        assert_eq!(cuda_paired_repo_file("onnx/model.onnx"), "onnx/model.onnx");
        assert_eq!(
            cuda_paired_repo_file("weird/export.onnx"),
            "weird/export.onnx"
        );
        assert_eq!(
            cuda_paired_repo("onnx-community/bge-reranker-v2-m3-ONNX"),
            "onnx-community/bge-reranker-v2-m3-ONNX"
        );
    }

    /// End-to-end check with the real model. Downloads ~544 MB (int8) on a
    /// CPU-only host or ~1.1 GB (fp16) where CUDA resolves, on first run
    /// (cached under ~/.rustsight/models afterwards), so it is skipped in
    /// normal offline runs:
    /// `cargo test -p rustsight-providers --features onnx-rerank -- --ignored`
    #[tokio::test]
    #[ignore = "downloads the reranker model on first run; run explicitly with --ignored"]
    async fn real_model_scores_relevant_docs_higher() {
        let rerank = OrtRerank::load(OrtRerankSpec::default()).await.unwrap();
        let docs: Vec<String> = vec![
            "The capital of France is Paris.".into(),
            "Quarterly revenue increased by 12 percent year over year.".into(),
            "Alice loves hiking in the mountains on weekends.".into(),
            "The printer in the office is out of toner again.".into(),
            "Rust compiles to native machine code.".into(),
        ];
        let out = rerank
            .rerank(
                "What is Alice's favorite outdoor activity?",
                &docs,
                docs.len(),
            )
            .await
            .unwrap();
        assert_eq!(out.len(), docs.len());
        assert_eq!(out[0].0, 2, "hiking doc should rank first: {out:?}");
        for (i, s) in out.iter().skip(1) {
            assert!(out[0].1 > *s + 0.05, "scores: {out:?} (doc {i} too close)");
        }
    }
}
