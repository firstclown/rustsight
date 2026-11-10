//! `download`: fetch the LongMemEval-S JSON from HuggingFace into the
//! default eval cache. The file is ~265 MB (each question embeds its full
//! haystack, so questions sharing a conversation duplicate it); downloads
//! stream to a `.part` file that is renamed into place on success.

use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// The original repo stores the S variant as an extensionless LFS file
/// (`longmemeval_s`, ~265 MB) — the `.json` suffix only exists in the
/// `longmemeval-cleaned` repo.
pub const LONGMEMEVAL_S_URL: &str =
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s";

pub fn default_dest() -> PathBuf {
    let home = std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".rustsight/eval/longmemeval_s.json")
}

pub async fn download(dest: Option<PathBuf>) -> Result<PathBuf, String> {
    let dest = dest.unwrap_or_else(default_dest);
    if dest.exists() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir failed: {e}"))?;
    }
    let client = reqwest::Client::new();
    let resp = client
        .get(LONGMEMEVAL_S_URL)
        .send()
        .await
        .map_err(|e| format!("download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "download failed: HTTP {} for {LONGMEMEVAL_S_URL}",
            resp.status()
        ));
    }
    let part = dest.with_extension("part");
    let mut file = std::fs::File::create(&part).map_err(|e| format!("create failed: {e}"))?;
    let mut resp = resp;
    let mut bytes: u64 = 0;
    let mut sha = Sha256::new();
    use std::io::Write;
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("download stream failed: {e}"))?
    {
        sha.update(&chunk);
        file.write_all(&chunk)
            .map_err(|e| format!("write failed: {e}"))?;
        bytes += chunk.len() as u64;
        if bytes % (10 * 1024 * 1024) < chunk.len() as u64 {
            eprintln!("rustsight-eval: downloaded {} MB", bytes / (1024 * 1024));
        }
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("rename failed: {e}"))?;
    eprintln!(
        "rustsight-eval: {} ({:.1} MB, sha256 {:02x}) saved to {}",
        LONGMEMEVAL_S_URL,
        bytes as f64 / (1024.0 * 1024.0),
        sha.finalize(),
        dest.display()
    );
    Ok(dest)
}
