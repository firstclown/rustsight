//! Cost accounting: thin wrappers around the engine's providers that tally
//! request counts and token estimates (the providers do not surface usage
//! yet; estimates use the configured `TokenEstimator`).

use async_trait::async_trait;
use rustsight_core::traits::{EmbedProvider, LlmProvider, LlmRequest, LlmResponse, TokenEstimator};
use rustsight_core::Result;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default)]
pub struct Usage {
    requests: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
}

impl Usage {
    pub(crate) fn add(&self, input: u64, output: u64) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.input_tokens.fetch_add(input, Ordering::Relaxed);
        self.output_tokens.fetch_add(output, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct UsageSnapshot {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl From<&Usage> for UsageSnapshot {
    fn from(u: &Usage) -> Self {
        Self {
            requests: u.requests.load(Ordering::Relaxed),
            input_tokens: u.input_tokens.load(Ordering::Relaxed),
            output_tokens: u.output_tokens.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CountersSnapshot {
    pub llm: UsageSnapshot,
    pub embed: UsageSnapshot,
    pub judge: UsageSnapshot,
}

#[derive(Clone)]
pub struct Counters {
    llm: Arc<Usage>,
    embed: Arc<Usage>,
    judge: Arc<Usage>,
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            llm: Arc::new(Usage::default()),
            embed: Arc::new(Usage::default()),
            judge: Arc::new(Usage::default()),
        }
    }
}

impl Counters {
    pub fn llm(&self) -> Arc<Usage> {
        Arc::clone(&self.llm)
    }

    pub fn embed(&self) -> Arc<Usage> {
        Arc::clone(&self.embed)
    }

    pub fn judge(&self) -> Arc<Usage> {
        Arc::clone(&self.judge)
    }

    pub fn snapshot(&self) -> CountersSnapshot {
        CountersSnapshot {
            llm: UsageSnapshot::from(self.llm.as_ref()),
            embed: UsageSnapshot::from(self.embed.as_ref()),
            judge: UsageSnapshot::from(self.judge.as_ref()),
        }
    }
}

pub struct CountingLlm {
    inner: Arc<dyn LlmProvider>,
    usage: Arc<Usage>,
    est: Arc<dyn TokenEstimator>,
}

impl CountingLlm {
    pub fn new(
        inner: Arc<dyn LlmProvider>,
        usage: Arc<Usage>,
        est: Arc<dyn TokenEstimator>,
    ) -> Self {
        Self { inner, usage, est }
    }
}

#[async_trait]
impl LlmProvider for CountingLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let resp = self.inner.complete(req).await?;
        let mut input = self.est.count(req.system.as_deref().unwrap_or(""));
        for m in &req.messages {
            input += self.est.count(&m.content);
        }
        let mut output = 0usize;
        if let Some(c) = &resp.content {
            output += self.est.count(c);
        }
        for call in &resp.tool_calls {
            output += self.est.count(&call.arguments.to_string());
        }
        self.usage.add(input as u64, output as u64);
        Ok(resp)
    }
}

pub struct CountingEmbed {
    inner: Arc<dyn EmbedProvider>,
    usage: Arc<Usage>,
    est: Arc<dyn TokenEstimator>,
}

impl CountingEmbed {
    pub fn new(
        inner: Arc<dyn EmbedProvider>,
        usage: Arc<Usage>,
        est: Arc<dyn TokenEstimator>,
    ) -> Self {
        Self { inner, usage, est }
    }
}

#[async_trait]
impl EmbedProvider for CountingEmbed {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let tokens: usize = texts.iter().map(|t| self.est.count(t)).sum();
        let out = self.inner.embed(texts).await?;
        self.usage.add(tokens as u64, 0);
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustsight_core::traits::{ChatMessage, LlmResponse};
    use rustsight_providers::{MockEmbed, MockLlm, SimpleTokenEstimator};

    #[tokio::test]
    async fn counts_llm_traffic() {
        let counters = Counters::default();
        let llm = CountingLlm::new(
            Arc::new(MockLlm::script(vec![LlmResponse {
                content: Some("four four four four".into()),
                tool_calls: vec![],
            }])),
            counters.llm(),
            Arc::new(SimpleTokenEstimator),
        );
        let req = LlmRequest {
            system: Some("system prompt".into()),
            messages: vec![ChatMessage::user("hello world")],
            ..Default::default()
        };
        llm.complete(&req).await.unwrap();
        let snap = counters.snapshot();
        assert_eq!(snap.llm.requests, 1);
        assert!(snap.llm.input_tokens > 0);
        assert!(snap.llm.output_tokens > 0);
        assert_eq!(snap.judge.requests, 0);
    }

    #[tokio::test]
    async fn counts_embed_tokens() {
        let counters = Counters::default();
        let embed = CountingEmbed::new(
            Arc::new(MockEmbed::default()),
            counters.embed(),
            Arc::new(SimpleTokenEstimator),
        );
        embed.embed(vec!["some text".into()]).await.unwrap();
        assert_eq!(counters.snapshot().embed.requests, 1);
        assert!(counters.snapshot().embed.input_tokens > 0);
    }
}
