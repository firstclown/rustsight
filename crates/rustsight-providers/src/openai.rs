use async_trait::async_trait;
use rustsight_core::traits::{
    ChatMessage, EmbedProvider, LlmProvider, LlmRequest, LlmResponse, RerankProvider, Role,
    ToolCall,
};
use rustsight_core::{Error, Result};
use serde_json::{json, Value};
use std::sync::Mutex;

pub struct OpenAiCompat {
    pub client: reqwest::Client,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Formats an error with its full cause chain. reqwest's own Display stops
/// at the first layer — "error decoding response body" — hiding whether the
/// body died to a timeout, a dropped connection, or bad bytes, exactly the
/// distinction that decides retry-vs-abort upstream.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        out.push_str(": ");
        out.push_str(&s.to_string());
        src = s.source();
    }
    out
}

/// Models that take `max_completion_tokens` only and need room for hidden
/// reasoning (upstream `_supports_reasoning_model`).
fn is_reasoning_model(model: &str) -> bool {
    let model = model.to_lowercase();
    if model.contains("deepseek") {
        return ["v4-pro", "reasoner", "r1", "thinking"]
            .iter()
            .any(|m| model.contains(m));
    }
    ["gpt-5", "gpt-6", "o1", "o3"]
        .iter()
        .any(|m| model.contains(m))
}

/// The request field and value for an output cap, following upstream's
/// per-model limits so a large default stays valid on smaller models.
fn output_limit(model: &str, base_url: &str, requested: u64) -> (&'static str, u64) {
    let lowered = model.to_lowercase();
    let mut limit = requested;
    if lowered.contains("gpt-4.1") || lowered.contains("gpt-4-") {
        limit = limit.min(32_000);
    } else if lowered.contains("gpt-4o") {
        limit = limit.min(16_384);
    }
    let reasoning = is_reasoning_model(model);
    if reasoning {
        limit = limit.max(16_000);
    }
    let native_openai =
        base_url.contains("api.openai.com") || base_url.contains(".openai.azure.com");
    let field = if reasoning || native_openai {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    (field, limit)
}

/// OpenRouter load-balances a model across upstreams, and one that ignores
/// `response_format` can answer with empty content and a normal finish
/// reason. Limit routing to upstreams that honour every parameter sent
/// (upstream `require_parameters`).
fn require_supported_parameters(body: &mut Value, base_url: &str) {
    let is_openrouter =
        reqwest::Url::parse(base_url).is_ok_and(|url| url.host_str() == Some("openrouter.ai"));
    if is_openrouter && body.get("response_format").is_some() {
        body["provider"] = json!({"require_parameters": true});
    }
}

fn map_req(req: &LlmRequest) -> Vec<Value> {
    let mut messages: Vec<Value> = vec![];
    if let Some(s) = &req.system {
        messages.push(json!({"role": "system", "content": s}));
    }
    for m in &req.messages {
        match m.role {
            Role::User => messages.push(json!({"role": "user", "content": m.content})),
            Role::Assistant => {
                let mut v = json!({"role": "assistant", "content": m.content});
                if !m.tool_calls.is_empty() {
                    v["tool_calls"] = m
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments.to_string()}
                            })
                        })
                        .collect();
                }
                messages.push(v);
            }
            Role::Tool => messages.push(json!({
                "role": "tool",
                "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
                "content": m.content
            })),
        }
    }
    messages
}

#[async_trait]
impl LlmProvider for OpenAiCompat {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": map_req(req),
            "temperature": req.temperature.unwrap_or(0.2),
        });
        if !req.tools.is_empty() {
            body["tools"] = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            // Forced named tool (upstream `tool_choice=done` on reflect's
            // closing ask) or the model's choice.
            body["tool_choice"] = match &req.tool_choice {
                Some(name) => json!({"type": "function", "function": {"name": name}}),
                None => json!("auto"),
            };
        }
        if req.json_mode {
            body["response_format"] = json!({"type": "json_object"});
            require_supported_parameters(&mut body, &self.base_url);
        }
        if let Some(mt) = req.max_tokens {
            let (field, limit) = output_limit(&self.model, &self.base_url, mt);
            body[field] = json!(limit);
        }
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::LlmTransport(error_chain(&e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::LlmTransport(error_chain(&e)))?;
        if !status.is_success() {
            return Err(Error::LlmStatus {
                status: status.as_u16(),
                message: format!("{}: {}", status, truncate(&text, 400)),
            });
        }
        let v: Value =
            serde_json::from_str(&text).map_err(|e| Error::Llm(format!("bad json: {e}")))?;
        let msg = &v["choices"][0]["message"];
        let content = msg["content"].as_str().map(|s| s.to_string());
        let tool_calls: Vec<ToolCall> = msg["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let name = tc["function"]["name"].as_str()?;
                        let args_raw = tc["function"]["arguments"].as_str().unwrap_or("{}");
                        let arguments: Value =
                            serde_json::from_str(args_raw).unwrap_or_else(|_| json!({}));
                        Some(ToolCall {
                            id: tc["id"].as_str().unwrap_or("call").to_string(),
                            name: name.to_string(),
                            arguments,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }
}

pub struct OpenAiEmbed {
    pub client: reqwest::Client,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub dim_cache: Mutex<usize>,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(serde::Deserialize)]
struct EmbedItem {
    index: usize,
    embedding: Vec<f64>,
}

#[async_trait]
impl EmbedProvider for OpenAiEmbed {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&json!({"model": self.model, "input": texts}))
            .send()
            .await
            .map_err(|e| Error::EmbedTransport(error_chain(&e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::EmbedTransport(error_chain(&e)))?;
        if !status.is_success() {
            return Err(Error::EmbedStatus {
                status: status.as_u16(),
                message: format!("{}: {}", status, truncate(&text, 400)),
            });
        }
        let v: EmbedResponse =
            serde_json::from_str(&text).map_err(|e| Error::Embed(format!("bad json: {e}")))?;
        let mut items = v.data;
        items.sort_by_key(|i| i.index);
        let out: Vec<Vec<f32>> = items
            .into_iter()
            .map(|i| i.embedding.into_iter().map(|x| x as f32).collect())
            .collect();
        let n = out.len();
        *self.dim_cache.lock().unwrap() = n.max(1);
        Ok(out)
    }

    fn dim(&self) -> usize {
        *self.dim_cache.lock().unwrap()
    }
}

pub struct ApiRerank {
    pub client: reqwest::Client,
    pub url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(serde::Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(serde::Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl RerankProvider for ApiRerank {
    async fn rerank(
        &self,
        query: &str,
        docs: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
        let resp = self
            .client
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": top_k
            }))
            .send()
            .await
            .map_err(|e| Error::Rerank(error_chain(&e)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::Rerank(error_chain(&e)))?;
        if !status.is_success() {
            return Err(Error::Rerank(format!(
                "{}: {}",
                status,
                truncate(&text, 400)
            )));
        }
        let v: RerankResponse =
            serde_json::from_str(&text).map_err(|e| Error::Rerank(format!("bad json: {e}")))?;
        Ok(v.results
            .into_iter()
            .map(|r| (r.index, r.relevance_score))
            .collect())
    }
}

/// The no-op reranker: no model, no relevance signal. Returns no scores;
/// the engine detects `is_passthrough()` and seeds the combined-scoring
/// base from RRF rank (upstream's `RRFPassthroughCrossEncoder`).
pub struct NoopRerank;

#[async_trait]
impl RerankProvider for NoopRerank {
    async fn rerank(
        &self,
        _query: &str,
        _docs: &[String],
        _top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        Ok(vec![])
    }

    fn is_passthrough(&self) -> bool {
        true
    }
}

pub fn chat_message_debug(m: &ChatMessage) -> String {
    format!("{:?}: {}", m.role, m.content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Nest(&'static str, Option<Box<Nest>>);

    impl std::fmt::Display for Nest {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for Nest {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.1
                .as_deref()
                .map(|inner| inner as &(dyn std::error::Error + 'static))
        }
    }

    #[test]
    fn json_requests_to_openrouter_require_supported_parameters() {
        let json_body = || json!({"response_format": {"type": "json_object"}});

        let mut body = json_body();
        require_supported_parameters(&mut body, "https://openrouter.ai/api/v1");
        assert_eq!(body["provider"], json!({"require_parameters": true}));

        let mut body = json_body();
        require_supported_parameters(&mut body, "https://api.openai.com/v1");
        assert!(body.get("provider").is_none());

        let mut body = json_body();
        require_supported_parameters(&mut body, "https://openrouter.ai.example.com/v1");
        assert!(body.get("provider").is_none());

        let mut body = json!({});
        require_supported_parameters(&mut body, "https://openrouter.ai/api/v1");
        assert!(body.get("provider").is_none());
    }

    #[test]
    fn output_limit_follows_the_model_and_endpoint() {
        let openrouter = "https://openrouter.ai/api/v1";
        let openai = "https://api.openai.com/v1";
        assert_eq!(
            output_limit("openai/gpt-oss-120b", openrouter, 64_000),
            ("max_tokens", 64_000)
        );
        assert_eq!(
            output_limit("gpt-4o-mini", openai, 64_000),
            ("max_completion_tokens", 16_384)
        );
        assert_eq!(
            output_limit("openai/gpt-4.1", openrouter, 64_000),
            ("max_tokens", 32_000)
        );
        assert_eq!(
            output_limit("openai/gpt-5", openrouter, 8_192),
            ("max_completion_tokens", 16_000)
        );
    }

    /// The chain must surface the layers reqwest's Display hides: a bare
    /// "error decoding response body" says nothing about retryability,
    /// while its cause ("operation timed out") decides it.
    #[test]
    fn error_chain_includes_every_cause_layer() {
        let err = Nest(
            "error decoding response body",
            Some(Box::new(Nest(
                "request or response body error",
                Some(Box::new(Nest("operation timed out", None))),
            ))),
        );
        assert_eq!(
            error_chain(&err),
            "error decoding response body: request or response body error: operation timed out"
        );
        assert_eq!(error_chain(&Nest("flat", None)), "flat");
    }
}
