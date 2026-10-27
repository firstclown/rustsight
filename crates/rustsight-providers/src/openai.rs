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
            body["tool_choice"] = json!("auto");
        }
        if req.json_mode {
            body["response_format"] = json!({"type": "json_object"});
        }
        if let Some(mt) = req.max_tokens {
            body["max_tokens"] = json!(mt);
        }
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Llm(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| Error::Llm(e.to_string()))?;
        if !status.is_success() {
            return Err(Error::Llm(format!("{}: {}", status, truncate(&text, 400))));
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
            .map_err(|e| Error::Embed(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| Error::Embed(e.to_string()))?;
        if !status.is_success() {
            return Err(Error::Embed(format!(
                "{}: {}",
                status,
                truncate(&text, 400)
            )));
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
            .map_err(|e| Error::Rerank(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::Rerank(e.to_string()))?;
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

pub struct NoopRerank;

#[async_trait]
impl RerankProvider for NoopRerank {
    async fn rerank(
        &self,
        _query: &str,
        docs: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        Ok((0..docs.len().min(top_k))
            .map(|i| (i, 1.0 - i as f32 * 0.001))
            .collect())
    }
}

pub fn chat_message_debug(m: &ChatMessage) -> String {
    format!("{:?}: {}", m.role, m.content)
}
