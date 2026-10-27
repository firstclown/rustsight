use async_trait::async_trait;
use rustsight_core::traits::{
    EmbedProvider, LlmProvider, LlmRequest, LlmResponse, RerankProvider, TokenEstimator, ToolCall,
};
use rustsight_core::{Error, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

enum Mode {
    Script(VecDeque<LlmResponse>),
    Demo,
}

pub struct MockLlm {
    mode: Mutex<Mode>,
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl MockLlm {
    pub fn script(responses: Vec<LlmResponse>) -> Self {
        Self {
            mode: Mutex::new(Mode::Script(responses.into_iter().collect())),
        }
    }

    pub fn demo() -> Self {
        Self {
            mode: Mutex::new(Mode::Demo),
        }
    }
}

const STOPWORDS: &[&str] = &[
    "i",
    "she",
    "he",
    "it",
    "they",
    "we",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "our",
    "his",
    "her",
    "their",
    "last",
    "next",
    "yesterday",
    "today",
    "tomorrow",
    "in",
    "on",
    "at",
    "a",
    "an",
    "and",
    "but",
    "so",
    "then",
    "because",
    "when",
    "while",
    "after",
    "before",
    "there",
    "here",
    "what",
    "who",
    "why",
    "how",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

fn demo_extraction(content: &str) -> Value {
    let content = content.split("CONTENT:\n").nth(1).unwrap_or(content);
    let content = content
        .split("\n\nRespond with")
        .next()
        .unwrap_or(content)
        .trim();
    let mut facts = vec![];
    for sentence in content.split(['.', '!', '?']) {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut entities: Vec<Value> = vec![];
        let mut seen = std::collections::HashSet::new();
        for w in &words {
            let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() > 2
                && clean.chars().next().is_some_and(|c| c.is_uppercase())
                && clean.chars().all(|c| c.is_alphabetic())
                && !STOPWORDS.contains(&clean.to_lowercase().as_str())
                && seen.insert(clean.to_lowercase())
            {
                entities.push(json!({"name": clean, "etype": "person"}));
            }
        }
        facts.push(json!({
            "text": sentence,
            "fact_type": "world",
            "entities": entities,
            "occurred_start": null,
            "occurred_end": null,
            "causal": []
        }));
        if facts.len() >= 6 {
            break;
        }
    }
    json!({"facts": facts})
}

fn demo_consolidation(user: &str) -> Value {
    let mut observations = vec![];
    let mut indexes = vec![];
    for (i, line) in user.lines().enumerate() {
        let line = line.trim();
        if line.starts_with('[') {
            let text = line.split("): ").nth(1).unwrap_or(line);
            indexes.push(i);
            let _ = text;
        }
    }
    if indexes.is_empty() {
        return json!({"observations": []});
    }
    let texts: Vec<&str> = user
        .lines()
        .filter(|l| {
            l.trim_start_matches('[')
                .starts_with(|c: char| c.is_ascii_digit())
        })
        .collect();
    let summary = if texts.is_empty() {
        "Consolidated profile.".to_string()
    } else {
        let joined = texts
            .iter()
            .map(|t| t.split("): ").nth(1).unwrap_or(t))
            .collect::<Vec<_>>()
            .join(" ");
        format!("Summary: {}", joined.chars().take(400).collect::<String>())
    };
    let evidence: Vec<Value> = (0..texts.len())
        .map(|i| json!({"fact_index": i, "quote": ""}))
        .collect();
    observations.push(json!({"text": summary, "evidence": evidence}));
    json!({"observations": observations})
}

#[async_trait]
impl LlmProvider for MockLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let mut guard = self.mode.lock().unwrap();
        match &mut *guard {
            Mode::Script(queue) => Ok(queue.pop_front().unwrap_or(LlmResponse {
                content: Some("{}".into()),
                tool_calls: vec![],
            })),
            Mode::Demo => {
                let system = req.system.clone().unwrap_or_default();
                if system.contains("FACT EXTRACTION ENGINE") {
                    let user_msg = req
                        .messages
                        .iter()
                        .rev()
                        .find(|m| m.role == rustsight_core::Role::User)
                        .map(|m| m.content.clone())
                        .unwrap_or_default();
                    return Ok(LlmResponse {
                        content: Some(demo_extraction(&user_msg).to_string()),
                        tool_calls: vec![],
                    });
                }
                if system.contains("OBJECTIVE OBSERVER") {
                    let user = req
                        .messages
                        .last()
                        .map(|m| m.content.clone())
                        .unwrap_or_default();
                    return Ok(LlmResponse {
                        content: Some(demo_consolidation(&user).to_string()),
                        tool_calls: vec![],
                    });
                }
                if system.contains("OPINION REINFORCEMENT") {
                    return Ok(LlmResponse {
                        content: Some(r#"{"changed":false}"#.into()),
                        tool_calls: vec![],
                    });
                }
                if system.contains("OPINION EXTRACTION") {
                    return Ok(LlmResponse {
                        content: Some(r#"{"opinions":[]}"#.into()),
                        tool_calls: vec![],
                    });
                }
                if system.contains("REFLECTIVE REASONING LOOP") {
                    let tool_msgs = req
                        .messages
                        .iter()
                        .filter(|m| m.role == rustsight_core::Role::Tool)
                        .count();
                    if tool_msgs == 0 {
                        let query = req
                            .messages
                            .iter()
                            .rev()
                            .find(|m| m.role == rustsight_core::Role::User)
                            .map(|m| m.content.clone())
                            .unwrap_or_default();
                        return Ok(LlmResponse {
                            content: None,
                            tool_calls: vec![ToolCall {
                                id: "c1".into(),
                                name: "recall".into(),
                                arguments: json!({"query": query, "token_budget": 800}),
                            }],
                        });
                    }
                    let answer = format!(
                        "Based on the {} tool result(s) retrieved from memory, here is the demo reflection.",
                        tool_msgs
                    );
                    return Ok(LlmResponse {
                        content: None,
                        tool_calls: vec![ToolCall {
                            id: "c2".into(),
                            name: "finish".into(),
                            arguments: json!({"answer": answer, "citations": []}),
                        }],
                    });
                }
                if system.contains("MENTAL MODEL WRITER") {
                    return Ok(LlmResponse {
                        content: Some(
                            "Standing answer (demo): the bank is currently small.".into(),
                        ),
                        tool_calls: vec![],
                    });
                }
                Ok(LlmResponse {
                    content: Some("ok".into()),
                    tool_calls: vec![],
                })
            }
        }
    }
}

pub struct MockEmbed {
    pub dim: usize,
}

impl Default for MockEmbed {
    fn default() -> Self {
        Self { dim: 512 }
    }
}

#[async_trait]
impl EmbedProvider for MockEmbed {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; self.dim];
                for w in t.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                    if w.is_empty() {
                        continue;
                    }
                    v[(fnv1a(w) % self.dim as u64) as usize] += 1.0;
                }
                if v.iter().all(|x| *x == 0.0) {
                    v[0] = 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

pub struct MockRerank;

#[async_trait]
impl RerankProvider for MockRerank {
    async fn rerank(
        &self,
        _query: &str,
        docs: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        Ok((0..docs.len().min(top_k)).map(|i| (i, 1.0)).collect())
    }
}

pub struct SimpleTokenEstimator;

impl TokenEstimator for SimpleTokenEstimator {
    fn count(&self, text: &str) -> usize {
        text.chars().count() / 4 + 1
    }
}

/// Wraps an LLM provider and fails the first `fail_calls` calls with the
/// given HTTP status (as a typed `Error::LlmStatus`), then delegates. Used to
/// exercise the engine's retry path against transient 429/5xx failures.
pub struct FlakyLlm {
    inner: Arc<dyn LlmProvider>,
    failures_left: AtomicUsize,
    status: u16,
}

impl FlakyLlm {
    pub fn new(inner: Arc<dyn LlmProvider>, fail_calls: usize, status: u16) -> Self {
        Self {
            inner,
            failures_left: AtomicUsize::new(fail_calls),
            status,
        }
    }
}

#[async_trait]
impl LlmProvider for FlakyLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let left = self.failures_left.load(Ordering::SeqCst);
        if left > 0 {
            self.failures_left.store(left - 1, Ordering::SeqCst);
            return Err(Error::LlmStatus {
                status: self.status,
                message: "simulated transient failure".into(),
            });
        }
        self.inner.complete(req).await
    }
}

/// Embedding-side twin of [`FlakyLlm`]: fails the first `fail_calls` calls
/// with `Error::EmbedStatus`, then delegates.
pub struct FlakyEmbed {
    inner: Arc<dyn EmbedProvider>,
    failures_left: AtomicUsize,
    status: u16,
}

impl FlakyEmbed {
    pub fn new(inner: Arc<dyn EmbedProvider>, fail_calls: usize, status: u16) -> Self {
        Self {
            inner,
            failures_left: AtomicUsize::new(fail_calls),
            status,
        }
    }
}

#[async_trait]
impl EmbedProvider for FlakyEmbed {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let left = self.failures_left.load(Ordering::SeqCst);
        if left > 0 {
            self.failures_left.store(left - 1, Ordering::SeqCst);
            return Err(Error::EmbedStatus {
                status: self.status,
                message: "simulated transient failure".into(),
            });
        }
        self.inner.embed(texts).await
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
}
