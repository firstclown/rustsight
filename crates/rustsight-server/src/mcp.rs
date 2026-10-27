use rustsight_core::types::{RecallRequest, RetainRequest};
use rustsight_core::Engine;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "retain",
                "description": "Store information in the memory bank (extracts facts, entities, links)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "content": {"type": "string"},
                        "context": {"type": "string"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "recall",
                "description": "Search the memory bank with hybrid retrieval (semantic, keyword, graph, temporal)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "query": {"type": "string"},
                        "token_budget": {"type": "integer", "default": 1000}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "reflect",
                "description": "Agentic reasoning over the memory bank; returns an answer with citations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
    })
}

async fn call_tool(
    engine: &Engine,
    default_bank: &Option<String>,
    params: &Value,
) -> Result<Value, String> {
    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    let bank_id = args
        .get("bank_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| default_bank.clone())
        .ok_or_else(|| "bank_id required (argument or --default-bank)".to_string())?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    match name {
        "retain" => {
            let content = args
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or("content required")?;
            let r = engine
                .retain(RetainRequest {
                    bank_id,
                    content: content.to_string(),
                    context: args
                        .get("context")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    timestamp_ms: None,
                })
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "memories_stored": r.memories_stored,
                "entities": r.entities,
                "edges_created": r.edges_created
            }))
        }
        "recall" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or("query required")?;
            let r = engine
                .recall(RecallRequest {
                    bank_id,
                    query: query.to_string(),
                    token_budget: args
                        .get("token_budget")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1000) as usize,
                    top_k_per_arm: 10,
                    kinds: None,
                })
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "memories": r.memories,
                "strategies_used": r.strategies_used
            }))
        }
        "reflect" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or("query required")?;
            let r = engine
                .reflect(&bank_id, query)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "answer": r.answer,
                "citations": r.citations,
                "rounds": r.rounds
            }))
        }
        other => Err(format!("unknown tool {other}")),
    }
}

pub async fn run_stdio(
    engine: Arc<Engine>,
    default_bank: Option<String>,
) -> rustsight_core::Result<()> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| rustsight_core::Error::Store(format!("stdin: {e}")))?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = v.get("id").cloned();
        let method = v
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string();
        if method.starts_with("notifications/") {
            continue;
        }
        let outcome: Result<Value, String> = match method.as_str() {
            "initialize" => Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "rustsight", "version": env!("CARGO_PKG_VERSION")}
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools_list()),
            "tools/call" => call_tool(&engine, &default_bank, &v["params"])
                .await
                .map(|result| json!({"content": [{"type": "text", "text": result.to_string()}]})),
            other => Err(format!("unknown method {other}")),
        };
        let resp = match outcome {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(msg) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32603, "message": msg}})
            }
        };
        let mut out = resp.to_string();
        out.push('\n');
        stdout
            .write_all(out.as_bytes())
            .await
            .map_err(|e| rustsight_core::Error::Store(format!("stdout: {e}")))?;
        stdout
            .flush()
            .await
            .map_err(|e| rustsight_core::Error::Store(format!("stdout flush: {e}")))?;
    }
    Ok(())
}
