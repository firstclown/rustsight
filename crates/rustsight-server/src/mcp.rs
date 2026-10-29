use rustsight_core::types::{RecallRequest, RetainRequest};
use rustsight_core::Engine;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::consolidate_interval_from_env;

const MAX_LINE_BYTES: usize = 10 * 1024 * 1024;

enum LineIn {
    Line(String),
    Eof,
    TooLarge,
}

async fn drain_to_newline<R: AsyncBufRead + Unpin>(reader: &mut R) -> std::io::Result<()> {
    loop {
        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            return Ok(());
        }
        match chunk.iter().position(|&b| b == b'\n') {
            Some(pos) => {
                reader.consume(pos + 1);
                return Ok(());
            }
            None => {
                let len = chunk.len();
                reader.consume(len);
            }
        }
    }
}

async fn read_line_capped<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    cap: usize,
) -> std::io::Result<LineIn> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            return Ok(if out.is_empty() {
                LineIn::Eof
            } else {
                LineIn::Line(String::from_utf8_lossy(&out).into_owned())
            });
        }
        if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
            let keep = pos.min(cap.saturating_sub(out.len()));
            out.extend_from_slice(&chunk[..keep]);
            let too_large = pos > keep;
            reader.consume(pos + 1);
            return if too_large {
                Ok(LineIn::TooLarge)
            } else {
                Ok(LineIn::Line(String::from_utf8_lossy(&out).into_owned()))
            };
        }
        let keep = chunk.len().min(cap.saturating_sub(out.len()));
        out.extend_from_slice(&chunk[..keep]);
        let len = chunk.len();
        reader.consume(len);
        if len > keep {
            drain_to_newline(reader).await?;
            return Ok(LineIn::TooLarge);
        }
    }
}

async fn send<W: AsyncWrite + Unpin>(
    stdout: &mut W,
    resp: &Value,
) -> Result<(), rustsight_core::Error> {
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
    Ok(())
}

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
    run_io(
        engine,
        default_bank,
        BufReader::new(tokio::io::stdin()),
        tokio::io::stdout(),
        MAX_LINE_BYTES,
    )
    .await
}

pub(crate) async fn run_io<R, W>(
    engine: Arc<Engine>,
    default_bank: Option<String>,
    mut reader: R,
    mut stdout: W,
    max_line: usize,
) -> rustsight_core::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let worker = engine.spawn_worker(consolidate_interval_from_env(), 50);
    loop {
        match read_line_capped(&mut reader, max_line).await {
            Err(e) => return Err(rustsight_core::Error::Store(format!("stdin: {e}"))),
            Ok(LineIn::Eof) => break,
            Ok(LineIn::TooLarge) => {
                send(
                    &mut stdout,
                    &json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "request too large"}}),
                )
                .await?;
            }
            Ok(LineIn::Line(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => {
                        send(
                            &mut stdout,
                            &json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "parse error"}}),
                        )
                        .await?;
                        continue;
                    }
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
                    "tools/call" => call_tool(&engine, &default_bank, &v["params"]).await.map(
                        |result| json!({"content": [{"type": "text", "text": result.to_string()}]}),
                    ),
                    other => Err(format!("unknown method {other}")),
                };
                let resp = match outcome {
                    Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                    Err(msg) => {
                        json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32603, "message": msg}})
                    }
                };
                send(&mut stdout, &resp).await?;
            }
        }
    }
    // stdin closed (parent went away): stop the background worker before
    // returning. A hard kill is safe too - dirty rows persist and the next
    // run re-discovers them.
    worker.abort();
    let _ = worker.await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustsight_core::SystemClock;
    use rustsight_providers::{MockEmbed, MockLlm, MockRerank, SimpleTokenEstimator};
    use rustsight_storage::SqliteStore;
    use std::time::Duration;
    use tokio::io::{duplex, BufReader as TestBufReader};

    fn test_engine() -> Arc<Engine> {
        Arc::new(Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            Arc::new(MockLlm::demo()),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(SystemClock),
        ))
    }

    struct McpPair {
        write: tokio::io::DuplexStream,
        read: TestBufReader<tokio::io::DuplexStream>,
        handle: tokio::task::JoinHandle<rustsight_core::Result<()>>,
    }

    async fn spawn_server(cap: usize) -> McpPair {
        let (req_w, req_r) = duplex(4096);
        let (resp_r, resp_w) = duplex(4096);
        let engine = test_engine();
        let handle = tokio::spawn(run_io(
            engine,
            Some("b".into()),
            TestBufReader::new(req_r),
            resp_w,
            cap,
        ));
        McpPair {
            write: req_w,
            read: TestBufReader::new(resp_r),
            handle,
        }
    }

    impl McpPair {
        async fn next_line(&mut self) -> String {
            let mut line = String::new();
            tokio::time::timeout(Duration::from_secs(5), self.read.read_line(&mut line))
                .await
                .expect("timed out waiting for response")
                .unwrap();
            line
        }

        async fn send_line(&mut self, line: &str) {
            self.write.write_all(line.as_bytes()).await.unwrap();
            self.write.flush().await.unwrap();
        }
    }

    #[tokio::test]
    async fn parse_error_gets_32700_then_server_survives() {
        let mut p = spawn_server(1024).await;
        p.send_line("this is not json\n").await;
        let resp = p.next_line().await;
        assert!(resp.contains("-32700"), "got: {resp}");
        p.send_line("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")
            .await;
        let resp = p.next_line().await;
        assert!(resp.contains("\"result\":{}"), "got: {resp}");
        assert!(resp.contains("\"id\":1"), "got: {resp}");
        drop(p.write);
        p.handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn oversized_line_gets_32600_then_server_survives() {
        let mut p = spawn_server(64).await;
        let big = format!("{}\n", "a".repeat(200));
        p.send_line(&big).await;
        let resp = p.next_line().await;
        assert!(resp.contains("-32600"), "got: {resp}");
        p.send_line("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n")
            .await;
        let resp = p.next_line().await;
        assert!(resp.contains("\"result\":{}"), "got: {resp}");
        drop(p.write);
        p.handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn line_exactly_at_cap_is_parsed() {
        let mut p = spawn_server(64).await;
        let exact = format!("{}\n", "b".repeat(63));
        p.send_line(&exact).await;
        let resp = p.next_line().await;
        assert!(resp.contains("-32700"), "got: {resp}");
        drop(p.write);
        p.handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn unterminated_oversized_line_still_replies() {
        let mut p = spawn_server(64).await;
        let big = format!(
            "{}{}\n",
            "c".repeat(100),
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"}"
        );
        p.send_line(&big).await;
        let resp = p.next_line().await;
        assert!(resp.contains("-32600"), "got: {resp}");
        p.send_line("{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"ping\"}\n")
            .await;
        let resp = p.next_line().await;
        assert!(resp.contains("\"id\":4"), "got: {resp}");
        drop(p.write);
        p.handle.await.unwrap().unwrap();
    }
}
