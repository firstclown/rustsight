use crate::compat::{
    curation_error, memory_list_item, memory_unit_json, parse_update_memory, ApiError,
};
use rustsight_core::types::{
    MemoryKind, MemoryListQuery, MemoryState, RecallRequest, RetainRequest, TagFilter,
};
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
            },
            {
                "name": "get_memory",
                "description": "Get a specific memory by ID. Returns the full memory unit including content, metadata, timestamps and curation state.",
                "annotations": {"readOnlyHint": true},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "memory_id": {"type": "string"}
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "list_memories",
                "description": "Browse stored memories with optional filtering (no relevance ranking).",
                "annotations": {"readOnlyHint": true},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "type": {"type": "string", "description": "world, experience or observation"},
                        "q": {"type": "string", "description": "Case-insensitive substring filter"},
                        "state": {"type": "string", "enum": ["valid", "invalidated"]},
                        "limit": {"type": "integer", "default": 100},
                        "offset": {"type": "integer", "default": 0},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "tags_match": {"type": "string", "enum": ["any", "all", "any_strict", "all_strict", "exact"], "default": "any"}
                    }
                }
            },
            {
                "name": "update_memory",
                "description": "Edit a memory unit to correct what was extracted. Pass any of text / context / occurred_start / occurred_end / mentioned_at / fact_type / entities. For context and the dates, \"\" clears the field and omitting it leaves it unchanged; entities replaces the fact's entity set ([] detaches all). The memory is re-embedded and its derived observations and links are recomputed. resolve_entities=false takes entity names literally (recommended for deliberate corrections). Only raw world/experience facts can be edited; to retire or restore a fact, use invalidate_memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "memory_id": {"type": "string"},
                        "text": {"type": "string"},
                        "context": {"type": "string"},
                        "occurred_start": {"type": "string"},
                        "occurred_end": {"type": "string"},
                        "mentioned_at": {"type": "string"},
                        "fact_type": {"type": "string", "enum": ["world", "experience"]},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "resolve_entities": {"type": "boolean", "default": true}
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "invalidate_memory",
                "description": "Soft-retire a memory unit (or restore a previously retired one). Invalidating moves the fact out of the active set: it is excluded from recall, consolidation and the knowledge graph, its links are pruned and its derived observations are deleted for recomputation, but it is kept for audit and is fully reversible. Pass restore=true to bring it back. Only raw world/experience facts can be invalidated.",
                "annotations": {"destructiveHint": true},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bank_id": {"type": "string"},
                        "memory_id": {"type": "string"},
                        "reason": {"type": "string", "description": "Optional free-text reason recorded when invalidating"},
                        "restore": {"type": "boolean", "default": false}
                    },
                    "required": ["memory_id"]
                }
            }
        ]
    })
}

fn api_err(e: ApiError) -> String {
    e.detail()
}

fn memory_id_arg(args: &Value) -> Result<String, String> {
    args.get("memory_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| "memory_id required".to_string())
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
                    ..Default::default()
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
            let resolved = engine
                .resolved_config(&bank_id)
                .map_err(|e| e.to_string())?;
            let r = engine
                .recall(RecallRequest {
                    bank_id,
                    query: query.to_string(),
                    token_budget: args
                        .get("token_budget")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1000) as usize,
                    top_k_per_arm: engine.recall_budget_for(&resolved).resolve(
                        None,
                        args.get("token_budget")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1000) as usize,
                    ),
                    kinds: None,
                    ..Default::default()
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
        "get_memory" => {
            let memory_id = memory_id_arg(&args)?;
            let unit = engine
                .get_memory_unit(&bank_id, &memory_id)
                .map_err(|e| api_err(curation_error(e)))?
                .ok_or_else(|| format!("Memory '{memory_id}' not found"))?;
            memory_unit_json(engine, &unit).map_err(api_err)
        }
        "list_memories" => {
            let str_arg = |k: &str| {
                args.get(k)
                    .and_then(|v| v.as_str())
                    .filter(|v| !v.is_empty())
            };
            let state = match str_arg("state") {
                None => None,
                Some(s) => Some(MemoryState::parse(s).ok_or_else(|| {
                    format!("Invalid state '{s}': expected 'valid' or 'invalidated'.")
                })?),
            };
            let tags: Vec<String> = args
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let mode = serde_json::from_value(json!(str_arg("tags_match").unwrap_or("any")))
                .map_err(|_| "tags_match must be one of any, all, any_strict, all_strict, exact")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
            let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let query = MemoryListQuery {
                state,
                kind: str_arg("type").map(|t| MemoryKind::parse(t).unwrap_or(MemoryKind::Opinion)),
                q: str_arg("q").map(String::from),
                tags: Some(TagFilter {
                    tags,
                    mode,
                    groups: vec![],
                }),
                limit,
                offset,
                ..Default::default()
            };
            let (units, total) = engine
                .list_memory_units(&bank_id, &query)
                .map_err(|e| e.to_string())?;
            let items: Vec<Value> = units.iter().map(memory_list_item).collect();
            Ok(json!({"items": items, "total": total, "limit": limit, "offset": offset}))
        }
        "update_memory" => {
            let memory_id = memory_id_arg(&args)?;
            // The edit tool only edits; retiring goes through
            // invalidate_memory (as upstream splits them).
            let mut body = args.clone();
            if let Some(obj) = body.as_object_mut() {
                for k in ["bank_id", "memory_id", "state", "reason"] {
                    obj.remove(k);
                }
            }
            let req = parse_update_memory(&bank_id, &memory_id, &body).map_err(api_err)?;
            let unit = engine
                .curate_memory(req)
                .await
                .map_err(|e| api_err(curation_error(e)))?;
            memory_unit_json(engine, &unit).map_err(api_err)
        }
        "invalidate_memory" => {
            let memory_id = memory_id_arg(&args)?;
            let restore = args
                .get("restore")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let mut req = rustsight_core::CurateRequest::new(bank_id, memory_id);
            req.state = Some(if restore {
                MemoryState::Valid
            } else {
                MemoryState::Invalidated
            });
            req.reason = args
                .get("reason")
                .and_then(|v| v.as_str())
                .map(String::from);
            let unit = engine
                .curate_memory(req)
                .await
                .map_err(|e| api_err(curation_error(e)))?;
            memory_unit_json(engine, &unit).map_err(api_err)
        }
        other => Err(format!("unknown tool {other}")),
    }
}

pub async fn run_stdio(
    engine: Arc<Engine>,
    default_bank: Option<String>,
) -> rustsight_core::Result<()> {
    // Bank webhooks fire for MCP retains too (RS-10b).
    let webhook_settings = Arc::new(crate::webhooks::WebhookSettings::from_env());
    crate::webhooks::set_settings(Arc::clone(&webhook_settings));
    let webhooks = crate::webhooks::start(&engine, webhook_settings);
    let result = run_io(
        engine,
        default_bank,
        BufReader::new(tokio::io::stdin()),
        tokio::io::stdout(),
        MAX_LINE_BYTES,
    )
    .await;
    webhooks.abort();
    let _ = webhooks.await;
    result
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

    async fn rpc(p: &mut McpPair, id: u64, method: &str, params: Value) -> Value {
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        p.send_line(&format!("{req}\n")).await;
        serde_json::from_str(&p.next_line().await).unwrap()
    }

    async fn tool(p: &mut McpPair, id: u64, name: &str, args: Value) -> Value {
        let resp = rpc(
            p,
            id,
            "tools/call",
            json!({"name": name, "arguments": args}),
        )
        .await;
        let text = resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} failed: {resp}"));
        serde_json::from_str(text).unwrap()
    }

    #[tokio::test]
    async fn curation_tools_are_listed_and_callable() {
        let mut p = spawn_server(MAX_LINE_BYTES).await;
        let tools = rpc(&mut p, 1, "tools/list", json!({})).await;
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for name in [
            "get_memory",
            "list_memories",
            "update_memory",
            "invalidate_memory",
        ] {
            assert!(names.contains(&name), "{names:?}");
        }
        let invalidate = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "invalidate_memory")
            .unwrap();
        assert_eq!(invalidate["annotations"]["destructiveHint"], true);

        tool(
            &mut p,
            2,
            "retain",
            json!({"content": "Mara repairs clocks."}),
        )
        .await;
        let list = tool(&mut p, 3, "list_memories", json!({"type": "world"})).await;
        assert_eq!(list["total"], 1, "{list}");
        let id = list["items"][0]["id"].as_str().unwrap().to_string();

        let got = tool(&mut p, 4, "get_memory", json!({"memory_id": id})).await;
        assert_eq!(got["state"], "valid");
        let edited = tool(
            &mut p,
            5,
            "update_memory",
            json!({"memory_id": id, "text": "Mara repairs watches."}),
        )
        .await;
        assert_eq!(edited["text"], "Mara repairs watches.");
        let gone = tool(
            &mut p,
            6,
            "invalidate_memory",
            json!({"memory_id": id, "reason": "wrong"}),
        )
        .await;
        assert_eq!(gone["state"], "invalidated");
        assert_eq!(gone["invalidation_reason"], "wrong");
        let archived = tool(&mut p, 7, "list_memories", json!({"state": "invalidated"})).await;
        assert_eq!(archived["total"], 1);
        let back = tool(
            &mut p,
            8,
            "invalidate_memory",
            json!({"memory_id": id, "restore": true}),
        )
        .await;
        assert_eq!(back["state"], "valid");

        // The edit tool never changes state.
        let resp = rpc(
            &mut p,
            9,
            "tools/call",
            json!({"name": "update_memory", "arguments": {"memory_id": id, "state": "invalidated"}}),
        )
        .await;
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("at least one field"));
        drop(p.write);
        p.handle.await.unwrap().unwrap();
    }
}
