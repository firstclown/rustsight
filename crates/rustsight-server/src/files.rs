//! File retain + attachments (RS-10c), Hindsight-compatible:
//!
//! - `POST /v1/default/banks/{id}/files/retain` — multipart `files` (one or
//!   more) plus a `request` JSON field (`parser`, `files_metadata`). Always
//!   async: every file's bytes are stored as an attachment of its document
//!   and a `file_convert_retain` operation queued; the response carries
//!   one operation id per file.
//! - `GET /v1/default/banks/{id}/attachments/{attachment_id}` — the original
//!   bytes under their declared Content-Type.
//!
//! The operation parses the stored bytes with the requested parser chain
//! and retains the text through the same path as a JSON retain item (Memory
//! Defense, tags, timestamp), so every fact carries `attachment_id` and
//! `filename` in its metadata. Upstream converts in one operation and
//! queues a second for the retain; here one operation covers both.

use crate::compat::{parse_item, ApiError};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rustsight_core::files::{Attachment, FileInput};
use rustsight_core::types::Operation;
use rustsight_core::Engine;
use serde_json::{json, Map, Value};
use std::sync::{Arc, OnceLock};

pub const DEFAULT_MAX_FILES: usize = 10;
pub const DEFAULT_MAX_BATCH_MB: usize = 100;
pub const DEFAULT_PARSER_CHAIN: &str = "text,html";

/// Per-file fields that belong in `files_metadata`, not at the top level
/// (upstream `_FILE_RETAIN_PER_FILE_FIELDS`).
const PER_FILE_FIELDS: [&str; 6] = [
    "context",
    "document_id",
    "metadata",
    "strategy",
    "tags",
    "timestamp",
];

#[derive(Debug, Clone)]
pub struct FileSettings {
    pub enabled: bool,
    pub max_files: usize,
    pub max_batch_bytes: usize,
    /// Chain used when a request names no parser.
    pub default_chain: Vec<String>,
}

impl Default for FileSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_files: DEFAULT_MAX_FILES,
            max_batch_bytes: DEFAULT_MAX_BATCH_MB * 1024 * 1024,
            default_chain: split_chain(DEFAULT_PARSER_CHAIN),
        }
    }
}

fn split_chain(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

fn env_either(name: &str) -> Option<String> {
    std::env::var(format!("RUSTSIGHT_{name}"))
        .or_else(|_| std::env::var(format!("HINDSIGHT_API_{name}")))
        .ok()
        .filter(|v| !v.trim().is_empty())
}

fn env_count(name: &str, default: usize) -> usize {
    match env_either(name) {
        None => default,
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or_else(|| {
                eprintln!("rustsight: invalid RUSTSIGHT_{name} {raw:?}, using {default}");
                default
            }),
    }
}

impl FileSettings {
    /// `RUSTSIGHT_ENABLE_FILE_UPLOAD_API` (default true),
    /// `RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE` (files, default 10),
    /// `RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE_MB` (default 100) — each
    /// also answers to `HINDSIGHT_API_*` — and `RUSTSIGHT_FILE_PARSER`
    /// (default `text,html`). The parser chain is deliberately not read
    /// from `HINDSIGHT_API_FILE_PARSER`: Hindsight's parser names
    /// (`markitdown`, `iris`) are not built in here.
    pub fn from_env() -> Self {
        let enabled = env_either("ENABLE_FILE_UPLOAD_API").is_none_or(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
        let default_chain = std::env::var("RUSTSIGHT_FILE_PARSER")
            .ok()
            .map(|v| split_chain(&v))
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| split_chain(DEFAULT_PARSER_CHAIN));
        Self {
            enabled,
            max_files: env_count("FILE_CONVERSION_MAX_BATCH_SIZE", DEFAULT_MAX_FILES),
            max_batch_bytes: env_count("FILE_CONVERSION_MAX_BATCH_SIZE_MB", DEFAULT_MAX_BATCH_MB)
                * 1024
                * 1024,
            default_chain,
        }
    }
}

fn settings() -> &'static FileSettings {
    static SETTINGS: OnceLock<FileSettings> = OnceLock::new();
    SETTINGS.get_or_init(FileSettings::from_env)
}

pub(crate) fn routes() -> Router<Arc<Engine>> {
    // Multipart bodies may be up to the batch cap; a little slack lets an
    // over-cap batch reach the handler's own 400 instead of a bare 413.
    let limit = settings().max_batch_bytes.saturating_add(8 * 1024 * 1024);
    Router::new()
        .route(
            "/v1/default/banks/{bank_id}/files/retain",
            post(files_retain).layer(DefaultBodyLimit::max(limit)),
        )
        .route(
            "/v1/default/banks/{bank_id}/attachments/{attachment_id}",
            get(get_attachment),
        )
}

/// `['a', 'b']`, as upstream's error messages print lists.
fn py_list(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|i| format!("'{i}'")).collect();
    format!("[{}]", quoted.join(", "))
}

fn parser_chain(raw: Option<&Value>, field: &str) -> Result<Option<Vec<String>>, ApiError> {
    match raw {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(vec![s.clone()])),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "Invalid request JSON: {field} must be a string or a list of strings"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(ApiError::bad_request(format!(
            "Invalid request JSON: {field} must be a string or a list of strings"
        ))),
    }
}

fn check_parsers(engine: &Engine, chain: &[String], context: &str) -> Result<(), ApiError> {
    let disallowed: Vec<String> = chain
        .iter()
        .filter(|p| !engine.parsers.contains(p))
        .cloned()
        .collect();
    if chain.is_empty() || !disallowed.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Parser(s) not available ({context}): {}. Available: {}",
            py_list(&disallowed),
            py_list(&engine.parsers.names())
        )));
    }
    Ok(())
}

/// The parsed `request` form field.
struct FileRequest {
    parser: Option<Vec<String>>,
    files_metadata: Option<Vec<Map<String, Value>>>,
}

fn parse_request(raw: &str) -> Result<FileRequest, ApiError> {
    let invalid = |msg: String| ApiError::bad_request(format!("Invalid request JSON: {msg}"));
    let value: Value = serde_json::from_str(raw).map_err(|e| invalid(e.to_string()))?;
    let Value::Object(obj) = value else {
        return Err(invalid("expected a JSON object".into()));
    };
    let misplaced: Vec<&str> = PER_FILE_FIELDS
        .iter()
        .copied()
        .filter(|f| obj.get(*f).is_some_and(|v| !v.is_null()))
        .collect();
    let mut errors = vec![];
    if !misplaced.is_empty() {
        errors.push(format!(
            "Per-file fields ({}) must be placed in the corresponding 'files_metadata' entry",
            misplaced.join(", ")
        ));
    }
    if obj.get("update_mode").is_some_and(|v| !v.is_null()) {
        errors.push(
            "'update_mode' is not supported by /files/retain, which always processes asynchronously"
                .into(),
        );
    }
    if !errors.is_empty() {
        return Err(invalid(errors.join("; ")));
    }
    let files_metadata = match obj.get("files_metadata") {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) => Some(
            items
                .iter()
                .map(|m| match m {
                    Value::Object(map) => Ok(map.clone()),
                    Value::Null => Ok(Map::new()),
                    _ => Err(invalid("files_metadata entries must be objects".into())),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Some(_) => return Err(invalid("files_metadata must be a list".into())),
    };
    Ok(FileRequest {
        parser: parser_chain(obj.get("parser"), "parser")?,
        files_metadata,
    })
}

struct Upload {
    filename: String,
    content_type: Option<String>,
    bytes: Bytes,
}

async fn read_multipart(mut form: Multipart) -> Result<(Vec<Upload>, Option<String>), ApiError> {
    let mut files = vec![];
    let mut request = None;
    loop {
        let field = form
            .next_field()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart body: {e}")))?;
        let Some(field) = field else { break };
        match field.name() {
            Some("files") => {
                let filename = field
                    .file_name()
                    .map(str::to_string)
                    .filter(|f| !f.trim().is_empty())
                    .unwrap_or_else(|| "unknown".into());
                let content_type = field.content_type().map(str::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("Invalid multipart body: {e}")))?;
                files.push(Upload {
                    filename,
                    content_type,
                    bytes,
                });
            }
            Some("request") => {
                request =
                    Some(field.text().await.map_err(|e| {
                        ApiError::bad_request(format!("Invalid multipart body: {e}"))
                    })?);
            }
            _ => {}
        }
    }
    Ok((files, request))
}

async fn files_retain(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    form: Multipart,
) -> Result<Json<Value>, ApiError> {
    let settings = settings();
    if !settings.enabled {
        return Err(ApiError::not_found(
            "File upload API is disabled. Set RUSTSIGHT_ENABLE_FILE_UPLOAD_API=true to enable.",
        ));
    }
    let (files, request) = read_multipart(form).await?;
    let request = request.ok_or_else(|| ApiError::unprocessable("Field 'request' is required"))?;
    if files.is_empty() {
        return Err(ApiError::unprocessable("Field 'files' is required"));
    }
    let request = parse_request(&request)?;
    if files.len() > settings.max_files {
        return Err(ApiError::bad_request(format!(
            "Too many files. Maximum {} files per request.",
            settings.max_files
        )));
    }
    if let Some(meta) = &request.files_metadata {
        if meta.len() != files.len() {
            return Err(ApiError::bad_request(format!(
                "files_metadata count ({}) must match files count ({})",
                meta.len(),
                files.len()
            )));
        }
    }
    if let Some(chain) = &request.parser {
        check_parsers(&engine, chain, "request-level parser")?;
    }
    let total: usize = files.iter().map(|f| f.bytes.len()).sum();
    if total > settings.max_batch_bytes {
        return Err(ApiError::bad_request(format!(
            "Total batch size ({:.1}MB) exceeds maximum of {}MB",
            total as f64 / (1024.0 * 1024.0),
            settings.max_batch_bytes / (1024 * 1024)
        )));
    }

    // Validate every file before storing anything.
    let mut planned = Vec::with_capacity(files.len());
    for (index, file) in files.into_iter().enumerate() {
        let mut meta = request
            .files_metadata
            .as_ref()
            .map(|m| m[index].clone())
            .unwrap_or_default();
        let chain = match parser_chain(meta.get("parser"), "files_metadata.parser")? {
            Some(chain) => chain,
            None => request
                .parser
                .clone()
                .unwrap_or_else(|| settings.default_chain.clone()),
        };
        check_parsers(&engine, &chain, &format!("file '{}'", file.filename))?;
        let document_id = meta
            .get("document_id")
            .and_then(|d| d.as_str())
            .filter(|d| !d.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("file_{}", uuid::Uuid::new_v4()));
        // The retain item the operation will run, minus its content.
        meta.remove("parser");
        meta.remove("strategy");
        meta.insert("document_id".into(), json!(document_id));
        meta.insert("content".into(), json!("(pending conversion)"));
        let item = Value::Object(meta);
        parse_item(&item, index)?;
        planned.push((file, chain, document_id, item));
    }

    let guard = engine.bank_read(&bank_id).await;
    let now = engine.clock.now_ms();
    let mut ops = Vec::with_capacity(planned.len());
    for (file, chain, document_id, item) in planned {
        let attachment = Attachment::new(
            &bank_id,
            &document_id,
            &file.filename,
            file.content_type.as_deref(),
            file.bytes.to_vec(),
            now,
        );
        engine.store.put_attachment(&attachment)?;
        let op = Operation {
            id: uuid::Uuid::new_v4().to_string(),
            bank_id: bank_id.clone(),
            kind: "file_convert_retain".into(),
            status: "pending".into(),
            error: None,
            payload: json!({
                "items": [item],
                "attachment_id": attachment.id,
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "parser": chain,
            }),
            created_at_ms: now,
            updated_at_ms: now,
        };
        engine.store.upsert_operation(&op)?;
        ops.push(op);
    }
    drop(guard);
    let operation_ids: Vec<String> = ops.iter().map(|op| op.id.clone()).collect();
    for op in ops {
        crate::compat::spawn_operation(Arc::clone(&engine), op);
    }
    Ok(Json(json!({ "operation_ids": operation_ids })))
}

/// The retain item a `file_convert_retain` operation runs: the stored
/// file converted to text, with `attachment_id` / `filename` added to its
/// metadata. Called by the operation runner under the bank guard.
pub(crate) fn converted_item(engine: &Engine, op: &Operation) -> Result<Value, ApiError> {
    let field = |name: &str| {
        op.payload
            .get(name)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
    };
    let attachment_id = field("attachment_id");
    let filename = field("filename").to_string();
    let mut item = op
        .payload
        .get("items")
        .and_then(|i| i.get(0))
        .cloned()
        .ok_or_else(|| ApiError::unprocessable("file operation has no item"))?;
    let attachment = engine
        .store
        .get_attachment(&op.bank_id, attachment_id)?
        .ok_or_else(|| {
            ApiError::not_found(format!(
                "Failed to parse file '{filename}': stored file is gone"
            ))
        })?;
    let chain: Vec<String> = op
        .payload
        .get("parser")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default();
    let converted = engine
        .parsers
        .convert_with_fallback(
            &chain,
            &FileInput {
                bytes: &attachment.bytes,
                filename: &filename,
                content_type: attachment.content_type.as_deref(),
            },
        )
        .map_err(|e| {
            let reason = match e {
                rustsight_core::Error::BadInput(m) => m,
                other => other.to_string(),
            };
            ApiError::unprocessable(format!("Failed to parse file '{filename}': {reason}"))
        })?;
    if converted.text.trim().is_empty() {
        return Err(ApiError::unprocessable(format!(
            "Failed to parse file '{filename}': no text content"
        )));
    }
    let obj = item.as_object_mut().expect("items are objects");
    obj.insert("content".into(), json!(converted.text));
    let mut metadata = match obj.remove("metadata") {
        Some(Value::Object(m)) => m,
        _ => Map::new(),
    };
    metadata.insert("attachment_id".into(), json!(attachment_id));
    metadata.insert("filename".into(), json!(filename));
    obj.insert("metadata".into(), Value::Object(metadata));
    Ok(item)
}

/// After a file retain succeeded: the document's content is now this
/// file's, so earlier uploads into the same document are dropped.
pub(crate) fn prune_replaced(engine: &Engine, op: &Operation) -> Result<(), ApiError> {
    let attachment_id = op
        .payload
        .get("attachment_id")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let document_id = op
        .payload
        .pointer("/items/0/document_id")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    engine
        .store
        .prune_document_attachments(&op.bank_id, document_id, attachment_id)?;
    Ok(())
}

async fn get_attachment(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, attachment_id)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let attachment = engine
        .store
        .get_attachment(&bank_id, &attachment_id)?
        .ok_or_else(|| ApiError::not_found("Attachment not found"))?;
    let content_type = attachment
        .content_type
        .unwrap_or_else(|| "application/octet-stream".into());
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            // Content-addressed: the bytes behind this URL never change.
            (
                header::CACHE_CONTROL,
                "private, max-age=31536000, immutable".into(),
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".into()),
        ],
        attachment.bytes,
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::ApiAuth;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use rustsight_core::SystemClock;
    use rustsight_providers::{MockEmbed, MockLlm, MockRerank, SimpleTokenEstimator};
    use rustsight_storage::SqliteStore;
    use tower::ServiceExt;

    const BOUNDARY: &str = "rs10cboundary";

    fn engine() -> Arc<Engine> {
        Arc::new(Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            Arc::new(MockLlm::demo()),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(SystemClock),
        ))
    }

    fn app(engine: &Arc<Engine>) -> Router {
        crate::http::router(Arc::clone(engine), Arc::new(ApiAuth::None))
    }

    /// `(filename, content type, bytes)` files plus the `request` field.
    fn multipart(files: &[(&str, &str, &[u8])], request: Option<Value>) -> Vec<u8> {
        let mut body = vec![];
        for (name, ct, bytes) in files {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\nContent-Type: {ct}\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(bytes);
            body.extend_from_slice(b"\r\n");
        }
        if let Some(request) = request {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n{request}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        body
    }

    async fn upload(app: &Router, bank: &str, body: Vec<u8>) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(Method::POST)
            .uri(format!("/v1/default/banks/{bank}/files/retain"))
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn get(app: &Router, uri: &str) -> (StatusCode, axum::http::HeaderMap, Bytes) {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, headers, bytes)
    }

    async fn wait_for(app: &Router, bank: &str, id: &str) -> Value {
        let uri = format!("/v1/default/banks/{bank}/operations/{id}");
        for _ in 0..500 {
            let (_, _, bytes) = get(app, &uri).await;
            let state: Value = serde_json::from_slice(&bytes).unwrap();
            if state["status"] != "pending" {
                return state;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("operation {id} did not finish");
    }

    const NOTES: &[u8] =
        b"Alice joined Google as an engineer. Alice works on search infrastructure.";
    const PAGE: &[u8] = b"<html><head><script>ignore()</script></head><body><p>Bob moved to Berlin.</p></body></html>";

    #[tokio::test]
    async fn uploaded_files_become_memories_with_their_attachment() {
        let engine = engine();
        let app = app(&engine);
        let (status, body) = upload(
            &app,
            "fb",
            multipart(
                &[
                    ("notes.md", "text/markdown", NOTES),
                    ("page.html", "text/html", PAGE),
                ],
                Some(json!({"files_metadata": [
                    {"document_id": "notes", "tags": ["team"], "metadata": {"source": "upload"}},
                    {}
                ]})),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ids = body["operation_ids"].as_array().unwrap().clone();
        assert_eq!(ids.len(), 2, "one operation per file");
        for id in &ids {
            let state = wait_for(&app, "fb", id.as_str().unwrap()).await;
            assert_eq!(state["status"], "completed", "{state}");
            assert_eq!(state["operation_type"], "file_convert_retain");
        }

        let memories = engine.store.list_memories("fb", 100).unwrap();
        let notes: Vec<_> = memories
            .iter()
            .filter(|m| m.document_id.as_deref() == Some("notes"))
            .collect();
        assert!(!notes.is_empty());
        let notes_id = Attachment::new("", "", "", None, NOTES.to_vec(), 0).id;
        for m in &notes {
            let meta = m.metadata.as_ref().unwrap();
            assert_eq!(meta["attachment_id"], notes_id);
            assert_eq!(meta["filename"], "notes.md");
            assert_eq!(meta["source"], "upload");
            assert_eq!(m.tags, vec!["team".to_string()]);
        }
        // The HTML file went through the default chain to `html`.
        let page: Vec<_> = memories
            .iter()
            .filter(|m| {
                m.metadata
                    .as_ref()
                    .is_some_and(|md| md["filename"] == "page.html")
            })
            .collect();
        assert!(!page.is_empty());
        assert!(page[0].document_id.as_deref().unwrap().starts_with("file_"));
        assert!(page.iter().all(|m| !m.text.contains("ignore")));

        let (status, headers, bytes) = get(
            &app,
            &format!("/v1/default/banks/fb/attachments/{notes_id}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bytes.as_ref(), NOTES);
        assert_eq!(headers[header::CONTENT_TYPE], "text/markdown");
        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        let (status, _, _) = get(&app, "/v1/default/banks/fb/attachments/000000000000").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = get(
            &app,
            &format!("/v1/default/banks/other/attachments/{notes_id}"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "attachments are bank-scoped");

        // The list shows which file an operation was for.
        let (_, _, bytes) = get(&app, "/v1/default/banks/fb/operations").await;
        let list: Value = serde_json::from_slice(&bytes).unwrap();
        let filenames: Vec<&str> = list["operations"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|o| o["filename"].as_str())
            .collect();
        assert!(filenames.contains(&"notes.md") && filenames.contains(&"page.html"));
    }

    #[tokio::test]
    async fn unparseable_files_fail_their_operation() {
        let engine = engine();
        let app = app(&engine);
        let (status, body) = upload(
            &app,
            "fx",
            multipart(
                &[("r.pdf", "application/pdf", b"%PDF-1.7\0\x01\x02")],
                Some(json!({})),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let state = wait_for(&app, "fx", body["operation_ids"][0].as_str().unwrap()).await;
        assert_eq!(state["status"], "failed");
        let err = state["error_message"].as_str().unwrap();
        assert!(err.starts_with("Failed to parse file 'r.pdf'"), "{err}");
        assert!(engine.store.list_memories("fx", 10).unwrap().is_empty());
    }

    #[tokio::test]
    async fn upload_requests_are_validated_like_upstream() {
        let engine = engine();
        let app = app(&engine);
        let one: &[(&str, &str, &[u8])] = &[("a.txt", "text/plain", b"hi")];
        let eleven: Vec<(&str, &str, &[u8])> = (0..11)
            .map(|_| ("a.txt", "text/plain", &b"hi"[..]))
            .collect();
        for (body, want, needle) in [
            (
                multipart(one, Some(json!({"parser": "markitdown"}))),
                StatusCode::BAD_REQUEST,
                "Parser(s) not available (request-level parser): ['markitdown']",
            ),
            (
                multipart(
                    one,
                    Some(json!({"files_metadata": [{"parser": ["text", "iris"]}]})),
                ),
                StatusCode::BAD_REQUEST,
                "file 'a.txt'",
            ),
            (
                multipart(&eleven, Some(json!({}))),
                StatusCode::BAD_REQUEST,
                "Too many files. Maximum 10",
            ),
            (
                multipart(one, Some(json!({"files_metadata": [{}, {}]}))),
                StatusCode::BAD_REQUEST,
                "files_metadata count (2) must match files count (1)",
            ),
            (
                multipart(one, Some(json!({"tags": ["x"]}))),
                StatusCode::BAD_REQUEST,
                "Per-file fields (tags)",
            ),
            (
                multipart(one, Some(json!({"update_mode": "append"}))),
                StatusCode::BAD_REQUEST,
                "'update_mode' is not supported",
            ),
            (
                multipart(one, Some(Value::String("{not json".into()))),
                StatusCode::BAD_REQUEST,
                "Invalid request JSON",
            ),
            (
                multipart(one, None),
                StatusCode::UNPROCESSABLE_ENTITY,
                "request",
            ),
            (
                multipart(&[], Some(json!({}))),
                StatusCode::UNPROCESSABLE_ENTITY,
                "files",
            ),
        ] {
            let (status, resp) = upload(&app, "fv", body).await;
            assert_eq!(status, want, "{resp}");
            let detail = resp["detail"].as_str().unwrap_or_default();
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}");
        }
        assert!(engine
            .store
            .get_attachment("fv", "8f434346648f")
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn reuploading_into_a_document_replaces_its_attachment() {
        let engine = engine();
        let app = app(&engine);
        for content in [&b"Alice likes tea."[..], &b"Alice likes coffee."[..]] {
            let (_, body) = upload(
                &app,
                "fr",
                multipart(
                    &[("pref.txt", "text/plain", content)],
                    Some(json!({"files_metadata": [{"document_id": "pref"}]})),
                ),
            )
            .await;
            wait_for(&app, "fr", body["operation_ids"][0].as_str().unwrap()).await;
        }
        let old = Attachment::new("", "", "", None, b"Alice likes tea.".to_vec(), 0).id;
        let new = Attachment::new("", "", "", None, b"Alice likes coffee.".to_vec(), 0).id;
        assert!(engine.store.get_attachment("fr", &old).unwrap().is_none());
        assert!(engine.store.get_attachment("fr", &new).unwrap().is_some());
        engine.forget_document("fr", "pref").unwrap();
        assert!(engine.store.get_attachment("fr", &new).unwrap().is_none());
    }
}
