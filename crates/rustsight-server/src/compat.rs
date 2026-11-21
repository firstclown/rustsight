//! Hindsight wire-compatible HTTP API (RS-9).
//!
//! Serves the subset of the upstream `hindsight-api` surface that Hindsight
//! clients (`@vectorize-io/hindsight-client`, `hindsight-client` for Python)
//! use for day-to-day memory work, with upstream's paths, request bodies and
//! response shapes, so a client can be pointed at a rustsight server
//! unchanged:
//!
//! - `GET /health`, `GET /version`
//! - `GET /openapi.json` (the request fields this server acts on)
//! - `GET /v1/default/banks`, `PUT|PATCH|DELETE /v1/default/banks/{bank_id}`,
//!   `GET /v1/default/banks/{bank_id}/profile`,
//!   `GET|PATCH|DELETE /v1/default/banks/{bank_id}/config`
//! - `POST /v1/default/banks/{bank_id}/memories` (retain, sync or async)
//! - `POST /v1/default/banks/{bank_id}/memories/recall`
//! - `POST /v1/default/banks/{bank_id}/reflect`
//! - `GET /v1/default/banks/{bank_id}/documents`,
//!   `GET|DELETE /v1/default/banks/{bank_id}/documents/{document_id}`
//! - `GET /v1/default/banks/{bank_id}/operations`,
//!   `GET /v1/default/banks/{bank_id}/operations/{operation_id}`,
//!   `POST /v1/default/banks/{bank_id}/operations/{operation_id}/retry`
//!
//! Shapes follow hindsight-api 1.2.x (`api/http.py`). Errors use FastAPI's
//! `{"detail": "..."}` body. What is accepted but not acted on is listed in
//! the README's compatibility matrix.

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use rustsight_core::labels::BankLabels;
use rustsight_core::tag_resolution::{resolve_fuzzy_leaf, tag_vocabulary};
use rustsight_core::types::*;
use rustsight_core::{BankGuard, Budget, Engine, Error};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;

/// The upstream release whose behavior this server mirrors (the prompt pin).
/// Reported as `api_version` so Hindsight clients that gate on the server
/// version see a real Hindsight version, not rustsight's own.
pub const HINDSIGHT_API_VERSION: &str = "1.1.0";

/// Fact types a Hindsight recall may name (upstream
/// `VALID_RECALL_FACT_TYPES`). Opinions are a rustsight-native kind and are
/// never returned through this API.
const RECALL_KINDS: [MemoryKind; 3] = [
    MemoryKind::World,
    MemoryKind::Experience,
    MemoryKind::Observation,
];

const DEFAULT_ASYNC_RETAIN_CONCURRENCY: usize = 2;
const MAX_ASYNC_RETAIN_CONCURRENCY: usize = 64;

/// `RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY`: a whole number, clamped to the
/// supported range. Anything else falls back to the default with a warning.
fn parse_retain_concurrency(raw: Option<&str>) -> usize {
    let Some(raw) = raw.map(str::trim).filter(|r| !r.is_empty()) else {
        return DEFAULT_ASYNC_RETAIN_CONCURRENCY;
    };
    match raw.parse::<usize>() {
        Ok(n) => n.clamp(1, MAX_ASYNC_RETAIN_CONCURRENCY),
        Err(_) => {
            eprintln!(
                "rustsight: invalid RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY {raw:?}, using {DEFAULT_ASYNC_RETAIN_CONCURRENCY}"
            );
            DEFAULT_ASYNC_RETAIN_CONCURRENCY
        }
    }
}

/// Background (async) retains run at most this many at a time; the rest
/// queue. Each one is a chain of LLM calls, so an unbounded fan-out would
/// trip provider rate limits.
fn async_retain_slots() -> &'static Semaphore {
    static SLOTS: OnceLock<Semaphore> = OnceLock::new();
    SLOTS.get_or_init(|| {
        let raw = std::env::var("RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY").ok();
        Semaphore::new(parse_retain_concurrency(raw.as_deref()))
    })
}

/// Routes that need no token (liveness and version probes).
pub fn public_routes() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/health/live", get(health))
        .route("/health/ready", get(health))
        .route("/version", get(version))
        .route("/v1/version", get(version))
        .route("/openapi.json", get(openapi))
}

/// Bank-scoped routes; the caller wraps them in the token middleware.
pub fn routes() -> Router<Arc<Engine>> {
    Router::new()
        .route("/v1/default/banks", get(list_banks))
        .route(
            "/v1/default/banks/{bank_id}",
            put(upsert_bank).patch(upsert_bank).delete(delete_bank),
        )
        .route(
            "/v1/default/banks/{bank_id}/config",
            get(get_bank_config)
                .patch(patch_bank_config)
                .delete(reset_bank_config),
        )
        .route("/v1/default/banks/{bank_id}/profile", get(bank_profile))
        .route("/v1/default/banks/{bank_id}/memories", post(retain))
        .route("/v1/default/banks/{bank_id}/memories/recall", post(recall))
        .route("/v1/default/banks/{bank_id}/reflect", post(reflect))
        .route("/v1/default/banks/{bank_id}/documents", get(list_documents))
        .route(
            "/v1/default/banks/{bank_id}/documents/{*document_id}",
            get(get_document).delete(delete_document),
        )
        .route(
            "/v1/default/banks/{bank_id}/operations",
            get(list_operations),
        )
        .route(
            "/v1/default/banks/{bank_id}/operations/{operation_id}",
            get(get_operation),
        )
        .route(
            "/v1/default/banks/{bank_id}/operations/{operation_id}/retry",
            post(retry_operation),
        )
}

// ── Errors ───────────────────────────────────────────────────────────────

/// FastAPI-shaped error: `{"detail": "..."}` with the matching status.
#[derive(Debug)]
pub struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }

    fn unprocessable(msg: impl Into<String>) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, msg.into())
    }

    fn not_found(msg: impl Into<String>) -> Self {
        Self(StatusCode::NOT_FOUND, msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "detail": self.1 }))).into_response()
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        match &e {
            Error::NotFound(_) => Self(StatusCode::NOT_FOUND, e.to_string()),
            Error::BadInput(_) | Error::Parse(_) => Self(StatusCode::BAD_REQUEST, e.to_string()),
            _ => {
                eprintln!("rustsight http: internal error: {e}");
                Self(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".into(),
                )
            }
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(r: JsonRejection) -> Self {
        Self::unprocessable(r.body_text())
    }
}

type ApiResult = Result<Json<Value>, ApiError>;

// ── Time helpers ─────────────────────────────────────────────────────────

/// ms-epoch → ISO 8601 with a UTC offset, the form Python's
/// `datetime.isoformat()` produces upstream (`2024-01-15T10:30:00+00:00`).
fn iso(ms: i64) -> Option<String> {
    DateTime::<Utc>::from_timestamp_millis(ms).map(|dt| dt.to_rfc3339())
}

/// Parse an ISO 8601 timestamp: RFC 3339, a naive datetime (taken as UTC),
/// or a bare date (midnight UTC).
fn parse_iso(raw: &str) -> Option<i64> {
    let raw = raw.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Some(dt.timestamp_millis());
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(raw, fmt) {
            return Some(naive.and_utc().timestamp_millis());
        }
    }
    NaiveDate::parse_from_str(raw, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|naive| naive.and_utc().timestamp_millis())
}

// ── Health / version ─────────────────────────────────────────────────────

async fn health() -> Json<Value> {
    Json(json!({ "status": "healthy", "database": "connected" }))
}

async fn version() -> Json<Value> {
    Json(json!({
        "api_version": HINDSIGHT_API_VERSION,
        "server": format!("rustsight/{}", env!("CARGO_PKG_VERSION")),
        "features": {
            "observations": true,
            "mcp": false,
            "worker": true,
            "bank_config_api": true,
            "bank_llm_health": false,
            "file_upload_api": false,
            "document_export_api": false,
            "document_import_api": false,
            "audit_log": false,
            "llm_trace": false,
            "store_document_text": true,
        },
    }))
}

/// The API description. The Hindsight part is limited to what clients probe
/// for: a recall field is listed only when this server acts on it, so a
/// client checking for a feature gets an honest answer. The native API's
/// full description (`openapi.json`) is merged in alongside.
async fn openapi() -> Json<Value> {
    let recall_fields = [
        "query",
        "types",
        "prefer_observations",
        "budget",
        "max_tokens",
        "trace",
        "query_timestamp",
        "include",
        "tags",
        "tags_match",
        "tag_groups",
        "min_scores",
    ];
    let properties: Map<String, Value> = recall_fields
        .iter()
        .map(|field| (field.to_string(), json!({})))
        .collect();
    let bank = "/v1/default/banks/{bank_id}";
    let mut doc = json!({
        "openapi": "3.1.0",
        "info": { "title": "rustsight", "version": HINDSIGHT_API_VERSION },
        "paths": {
            "/v1/default/banks": { "get": {} },
            (bank): { "put": {}, "patch": {}, "delete": {} },
            (format!("{bank}/config")): { "get": {}, "patch": {}, "delete": {} },
            (format!("{bank}/profile")): { "get": {} },
            (format!("{bank}/memories")): { "post": {} },
            (format!("{bank}/memories/recall")): {
                "post": { "requestBody": { "content": { "application/json": {
                    "schema": { "$ref": "#/components/schemas/RecallRequest" }
                } } } }
            },
            (format!("{bank}/reflect")): { "post": {} },
            (format!("{bank}/documents")): { "get": {} },
            (format!("{bank}/documents/{{document_id}}")): { "get": {}, "delete": {} },
            (format!("{bank}/operations")): { "get": {} },
            (format!("{bank}/operations/{{operation_id}}")): { "get": {} },
            (format!("{bank}/operations/{{operation_id}}/retry")): { "post": {} },
        },
        "components": { "schemas": { "RecallRequest": {
            "type": "object",
            "required": ["query"],
            "properties": properties,
        } } },
    });
    crate::http::merge_native_openapi(&mut doc);
    Json(doc)
}

// ── Banks ────────────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct DispositionBody {
    skepticism: Option<i64>,
    literalism: Option<i64>,
    empathy: Option<i64>,
}

#[derive(Deserialize, Default)]
struct BankBody {
    name: Option<String>,
    mission: Option<String>,
    reflect_mission: Option<String>,
    background: Option<String>,
    disposition: Option<DispositionBody>,
    disposition_skepticism: Option<i64>,
    disposition_literalism: Option<i64>,
    disposition_empathy: Option<i64>,
    retain_mission: Option<String>,
    observations_mission: Option<String>,
    enable_observations: Option<bool>,
    entity_labels: Option<Value>,
    entities_allow_free_form: Option<bool>,
}

fn bank_or_default(engine: &Engine, bank_id: &str) -> Result<Bank, ApiError> {
    Ok(engine.store.get_bank(bank_id)?.unwrap_or(Bank {
        id: bank_id.to_string(),
        mission: None,
        directives: None,
        background: None,
        disposition: Disposition::default(),
    }))
}

fn profile_json(engine: &Engine, bank: &Bank) -> Result<Value, ApiError> {
    let config = engine.store.bank_config(&bank.id)?;
    let name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&bank.id);
    let mission = bank.mission.clone().unwrap_or_default();
    Ok(json!({
        "bank_id": bank.id,
        "name": name,
        "disposition": {
            "skepticism": bank.disposition.skepticism,
            "literalism": bank.disposition.literalism,
            "empathy": bank.disposition.empathy,
        },
        "mission": mission,
        "background": mission,
    }))
}

fn trait_value(name: &str, v: i64) -> Result<u8, ApiError> {
    if (1..=5).contains(&v) {
        Ok(v as u8)
    } else {
        Err(ApiError::unprocessable(format!(
            "disposition.{name} must be between 1 and 5"
        )))
    }
}

async fn upsert_bank(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    body: Result<Json<BankBody>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body?;
    if let Some(labels) = &body.entity_labels {
        BankLabels::parse(Some(labels), true)?;
    }
    let mut bank = bank_or_default(&engine, &bank_id)?;
    // `reflect_mission` is the current name; `mission` and `background` are
    // upstream's deprecated aliases for it.
    if let Some(m) = body.reflect_mission.or(body.mission).or(body.background) {
        bank.mission = Some(m).filter(|m| !m.trim().is_empty());
    }
    let d = body.disposition.unwrap_or_default();
    if let Some(v) = body.disposition_skepticism.or(d.skepticism) {
        bank.disposition.skepticism = trait_value("skepticism", v)?;
    }
    if let Some(v) = body.disposition_literalism.or(d.literalism) {
        bank.disposition.literalism = trait_value("literalism", v)?;
    }
    if let Some(v) = body.disposition_empathy.or(d.empathy) {
        bank.disposition.empathy = trait_value("empathy", v)?;
    }
    engine.store.ensure_bank(&bank)?;
    engine.store.update_bank(&bank)?;

    let mut config = match engine.store.bank_config(&bank_id)? {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    // A bank configured through this API follows Hindsight's split: the
    // mission steers reflect only; retain and consolidation have their own.
    config.insert("mission_scope".into(), json!("reflect"));
    for (key, value) in [
        ("name", body.name),
        ("retain_mission", body.retain_mission),
        ("observations_mission", body.observations_mission),
    ] {
        if let Some(v) = value {
            config.insert(key.into(), json!(v));
        }
    }
    for (key, value) in [
        ("enable_observations", body.enable_observations),
        ("entities_allow_free_form", body.entities_allow_free_form),
    ] {
        if let Some(v) = value {
            config.insert(key.into(), json!(v));
        }
    }
    if let Some(labels) = body.entity_labels {
        config.insert("entity_labels".into(), labels);
    }
    engine
        .store
        .set_bank_config(&bank_id, &Value::Object(config))?;
    // Observations may have just been switched back on.
    engine.wake_worker();
    Ok(Json(profile_json(&engine, &bank)?))
}

async fn delete_bank(State(engine): State<Arc<Engine>>, Path(bank_id): Path<String>) -> ApiResult {
    let deleted = engine.delete_bank(&bank_id).await?;
    Ok(Json(json!({
        "success": true,
        "message": format!("Bank '{bank_id}' and all associated data deleted successfully"),
        "deleted_count": deleted.memories + deleted.entities + deleted.documents,
    })))
}

// ── Bank config ──────────────────────────────────────────────────────────

/// The per-bank settings this server acts on. Anything else is refused:
/// accepting a setting that changes nothing would misreport what ran.
const CONFIG_KEYS: [&str; 9] = [
    "retain_mission",
    "observations_mission",
    "reflect_mission",
    "enable_observations",
    "entity_labels",
    "entities_allow_free_form",
    "disposition_skepticism",
    "disposition_literalism",
    "disposition_empathy",
];

fn is_config_key(key: &str) -> bool {
    CONFIG_KEYS.contains(&key)
}

/// Upstream accepts both `retain_mission` and `HINDSIGHT_API_RETAIN_MISSION`.
fn config_key(raw: &str) -> String {
    let lower = raw.to_lowercase();
    lower
        .strip_prefix("hindsight_api_")
        .unwrap_or(&lower)
        .to_string()
}

#[derive(Deserialize)]
struct BankConfigUpdate {
    updates: Map<String, Value>,
}

fn bank_config_map(engine: &Engine, bank_id: &str) -> Result<Map<String, Value>, ApiError> {
    Ok(match engine.store.bank_config(bank_id)? {
        Value::Object(map) => map,
        _ => Map::new(),
    })
}

fn bank_config_json(engine: &Engine, bank: &Bank) -> Result<Value, ApiError> {
    let stored = bank_config_map(engine, &bank.id)?;
    let overrides: Map<String, Value> = stored
        .iter()
        .filter(|(key, _)| is_config_key(key))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let setting = |key: &str, default: Value| stored.get(key).cloned().unwrap_or(default);
    Ok(json!({
        "bank_id": bank.id,
        "config": {
            "retain_mission": setting("retain_mission", Value::Null),
            "observations_mission": setting("observations_mission", Value::Null),
            "reflect_mission": bank.mission,
            "enable_observations": setting("enable_observations", json!(true)),
            "entity_labels": setting("entity_labels", Value::Null),
            "entities_allow_free_form": setting("entities_allow_free_form", json!(true)),
            "disposition_skepticism": bank.disposition.skepticism,
            "disposition_literalism": bank.disposition.literalism,
            "disposition_empathy": bank.disposition.empathy,
        },
        "overrides": overrides,
    }))
}

/// Validate one update and apply it to the in-memory bank; nothing is
/// stored until every update in the request has passed.
fn apply_config_update(bank: &mut Bank, key: &str, value: &Value) -> Result<(), ApiError> {
    let wrong_type = |expected: &str| ApiError::bad_request(format!("{key} must be {expected}"));
    match key {
        "retain_mission" | "observations_mission" | "reflect_mission" => {
            if !(value.is_string() || value.is_null()) {
                return Err(wrong_type("a string or null"));
            }
            if key == "reflect_mission" {
                bank.mission = value
                    .as_str()
                    .filter(|m| !m.trim().is_empty())
                    .map(str::to_string);
            }
        }
        "enable_observations" | "entities_allow_free_form" => {
            if !(value.is_boolean() || value.is_null()) {
                return Err(wrong_type("a boolean or null"));
            }
        }
        "entity_labels" => {
            BankLabels::parse(Some(value), true)?;
        }
        _ => {
            let name = key.trim_start_matches("disposition_");
            let defaults = Disposition::default();
            let (slot, default) = match name {
                "skepticism" => (&mut bank.disposition.skepticism, defaults.skepticism),
                "literalism" => (&mut bank.disposition.literalism, defaults.literalism),
                _ => (&mut bank.disposition.empathy, defaults.empathy),
            };
            *slot = match value {
                Value::Null => default,
                other => {
                    let raw = other.as_i64().ok_or_else(|| wrong_type("an integer"))?;
                    trait_value(name, raw).map_err(|ApiError(_, m)| ApiError::bad_request(m))?
                }
            };
        }
    }
    Ok(())
}

fn store_bank_settings(
    engine: &Engine,
    bank: &Bank,
    mut config: Map<String, Value>,
) -> Result<(), ApiError> {
    // A bank configured through this API follows Hindsight's mission split.
    config.insert("mission_scope".into(), json!("reflect"));
    engine.store.ensure_bank(bank)?;
    engine.store.update_bank(bank)?;
    engine
        .store
        .set_bank_config(&bank.id, &Value::Object(config))?;
    engine.wake_worker();
    Ok(())
}

async fn get_bank_config(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> ApiResult {
    let bank = bank_or_default(&engine, &bank_id)?;
    Ok(Json(bank_config_json(&engine, &bank)?))
}

async fn patch_bank_config(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    body: Result<Json<BankConfigUpdate>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body?;
    let mut bank = bank_or_default(&engine, &bank_id)?;
    let mut config = bank_config_map(&engine, &bank_id)?;
    for (raw_key, value) in &body.updates {
        let key = config_key(raw_key);
        if !is_config_key(&key) {
            return Err(ApiError::bad_request(format!(
                "Unsupported bank config key {raw_key:?}; supported keys: {}",
                CONFIG_KEYS.join(", ")
            )));
        }
        apply_config_update(&mut bank, &key, value)?;
        if value.is_null() {
            config.remove(&key);
        } else {
            config.insert(key, value.clone());
        }
    }
    store_bank_settings(&engine, &bank, config)?;
    Ok(Json(bank_config_json(&engine, &bank)?))
}

async fn reset_bank_config(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> ApiResult {
    let mut bank = bank_or_default(&engine, &bank_id)?;
    let mut config = bank_config_map(&engine, &bank_id)?;
    let overridden: Vec<String> = config
        .keys()
        .filter(|key| is_config_key(key))
        .cloned()
        .collect();
    for key in overridden {
        apply_config_update(&mut bank, &key, &Value::Null)?;
        config.remove(&key);
    }
    store_bank_settings(&engine, &bank, config)?;
    Ok(Json(bank_config_json(&engine, &bank)?))
}

async fn bank_profile(State(engine): State<Arc<Engine>>, Path(bank_id): Path<String>) -> ApiResult {
    // Upstream auto-creates a bank with defaults on first profile read.
    let bank = bank_or_default(&engine, &bank_id)?;
    engine.store.ensure_bank(&bank)?;
    Ok(Json(profile_json(&engine, &bank)?))
}

async fn list_banks(State(engine): State<Arc<Engine>>) -> ApiResult {
    let banks = engine.store.list_banks()?;
    let mut items = Vec::with_capacity(banks.len());
    for bank in &banks {
        items.push(profile_json(&engine, bank)?);
    }
    let total = items.len();
    Ok(Json(
        json!({ "banks": items, "total": total, "limit": total, "offset": 0 }),
    ))
}

// ── Retain ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RetainBody {
    items: Vec<Value>,
    #[serde(rename = "async", default)]
    is_async: bool,
    #[serde(default)]
    document_tags: Option<Vec<String>>,
    #[serde(default)]
    operation_id: Option<String>,
}

/// One retain item, normalized from the wire shape.
struct RetainItem {
    content: String,
    timestamp_ms: Option<i64>,
    timestamp_unset: bool,
    context: Option<String>,
    metadata: Option<BTreeMap<String, String>>,
    document_id: Option<String>,
    entities: Vec<ExtractedEntity>,
    tags: Vec<String>,
    update_mode: Option<UpdateMode>,
}

fn string_list(v: &Value, field: &str) -> Result<Vec<String>, ApiError> {
    match v {
        Value::Null => Ok(vec![]),
        Value::Array(items) => items
            .iter()
            .map(|t| {
                t.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ApiError::unprocessable(format!("{field} must be strings")))
            })
            .collect(),
        // MCP bridges sometimes send a JSON array serialized as a string;
        // a plain string is a single tag (upstream `coerce_tags`).
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(items)) => string_list(&Value::Array(items), field),
            _ => Ok(vec![s.clone()]),
        },
        _ => Err(ApiError::unprocessable(format!(
            "{field} must be a list of strings"
        ))),
    }
}

/// `content` is a plain string or an ordered list of content blocks; only
/// text blocks are supported (no vision-capable retain path here).
fn item_content(v: &Value, index: usize) -> Result<String, ApiError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut parts = vec![];
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => parts.push(
                        block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    ),
                    other => {
                        return Err(ApiError::bad_request(format!(
                            "items[{index}].content: {} content blocks are not supported by \
                             this server; send text",
                            other.unwrap_or("untyped")
                        )))
                    }
                }
            }
            Ok(parts.join("\n"))
        }
        _ => Err(ApiError::unprocessable(format!(
            "items[{index}].content must be a string or a list of content blocks"
        ))),
    }
}

fn parse_item(v: &Value, index: usize) -> Result<RetainItem, ApiError> {
    let obj = v
        .as_object()
        .ok_or_else(|| ApiError::unprocessable(format!("items[{index}] must be an object")))?;
    let content = item_content(obj.get("content").unwrap_or(&Value::Null), index)?;
    if content.trim().is_empty() {
        return Err(ApiError::unprocessable(format!(
            "items[{index}].content must not be empty"
        )));
    }
    let (timestamp_ms, timestamp_unset) = match obj.get("timestamp") {
        None | Some(Value::Null) => (None, false),
        Some(Value::String(s)) if s == "unset" => (None, true),
        Some(Value::String(s)) => match parse_iso(s) {
            Some(ms) => (Some(ms), false),
            None => {
                return Err(ApiError::unprocessable(format!(
                    "items[{index}].timestamp is not an ISO 8601 datetime: {s:?}"
                )))
            }
        },
        Some(_) => {
            return Err(ApiError::unprocessable(format!(
                "items[{index}].timestamp must be an ISO 8601 string"
            )))
        }
    };
    let metadata = match obj.get("metadata") {
        None | Some(Value::Null) => None,
        Some(Value::Object(map)) => Some(
            map.iter()
                .map(|(k, v)| {
                    let value = match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    (k.clone(), value)
                })
                .collect::<BTreeMap<_, _>>(),
        ),
        Some(_) => {
            return Err(ApiError::unprocessable(format!(
                "items[{index}].metadata must be an object"
            )))
        }
    };
    let entities = match obj.get("entities") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|e| {
                let name = e.get("text").and_then(|t| t.as_str())?.trim();
                (!name.is_empty()).then(|| ExtractedEntity {
                    name: name.to_string(),
                    // Upstream defaults an untyped entity to CONCEPT.
                    etype: EntityType::parse(
                        e.get("type").and_then(|t| t.as_str()).unwrap_or("concept"),
                    ),
                })
            })
            .collect(),
        Some(_) => {
            return Err(ApiError::unprocessable(format!(
                "items[{index}].entities must be a list"
            )))
        }
    };
    let update_mode = match obj.get("update_mode").and_then(|v| v.as_str()) {
        None => None,
        Some("replace") => Some(UpdateMode::Replace),
        Some("append") => Some(UpdateMode::Append),
        Some(other) => {
            return Err(ApiError::unprocessable(format!(
                "items[{index}].update_mode must be 'replace' or 'append', got {other:?}"
            )))
        }
    };
    let document_id = obj
        .get("document_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .filter(|d| !d.trim().is_empty());
    if update_mode == Some(UpdateMode::Append) && document_id.is_none() {
        return Err(ApiError::bad_request(format!(
            "items[{index}]: update_mode 'append' requires a document_id"
        )));
    }
    Ok(RetainItem {
        content,
        timestamp_ms,
        timestamp_unset,
        context: obj
            .get("context")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|c| !c.trim().is_empty()),
        metadata,
        document_id,
        entities,
        tags: string_list(obj.get("tags").unwrap_or(&Value::Null), "tags")?,
        update_mode,
    })
}

/// Give every item a concrete `document_id`, the way upstream does: when no
/// item names one they all share one generated document; otherwise each
/// item without one gets its own. Written back into the body so a queued
/// (async) request replays against the same documents after a restart.
fn assign_document_ids(items: &mut [Value]) {
    let any_named = items.iter().any(|i| {
        i.get("document_id")
            .and_then(|d| d.as_str())
            .is_some_and(|d| !d.trim().is_empty())
    });
    let shared = uuid::Uuid::new_v4().to_string();
    for item in items.iter_mut() {
        let Some(obj) = item.as_object_mut() else {
            continue;
        };
        let named = obj
            .get("document_id")
            .and_then(|d| d.as_str())
            .is_some_and(|d| !d.trim().is_empty());
        if !named {
            let id = if any_named {
                uuid::Uuid::new_v4().to_string()
            } else {
                shared.clone()
            };
            obj.insert("document_id".into(), json!(id));
        }
    }
}

/// Retain every item in order. Items that share a `document_id` within one
/// request build one document: the first applies its `update_mode`, the
/// rest append to it.
async fn run_retain(
    engine: &Engine,
    guard: &BankGuard,
    bank_id: &str,
    items: &[Value],
    document_tags: &[String],
) -> Result<usize, ApiError> {
    let mut seen_documents: HashSet<String> = HashSet::new();
    for (index, raw) in items.iter().enumerate() {
        let item = parse_item(raw, index)?;
        let mut tags = item.tags;
        for t in document_tags {
            if !tags.contains(t) {
                tags.push(t.clone());
            }
        }
        let repeated = item
            .document_id
            .as_ref()
            .is_some_and(|d| !seen_documents.insert(d.clone()));
        let update_mode = if repeated {
            UpdateMode::Append
        } else {
            item.update_mode.unwrap_or_default()
        };
        engine
            .retain_guarded(
                guard,
                RetainRequest {
                    bank_id: bank_id.to_string(),
                    content: item.content,
                    context: item.context,
                    timestamp_ms: item.timestamp_ms,
                    timestamp_unset: item.timestamp_unset,
                    document_id: item.document_id,
                    update_mode,
                    tags,
                    metadata: item.metadata,
                    entities: item.entities,
                },
            )
            .await?;
    }
    Ok(items.len())
}

async fn finish_operation(engine: &Engine, mut op: Operation, outcome: Result<usize, ApiError>) {
    match outcome {
        Ok(_) => op.status = "completed".into(),
        Err(ApiError(_, detail)) => {
            op.status = "failed".into();
            op.error = Some(detail);
        }
    }
    op.updated_at_ms = engine.clock.now_ms();
    if let Err(err) = engine.store.upsert_operation(&op) {
        eprintln!("rustsight: recording operation {} failed: {err}", op.id);
    }
}

fn spawn_operation(engine: Arc<Engine>, op: Operation) {
    tokio::spawn(async move {
        let _slot = async_retain_slots().acquire().await;
        // Held through the final status write, so a bank delete either
        // waits for that write or removes the operation before it starts.
        let guard = engine.bank_read(&op.bank_id).await;
        match engine.store.get_operation(&op.id) {
            Ok(Some(current)) if current.status == "pending" => {}
            _ => return,
        }
        let items: Vec<Value> = op
            .payload
            .get("items")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let document_tags = string_list(
            op.payload.get("document_tags").unwrap_or(&Value::Null),
            "document_tags",
        )
        .unwrap_or_default();
        let outcome = run_retain(&engine, &guard, &op.bank_id, &items, &document_tags).await;
        if let Err(ApiError(_, detail)) = &outcome {
            eprintln!(
                "rustsight: async retain {} (bank {}) failed: {detail}",
                op.id, op.bank_id
            );
        }
        finish_operation(&engine, op, outcome).await;
    });
}

/// Re-queue async retains that were accepted but not finished when the
/// server last stopped. Call once at startup.
pub fn resume_pending_operations(engine: &Arc<Engine>) -> usize {
    let pending = match engine.store.pending_operations() {
        Ok(ops) => ops,
        Err(err) => {
            eprintln!("rustsight: listing pending operations failed: {err}");
            return 0;
        }
    };
    let count = pending.len();
    for op in pending {
        spawn_operation(Arc::clone(engine), op);
    }
    count
}

async fn retain(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    body: Result<Json<RetainBody>, JsonRejection>,
) -> ApiResult {
    let Json(mut body) = body?;
    if body.items.is_empty() {
        return Err(ApiError::unprocessable("items must not be empty"));
    }
    // Validate everything up front so an async request cannot be accepted
    // and then fail on a malformed item nobody is waiting to hear about.
    for (index, raw) in body.items.iter().enumerate() {
        parse_item(raw, index)?;
    }
    assign_document_ids(&mut body.items);
    let document_tags = body.document_tags.clone().unwrap_or_default();

    if !body.is_async {
        let guard = engine.bank_read(&bank_id).await;
        let count = run_retain(&engine, &guard, &bank_id, &body.items, &document_tags).await?;
        return Ok(Json(json!({
            "success": true,
            "bank_id": bank_id,
            "items_count": count,
            "async": false,
        })));
    }

    let operation_id = match body.operation_id.as_deref() {
        Some(raw) => uuid::Uuid::parse_str(raw)
            .map(|u| u.to_string())
            .map_err(|_| ApiError::unprocessable("operation_id must be a valid UUID"))?,
        None => uuid::Uuid::new_v4().to_string(),
    };
    let items_count = body.items.len();
    // A resubmitted operation_id returns the original operation and queues
    // no new work (idempotent async retries).
    if let Some(existing) = engine.store.get_operation(&operation_id)? {
        if existing.bank_id != bank_id {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "operation_id belongs to a different operation".into(),
            ));
        }
    } else {
        let now = engine.clock.now_ms();
        let op = Operation {
            id: operation_id.clone(),
            bank_id: bank_id.clone(),
            kind: "retain".into(),
            status: "pending".into(),
            error: None,
            payload: json!({ "items": body.items, "document_tags": document_tags }),
            created_at_ms: now,
            updated_at_ms: now,
        };
        engine.store.upsert_operation(&op)?;
        spawn_operation(Arc::clone(&engine), op);
    }
    Ok(Json(json!({
        "success": true,
        "bank_id": bank_id,
        "items_count": items_count,
        "async": true,
        "operation_id": operation_id,
    })))
}

/// Consolidation runs as a queue rather than as operation rows. While a bank
/// has work queued it is reported as one pending operation under this id, so
/// a client waiting for "no operations left" also waits for observations.
fn consolidation_operation_id(bank_id: &str) -> String {
    format!("consolidation-{bank_id}")
}

fn consolidation_pending(engine: &Engine, bank_id: &str) -> Result<bool, ApiError> {
    Ok(engine.observations_enabled(bank_id)? && engine.store.count_dirty_entities(bank_id)? > 0)
}

fn operation_json(op: &Operation) -> Value {
    let items = op.payload.get("items").and_then(|v| v.as_array());
    let mut documents = items
        .into_iter()
        .flatten()
        .map(|item| item.get("document_id").and_then(|d| d.as_str()));
    let first = documents.next().flatten();
    let document_id = first.filter(|_| documents.all(|d| d == first));
    json!({
        "id": op.id,
        "task_type": op.kind,
        "items_count": items.map_or(0, Vec::len),
        "document_id": document_id,
        "created_at": iso(op.created_at_ms),
        "updated_at": iso(op.updated_at_ms),
        "completed_at": if op.status == "pending" { None } else { iso(op.updated_at_ms) },
        "status": op.status,
        "error_message": op.error,
    })
}

#[derive(Deserialize)]
struct OperationsQuery {
    status: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_operations(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    Query(query): Query<OperationsQuery>,
) -> ApiResult {
    let limit = query.limit.unwrap_or(20);
    if !(1..=100).contains(&limit) {
        return Err(ApiError::unprocessable("limit must be between 1 and 100"));
    }
    let offset = query.offset.unwrap_or(0);
    let status = query.status.as_deref();
    let kind = query.kind.as_deref();
    let consolidating = matches!(status, None | Some("pending"))
        && matches!(kind, None | Some("consolidation"))
        && consolidation_pending(&engine, &bank_id)?;
    let synthetic = usize::from(consolidating);
    // The consolidation entry sits first, ahead of the stored rows.
    let stored_offset = offset.saturating_sub(synthetic);
    let stored_limit = limit - usize::from(consolidating && offset == 0);
    let (rows, stored_total) =
        engine
            .store
            .list_operations(&bank_id, status, kind, stored_limit, stored_offset)?;
    let mut operations = Vec::with_capacity(limit);
    if consolidating && offset == 0 {
        let now = iso(engine.clock.now_ms());
        operations.push(json!({
            "id": consolidation_operation_id(&bank_id),
            "task_type": "consolidation",
            "items_count": 0,
            "document_id": Value::Null,
            "created_at": now,
            "updated_at": now,
            "completed_at": Value::Null,
            "status": "pending",
            "error_message": Value::Null,
        }));
    }
    operations.extend(rows.iter().map(operation_json));
    Ok(Json(json!({
        "bank_id": bank_id,
        "total": stored_total + synthetic,
        "limit": limit,
        "offset": offset,
        "operations": operations,
    })))
}

async fn retry_operation(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, operation_id)): Path<(String, String)>,
) -> ApiResult {
    if uuid::Uuid::parse_str(&operation_id).is_err() {
        return Err(ApiError::bad_request(format!(
            "Invalid operation_id: {operation_id}"
        )));
    }
    // Under the bank guard so a concurrent delete cannot be undone by the
    // status write below.
    let guard = engine.bank_read(&bank_id).await;
    let mut op = engine
        .store
        .get_operation(&operation_id)?
        .filter(|op| op.bank_id == bank_id)
        .ok_or_else(|| ApiError::not_found(format!("Operation {operation_id} not found")))?;
    if op.status != "failed" {
        return Err(ApiError(
            StatusCode::CONFLICT,
            format!(
                "Operation {operation_id} is {} and cannot be retried",
                op.status
            ),
        ));
    }
    op.status = "pending".into();
    op.error = None;
    op.updated_at_ms = engine.clock.now_ms();
    engine.store.upsert_operation(&op)?;
    drop(guard);
    spawn_operation(Arc::clone(&engine), op);
    Ok(Json(json!({
        "success": true,
        "message": format!("Operation {operation_id} queued for retry"),
        "operation_id": operation_id,
    })))
}

async fn get_operation(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, operation_id)): Path<(String, String)>,
) -> ApiResult {
    if operation_id == consolidation_operation_id(&bank_id) {
        let status = if consolidation_pending(&engine, &bank_id)? {
            "pending"
        } else {
            "completed"
        };
        return Ok(Json(json!({
            "operation_id": operation_id,
            "id": operation_id,
            "status": status,
            "operation_type": "consolidation",
            "created_at": Value::Null,
            "updated_at": Value::Null,
            "completed_at": Value::Null,
            "error_message": Value::Null,
        })));
    }
    let op = engine
        .store
        .get_operation(&operation_id)?
        .filter(|op| op.bank_id == bank_id)
        .ok_or_else(|| ApiError::not_found(format!("Operation {operation_id} not found")))?;
    let finished = op.status != "pending";
    Ok(Json(json!({
        "operation_id": op.id,
        "id": op.id,
        "status": op.status,
        "operation_type": op.kind,
        "created_at": iso(op.created_at_ms),
        "updated_at": iso(op.updated_at_ms),
        "completed_at": if finished { iso(op.updated_at_ms) } else { None },
        "error_message": op.error,
    })))
}

// ── Tag filters ──────────────────────────────────────────────────────────

fn parse_tags_match(raw: Option<&str>, default: TagsMatch) -> Result<TagsMatch, ApiError> {
    match raw {
        None => Ok(default),
        Some("any") => Ok(TagsMatch::Any),
        Some("all") => Ok(TagsMatch::All),
        Some("any_strict") => Ok(TagsMatch::AnyStrict),
        Some("all_strict") => Ok(TagsMatch::AllStrict),
        Some("exact") => Ok(TagsMatch::Exact),
        Some(other) => Err(ApiError::unprocessable(format!(
            "tags_match must be one of any, all, any_strict, all_strict, exact; got {other:?}"
        ))),
    }
}

/// The bank's tag vocabulary, read only if a fuzzy leaf asks for it and at
/// most once per request.
struct TagVocabulary<'a> {
    engine: &'a Engine,
    bank_id: &'a str,
    tags: Option<Vec<String>>,
}

impl TagVocabulary<'_> {
    fn get(&mut self) -> Result<&[String], ApiError> {
        if self.tags.is_none() {
            let index = self.engine.store.tag_index(self.bank_id)?;
            self.tags = Some(tag_vocabulary(&index).map_err(ApiError::unprocessable)?);
        }
        Ok(self.tags.as_deref().unwrap_or_default())
    }
}

fn parse_tag_group_list(
    v: &Value,
    vocabulary: &mut TagVocabulary<'_>,
) -> Result<Vec<TagGroup>, ApiError> {
    v.as_array()
        .ok_or_else(|| ApiError::unprocessable("tag group 'and'/'or' must be a list"))?
        .iter()
        .map(|group| parse_tag_group(group, vocabulary))
        .collect()
}

fn parse_tag_group(v: &Value, vocabulary: &mut TagVocabulary<'_>) -> Result<TagGroup, ApiError> {
    let obj = v
        .as_object()
        .ok_or_else(|| ApiError::unprocessable("each tag group must be an object"))?;
    if let Some(inner) = obj.get("and") {
        return Ok(TagGroup::And(parse_tag_group_list(inner, vocabulary)?));
    }
    if let Some(inner) = obj.get("or") {
        return Ok(TagGroup::Or(parse_tag_group_list(inner, vocabulary)?));
    }
    if let Some(inner) = obj.get("not") {
        return Ok(TagGroup::Not(Box::new(parse_tag_group(inner, vocabulary)?)));
    }
    let tags = string_list(obj.get("tags").unwrap_or(&Value::Null), "tags")?;
    // A leaf defaults to any_strict upstream (unlike the flat filter).
    let mode = parse_tags_match(
        obj.get("match").and_then(|m| m.as_str()),
        TagsMatch::AnyStrict,
    )?;
    match obj.get("resolve") {
        None | Some(Value::Null) => Ok(TagGroup::Leaf { tags, mode }),
        Some(Value::String(r)) if r == "exact" => Ok(TagGroup::Leaf { tags, mode }),
        Some(Value::String(r)) if r == "fuzzy" => {
            resolve_fuzzy_leaf(tags, mode, vocabulary.get()?).map_err(ApiError::unprocessable)
        }
        Some(other) => Err(ApiError::unprocessable(format!(
            "tag group resolve must be 'exact' or 'fuzzy', got {other}"
        ))),
    }
}

/// The request's tag scope from `tags` + `tags_match`, or `tag_groups`.
fn tag_filter(
    engine: &Engine,
    bank_id: &str,
    tags: &Option<Value>,
    tags_match: Option<&str>,
    tag_groups: &Option<Vec<Value>>,
) -> Result<Option<TagFilter>, ApiError> {
    let tags_given = matches!(tags, Some(v) if !v.is_null());
    if tags_given && tag_groups.is_some() {
        return Err(ApiError::unprocessable(
            "'tags' and 'tag_groups' are mutually exclusive. Use 'tag_groups' for compound filtering.",
        ));
    }
    let mut vocabulary = TagVocabulary {
        engine,
        bank_id,
        tags: None,
    };
    let filter = TagFilter {
        tags: match tags {
            Some(v) => string_list(v, "tags")?,
            None => vec![],
        },
        mode: parse_tags_match(tags_match, TagsMatch::Any)?,
        groups: match tag_groups {
            Some(groups) => groups
                .iter()
                .map(|group| parse_tag_group(group, &mut vocabulary))
                .collect::<Result<_, _>>()?,
            None => vec![],
        },
    };
    Ok((!filter.is_noop()).then_some(filter))
}

fn parse_budget(raw: Option<&str>, default: Budget) -> Result<Budget, ApiError> {
    match raw {
        None => Ok(default),
        Some("low") => Ok(Budget::Low),
        Some("mid") => Ok(Budget::Mid),
        Some("high") => Ok(Budget::High),
        Some(other) => Err(ApiError::unprocessable(format!(
            "budget must be one of low, mid, high; got {other:?}"
        ))),
    }
}

// ── Recall ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MaxTokens {
    max_tokens: Option<i64>,
}

#[derive(Deserialize, Default)]
struct RecallInclude {
    /// Present-and-null disables entity inclusion; absent means the default
    /// (enabled, 500 tokens) — hence the double Option.
    #[serde(default, deserialize_with = "present")]
    entities: Option<Option<MaxTokens>>,
    #[serde(default)]
    source_facts: Option<MaxTokens>,
    #[serde(default)]
    chunks: Option<MaxTokens>,
}

/// Deserialize a field so that an explicit `null` is distinguishable from
/// an absent key (`Some(None)` vs `None`).
fn present<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

#[derive(Deserialize, Default)]
struct MinScores {
    #[serde(rename = "final")]
    final_: Option<f32>,
}

#[derive(Deserialize)]
struct RecallBody {
    query: String,
    types: Option<Vec<String>>,
    prefer_observations: Option<bool>,
    budget: Option<String>,
    max_tokens: Option<usize>,
    #[serde(default)]
    trace: bool,
    query_timestamp: Option<String>,
    #[serde(default)]
    include: RecallInclude,
    tags: Option<Value>,
    tags_match: Option<String>,
    tag_groups: Option<Vec<Value>>,
    min_scores: Option<MinScores>,
}

fn parse_kinds(types: &[String], field: &str) -> Result<Vec<MemoryKind>, ApiError> {
    types
        .iter()
        .map(|t| {
            MemoryKind::parse(t)
                .filter(|k| RECALL_KINDS.contains(k))
                .ok_or_else(|| {
                    ApiError::unprocessable(format!(
                        "{field}: unknown fact type {t:?}; expected world, experience, observation"
                    ))
                })
        })
        .collect()
}

fn fact_json(m: &MemoryRef, entity_names: &HashMap<String, String>) -> Value {
    let entities: Vec<&String> = m
        .entity_ids
        .iter()
        .filter_map(|id| entity_names.get(id))
        .collect();
    json!({
        "id": m.id,
        "text": m.text,
        "type": m.kind.as_str(),
        "entities": if entities.is_empty() { Value::Null } else { json!(entities) },
        "context": m.context,
        "occurred_start": m.occurred_start_ms.and_then(iso),
        "occurred_end": m.occurred_end_ms.and_then(iso),
        "mentioned_at": m.mentioned_at_ms.and_then(iso),
        "document_id": m.document_id,
        "metadata": m.metadata,
        "chunk_id": Value::Null,
        "tags": if m.tags.is_empty() { Value::Null } else { json!(m.tags) },
    })
}

/// The longest prefix of `text` within `max_tokens`.
fn truncate_to_tokens(engine: &Engine, text: &str, max_tokens: usize) -> String {
    let boundaries: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain([text.len()])
        .collect();
    let (mut fits, mut too_long) = (0, boundaries.len());
    while too_long - fits > 1 {
        let middle = (fits + too_long) / 2;
        if engine.tokens.count(&text[..boundaries[middle]]) <= max_tokens {
            fits = middle;
        } else {
            too_long = middle;
        }
    }
    text[..boundaries[fits]].to_string()
}

/// Source chunks for ranked results, best result first, each chunk once. An
/// observation contributes the chunks of its evidence at its own rank. The
/// chunk that crosses the budget is cut to fit and ends the list. Under a
/// tag scope a chunk is visible only if its document is: it can hold text
/// from outside the scope that the fact extracted from it does not.
fn recall_chunks(
    engine: &Engine,
    bank_id: &str,
    ordered_ids: &[String],
    scope: Option<&TagFilter>,
    max_tokens: usize,
) -> Result<Map<String, Value>, ApiError> {
    let mut chunks: HashMap<String, Chunk> = engine
        .store
        .get_chunks(bank_id, ordered_ids)?
        .into_iter()
        .map(|chunk| (chunk.id.clone(), chunk))
        .collect();
    let mut remaining = max_tokens;
    let mut out = Map::new();
    for id in ordered_ids {
        if remaining == 0 {
            break;
        }
        let Some(chunk) = chunks.remove(id) else {
            continue;
        };
        if scope.is_some_and(|f| !f.matches(&chunk.document_tags)) {
            continue;
        }
        let cost = engine.tokens.count(&chunk.text);
        let truncated = cost > remaining;
        let text = if truncated {
            truncate_to_tokens(engine, &chunk.text, remaining)
        } else {
            chunk.text
        };
        remaining = if truncated { 0 } else { remaining - cost };
        out.insert(
            chunk.id.clone(),
            json!({
                "id": chunk.id,
                "text": text,
                "chunk_index": chunk.chunk_index,
                "truncated": truncated,
            }),
        );
    }
    Ok(out)
}

async fn recall(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    body: Result<Json<RecallBody>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body?;
    let started = std::time::Instant::now();
    if !body.query.chars().any(char::is_alphanumeric) {
        return Err(ApiError::unprocessable(
            "query must contain at least one word character after normalization",
        ));
    }
    let kinds = match &body.types {
        Some(types) if !types.is_empty() => parse_kinds(types, "types")?,
        _ => RECALL_KINDS.to_vec(),
    };
    let now_ms = match body.query_timestamp.as_deref() {
        Some(raw) => Some(parse_iso(raw).ok_or_else(|| {
            ApiError::bad_request(format!(
                "Invalid query_timestamp format. Expected ISO format (e.g., '2023-05-30T23:40:00'): {raw:?}"
            ))
        })?),
        None => None,
    };
    let budget = parse_budget(body.budget.as_deref(), Budget::Mid)?;
    let max_tokens = body.max_tokens.unwrap_or(4096).max(1);
    let tags = tag_filter(
        &engine,
        &bank_id,
        &body.tags,
        body.tags_match.as_deref(),
        &body.tag_groups,
    )?;
    let chunk_scope = tags.clone();

    let response = engine
        .recall(RecallRequest {
            bank_id: bank_id.clone(),
            query: body.query.clone(),
            token_budget: max_tokens,
            top_k_per_arm: engine
                .config
                .recall_budget
                .resolve(Some(budget), max_tokens),
            kinds: Some(kinds.clone()),
            tags,
            now_ms,
        })
        .await?;
    let mut memories = response.memories;

    // `min_scores.final` is a post-ranking floor (inclusive).
    if let Some(floor) = body.min_scores.as_ref().and_then(|m| m.final_) {
        memories.retain(|m| m.score >= floor);
    }

    let has_observations = memories.iter().any(|m| m.kind == MemoryKind::Observation);
    let want_source_facts = body.include.source_facts.is_some();
    let include_entities = !matches!(body.include.entities, Some(None));
    let want_chunks = body.include.chunks.is_some();
    let observations = if has_observations || include_entities {
        engine.store.observations_for_bank(&bank_id)?
    } else {
        vec![]
    };
    let evidence_of: HashMap<&str, Vec<&str>> = observations
        .iter()
        .map(|o| {
            (
                o.id.as_str(),
                o.evidence.iter().map(|ev| ev.memory_id.as_str()).collect(),
            )
        })
        .collect();

    // prefer_observations: an observation in the results supersedes the raw
    // facts it was consolidated from.
    if body.prefer_observations == Some(true) && has_observations {
        let superseded: HashSet<String> = memories
            .iter()
            .filter(|m| m.kind == MemoryKind::Observation)
            .flat_map(|m| evidence_of.get(m.id.as_str()).into_iter().flatten())
            .map(|id| id.to_string())
            .collect();
        memories.retain(|m| m.kind == MemoryKind::Observation || !superseded.contains(&m.id));
    }

    let entity_names: HashMap<String, String> = engine
        .store
        .entities_for_bank(&bank_id)?
        .into_iter()
        .map(|e| (e.id, e.name))
        .collect();

    // Facts in rank order, with an observation standing in for its evidence.
    let ranked_fact_ids: Vec<String> = memories
        .iter()
        .flat_map(|m| match evidence_of.get(m.id.as_str()) {
            Some(evidence) if m.kind == MemoryKind::Observation => {
                evidence.iter().map(|id| id.to_string()).collect()
            }
            _ => vec![m.id.clone()],
        })
        .collect();
    let chunk_of = engine.store.memory_chunk_ids(&bank_id, &ranked_fact_ids)?;
    let chunks = match &body.include.chunks {
        Some(options) => {
            let mut seen = HashSet::new();
            let ordered: Vec<String> = ranked_fact_ids
                .iter()
                .filter_map(|id| chunk_of.get(id))
                .filter(|id| seen.insert(id.as_str()))
                .cloned()
                .collect();
            let budget = options.max_tokens.unwrap_or(8192).max(0) as usize;
            recall_chunks(&engine, &bank_id, &ordered, chunk_scope.as_ref(), budget)?
        }
        None => Map::new(),
    };
    let with_chunk_id = |mut fact: Value, id: &str| {
        fact["chunk_id"] = json!(chunk_of.get(id));
        fact
    };

    let mut results: Vec<Value> = Vec::with_capacity(memories.len());
    let mut source_facts = Map::new();
    let mut source_facts_truncated = false;
    let mut source_budget = body
        .include
        .source_facts
        .as_ref()
        .map(|s| s.max_tokens.unwrap_or(4096))
        .unwrap_or(0);
    for m in &memories {
        let mut fact = with_chunk_id(fact_json(m, &entity_names), &m.id);
        fact["scores"] = json!({ "final": m.score });
        if want_source_facts && m.kind == MemoryKind::Observation {
            let ids: Vec<String> = evidence_of
                .get(m.id.as_str())
                .into_iter()
                .flatten()
                .map(|id| id.to_string())
                .collect();
            for source in engine.store.get_refs(&bank_id, &ids)? {
                if source_facts.contains_key(&source.id) {
                    continue;
                }
                let cost = engine.tokens.count(&source.text) as i64;
                if source_budget >= 0 && cost > source_budget {
                    source_facts_truncated = true;
                    continue;
                }
                if source_budget >= 0 {
                    source_budget -= cost;
                }
                source_facts.insert(
                    source.id.clone(),
                    with_chunk_id(fact_json(&source, &entity_names), &source.id),
                );
            }
            fact["source_fact_ids"] = json!(ids);
        }
        results.push(fact);
    }

    // Entity states: each entity named by a result, with its observations,
    // within the entity token budget.
    let entities = if include_entities {
        let mut budget = match &body.include.entities {
            Some(Some(opts)) => opts.max_tokens.unwrap_or(500),
            _ => 500,
        };
        let mut seen: HashSet<&str> = HashSet::new();
        let mut states = Map::new();
        for entity_id in memories.iter().flat_map(|m| m.entity_ids.iter()) {
            let Some(name) = entity_names.get(entity_id) else {
                continue;
            };
            if !seen.insert(entity_id.as_str()) {
                continue;
            }
            let mut entity_observations = vec![];
            for o in observations.iter().filter(|o| &o.entity_id == entity_id) {
                let cost = engine.tokens.count(&o.text) as i64;
                if cost > budget {
                    break;
                }
                budget -= cost;
                entity_observations
                    .push(json!({ "text": o.text, "mentioned_at": iso(o.updated_at_ms) }));
            }
            if !entity_observations.is_empty() {
                states.insert(
                    name.clone(),
                    json!({
                        "entity_id": entity_id,
                        "canonical_name": name,
                        "observations": entity_observations,
                    }),
                );
            }
        }
        (!states.is_empty()).then_some(Value::Object(states))
    } else {
        None
    };

    let trace = body.trace.then(|| {
        json!({
            "query": body.query,
            "num_results": results.len(),
            "time_seconds": started.elapsed().as_secs_f64(),
            "strategies_used": response.strategies_used,
        })
    });
    Ok(Json(json!({
        "results": results,
        "trace": trace,
        "entities": entities,
        "chunks": if want_chunks { Value::Object(chunks) } else { Value::Null },
        "source_facts": if want_source_facts { Value::Object(source_facts) } else { Value::Null },
        "source_facts_truncated": want_source_facts.then_some(source_facts_truncated),
    })))
}

// ── Reflect ──────────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct ReflectInclude {
    facts: Option<Value>,
}

#[derive(Deserialize)]
struct ReflectBody {
    query: String,
    budget: Option<String>,
    context: Option<String>,
    max_tokens: Option<usize>,
    #[serde(default)]
    include: ReflectInclude,
    response_schema: Option<Value>,
    tags: Option<Value>,
    tags_match: Option<String>,
    tag_groups: Option<Vec<Value>>,
    fact_types: Option<Vec<String>>,
}

async fn reflect(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    body: Result<Json<ReflectBody>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body?;
    let context = body.context.filter(|c| !c.trim().is_empty());
    if body.query.trim().is_empty() && context.is_none() {
        return Err(ApiError::unprocessable(
            "'query' must not be empty or whitespace-only.",
        ));
    }
    let kinds = match &body.fact_types {
        Some(types) if types.is_empty() => {
            return Err(ApiError::unprocessable(
                "fact_types must not be empty. Use null to include all fact types.",
            ))
        }
        Some(types) => Some(parse_kinds(types, "fact_types")?),
        None => None,
    };
    if let Some(schema) = &body.response_schema {
        if !schema.is_object() {
            return Err(ApiError::unprocessable(
                "response_schema must be a JSON Schema object",
            ));
        }
    }
    // The API's reflect default is `low` (upstream halves its iteration
    // budget at that level).
    let budget = parse_budget(body.budget.as_deref(), Budget::Low)?;
    let options = ReflectOptions {
        context,
        max_rounds: Some(engine.reflect_rounds_for_budget(budget)),
        max_tokens: Some(body.max_tokens.unwrap_or(4096).max(1)),
        tags: tag_filter(
            &engine,
            &bank_id,
            &body.tags,
            body.tags_match.as_deref(),
            &body.tag_groups,
        )?,
        kinds,
        response_schema: body.response_schema,
    };
    let response = engine.reflect_with(&bank_id, &body.query, &options).await?;

    let based_on = if body.include.facts.is_some_and(|f| !f.is_null()) {
        // The memories the answer cites; an answer that cited nothing is
        // based on everything the loop retrieved.
        let ids = if response.citations.is_empty() {
            &response.retrieved_ids
        } else {
            &response.citations
        };
        let entity_names: HashMap<String, String> = HashMap::new();
        let memories: Vec<Value> = engine
            .store
            .get_refs(&bank_id, ids)?
            .iter()
            .map(|m| {
                let mut fact = fact_json(m, &entity_names);
                if let Some(obj) = fact.as_object_mut() {
                    obj.remove("entities");
                }
                fact
            })
            .collect();
        json!({ "memories": memories, "mental_models": [], "directives": [] })
    } else {
        Value::Null
    };
    Ok(Json(json!({
        "text": response.answer,
        "based_on": based_on,
        "structured_output": response.structured_output,
        "structured_output_error": response.structured_output_error,
        "usage": Value::Null,
        "trace": Value::Null,
    })))
}

// ── Documents ────────────────────────────────────────────────────────────

fn document_json(engine: &Engine, doc: &Document, with_text: bool) -> Result<Value, ApiError> {
    let mut out = json!({
        "id": doc.id,
        "bank_id": doc.bank_id,
        "content_hash": doc.content_hash,
        "created_at": iso(doc.created_at_ms),
        "updated_at": iso(doc.updated_at_ms),
        "memory_unit_count": engine.store.count_document_memories(&doc.bank_id, &doc.id)?,
        "tags": doc.tags,
        "document_metadata": doc.metadata,
        "retain_params": doc.retain_params,
    });
    if with_text {
        out["original_text"] = json!(doc.original_text);
    } else {
        out["text_length"] = json!(doc.original_text.chars().count());
    }
    Ok(out)
}

async fn get_document(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, document_id)): Path<(String, String)>,
) -> ApiResult {
    let doc = engine
        .store
        .get_document(&bank_id, &document_id)?
        .ok_or_else(|| ApiError::not_found("Document not found"))?;
    Ok(Json(document_json(&engine, &doc, true)?))
}

async fn list_documents(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> ApiResult {
    let docs = engine.store.list_documents(&bank_id, 1000)?;
    let mut items = Vec::with_capacity(docs.len());
    for doc in &docs {
        items.push(document_json(&engine, doc, false)?);
    }
    let total = items.len();
    Ok(Json(
        json!({ "items": items, "total": total, "limit": 1000, "offset": 0 }),
    ))
}

async fn delete_document(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, document_id)): Path<(String, String)>,
) -> ApiResult {
    if engine.store.get_document(&bank_id, &document_id)?.is_none() {
        return Err(ApiError::not_found("Document not found"));
    }
    let deleted = engine.forget_document(&bank_id, &document_id)?;
    Ok(Json(json!({
        "success": true,
        "message": format!("Document '{document_id}' and {deleted} associated memory units deleted successfully"),
        "document_id": document_id,
        "memory_units_deleted": deleted,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::ApiAuth;
    use axum::body::Body;
    use axum::http::{header, Method, Request};
    use rustsight_core::SystemClock;
    use rustsight_providers::{MockEmbed, MockLlm, MockRerank, SimpleTokenEstimator};
    use rustsight_storage::SqliteStore;
    use tower::ServiceExt;

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

    async fn call(
        app: &Router,
        method: Method,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(v) => {
                req = req.header(header::CONTENT_TYPE, "application/json");
                Body::from(v.to_string())
            }
            None => Body::empty(),
        };
        let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    async fn retain_one(app: &Router, bank: &str, item: Value) {
        let (status, body) = call(
            app,
            Method::POST,
            &format!("/v1/default/banks/{bank}/memories"),
            Some(json!({ "items": [item] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["success"], true);
        assert_eq!(body["async"], false);
    }

    async fn wait_for_operation(app: &Router, bank: &str, id: &str) -> Value {
        let uri = format!("/v1/default/banks/{bank}/operations/{id}");
        for _ in 0..500 {
            let state = call(app, Method::GET, &uri, None).await.1;
            if state["status"] != "pending" {
                return state;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("operation {id} did not finish");
    }

    fn scripted_engine(responses: Vec<&str>) -> Arc<Engine> {
        let responses = responses
            .into_iter()
            .map(|content| rustsight_core::LlmResponse {
                content: Some(content.to_string()),
                ..Default::default()
            })
            .collect();
        Arc::new(Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            Arc::new(MockLlm::script(responses)),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(SystemClock),
        ))
    }

    #[test]
    fn retain_concurrency_parses_clamps_and_falls_back() {
        assert_eq!(parse_retain_concurrency(None), 2);
        assert_eq!(parse_retain_concurrency(Some(" 8 ")), 8);
        assert_eq!(parse_retain_concurrency(Some("0")), 1);
        assert_eq!(parse_retain_concurrency(Some("9999")), 64);
        assert_eq!(parse_retain_concurrency(Some("many")), 2);
    }

    #[tokio::test]
    async fn deleting_a_bank_removes_everything_and_a_recreated_bank_is_empty() {
        let engine = engine();
        let app = app(&engine);
        retain_one(
            &app,
            "gone",
            json!({"content": "Alice joined Google.", "document_id": "d1"}),
        )
        .await;
        retain_one(&app, "kept", json!({"content": "Bob moved to Oslo."})).await;
        let (status, body) = call(&app, Method::DELETE, "/v1/default/banks/gone", None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body["deleted_count"].as_u64().unwrap() > 0);
        assert!(engine.store.get_bank("gone").unwrap().is_none());
        assert!(!engine.store.list_memories("kept", 10).unwrap().is_empty());

        call(&app, Method::PUT, "/v1/default/banks/gone", Some(json!({}))).await;
        let (_, recalled) = call(
            &app,
            Method::POST,
            "/v1/default/banks/gone/memories/recall",
            Some(json!({"query": "Where does Alice work?"})),
        )
        .await;
        assert_eq!(recalled["results"], json!([]));
        let (status, again) = call(&app, Method::DELETE, "/v1/default/banks/never", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(again["deleted_count"], 0);
    }

    #[tokio::test]
    async fn operations_list_and_retry_a_failed_retain() {
        let engine = scripted_engine(vec!["not json", "not json"]);
        let app = app(&engine);
        let id = "6f1c1fb0-1111-4a55-9d0a-1c7f5a3d2e11";
        let (status, _) = call(
            &app,
            Method::POST,
            "/v1/default/banks/ops/memories",
            Some(json!({
                "items": [{"content": "word", "document_id": "d1"}],
                "async": true,
                "operation_id": id,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            wait_for_operation(&app, "ops", id).await["status"],
            "failed"
        );

        let (_, listed) = call(
            &app,
            Method::GET,
            "/v1/default/banks/ops/operations?status=failed&limit=1",
            None,
        )
        .await;
        assert_eq!(listed["total"], 1);
        let op = &listed["operations"][0];
        assert_eq!(op["id"], id);
        assert_eq!(op["task_type"], "retain");
        assert_eq!(op["items_count"], 1);
        assert_eq!(op["document_id"], "d1");
        assert!(op["error_message"].is_string());

        let retry = format!("/v1/default/banks/ops/operations/{id}/retry");
        let (status, body) = call(&app, Method::POST, &retry, None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["operation_id"], id);
        wait_for_operation(&app, "ops", id).await;
        let (status, _) = call(
            &app,
            Method::POST,
            "/v1/default/banks/ops/operations/nope/retry",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn queued_consolidation_counts_as_an_open_operation_only_when_enabled() {
        let engine = engine();
        let app = app(&engine);
        retain_one(&app, "b", json!({"content": "Alice joined Google."})).await;
        let list = "/v1/default/banks/b/operations?limit=1";
        let (_, listed) = call(&app, Method::GET, list, None).await;
        assert_eq!(listed["total"], 1);
        assert_eq!(listed["operations"][0]["task_type"], "consolidation");

        call(
            &app,
            Method::PUT,
            "/v1/default/banks/b",
            Some(json!({"enable_observations": false})),
        )
        .await;
        let (_, listed) = call(&app, Method::GET, list, None).await;
        assert_eq!(listed["total"], 0);
        assert_eq!(engine.consolidate_once("b", 10).await.unwrap(), 0);
        assert!(engine.store.observations_for_bank("b").unwrap().is_empty());
        assert!(engine.store.count_dirty_entities("b").unwrap() > 0);
    }

    #[tokio::test]
    async fn recall_returns_source_chunks_within_budget_and_tag_scope() {
        let engine = engine();
        let app = app(&engine);
        retain_one(
            &app,
            "c",
            json!({"content": "Alice joined Google today.", "document_id": "d1", "tags": ["a"]}),
        )
        .await;
        let recall = "/v1/default/banks/c/memories/recall";
        let (_, body) = call(
            &app,
            Method::POST,
            recall,
            Some(json!({"query": "Alice", "include": {"chunks": {"max_tokens": 100}}})),
        )
        .await;
        let chunk_id = body["results"][0]["chunk_id"].as_str().unwrap().to_string();
        assert_eq!(chunk_id, "c_d1_0");
        assert_eq!(
            body["chunks"][&chunk_id]["text"],
            "Alice joined Google today."
        );
        assert_eq!(body["chunks"][&chunk_id]["truncated"], false);

        let (_, body) = call(
            &app,
            Method::POST,
            recall,
            Some(json!({"query": "Alice", "include": {"chunks": {"max_tokens": 1}}})),
        )
        .await;
        assert_eq!(body["chunks"][&chunk_id]["truncated"], true);
        let cut = body["chunks"][&chunk_id]["text"].as_str().unwrap();
        assert!(cut.len() < "Alice joined Google today.".len());

        let (_, body) = call(
            &app,
            Method::POST,
            recall,
            Some(json!({
                "query": "Alice",
                "tags": ["a", "other"],
                "tags_match": "any",
                "include": {"chunks": {}},
            })),
        )
        .await;
        assert!(body["chunks"].get(&chunk_id).is_some());
        let (_, without) = call(&app, Method::POST, recall, Some(json!({"query": "Alice"}))).await;
        assert_eq!(without["chunks"], Value::Null);
    }

    #[tokio::test]
    async fn appended_content_gets_the_next_chunk_and_replace_starts_over() {
        let engine = engine();
        let app = app(&engine);
        let chunk_ids = |engine: &Arc<Engine>| {
            let memories: Vec<String> = engine
                .store
                .list_memories("c", 10)
                .unwrap()
                .into_iter()
                .map(|m| m.id)
                .collect();
            let found = engine.store.memory_chunk_ids("c", &memories).unwrap();
            let mut ids: Vec<String> = found.into_values().collect();
            ids.sort();
            ids.dedup();
            ids
        };
        retain_one(
            &app,
            "c",
            json!({"content": "Alice joined Google.", "document_id": "d"}),
        )
        .await;
        retain_one(
            &app,
            "c",
            json!({"content": "Bob moved to Oslo.", "document_id": "d", "update_mode": "append"}),
        )
        .await;
        assert_eq!(chunk_ids(&engine), vec!["c_d_0", "c_d_1"]);
        retain_one(
            &app,
            "c",
            json!({"content": "Carol sails.", "document_id": "d"}),
        )
        .await;
        assert_eq!(chunk_ids(&engine), vec!["c_d_0"]);
    }

    struct HangingLlm;

    #[async_trait::async_trait]
    impl rustsight_core::LlmProvider for HangingLlm {
        async fn complete(
            &self,
            _req: &rustsight_core::LlmRequest,
        ) -> rustsight_core::Result<rustsight_core::LlmResponse> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn deleting_a_bank_cancels_its_running_retain_without_leftovers() {
        let engine = Arc::new(Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            Arc::new(HangingLlm),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(SystemClock),
        ));
        let app = app(&engine);
        let id = "7a2c1fb0-2222-4a55-9d0a-1c7f5a3d2e11";
        call(
            &app,
            Method::POST,
            "/v1/default/banks/busy/memories",
            Some(json!({"items": [{"content": "word"}], "async": true, "operation_id": id})),
        )
        .await;
        for _ in 0..200 {
            if engine.store.get_bank("busy").unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let deleted = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            call(&app, Method::DELETE, "/v1/default/banks/busy", None),
        )
        .await
        .expect("delete must not wait for the provider");
        assert_eq!(deleted.0, StatusCode::OK);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(engine.store.get_operation(id).unwrap().is_none());
        assert!(engine.store.get_bank("busy").unwrap().is_none());
    }

    #[tokio::test]
    async fn config_labels_become_tags_that_fuzzy_filters_find() {
        let extraction = r#"{"facts":[{"what":"Kubernetes runs the cluster","fact_type":"world",
            "entities":["Kubernetes"],"labels":{"name":["kubernetes"]}}]}"#;
        let engine = scripted_engine(vec![extraction]);
        let app = app(&engine);
        let config = "/v1/default/banks/l/config";
        let (status, body) = call(
            &app,
            Method::PATCH,
            config,
            Some(json!({"updates": {
                "HINDSIGHT_API_ENTITY_LABELS": [{"key": "name", "type": "multi-text", "tag": true}],
                "retain_mission": "Keep names.",
            }})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["config"]["retain_mission"], "Keep names.");
        assert!(body["overrides"]["entity_labels"].is_array());
        let (status, _) = call(
            &app,
            Method::PATCH,
            config,
            Some(json!({"updates": {"llm_model": "x"}})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        retain_one(
            &app,
            "l",
            json!({"content": "Kubernetes runs the cluster."}),
        )
        .await;
        let memories = engine.store.list_memories("l", 10).unwrap();
        assert_eq!(memories[0].tags, vec!["name:kubernetes"]);

        let recall = "/v1/default/banks/l/memories/recall";
        let filter = |token: &str| {
            json!({"query": "cluster", "tag_groups": [
                {"tags": [token], "match": "any_strict", "resolve": "fuzzy"}
            ]})
        };
        let (status, hit) = call(&app, Method::POST, recall, Some(filter("name:kubernetis"))).await;
        assert_eq!(status, StatusCode::OK, "{hit}");
        assert_eq!(hit["results"].as_array().unwrap().len(), 1);
        let (_, miss) = call(&app, Method::POST, recall, Some(filter("name:zzzz"))).await;
        assert_eq!(miss["results"], json!([]));
        let (status, _) = call(
            &app,
            Method::POST,
            recall,
            Some(json!({"query": "cluster", "tag_groups": [{"tags": ["x"], "resolve": "nope"}]})),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        let (_, reset) = call(&app, Method::DELETE, config, None).await;
        assert_eq!(reset["overrides"], json!({}));
        assert_eq!(reset["config"]["entity_labels"], Value::Null);
    }

    #[tokio::test]
    async fn openapi_lists_only_recall_fields_that_work() {
        let engine = engine();
        let (status, spec) = call(&app(&engine), Method::GET, "/openapi.json", None).await;
        assert_eq!(status, StatusCode::OK);
        let fields = &spec["components"]["schemas"]["RecallRequest"]["properties"];
        assert!(fields.get("tag_groups").is_some());
        assert!(fields.get("query_tag_gate").is_none());
    }

    #[tokio::test]
    async fn probes_are_public_and_bank_routes_need_the_token() {
        let app = crate::http::router(engine(), Arc::new(ApiAuth::Token("sekrit".into())));
        for uri in ["/health", "/version", "/v1/version"] {
            let (status, _) = call(&app, Method::GET, uri, None).await;
            assert_eq!(status, StatusCode::OK, "{uri}");
        }
        let (status, _) = call(&app, Method::GET, "/v1/default/banks", None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (_, version) = call(&app, Method::GET, "/version", None).await;
        assert_eq!(version["api_version"], HINDSIGHT_API_VERSION);
        assert_eq!(version["features"]["observations"], true);
    }

    #[tokio::test]
    async fn bank_upsert_is_idempotent_and_scopes_the_mission_to_reflect() {
        let engine = engine();
        let app = app(&engine);
        let (status, body) = call(
            &app,
            Method::PUT,
            "/v1/default/banks/u1",
            Some(json!({
                "name": "Joe",
                "reflect_mission": "Remember things for Joe.",
                "disposition": {"skepticism": 4, "literalism": 2, "empathy": 5},
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body,
            json!({
                "bank_id": "u1",
                "name": "Joe",
                "disposition": {"skepticism": 4, "literalism": 2, "empathy": 5},
                "mission": "Remember things for Joe.",
                "background": "Remember things for Joe.",
            })
        );
        // A second PUT with fewer fields keeps what was set (no 409).
        let (status, again) =
            call(&app, Method::PUT, "/v1/default/banks/u1", Some(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(again, body);
        let (_, profile) = call(&app, Method::GET, "/v1/default/banks/u1/profile", None).await;
        assert_eq!(profile, body);
        // Hindsight's reflect_mission never steers extraction.
        let config = engine.store.bank_config("u1").unwrap();
        assert_eq!(config["mission_scope"], "reflect");

        let (status, err) = call(
            &app,
            Method::PUT,
            "/v1/default/banks/u1",
            Some(json!({"disposition": {"skepticism": 9, "literalism": 3, "empathy": 3}})),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(err["detail"].as_str().unwrap().contains("skepticism"));
    }

    #[tokio::test]
    async fn retain_then_recall_round_trips_hindsight_shapes() {
        let engine = engine();
        let app = app(&engine);
        retain_one(
            &app,
            "u1",
            json!({
                "content": "Alice joined Google.",
                "timestamp": "2026-03-02T12:00:00Z",
                "context": "career",
                "metadata": {"type": "note"},
                "document_id": "doc:alice",
                "tags": ["user:a"],
            }),
        )
        .await;
        retain_one(
            &app,
            "u1",
            json!({"content": "Bob joined Stripe.", "tags": ["user:b"]}),
        )
        .await;

        let (status, body) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/memories/recall",
            Some(json!({
                "query": "Who joined which company?",
                "types": ["world", "experience"],
                "budget": "mid",
                "max_tokens": 2048,
                "tags": ["user:a"],
                "tags_match": "any_strict",
                "trace": true,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let results = body["results"].as_array().unwrap();
        assert_eq!(results.len(), 1, "tag scope excludes Bob: {body}");
        let fact = &results[0];
        assert_eq!(fact["text"], "Alice joined Google");
        assert_eq!(fact["type"], "world");
        assert_eq!(fact["context"], "career");
        assert_eq!(fact["document_id"], "doc:alice");
        assert_eq!(fact["metadata"], json!({"type": "note"}));
        assert_eq!(fact["tags"], json!(["user:a"]));
        assert_eq!(fact["mentioned_at"], "2026-03-02T12:00:00+00:00");
        assert!(fact["entities"]
            .as_array()
            .unwrap()
            .contains(&json!("Alice")));
        assert_eq!(body["trace"]["num_results"], 1);

        // Unscoped recall sees both.
        let (_, all) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/memories/recall",
            Some(json!({"query": "Who joined which company?"})),
        )
        .await;
        assert_eq!(all["results"].as_array().unwrap().len(), 2, "{all}");
    }

    #[tokio::test]
    async fn retain_and_recall_validation_errors_carry_detail() {
        let app = app(&engine());
        let cases = [
            (
                "memories",
                json!({"items": []}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "memories",
                json!({"items": [{"content": "  "}]}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "memories",
                json!({"items": [{"content": "x", "timestamp": "yesterday-ish"}]}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "memories",
                json!({"items": [{"content": "x", "update_mode": "append"}]}),
                StatusCode::BAD_REQUEST,
            ),
            (
                "memories/recall",
                json!({"query": "   "}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "memories/recall",
                json!({"query": "x", "types": ["opinion"]}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "memories/recall",
                json!({"query": "x", "tags": ["a"], "tag_groups": [{"tags": ["b"]}]}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "reflect",
                json!({"query": " "}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "reflect",
                json!({"query": "x", "budget": "enormous"}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "reflect",
                json!({"nope": 1}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
        ];
        for (path, body, expected) in cases {
            let (status, err) = call(
                &app,
                Method::POST,
                &format!("/v1/default/banks/u1/{path}"),
                Some(body.clone()),
            )
            .await;
            assert_eq!(status, expected, "{path} {body} -> {err}");
            assert!(err["detail"].is_string(), "{path} {body} -> {err}");
        }
    }

    #[tokio::test]
    async fn documents_upsert_append_fetch_and_delete() {
        let engine = engine();
        let app = app(&engine);
        let doc = "tracker/entry:2026-07-18";
        retain_one(
            &app,
            "u1",
            json!({"content": "Alice joined Google.", "document_id": doc}),
        )
        .await;
        retain_one(
            &app,
            "u1",
            json!({"content": "Alice leads search.", "document_id": doc, "update_mode": "append"}),
        )
        .await;
        let uri = format!("/v1/default/banks/u1/documents/{doc}");
        let (status, body) = call(&app, Method::GET, &uri, None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["id"], doc);
        assert_eq!(
            body["original_text"],
            "Alice joined Google.\nAlice leads search."
        );
        assert_eq!(body["memory_unit_count"], 2);

        // Replace is the default: the document now holds only the new text.
        retain_one(
            &app,
            "u1",
            json!({"content": "Bob joined Stripe.", "document_id": doc}),
        )
        .await;
        let (_, body) = call(&app, Method::GET, &uri, None).await;
        assert_eq!(body["original_text"], "Bob joined Stripe.");
        assert_eq!(body["memory_unit_count"], 1);

        let (status, missing) = call(
            &app,
            Method::GET,
            "/v1/default/banks/u1/documents/nope",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(missing["detail"], "Document not found");

        let (status, deleted) = call(&app, Method::DELETE, &uri, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted["memory_units_deleted"], 1);
        let (status, _) = call(&app, Method::GET, &uri, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn items_without_document_ids_share_one_generated_document() {
        let engine = engine();
        let app = app(&engine);
        let (status, _) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/memories",
            Some(json!({"items": [
                {"content": "Alice joined Google."},
                {"content": "Bob joined Stripe."},
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let docs = engine.store.list_documents("u1", 10).unwrap();
        assert_eq!(docs.len(), 1, "{docs:?}");
        assert_eq!(
            docs[0].original_text,
            "Alice joined Google.\nBob joined Stripe."
        );
        assert_eq!(
            engine
                .store
                .count_document_memories("u1", &docs[0].id)
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn async_retain_returns_an_operation_that_completes() {
        let engine = engine();
        let app = app(&engine);
        let operation_id = "0b0f5d0e-7f43-4a55-9d0a-1c7f5a3d2e11";
        let request = json!({
            "items": [{"content": "Carol joined Anthropic.", "metadata": {"type": "session_summary"}}],
            "async": true,
            "operation_id": operation_id,
        });
        let (status, body) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/memories",
            Some(request.clone()),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["async"], true);
        assert_eq!(body["operation_id"], operation_id);

        let uri = format!("/v1/default/banks/u1/operations/{operation_id}");
        let mut state = Value::Null;
        for _ in 0..200 {
            state = call(&app, Method::GET, &uri, None).await.1;
            if state["status"] == "completed" {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(state["status"], "completed", "{state}");
        assert_eq!(engine.store.list_memories("u1", 10).unwrap().len(), 1);

        // Resubmitting the same operation_id queues no new work.
        let (status, _) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/memories",
            Some(request),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(engine.store.list_memories("u1", 10).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn reflect_returns_text_and_based_on() {
        let engine = engine();
        let app = app(&engine);
        retain_one(
            &app,
            "u1",
            json!({"content": "Alice joined Google. Alice works on search infrastructure."}),
        )
        .await;
        // The demo LLM answers from `search_observations`, so give it some.
        engine.consolidate_once("u1", 50).await.unwrap();
        let (status, body) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/reflect",
            Some(json!({
                "query": "Where does Alice work?",
                "context": "for a bio",
                "budget": "low",
                "include": {"facts": {}},
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!body["text"].as_str().unwrap().is_empty());
        let memories = body["based_on"]["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "{body}");
        assert!(memories[0]["id"].is_string() && memories[0]["text"].is_string());
        // Without include.facts the field is absent (null).
        let (_, plain) = call(
            &app,
            Method::POST,
            "/v1/default/banks/u1/reflect",
            Some(json!({"query": "Where does Alice work?"})),
        )
        .await;
        assert!(plain["based_on"].is_null());
    }

    #[test]
    fn iso_timestamps_parse_and_render() {
        assert_eq!(parse_iso("2024-01-15T10:30:00Z"), Some(1_705_314_600_000));
        assert_eq!(parse_iso("2024-01-15T10:30:00"), Some(1_705_314_600_000));
        assert_eq!(
            parse_iso("2024-01-15T12:30:00+02:00"),
            Some(1_705_314_600_000)
        );
        assert_eq!(parse_iso("2024-01-15"), Some(1_705_276_800_000));
        assert_eq!(parse_iso("not a date"), None);
        assert_eq!(
            iso(1_705_314_600_000).as_deref(),
            Some("2024-01-15T10:30:00+00:00")
        );
    }
}
