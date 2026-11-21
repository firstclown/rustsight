use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post, MethodRouter};
use axum::{Json, Router};
use rustsight_core::types::*;
use rustsight_core::{Engine, Error};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use crate::auth::{require_token, ApiAuth};
use crate::consolidate_interval_from_env;

/// The native API's public routes (no token). Together with
/// [`native_routes`] this is the single list the router is built from, and
/// the tests hold `openapi.json` to exactly these paths.
fn native_public_routes() -> Vec<(&'static str, MethodRouter)> {
    vec![("/healthz", get(healthz))]
}

/// The native API's token-protected routes.
fn native_routes() -> Vec<(&'static str, MethodRouter<Arc<Engine>>)> {
    vec![
        ("/banks", get(list_banks).post(upsert_bank)),
        ("/banks/{bank_id}", get(get_bank)),
        ("/banks/{bank_id}/retain", post(retain)),
        ("/banks/{bank_id}/recall", post(recall)),
        ("/banks/{bank_id}/reflect", post(reflect)),
        ("/banks/{bank_id}/consolidate", post(consolidate)),
        ("/banks/{bank_id}/memories", get(list_memories)),
        ("/banks/{bank_id}/entities", get(list_entities)),
        ("/banks/{bank_id}/observations", get(list_observations)),
        ("/banks/{bank_id}/opinions", get(list_opinions)),
        (
            "/banks/{bank_id}/mental-models",
            get(list_mental_models).post(refresh_mental_model),
        ),
    ]
}

/// All routes except `/healthz` sit behind the token middleware. Returns a
/// stateless `Router<()>` ready to serve or test.
pub fn router(engine: Arc<Engine>, auth: Arc<ApiAuth>) -> Router {
    let protected = native_routes()
        .into_iter()
        .fold(Router::new(), |r, (path, method)| r.route(path, method))
        // Hindsight wire-compatible surface (`/v1/default/banks/...`).
        .merge(crate::compat::routes())
        .with_state(engine)
        .layer(axum::middleware::from_fn_with_state(auth, require_token));
    native_public_routes()
        .into_iter()
        .fold(Router::new(), |r, (path, method)| r.route(path, method))
        .merge(crate::compat::public_routes())
        .merge(protected)
}

/// The native API's OpenAPI description (`crates/rustsight-server/openapi.json`,
/// hand-written; the tests below keep it in step with the router and the
/// core types' serialized shapes).
const NATIVE_OPENAPI: &str = include_str!("../openapi.json");

fn native_openapi() -> &'static Value {
    static SPEC: OnceLock<Value> = OnceLock::new();
    SPEC.get_or_init(|| serde_json::from_str(NATIVE_OPENAPI).expect("openapi.json is valid JSON"))
}

/// Fold the native API description into `doc` (the Hindsight-compatible
/// document served at `/openapi.json`): native paths, components, tags and
/// the bearer security scheme are added; the compat document's `info` and
/// its own schemas stay as they are, so Hindsight clients probing it see
/// no change.
pub(crate) fn merge_native_openapi(doc: &mut Value) {
    let native = native_openapi();
    let Some(out) = doc.as_object_mut() else {
        return;
    };
    for key in ["paths", "components"] {
        let target = out
            .entry(key)
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .expect("openapi section is an object");
        for (name, value) in native[key].as_object().expect("native section") {
            match (target.get_mut(name), value) {
                // components.schemas / .responses / ...: merge entry by entry.
                (Some(Value::Object(existing)), Value::Object(add)) if key == "components" => {
                    for (k, v) in add {
                        existing.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
                (Some(_), _) => {}
                (None, _) => {
                    target.insert(name.clone(), value.clone());
                }
            }
        }
    }
    for key in ["security", "tags"] {
        out.entry(key).or_insert_with(|| native[key].clone());
    }
    if let Some(info) = out.get_mut("info").and_then(Value::as_object_mut) {
        info.entry("description")
            .or_insert_with(|| native["info"]["description"].clone());
    }
}

/// Resolve `addr` (literal `ip:port` or hostname) to the socket we bind.
async fn bind_addr(addr: &str) -> Result<SocketAddr, Error> {
    if let Ok(sa) = addr.parse::<SocketAddr>() {
        return Ok(sa);
    }
    tokio::net::lookup_host(addr)
        .await
        .map_err(|e| Error::BadInput(format!("resolve addr {addr:?}: {e}")))?
        .next()
        .ok_or_else(|| Error::BadInput(format!("addr {addr:?} resolved to nothing")))
}

pub async fn serve(
    engine: Arc<Engine>,
    addr: &str,
    auth: ApiAuth,
    allow_unauth: bool,
) -> Result<(), Error> {
    let socket = bind_addr(addr).await?;
    if !socket.ip().is_loopback() && auth == ApiAuth::None && !allow_unauth {
        return Err(Error::BadInput(format!(
            "refusing to serve on non-loopback {addr} without authentication: set \
             RUSTSIGHT_API_TOKEN (or RUSTSIGHT_API_TOKEN_FILE), or pass --allow-unauth to \
             accept unauthenticated access deliberately"
        )));
    }
    let worker = engine.spawn_worker(consolidate_interval_from_env(), 50);
    let webhook_settings = Arc::new(crate::webhooks::WebhookSettings::from_env());
    crate::webhooks::set_settings(Arc::clone(&webhook_settings));
    let webhooks = crate::webhooks::start(&engine, webhook_settings);
    let resumed = crate::compat::resume_pending_operations(&engine);
    if resumed > 0 {
        eprintln!("rustsight: resuming {resumed} unfinished async retain operation(s)");
    }
    let listener = tokio::net::TcpListener::bind(socket)
        .await
        .map_err(|e| Error::BadInput(format!("bind {addr}: {e}")))?;
    axum::serve(listener, router(engine, Arc::new(auth)))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Error::Store(format!("server: {e}")))?;
    worker.abort();
    let _ = worker.await;
    webhooks.abort();
    let _ = webhooks.await;
    Ok(())
}

/// Resolves on SIGINT (ctrl-c) or SIGTERM. Process supervisors — including
/// the ones that manage `hindsight-api` — stop a server with SIGTERM and
/// expect in-flight requests to drain.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

fn http_err(e: Error) -> (StatusCode, String) {
    let status = match &e {
        Error::NotFound(_) => StatusCode::NOT_FOUND,
        Error::BadInput(_) | Error::Parse(_) => StatusCode::BAD_REQUEST,
        Error::Blocked { .. } => StatusCode::UNPROCESSABLE_ENTITY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let body = match &e {
        Error::NotFound(_) | Error::BadInput(_) | Error::Parse(_) | Error::Blocked { .. } => {
            e.to_string()
        }
        _ => {
            eprintln!("rustsight http: internal error: {e}");
            "internal error".to_string()
        }
    };
    (status, body)
}

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn list_banks(
    State(engine): State<Arc<Engine>>,
) -> Result<Json<Vec<Bank>>, (StatusCode, String)> {
    engine.store.list_banks().map(Json).map_err(http_err)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BankBody {
    bank_id: String,
    mission: Option<String>,
    directives: Option<String>,
    background: Option<String>,
    skepticism: Option<u8>,
    literalism: Option<u8>,
    empathy: Option<u8>,
    bias_strength: Option<f32>,
}

async fn upsert_bank(
    State(engine): State<Arc<Engine>>,
    Json(body): Json<BankBody>,
) -> Result<Json<Bank>, (StatusCode, String)> {
    let mut bank = engine.provision_bank(&body.bank_id).map_err(http_err)?;
    if body.mission.is_some() {
        bank.mission = body.mission;
    }
    if body.directives.is_some() {
        bank.directives = body.directives;
    }
    if body.background.is_some() {
        bank.background = body.background;
    }
    if let Some(v) = body.skepticism {
        bank.disposition.skepticism = v.clamp(1, 5);
    }
    if let Some(v) = body.literalism {
        bank.disposition.literalism = v.clamp(1, 5);
    }
    if let Some(v) = body.empathy {
        bank.disposition.empathy = v.clamp(1, 5);
    }
    if let Some(v) = body.bias_strength {
        bank.disposition.bias_strength = v.clamp(0.0, 1.0);
    }
    engine.store.ensure_bank(&bank).map_err(http_err)?;
    engine.store.update_bank(&bank).map_err(http_err)?;
    Ok(Json(bank))
}

async fn get_bank(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<Bank>, (StatusCode, String)> {
    engine
        .store
        .get_bank(&bank_id)
        .map_err(http_err)?
        .map(Json)
        .ok_or_else(|| http_err(Error::NotFound(format!("bank {bank_id}"))))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetainBody {
    content: String,
    context: Option<String>,
    timestamp_ms: Option<i64>,
}

async fn retain(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    Json(body): Json<RetainBody>,
) -> Result<Json<RetainResponse>, (StatusCode, String)> {
    engine
        .retain(RetainRequest {
            bank_id,
            content: body.content,
            context: body.context,
            timestamp_ms: body.timestamp_ms,
            ..Default::default()
        })
        .await
        .map(Json)
        .map_err(http_err)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecallBody {
    query: String,
    token_budget: Option<usize>,
    top_k_per_arm: Option<usize>,
}

async fn recall(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    Json(body): Json<RecallBody>,
) -> Result<Json<RecallResponse>, (StatusCode, String)> {
    engine
        .recall(RecallRequest {
            bank_id,
            query: body.query,
            token_budget: body.token_budget.unwrap_or(1000),
            top_k_per_arm: body
                .top_k_per_arm
                .unwrap_or_else(|| engine.config.recall_budget.resolve(None, 1000)),
            kinds: None,
            ..Default::default()
        })
        .await
        .map(Json)
        .map_err(http_err)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReflectBody {
    query: String,
}

async fn reflect(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    Json(body): Json<ReflectBody>,
) -> Result<Json<ReflectResponse>, (StatusCode, String)> {
    engine
        .reflect(&bank_id, &body.query)
        .await
        .map(Json)
        .map_err(http_err)
}

#[derive(Serialize)]
struct ConsolidateResponse {
    observations_updated: usize,
}

async fn consolidate(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<ConsolidateResponse>, (StatusCode, String)> {
    engine
        .consolidate_once(&bank_id, 50)
        .await
        .map(|n| {
            Json(ConsolidateResponse {
                observations_updated: n,
            })
        })
        .map_err(http_err)
}

async fn list_memories(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<Vec<Memory>>, (StatusCode, String)> {
    engine
        .store
        .list_memories(&bank_id, 500)
        .map(Json)
        .map_err(http_err)
}

async fn list_entities(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<Vec<Entity>>, (StatusCode, String)> {
    engine
        .store
        .entities_for_bank(&bank_id)
        .map(Json)
        .map_err(http_err)
}

async fn list_observations(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<Vec<Observation>>, (StatusCode, String)> {
    engine
        .store
        .observations_for_bank(&bank_id)
        .map(Json)
        .map_err(http_err)
}

async fn list_opinions(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<Vec<Opinion>>, (StatusCode, String)> {
    engine
        .store
        .list_opinions(&bank_id)
        .map(Json)
        .map_err(http_err)
}

async fn list_mental_models(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
) -> Result<Json<Vec<MentalModel>>, (StatusCode, String)> {
    engine
        .store
        .list_mental_models(&bank_id)
        .map(Json)
        .map_err(http_err)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MentalModelBody {
    question: String,
    path: Option<String>,
}

async fn refresh_mental_model(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    Json(body): Json<MentalModelBody>,
) -> Result<Json<MentalModel>, (StatusCode, String)> {
    engine
        .refresh_mental_model(&bank_id, &body.question, body.path.as_deref())
        .await
        .map(Json)
        .map_err(http_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use rustsight_core::SystemClock;
    use rustsight_providers::{MockEmbed, MockLlm, MockRerank, SimpleTokenEstimator};
    use rustsight_storage::SqliteStore;
    use tower::ServiceExt;

    fn test_router(auth: ApiAuth) -> Router {
        let engine = Arc::new(Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            Arc::new(MockLlm::demo()),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(SystemClock),
        ));
        router(engine, Arc::new(auth))
    }

    async fn get(router: Router, uri: &str, bearer: Option<&str>) -> axum::response::Response {
        let mut req = Request::builder().uri(uri);
        if let Some(token) = bearer {
            req = req.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        router
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn token_auth_rejects_missing_and_wrong_tokens() {
        let r = test_router(ApiAuth::Token("sekrit".into()));
        let resp = get(r.clone(), "/banks", None).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let resp = get(r, "/banks", Some("wrong")).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn token_auth_accepts_correct_token() {
        let r = test_router(ApiAuth::Token("sekrit".into()));
        let resp = get(r, "/banks", Some("sekrit")).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn healthz_stays_public() {
        let r = test_router(ApiAuth::Token("sekrit".into()));
        let resp = get(r, "/healthz", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn no_auth_passes_through() {
        let r = test_router(ApiAuth::None);
        let resp = get(r, "/banks", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Every native path the router serves, read from the same lists the
    /// router is built from — nothing hand-maintained in the tests.
    fn routed_paths() -> Vec<String> {
        let mut paths: Vec<String> = native_public_routes()
            .into_iter()
            .map(|(p, _)| p.to_string())
            .chain(native_routes().into_iter().map(|(p, _)| p.to_string()))
            .collect();
        paths.sort();
        paths
    }

    const HTTP_METHODS: [&str; 5] = ["get", "post", "put", "patch", "delete"];

    fn spec_methods(item: &Value) -> Vec<String> {
        let mut methods: Vec<String> = item
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| HTTP_METHODS.contains(&k.as_str()))
            .cloned()
            .collect();
        methods.sort();
        methods
    }

    #[test]
    fn native_spec_documents_exactly_the_routed_paths() {
        let mut documented: Vec<String> = native_openapi()["paths"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        documented.sort();
        assert_eq!(documented, routed_paths());
    }

    /// For every native path, each documented method reaches a handler and
    /// every other method gets 405 — so a method added to (or dropped from)
    /// a route without touching `openapi.json` fails here.
    #[tokio::test]
    async fn native_spec_methods_match_the_router() {
        let router = test_router(ApiAuth::None);
        let paths = native_openapi()["paths"].as_object().unwrap();
        for path in routed_paths() {
            let documented = spec_methods(&paths[&path]);
            let uri = path.replace("{bank_id}", "spec-probe");
            for method in HTTP_METHODS {
                let mut req = Request::builder()
                    .method(method.to_uppercase().as_str())
                    .uri(&uri);
                // `{}` fails body validation (422) for every route with a
                // required field, which proves the route matched without
                // running the pipeline behind it.
                let body = if method == "get" {
                    Body::empty()
                } else {
                    req = req.header(header::CONTENT_TYPE, "application/json");
                    Body::from("{}")
                };
                let resp = router
                    .clone()
                    .oneshot(req.body(body).unwrap())
                    .await
                    .unwrap();
                let status = resp.status();
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .unwrap();
                if documented.iter().any(|m| m == method) {
                    assert_ne!(status, StatusCode::METHOD_NOT_ALLOWED, "{method} {path}");
                    // The router's own fallback 404 has an empty body; a
                    // handler's 404 says what was not found.
                    assert!(
                        !(status == StatusCode::NOT_FOUND && bytes.is_empty()),
                        "{method} {path} is not routed"
                    );
                } else {
                    assert_eq!(
                        status,
                        StatusCode::METHOD_NOT_ALLOWED,
                        "{method} {path} is routed but not in openapi.json"
                    );
                }
            }
        }
    }

    /// A value of the schema's type, enough to deserialize.
    fn sample_value(schema: &Value) -> Value {
        match schema["type"].as_str() {
            Some("string") => json!("x"),
            Some("integer") => json!(3),
            Some("number") => json!(0.5),
            Some("boolean") => json!(true),
            other => panic!("no sample for schema type {other:?}"),
        }
    }

    /// The request schema lists exactly the body struct's fields: a body
    /// with every documented property parses, so does one with only the
    /// required ones, dropping any required one fails, and an extra
    /// property is refused (`deny_unknown_fields` + `additionalProperties:
    /// false`) — so the server never silently ignores a field.
    fn assert_request_schema_matches<T: serde::de::DeserializeOwned>(name: &str) {
        let schema = &native_openapi()["components"]["schemas"][name];
        assert_eq!(schema["additionalProperties"], json!(false), "{name}");
        let props = schema["properties"].as_object().unwrap();
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let full: Map<String, Value> = props
            .iter()
            .map(|(k, v)| (k.clone(), sample_value(v)))
            .collect();
        let parse = |m: &Map<String, Value>| serde_json::from_value::<T>(Value::Object(m.clone()));
        assert!(
            parse(&full).is_ok(),
            "{name}: full body refused: {:?}",
            parse(&full).err()
        );
        let minimal: Map<String, Value> = full
            .iter()
            .filter(|(k, _)| required.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert!(
            parse(&minimal).is_ok(),
            "{name}: optional field is actually required"
        );
        for field in &required {
            let mut missing = full.clone();
            missing.remove(*field);
            assert!(
                parse(&missing).is_err(),
                "{name}: {field} is documented required but optional"
            );
        }
        let mut extra = full.clone();
        extra.insert("not_a_documented_field".into(), json!(1));
        let err = parse(&extra)
            .err()
            .unwrap_or_else(|| panic!("{name}: unknown field accepted"))
            .to_string();
        // serde's unknown-field error names every field the struct accepts
        // ("expected `a`", "`a` or `b`", "one of `a`, `b`, `c`"): that
        // list must be exactly the documented properties, which catches an
        // optional struct field the schema forgot.
        let expected_part = err
            .split_once("expected")
            .unwrap_or_else(|| panic!("{name}: unexpected error {err}"))
            .1;
        let mut accepted: Vec<&str> = expected_part.split('`').skip(1).step_by(2).collect();
        accepted.sort();
        let mut documented: Vec<&str> = props.keys().map(String::as_str).collect();
        documented.sort();
        assert_eq!(accepted, documented, "{name}: struct fields vs schema");
    }

    #[test]
    fn native_request_schemas_match_body_structs() {
        assert_request_schema_matches::<BankBody>("NativeBankRequest");
        assert_request_schema_matches::<RetainBody>("NativeRetainRequest");
        assert_request_schema_matches::<RecallBody>("NativeRecallRequest");
        assert_request_schema_matches::<ReflectBody>("NativeReflectRequest");
        assert_request_schema_matches::<MentalModelBody>("NativeMentalModelRequest");
        // Every Native*Request schema is covered above.
        let covered = 5;
        let request_schemas = native_openapi()["components"]["schemas"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| k.starts_with("Native") && k.ends_with("Request"))
            .count();
        assert_eq!(request_schemas, covered);
    }

    #[tokio::test]
    async fn native_bodies_refuse_unknown_fields() {
        let router = test_router(ApiAuth::None);
        let req = Request::builder()
            .method("POST")
            .uri("/banks/demo/recall")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"query": "q", "kinds": ["world"]}"#))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(body.contains("unknown field `kinds`"), "{body}");
    }

    fn collect_refs(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(map) => {
                for (k, v) in map {
                    match (k.as_str(), v) {
                        ("$ref", Value::String(r)) => out.push(r.clone()),
                        _ => collect_refs(v, out),
                    }
                }
            }
            Value::Array(items) => items.iter().for_each(|v| collect_refs(v, out)),
            _ => {}
        }
    }

    #[test]
    fn native_spec_refs_resolve() {
        let spec = native_openapi();
        let mut refs = Vec::new();
        collect_refs(spec, &mut refs);
        assert!(!refs.is_empty());
        for r in refs {
            let pointer = r.strip_prefix('#').expect("local ref");
            assert!(spec.pointer(pointer).is_some(), "dangling {r}");
        }
    }

    /// The schema's properties must be exactly the keys the type
    /// serializes, and every serialized key is listed as required (the
    /// server always emits it, `null` included).
    fn assert_schema_matches<T: Serialize>(name: &str, value: &T) {
        let schema = &native_openapi()["components"]["schemas"][name];
        assert!(schema.is_object(), "schema {name} missing");
        let json = serde_json::to_value(value).unwrap();
        let mut keys: Vec<&String> = json.as_object().unwrap().keys().collect();
        keys.sort();
        let mut props: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
        props.sort();
        assert_eq!(props, keys, "{name} properties");
        let mut required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        required.sort();
        let keys: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        assert_eq!(required, keys, "{name} required");
    }

    fn assert_enum_matches<T: Serialize>(name: &str, values: &[T]) {
        let listed = &native_openapi()["components"]["schemas"][name]["enum"];
        let serialized: Vec<Value> = values
            .iter()
            .map(|v| serde_json::to_value(v).unwrap())
            .collect();
        assert_eq!(listed, &Value::Array(serialized), "{name} enum");
    }

    #[test]
    fn native_spec_schemas_match_serialized_types() {
        let evidence = Evidence {
            memory_id: "m1".into(),
            quote: "q".into(),
        };
        let point = ConfidencePoint {
            confidence: 0.5,
            at_ms: 1,
            reason: "r".into(),
        };
        let memory_ref = MemoryRef {
            id: "m1".into(),
            kind: MemoryKind::World,
            text: "t".into(),
            score: 1.0,
            occurred_start_ms: None,
            occurred_end_ms: None,
            mentioned_at_ms: Some(1),
            entity_ids: vec![],
            proof_count: None,
            context: None,
            document_id: None,
            tags: vec![],
            metadata: None,
        };
        assert_schema_matches("Disposition", &Disposition::default());
        assert_schema_matches(
            "Bank",
            &Bank {
                id: "b".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            },
        );
        assert_schema_matches(
            "Memory",
            &Memory {
                id: "m1".into(),
                bank_id: "b".into(),
                kind: MemoryKind::World,
                text: "t".into(),
                confidence: None,
                occurred_start_ms: None,
                occurred_end_ms: None,
                mentioned_at_ms: 1,
                source: None,
                context: None,
                document_id: None,
                tags: vec![],
                metadata: None,
            },
        );
        assert_schema_matches(
            "Entity",
            &Entity {
                id: "e".into(),
                bank_id: "b".into(),
                name: "n".into(),
                etype: EntityType::Person,
                last_mentioned_ms: 1,
            },
        );
        assert_schema_matches("Evidence", &evidence);
        assert_schema_matches("ConfidencePoint", &point);
        assert_schema_matches(
            "Observation",
            &Observation {
                id: "o".into(),
                bank_id: "b".into(),
                entity_id: "e".into(),
                text: "t".into(),
                evidence: vec![evidence.clone()],
                proof_count: 1,
                history: vec![],
                updated_at_ms: 1,
                created_at_ms: 1,
                embedding: None,
            },
        );
        assert_schema_matches(
            "Opinion",
            &Opinion {
                id: "p".into(),
                bank_id: "b".into(),
                text: "t".into(),
                confidence: 0.5,
                evidence: vec![evidence],
                history: vec![point],
                entity_ids: vec![],
                updated_at_ms: 1,
                created_at_ms: 1,
                embedding: None,
            },
        );
        assert_schema_matches(
            "MentalModel",
            &MentalModel {
                id: "mm".into(),
                bank_id: "b".into(),
                question: "q".into(),
                path: None,
                content: "c".into(),
                last_refreshed_ms: None,
                created_at_ms: 1,
                name: None,
                tags: vec![],
                max_tokens: None,
                trigger: None,
            },
        );
        assert_schema_matches("MemoryRef", &memory_ref);
        assert_schema_matches(
            "RetainResponse",
            &RetainResponse {
                memories_stored: 0,
                entities: vec![],
                edges_created: 0,
                reasks: 0,
                document_id: None,
                defense: None,
            },
        );
        assert_schema_matches(
            "RecallResponse",
            &RecallResponse {
                memories: vec![memory_ref],
                window: None,
                strategies_used: vec![],
            },
        );
        assert_schema_matches(
            "ReflectResponse",
            &ReflectResponse {
                answer: "a".into(),
                citations: vec![],
                rounds: 1,
                retrieved: 0,
                retrieved_ids: vec![],
                answer_path: AnswerPath::Loop,
                synthesized: false,
                loop_ms: 0,
                synthesis_ms: None,
                structured_output: None,
                structured_output_error: None,
            },
        );
        assert_schema_matches(
            "ConsolidateResponse",
            &ConsolidateResponse {
                observations_updated: 0,
            },
        );
        assert_enum_matches(
            "MemoryKind",
            &[
                MemoryKind::World,
                MemoryKind::Experience,
                MemoryKind::Observation,
                MemoryKind::Opinion,
            ],
        );
        assert_enum_matches(
            "EntityType",
            &[
                EntityType::Person,
                EntityType::Organization,
                EntityType::Location,
                EntityType::Product,
                EntityType::Concept,
                EntityType::Other,
            ],
        );
        assert_enum_matches(
            "AnswerPath",
            &[
                AnswerPath::Loop,
                AnswerPath::ForcedDone,
                AnswerPath::Synthesis,
            ],
        );
    }

    #[tokio::test]
    async fn served_openapi_carries_native_and_compat_paths_without_a_token() {
        let r = test_router(ApiAuth::Token("sekrit".into()));
        let resp = get(r, "/openapi.json", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let doc: Value = serde_json::from_slice(&bytes).unwrap();
        let paths = doc["paths"].as_object().unwrap();
        for path in routed_paths() {
            assert!(paths.contains_key(&path), "{path} missing");
        }
        assert!(paths.contains_key("/v1/default/banks/{bank_id}/memories/recall"));
        let schemas = &doc["components"]["schemas"];
        // The compat probe schema survives the merge untouched.
        assert!(schemas["RecallRequest"]["properties"]["tag_groups"].is_object());
        assert!(schemas["NativeRecallRequest"]["properties"]["token_budget"].is_object());
        assert_eq!(
            doc["components"]["securitySchemes"]["bearerAuth"]["scheme"],
            "bearer"
        );
        assert_eq!(doc["info"]["title"], "rustsight");
    }

    #[test]
    fn http_err_hides_internal_detail() {
        let (status, body) = http_err(Error::Store("no such table: memories".into()));
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body, "internal error");
        let (status, body) = http_err(Error::Llm("upstream said secret-token".into()));
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body, "internal error");
        let (status, _) = http_err(Error::LlmStatus {
            status: 429,
            message: "rate limited".into(),
        });
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn http_err_keeps_caller_facing_messages() {
        let (status, body) = http_err(Error::NotFound("bank demo".into()));
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body, "not found: bank demo");
        let (status, body) = http_err(Error::BadInput("empty query".into()));
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "invalid input: empty query");
        let (status, body) = http_err(Error::Parse("bad json".into()));
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "parse error: bad json");
    }
}
