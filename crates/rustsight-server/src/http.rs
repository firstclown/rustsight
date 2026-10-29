use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rustsight_core::types::*;
use rustsight_core::{Engine, Error};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::auth::{require_token, ApiAuth};
use crate::consolidate_interval_from_env;

/// All routes except `/healthz` sit behind the token middleware. Returns a
/// stateless `Router<()>` ready to serve or test.
pub fn router(engine: Arc<Engine>, auth: Arc<ApiAuth>) -> Router {
    let protected = Router::new()
        .route("/banks", get(list_banks).post(upsert_bank))
        .route("/banks/{bank_id}", get(get_bank))
        .route("/banks/{bank_id}/retain", post(retain))
        .route("/banks/{bank_id}/recall", post(recall))
        .route("/banks/{bank_id}/reflect", post(reflect))
        .route("/banks/{bank_id}/consolidate", post(consolidate))
        .route("/banks/{bank_id}/memories", get(list_memories))
        .route("/banks/{bank_id}/entities", get(list_entities))
        .route("/banks/{bank_id}/observations", get(list_observations))
        .route("/banks/{bank_id}/opinions", get(list_opinions))
        .route(
            "/banks/{bank_id}/mental-models",
            get(list_mental_models).post(refresh_mental_model),
        )
        .with_state(engine)
        .layer(axum::middleware::from_fn_with_state(auth, require_token));
    Router::new()
        .route("/healthz", get(healthz))
        .merge(protected)
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
    let listener = tokio::net::TcpListener::bind(socket)
        .await
        .map_err(|e| Error::BadInput(format!("bind {addr}: {e}")))?;
    axum::serve(listener, router(engine, Arc::new(auth)))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| Error::Store(format!("server: {e}")))?;
    worker.abort();
    let _ = worker.await;
    Ok(())
}

fn http_err(e: Error) -> (StatusCode, String) {
    let status = match &e {
        Error::NotFound(_) => StatusCode::NOT_FOUND,
        Error::BadInput(_) | Error::Parse(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let body = match &e {
        Error::NotFound(_) | Error::BadInput(_) | Error::Parse(_) => e.to_string(),
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
    let existing = engine.store.get_bank(&body.bank_id).map_err(http_err)?;
    let mut bank = existing.unwrap_or(Bank {
        id: body.bank_id.clone(),
        mission: None,
        directives: None,
        background: None,
        disposition: Disposition::default(),
    });
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
        })
        .await
        .map(Json)
        .map_err(http_err)
}

#[derive(Deserialize)]
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
            top_k_per_arm: body.top_k_per_arm.unwrap_or(10),
            kinds: None,
        })
        .await
        .map(Json)
        .map_err(http_err)
}

#[derive(Deserialize)]
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
