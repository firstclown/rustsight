use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rustsight_core::types::*;
use rustsight_core::{Engine, Error};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use crate::consolidate_interval_from_env;

pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
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
}

pub async fn serve(engine: Arc<Engine>, addr: &str) -> Result<(), Error> {
    let worker = engine.spawn_worker(consolidate_interval_from_env(), 50);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| Error::BadInput(format!("bind {addr}: {e}")))?;
    axum::serve(listener, router(engine))
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
    (status, e.to_string())
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
