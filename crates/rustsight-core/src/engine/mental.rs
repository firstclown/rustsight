use crate::engine::{with_retry, Engine};
use crate::prompts::reflect::build_mental_model_prompt;
use crate::traits::{ChatMessage, LlmRequest};
use crate::types::*;
use crate::Result;

pub(super) async fn refresh(
    e: &Engine,
    bank_id: &str,
    question: &str,
    path: Option<&str>,
) -> Result<MentalModel> {
    if question.trim().is_empty() {
        return Err(crate::Error::BadInput("question is empty".into()));
    }
    let r = e
        .recall(RecallRequest {
            bank_id: bank_id.to_string(),
            query: question.to_string(),
            token_budget: 1500,
            // Mental-model refresh runs at the default (MID) recall budget.
            top_k_per_arm: e
                .recall_budget_for(&e.resolved_config(bank_id)?)
                .resolve(None, 1500),
            kinds: None,
            ..Default::default()
        })
        .await?;
    let block = if r.memories.is_empty() {
        "(no memories found)".to_string()
    } else {
        r.memories
            .iter()
            .map(|m| format!("[{}] {}: {}", m.kind.as_str(), m.id, m.text))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let (system, user) = build_mental_model_prompt(question, &block);
    let llm = e.llm_for_reflect();
    let request = LlmRequest {
        system: Some(system),
        messages: vec![ChatMessage::user(user)],
        temperature: Some(0.2),
        ..Default::default()
    };
    let resp = with_retry(e, || llm.complete(&request)).await?;
    let content = resp.content.unwrap_or_default();
    let now = e.clock.now_ms();
    let existing = e
        .store
        .list_mental_models(bank_id)?
        .into_iter()
        .find(|m| m.question == question);
    let model = match existing {
        Some(mut m) => {
            m.content = content;
            m.path = path.map(|p| p.to_string()).or(m.path);
            m.last_refreshed_ms = Some(now);
            e.store.upsert_mental_model(&m)?;
            m
        }
        None => {
            let m = MentalModel {
                id: uuid::Uuid::new_v4().to_string(),
                bank_id: bank_id.to_string(),
                question: question.to_string(),
                path: path.map(|p| p.to_string()),
                content,
                last_refreshed_ms: Some(now),
                created_at_ms: now,
                name: None,
                tags: vec![],
                max_tokens: None,
                trigger: None,
            };
            e.store.upsert_mental_model(&m)?;
            m
        }
    };
    Ok(model)
}
