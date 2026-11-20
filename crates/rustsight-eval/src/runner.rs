//! The `run` command: ingest → retain per conversation → consolidate →
//! answer via `reflect` → judge → report. Resumable at every unit boundary
//! (conversation / question), parallel across conversations, never holding
//! the store lock across provider calls (the engine already guarantees
//! that; the runner just honors the same shape).
//!
//! Consolidation is tolerant by design: a pass that errors, stalls, or hits
//! its deadline only leaves that bank partially consolidated — the bank
//! stays answerable and the run moves on. The original all-or-nothing loop
//! turned one bad pass into "conversation was not retained" and silently
//! (errors only printed after every bank had been spawned) doomed a
//! 21-hour LongMemEval attempt; see NEXT_STEPS.md RS-6.

use crate::counting::{Counters, CountingEmbed, CountingLlm};
use crate::dataset::{self, Dataset, Question};
use crate::judge::{Judge, JudgeEnv};
use crate::report::{self, QuestionTrace, ReportSummary};
use crate::state::{ConversationState, QuestionState, RunState};
use chrono::NaiveDate;
use rustsight_core::traits::{EmbedProvider, LlmProvider, TokenEstimator};
use rustsight_core::types::{ReflectResponse, RetainRequest};
use rustsight_core::{Engine, EngineConfig, FixedClock};
use rustsight_providers::{
    build_mock_providers, build_providers, retain_max_completion_tokens, ProviderEnv,
    SimpleTokenEstimator,
};
use rustsight_storage::SqliteStore;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::Semaphore;

/// Consecutive consolidation passes with no dirty-queue progress after
/// which the harness keeps the bank as-is (partially consolidated) instead
/// of spinning on it forever.
const MAX_STALLED_PASSES: usize = 3;

#[derive(Clone)]
pub struct RunConfig {
    pub out: PathBuf,
    pub dataset_path: PathBuf,
    pub conversations_parallel: usize,
    pub limit_questions: Option<usize>,
    pub session_spacing_days: i64,
    pub mock: bool,
    pub consolidate_max: usize,
    pub consolidate_pass_limit: usize,
    /// Dirty entities synthesized concurrently inside one consolidation
    /// pass. Mirrors the engine's `EngineConfig::consolidate_concurrency`.
    pub consolidate_concurrency: usize,
    /// Seconds between progress heartbeats on stderr; 0 disables them.
    pub heartbeat_secs: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            out: PathBuf::from("results"),
            dataset_path: PathBuf::from(dataset::DEFAULT_DATASET_FILE),
            conversations_parallel: 2,
            limit_questions: None,
            session_spacing_days: 1,
            mock: false,
            consolidate_max: 50,
            consolidate_pass_limit: 200,
            consolidate_concurrency: 4,
            heartbeat_secs: 60,
        }
    }
}

pub struct RunOutput {
    pub summary: ReportSummary,
    pub report_md: PathBuf,
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn trace_path(out: &Path, question_id: &str) -> PathBuf {
    out.join("traces")
        .join(format!("{}.json", sanitize_id(question_id)))
}

fn write_trace(out: &Path, trace: &QuestionTrace) -> Result<(), String> {
    let path = trace_path(out, &trace.question_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir traces failed: {e}"))?;
    }
    std::fs::write(
        &path,
        serde_json::to_string_pretty(trace).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("trace write failed: {e}"))
}

fn read_trace(out: &Path, question_id: &str) -> Option<QuestionTrace> {
    let path = trace_path(out, question_id);
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn missing_traces(questions: &[&Question], out: &Path) -> usize {
    questions
        .iter()
        .filter(|q| read_trace(out, &q.question_id).is_none())
        .count()
}

fn fallback_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is a valid date")
}

#[derive(Clone, Copy)]
struct RetainParams {
    spacing_days: i64,
    consolidate_max: usize,
    pass_limit: usize,
}

/// Shared, heartbeat-visible progress counters.
#[derive(Default)]
struct Progress {
    phase: &'static str,
    banks_done: usize,
    banks_failed: usize,
    banks_partial: usize,
    banks_total: usize,
    questions_answered: usize,
    questions_total: usize,
    questions_judged: usize,
}

/// Retain every session of one conversation, then drain its dirty-entity
/// queue. Returns the residual dirty count: 0 means fully consolidated,
/// anything else means the bank is retained but partially consolidated
/// (stalled passes or the pass limit) — still answerable, never a reason
/// to re-retain the conversation.
async fn retain_conversation(
    engine: &Engine,
    clock: &FixedClock,
    bank_id: &str,
    sessions: &[String],
    anchor: NaiveDate,
    params: &RetainParams,
) -> Result<usize, String> {
    clock.set(dataset::question_timestamp(anchor));
    let timestamps = dataset::session_timestamps(sessions.len(), anchor, params.spacing_days);
    for (session, ts) in sessions.iter().zip(timestamps) {
        clock.set(ts);
        engine
            .retain(RetainRequest {
                bank_id: bank_id.to_string(),
                content: session.clone(),
                context: None,
                timestamp_ms: Some(ts),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("retain for bank {bank_id} failed: {e}"))?;
    }
    clock.set(dataset::question_timestamp(anchor));
    let mut passes = 0usize;
    let mut stalled = 0usize;
    loop {
        let before = engine
            .store
            .count_dirty_entities(bank_id)
            .map_err(|e| format!("dirty check for bank {bank_id} failed: {e}"))?;
        if before == 0 {
            return Ok(0);
        }
        // Pass errors are expected occasionally (wedged provider calls hit
        // the hard wall; per-entity failures report through the engine).
        // Progress is measured on the dirty queue, so retrying is safe.
        if let Err(err) = engine
            .consolidate_once(bank_id, params.consolidate_max)
            .await
        {
            eprintln!("rustsight-eval: bank {bank_id}: consolidation pass failed: {err}");
        }
        let after = engine
            .store
            .count_dirty_entities(bank_id)
            .map_err(|e| format!("dirty check for bank {bank_id} failed: {e}"))?;
        passes += 1;
        if after < before {
            stalled = 0;
        } else {
            stalled += 1;
        }
        if stalled >= MAX_STALLED_PASSES {
            eprintln!(
                "rustsight-eval: bank {bank_id}: consolidation stalled after {passes} passes \
                 ({after} dirty left); keeping the bank partially consolidated"
            );
            return Ok(after);
        }
        if passes > params.pass_limit {
            eprintln!(
                "rustsight-eval: bank {bank_id}: hit the {}-pass limit ({after} dirty left); \
                 keeping the bank partially consolidated",
                params.pass_limit
            );
            return Ok(after);
        }
    }
}

/// Reflect with one blind retry. Even with the engine's in-loop nudges, a
/// reflect call can fail outright (persistent empty responses, provider
/// wedging after retries) — and in the first fullstack smoke run that
/// dropped 2 of 7 questions as permanent errors. A second attempt costs
/// one round trip; a dropped question costs the whole datapoint. The
/// first failure is logged; a second failure becomes the trace error.
async fn reflect_with_retry(
    engine: &Engine,
    bank: &str,
    query: &str,
    question_id: &str,
) -> Result<ReflectResponse, String> {
    match engine.reflect(bank, query).await {
        Ok(resp) => Ok(resp),
        Err(first) => {
            eprintln!("rustsight-eval: reflect for {question_id} failed ({first}); retrying once");
            engine
                .reflect(bank, query)
                .await
                .map_err(|last| last.to_string())
        }
    }
}

struct ProviderSetup {
    engine: Arc<Engine>,
    clock: Arc<FixedClock>,
    judge: Judge,
    counters: Counters,
    providers_info: serde_json::Value,
    judge_info: serde_json::Value,
}

async fn build_setup(cfg: &RunConfig) -> Result<ProviderSetup, String> {
    let clock = Arc::new(FixedClock::new(0));
    let db_path = cfg.out.join("banks.db");
    let store = SqliteStore::open_with_clock(&db_path, clock.clone())
        .map_err(|e| format!("failed to open {}: {e}", db_path.display()))?;
    let counters = Counters::default();
    let est: Arc<dyn TokenEstimator> = Arc::new(SimpleTokenEstimator);

    let (llm, embed, rerank, judge, providers_info, judge_info) = if cfg.mock {
        let (llm, embed, rerank) = build_mock_providers();
        (
            llm,
            embed,
            rerank,
            Judge::Fixed { correct: true },
            serde_json::json!({"llm": "mock", "embed": "mock", "rerank": "mock"}),
            serde_json::json!({"mode": "fixed(mock)"}),
        )
    } else {
        let env = ProviderEnv::from_env()?;
        if env.llm_api_key.is_empty() {
            return Err(
                "no API key configured: set RUSTSIGHT_LLM_API_KEY (or pass --mock)".to_string(),
            );
        }
        let judge_env = JudgeEnv::from_env().map_err(|missing| {
            format!("judge configuration incomplete: {missing} (or pass --mock)")
        })?;
        let (llm, embed, rerank) = build_providers(&env).await?;
        let judge = judge_env.build(counters.judge(), est.clone())?;
        let providers_info = serde_json::json!({
            "llm": {"base_url": env.llm_base_url, "model": env.llm_model},
            "embed": {"base_url": env.embed_base_url, "model": env.embed_model},
            "rerank": {"mode": format!("{:?}", env.rerank_mode)},
        });
        let judge_info = serde_json::json!({
            "base_url": judge_env.base_url,
            "model": judge_env.model,
        });
        (llm, embed, rerank, judge, providers_info, judge_info)
    };

    let llm: Arc<dyn LlmProvider> = Arc::new(CountingLlm::new(llm, counters.llm(), est.clone()));
    let embed: Arc<dyn EmbedProvider> =
        Arc::new(CountingEmbed::new(embed, counters.embed(), est.clone()));
    let engine = Arc::new(
        Engine::new(Arc::new(store), llm, embed, rerank, est, clock.clone()).with_config(
            EngineConfig {
                consolidate_concurrency: cfg.consolidate_concurrency.max(1),
                retain_max_completion_tokens: Some(retain_max_completion_tokens()),
                ..Default::default()
            },
        ),
    );
    Ok(ProviderSetup {
        engine,
        clock,
        judge,
        counters,
        providers_info,
        judge_info,
    })
}

pub async fn run(cfg: RunConfig) -> Result<RunOutput, String> {
    let started = Instant::now();
    let file = std::fs::File::open(&cfg.dataset_path)
        .map_err(|e| format!("cannot read dataset {}: {e}", cfg.dataset_path.display()))?;
    let data: Dataset = dataset::parse_dataset_reader(std::io::BufReader::new(file))?;
    let questions: Vec<&Question> = match cfg.limit_questions {
        Some(n) => data.questions.iter().take(n).collect(),
        None => data.questions.iter().collect(),
    };
    if questions.is_empty() {
        return Err("no questions selected".into());
    }

    std::fs::create_dir_all(cfg.out.join("traces"))
        .map_err(|e| format!("cannot create {}: {e}", cfg.out.display()))?;
    let state_path = cfg.out.join("state.json");
    let state = Arc::new(Mutex::new(RunState::load(&state_path)?));

    let setup = build_setup(&cfg).await?;
    let ProviderSetup {
        engine,
        clock,
        judge,
        counters,
        providers_info,
        judge_info,
    } = setup;

    let needed_banks: HashSet<&str> = questions.iter().map(|q| q.bank_id.as_str()).collect();
    let conversations_by_bank: BTreeMap<&str, (&Vec<String>, NaiveDate)> = data
        .conversations
        .iter()
        .filter(|c| needed_banks.contains(c.bank_id.as_str()))
        .map(|c| (c.bank_id.as_str(), (&c.sessions, c.anchor_date)))
        .collect();

    // ---- Heartbeat ----
    let progress = Arc::new(Mutex::new(Progress {
        phase: "retain",
        banks_total: conversations_by_bank.len(),
        questions_total: questions.len(),
        ..Default::default()
    }));
    let heartbeat = spawn_heartbeat(&progress, cfg.heartbeat_secs, started);

    // ---- Phase 1: retain + consolidate (parallel across conversations) ----
    let parallel = if cfg.mock {
        1
    } else {
        cfg.conversations_parallel.max(1)
    };
    let semaphore = Arc::new(Semaphore::new(parallel));
    let retain_params = RetainParams {
        spacing_days: cfg.session_spacing_days,
        consolidate_max: cfg.consolidate_max,
        pass_limit: cfg.consolidate_pass_limit,
    };
    let mut retain_futs = Vec::new();
    for (bank, (sessions, anchor)) in &conversations_by_bank {
        let already = state
            .lock()
            .unwrap()
            .conversations
            .get(*bank)
            .is_some_and(|c| c.retained && c.consolidated);
        if already {
            continue;
        }
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| format!("semaphore closed: {e}"))?;
        let engine = engine.clone();
        let clock = clock.clone();
        let bank = (*bank).to_string();
        let sessions: Vec<String> = (*sessions).clone();
        let anchor: NaiveDate = *anchor;
        let params = retain_params;
        let state = state.clone();
        let state_path = state_path.clone();
        let progress = progress.clone();
        retain_futs.push(tokio::spawn(async move {
            let _permit = permit;
            eprintln!(
                "rustsight-eval: retaining bank {bank} ({} sessions)",
                sessions.len()
            );
            // The task itself records the outcome: state update, save, and
            // the log line all land the moment the conversation finishes —
            // a failure is never deferred behind the spawn loop again.
            let outcome =
                retain_conversation(&engine, &clock, &bank, &sessions, anchor, &params).await;
            let mut prog = progress.lock().unwrap();
            prog.banks_done += 1;
            match outcome {
                Ok(residual) => {
                    let partial = residual > 0;
                    if partial {
                        prog.banks_partial += 1;
                    }
                    drop(prog);
                    eprintln!(
                        "rustsight-eval: bank {bank} retained + consolidated{}",
                        if partial {
                            format!(" (partially: {residual} dirty left)")
                        } else {
                            String::new()
                        }
                    );
                    let mut st = state.lock().unwrap();
                    st.conversations.insert(
                        bank,
                        ConversationState {
                            retained: true,
                            consolidated: true,
                            partial,
                        },
                    );
                    if let Err(e) = st.save(&state_path) {
                        eprintln!("rustsight-eval: {e}");
                    }
                    true
                }
                Err(e) => {
                    prog.banks_failed += 1;
                    drop(prog);
                    eprintln!("rustsight-eval: {e}");
                    false
                }
            }
        }));
    }
    let mut retained_any = false;
    let mut saw_retain_error = false;
    for fut in retain_futs {
        let ok = fut
            .await
            .map_err(|e| format!("retain task panicked: {e}"))?;
        if ok {
            retained_any = true;
        } else {
            saw_retain_error = true;
        }
    }
    if saw_retain_error && !retained_any {
        heartbeat.abort();
        return Err("every conversation failed to retain; aborting".into());
    }

    let retained_banks: HashSet<String> = state
        .lock()
        .unwrap()
        .conversations
        .iter()
        .filter(|(_, c)| c.retained && c.consolidated)
        .map(|(b, _)| b.clone())
        .collect();
    let partial_banks = state
        .lock()
        .unwrap()
        .conversations
        .iter()
        .filter(|(_, c)| c.partial)
        .count();

    // ---- Phase 2: answer (parallel across banks, sequential within) ----
    progress.lock().unwrap().phase = "answer";
    eprintln!(
        "rustsight-eval: phase 2/4 — answering {} questions ({} banks retained, {partial_banks} partial)",
        questions.len(),
        retained_banks.len()
    );
    let mut questions_by_bank: BTreeMap<String, Vec<Question>> = BTreeMap::new();
    for q in &questions {
        questions_by_bank
            .entry(q.bank_id.clone())
            .or_default()
            .push((*q).clone());
    }
    let mut answer_futs: Vec<tokio::task::JoinHandle<Result<(), String>>> = Vec::new();
    for (bank, bank_questions) in &questions_by_bank {
        {
            let st = state.lock().unwrap();
            if bank_questions
                .iter()
                .all(|q| st.questions.get(&q.question_id).is_some_and(|s| s.answered))
            {
                continue;
            }
        }
        if !retained_banks.contains(bank) {
            for q in bank_questions {
                let trace = QuestionTrace {
                    question_id: q.question_id.clone(),
                    bank_id: bank.clone(),
                    ability: q.ability.as_str().to_string(),
                    abstention: q.ability == dataset::Ability::Abstention,
                    question: q.question.clone(),
                    gold_answer: q.gold_answer.clone(),
                    evidence_session_ids: q.evidence_session_ids.clone(),
                    error: Some("conversation was not retained".into()),
                    ..Default::default()
                };
                write_trace(&cfg.out, &trace)?;
                let mut st = state.lock().unwrap();
                st.questions.insert(
                    q.question_id.clone(),
                    QuestionState {
                        answered: true,
                        judged: false,
                    },
                );
            }
            state.lock().unwrap().save(&state_path).ok();
            continue;
        }
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| format!("semaphore closed: {e}"))?;
        let engine = engine.clone();
        let clock = clock.clone();
        let state = state.clone();
        let state_path = state_path.clone();
        let out = cfg.out.clone();
        let bank = bank.clone();
        let bank_questions = bank_questions.clone();
        let progress = progress.clone();
        answer_futs.push(tokio::spawn(async move {
            let _permit = permit;
            for q in &bank_questions {
                let skip = state
                    .lock()
                    .unwrap()
                    .questions
                    .get(&q.question_id)
                    .is_some_and(|s| s.answered);
                if skip {
                    continue;
                }
                let date = q.question_date.unwrap_or_else(fallback_date);
                clock.set(dataset::question_timestamp(date));
                let query = format!(
                    "Today's date is {}. {}",
                    dataset::date_string(date),
                    q.question
                );
                let t0 = Instant::now();
                let mut trace = QuestionTrace {
                    question_id: q.question_id.clone(),
                    bank_id: bank.clone(),
                    ability: q.ability.as_str().to_string(),
                    abstention: q.ability == dataset::Ability::Abstention,
                    question: q.question.clone(),
                    gold_answer: q.gold_answer.clone(),
                    evidence_session_ids: q.evidence_session_ids.clone(),
                    ..Default::default()
                };
                match reflect_with_retry(&engine, &bank, &query, &q.question_id).await {
                    Ok(resp) => {
                        trace.hypothesis = Some(resp.answer);
                        trace.citations = resp.citations;
                        trace.retrieved_ids = resp.retrieved_ids;
                        trace.retrieved = resp.retrieved;
                        trace.rounds = resp.rounds;
                    }
                    Err(e) => {
                        trace.error = Some(e);
                    }
                }
                trace.answer_ms = t0.elapsed().as_millis() as u64;
                write_trace(&out, &trace)?;
                {
                    let mut st = state.lock().unwrap();
                    st.questions.insert(
                        q.question_id.clone(),
                        QuestionState {
                            answered: true,
                            judged: false,
                        },
                    );
                    if let Err(e) = st.save(&state_path) {
                        eprintln!("rustsight-eval: {e}");
                    }
                }
                {
                    let mut prog = progress.lock().unwrap();
                    prog.questions_answered += 1;
                    if prog.questions_answered.is_multiple_of(25) {
                        eprintln!(
                            "rustsight-eval: answered {}/{} questions",
                            prog.questions_answered, prog.questions_total
                        );
                    }
                }
            }
            Ok(())
        }));
    }
    let mut answer_errors = 0usize;
    for fut in answer_futs {
        match fut.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                answer_errors += 1;
                eprintln!("rustsight-eval: {e}");
            }
            Err(e) => {
                heartbeat.abort();
                return Err(format!("answer task panicked: {e}"));
            }
        }
    }
    if answer_errors > 0 && missing_traces(&questions, &cfg.out) == questions.len() {
        heartbeat.abort();
        return Err("every answer task failed; aborting".into());
    }

    // ---- Phase 3: judge (sequential; reads hypotheses from traces) ----
    progress.lock().unwrap().phase = "judge";
    eprintln!(
        "rustsight-eval: phase 3/4 — judging {} answers",
        questions.len()
    );
    for q in &questions {
        let judged = state
            .lock()
            .unwrap()
            .questions
            .get(&q.question_id)
            .is_some_and(|s| s.judged);
        if judged {
            continue;
        }
        let Some(mut trace) = read_trace(&cfg.out, &q.question_id) else {
            continue;
        };
        let Some(hypothesis) = trace.hypothesis.clone() else {
            let mut st = state.lock().unwrap();
            st.questions.insert(
                q.question_id.clone(),
                QuestionState {
                    answered: true,
                    judged: true,
                },
            );
            st.save(&state_path).ok();
            continue;
        };
        let t0 = Instant::now();
        match judge
            .judge(&q.question, &q.gold_answer, &hypothesis, trace.abstention)
            .await
        {
            Ok(verdict) => {
                trace.correct = Some(verdict.correct);
                trace.judge_raw = Some(verdict.raw);
                trace.judge_parsed = Some(verdict.parsed);
            }
            Err(e) => {
                trace.judge_raw = Some(format!("judge call failed: {e}"));
                trace.judge_parsed = Some(false);
            }
        }
        trace.judge_ms = Some(t0.elapsed().as_millis() as u64);
        write_trace(&cfg.out, &trace)?;
        {
            let mut st = state.lock().unwrap();
            st.questions.insert(
                q.question_id.clone(),
                QuestionState {
                    answered: true,
                    judged: true,
                },
            );
            st.save(&state_path).ok();
        }
        let mut prog = progress.lock().unwrap();
        prog.questions_judged += 1;
        if prog.questions_judged.is_multiple_of(25) {
            eprintln!(
                "rustsight-eval: judged {}/{} answers",
                prog.questions_judged, prog.questions_total
            );
        }
    }

    // ---- Phase 4: report ----
    heartbeat.abort();
    progress.lock().unwrap().phase = "report";
    let traces: Vec<QuestionTrace> = questions
        .iter()
        .filter_map(|q| read_trace(&cfg.out, &q.question_id))
        .collect();
    let summary = report::aggregate(&traces);
    let counters_snapshot = counters.snapshot();
    let run_info = serde_json::json!({
        "providers": providers_info,
        "judge": judge_info,
        "query_style": "date-prefixed reflect query; clock anchored per question",
        "dataset": {
            "path": cfg.dataset_path,
            "questions_selected": questions.len(),
            "conversations": conversations_by_bank.len(),
            "partial_conversations": partial_banks,
        },
        "config": {
            "session_spacing_days": cfg.session_spacing_days,
            "conversations_parallel": parallel,
            "consolidate_max": cfg.consolidate_max,
            "consolidate_concurrency": cfg.consolidate_concurrency,
            "limit_questions": cfg.limit_questions,
        },
    });
    report::write_report(
        &cfg.out,
        &summary,
        &counters_snapshot,
        &run_info,
        started.elapsed().as_secs(),
    )?;

    eprintln!(
        "rustsight-eval: done — {}/{} correct ({:.1}%), report at {}",
        summary.correct,
        summary.total,
        summary.accuracy * 100.0,
        cfg.out.join("report.md").display()
    );
    for a in &summary.by_ability {
        eprintln!(
            "  {:>26} {}/{} ({:.1}%)",
            a.ability,
            a.correct,
            a.judged,
            a.accuracy * 100.0
        );
    }

    Ok(RunOutput {
        summary,
        report_md: cfg.out.join("report.md"),
    })
}

/// Print a one-line heartbeat on stderr every `secs` seconds so long runs
/// (and their owners) can see liveness: phase, unit counters, elapsed.
fn spawn_heartbeat(
    progress: &Arc<Mutex<Progress>>,
    secs: u64,
    started: Instant,
) -> tokio::task::JoinHandle<()> {
    let progress = Arc::clone(progress);
    tokio::spawn(async move {
        if secs == 0 {
            return;
        }
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(secs.max(1)));
        interval.tick().await; // the first tick fires immediately; skip it
        loop {
            interval.tick().await;
            let p = progress.lock().unwrap();
            eprintln!(
                "rustsight-eval: heartbeat [{}] {}/{} banks done ({} partial, {} failed), \
                 {}/{} answered, {}/{} judged, {}s elapsed",
                p.phase,
                p.banks_done,
                p.banks_total,
                p.banks_partial,
                p.banks_failed,
                p.questions_answered,
                p.questions_total,
                p.questions_judged,
                p.questions_total,
                started.elapsed().as_secs()
            );
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustsight_core::traits::{LlmResponse, ToolCall};
    use rustsight_providers::{MockEmbed, MockLlm, MockRerank};
    use serde_json::json;

    fn resp(content: &str) -> LlmResponse {
        LlmResponse {
            content: Some(content.into()),
            tool_calls: vec![],
        }
    }

    fn tool_resp(id: &str, name: &str, args: serde_json::Value) -> LlmResponse {
        LlmResponse {
            content: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: args,
            }],
        }
    }

    fn alice_facts_resp() -> LlmResponse {
        let facts = [
            "Alice joined Google as an engineer.",
            "Alice works on Search infrastructure.",
        ]
        .iter()
        .map(|text| {
            json!({"text": text, "fact_type": "world",
                   "entities": [{"name": "Alice", "etype": "person"}],
                   "occurred_start": null, "occurred_end": null, "causal": []})
        })
        .collect::<Vec<_>>();
        resp(&json!({"facts": facts}).to_string())
    }

    fn test_engine(llm: MockLlm, clock: &Arc<FixedClock>) -> Engine {
        Engine::new(
            Arc::new(SqliteStore::open_in_memory_with_clock(clock.clone()).unwrap()),
            Arc::new(llm),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            clock.clone(),
        )
    }

    /// A bank whose consolidation never makes progress (every synthesis
    /// response is malformed) must still count as retained: memories are
    /// stored, the residual dirty count comes back instead of an error,
    /// and the run keeps going. This is the regression test for the
    /// all-or-nothing loop that silently killed the 21-hour LongMemEval
    /// attempt: one bad pass per bank marked every conversation failed.
    #[tokio::test]
    async fn stalled_consolidation_keeps_bank_answerable() {
        let clock = Arc::new(FixedClock::new(1_800_000_000_000));
        // extraction, then three malformed synthesis responses — one per
        // stalled pass before the harness gives up (MAX_STALLED_PASSES).
        let engine = test_engine(
            MockLlm::script(vec![
                alice_facts_resp(),
                resp("not json at all"),
                resp("still not json"),
                resp("nope"),
            ]),
            &clock,
        );
        let params = RetainParams {
            spacing_days: 1,
            consolidate_max: 50,
            pass_limit: 200,
        };
        let anchor = NaiveDate::from_ymd_opt(2024, 5, 10).expect("valid date");
        let residual = retain_conversation(
            &engine,
            &clock,
            "stall",
            &["Alice joined Google. Alice works on Search.".to_string()],
            anchor,
            &params,
        )
        .await
        .expect("stalled consolidation must not fail the conversation");
        assert!(residual > 0, "residual={residual}");
        // the retain itself stored memories: the bank is answerable
        assert!(
            !engine.store.list_memories("stall", 10).unwrap().is_empty(),
            "stalled bank must keep its memories"
        );
        assert_eq!(
            engine.store.count_dirty_entities("stall").unwrap(),
            residual
        );
    }

    /// The happy path: a clean synthesis drains the dirty queue to zero.
    #[tokio::test]
    async fn healthy_conversation_consolidates_fully() {
        let clock = Arc::new(FixedClock::new(1_800_000_000_000));
        let engine = test_engine(
            MockLlm::script(vec![
                alice_facts_resp(),
                resp(
                    r#"{"observations":[{"text":"Alice works at Google on Search.","evidence":[{"fact_index":0,"quote":""}]}]}"#,
                ),
            ]),
            &clock,
        );
        let params = RetainParams {
            spacing_days: 1,
            consolidate_max: 50,
            pass_limit: 200,
        };
        let anchor = NaiveDate::from_ymd_opt(2024, 5, 10).expect("valid date");
        let residual = retain_conversation(
            &engine,
            &clock,
            "healthy",
            &["Alice joined Google. Alice works on Search.".to_string()],
            anchor,
            &params,
        )
        .await
        .expect("healthy conversation must retain");
        assert_eq!(residual, 0);
        assert!(engine.store.count_dirty_entities("healthy").unwrap() == 0);
        assert!(!engine
            .store
            .observations_for_bank("healthy")
            .unwrap()
            .is_empty());
    }

    /// The first fullstack smoke run lost 2/7 questions to one-shot
    /// reflect failures (empty model responses). A failed first attempt
    /// must be retried once; the retry's answer becomes the question's
    /// answer.
    ///
    /// The reflect pipeline is loop → forced-done → final synthesis (only
    /// when the loop produced nothing) → backgrounded opinions. Here a
    /// provider hiccup empties the loop twice, the forced-done round still
    /// lands a `finish`, and that closing answer IS the final answer
    /// (upstream precedence: the loop's done call wins).
    #[tokio::test]
    async fn reflect_with_retry_recovers_a_failed_first_attempt() {
        let clock = Arc::new(FixedClock::new(1_800_000_000_000));
        let engine = test_engine(
            MockLlm::script(vec![
                // first reflect attempt: two consecutive empty responses
                // (the engine nudges once, then breaks to the answer path)
                LlmResponse {
                    content: None,
                    tool_calls: vec![],
                },
                LlmResponse {
                    content: None,
                    tool_calls: vec![],
                },
                // forced-done round: clean finish — this answer wins
                tool_resp("1", "finish", json!({"answer": "Recovered."})),
            ]),
            &clock,
        );
        let r = reflect_with_retry(&engine, "b", "q", "q1")
            .await
            .expect("retry must recover");
        assert_eq!(r.answer, "Recovered.");
        assert_eq!(
            r.answer_path,
            rustsight_core::AnswerPath::ForcedDone,
            "the closing finish is the answer"
        );
    }

    /// A provider that never says anything still fails reflect (after the
    /// harness retry) — the last error surfaces for the trace.
    #[tokio::test]
    async fn reflect_with_retry_surfaces_the_last_error() {
        use rustsight_core::traits::LlmRequest;
        struct ForeverEmpty;
        #[async_trait::async_trait]
        impl LlmProvider for ForeverEmpty {
            async fn complete(&self, _req: &LlmRequest) -> rustsight_core::Result<LlmResponse> {
                Ok(LlmResponse {
                    content: None,
                    tool_calls: vec![],
                })
            }
        }
        let clock = Arc::new(FixedClock::new(1_800_000_000_000));
        let engine = Engine::new(
            Arc::new(SqliteStore::open_in_memory_with_clock(clock.clone()).unwrap()),
            Arc::new(ForeverEmpty),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            clock.clone(),
        );
        let err = reflect_with_retry(&engine, "b", "q", "q1")
            .await
            .expect_err("persistent emptiness must fail both attempts");
        assert!(
            err.contains("produced no answer"),
            "unexpected error: {err}"
        );
    }
}
