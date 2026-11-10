//! End-to-end harness run against mock providers: dataset ingest → retain →
//! consolidate → reflect → fixed judge → report. Fully offline (RS-6 DoD
//! pattern: deterministic by construction, no network, no env vars).

use rustsight_eval::compare as record_compare;
use rustsight_eval::runner::{self, RunConfig};
use std::path::PathBuf;

fn fixture_dataset() -> String {
    // Conversation A: two sessions, two questions (one abstention).
    // Conversation B: one session, one temporal question. Capitalized
    // names give the demo extraction mock entities; sentences give it
    // facts (it splits on '.', takes up to 6).
    let a1 = r#"[{"role":"user","content":"Alice joined Google. Alice works on Search infrastructure."},{"role":"assistant","content":"That is great news."}]"#;
    let a2 = r#"[{"role":"user","content":"Alice moved to the Maps team. Alice enjoys hiking."},{"role":"assistant","content":"Noted."}]"#;
    let b1 = r#"[{"role":"user","content":"Bob started running. Bob runs marathons now."},{"role":"assistant","content":"Impressive."}]"#;
    format!(
        r#"[
  {{"question_id":"conv_a_q1","question_type":"single-session-user","question":"Where does Alice work?","answer":"Google","question_date":"2024-05-10","haystack_session_ids":[1,2],"haystack_sessions":[{a1},{a2}],"answer_session_ids":[1]}},
  {{"question_id":"conv_a_q2_abs","question_type":"single-session-user","question":"What is Alice's favorite food?","answer":"Not mentioned","question_date":"2024-05-11","haystack_session_ids":[1,2],"haystack_sessions":[{a1},{a2}],"answer_session_ids":[]}},
  {{"question_id":"conv_b_q1","question_type":"temporal-reasoning","question":"What hobby did Bob pick up?","answer":"running","question_date":"2024-06-01","haystack_session_ids":["s1"],"haystack_sessions":[{b1}],"answer_session_ids":["s1"]}}
]"#
    )
}

fn temp_run_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rs-eval-run-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn mock_run_end_to_end() {
    let dir = temp_run_dir("e2e");
    let dataset_path = dir.join("dataset.json");
    std::fs::write(&dataset_path, fixture_dataset()).unwrap();

    let out = dir.join("results");
    let cfg = RunConfig {
        out: out.clone(),
        dataset_path,
        conversations_parallel: 2,
        limit_questions: None,
        session_spacing_days: 1,
        mock: true,
        consolidate_max: 50,
        consolidate_pass_limit: 50,
        consolidate_concurrency: 2,
        heartbeat_secs: 0,
    };
    let result = runner::run(cfg).await.expect("mock run should succeed");

    assert_eq!(result.summary.total, 3);
    assert_eq!(result.summary.judged, 3);
    assert_eq!(result.summary.correct, 3);
    assert_eq!(result.summary.accuracy, 1.0);
    assert_eq!(result.summary.by_ability.len(), 3);

    for qid in ["conv_a_q1", "conv_a_q2_abs", "conv_b_q1"] {
        let trace = std::fs::read_to_string(out.join("traces").join(format!("{qid}.json")))
            .unwrap_or_else(|e| panic!("trace {qid} missing: {e}"));
        let trace: serde_json::Value = serde_json::from_str(&trace).unwrap();
        assert_eq!(trace["question_id"], qid, "trace id mismatch");
        assert!(trace["hypothesis"].is_string(), "{qid} has no hypothesis");
        assert_eq!(trace["correct"], true);
    }
    assert!(out.join("report.md").exists());
    assert!(out.join("report.json").exists());
    assert!(out.join("banks.db").exists());
    let report = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(report.contains("**overall**"));
    assert!(report.contains("abstention"));

    let state = std::fs::read_to_string(out.join("state.json")).unwrap();
    let state: serde_json::Value = serde_json::from_str(&state).unwrap();
    assert_eq!(state["conversations"].as_object().unwrap().len(), 2);
    for (_, q) in state["questions"].as_object().unwrap() {
        assert_eq!(q["answered"], true);
        assert_eq!(q["judged"], true);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn rerun_is_resumable_and_idempotent() {
    let dir = temp_run_dir("resume");
    let dataset_path = dir.join("dataset.json");
    std::fs::write(&dataset_path, fixture_dataset()).unwrap();
    let out = dir.join("results");
    let cfg = RunConfig {
        out: out.clone(),
        dataset_path: dataset_path.clone(),
        conversations_parallel: 1,
        limit_questions: None,
        session_spacing_days: 1,
        mock: true,
        consolidate_max: 50,
        consolidate_pass_limit: 50,
        consolidate_concurrency: 2,
        heartbeat_secs: 0,
    };
    runner::run(cfg.clone()).await.expect("first run");
    // A second run over the same directory must skip everything (state)
    // and still produce a complete report.
    let second = runner::run(cfg).await.expect("second run");
    assert_eq!(second.summary.total, 3);
    assert_eq!(second.summary.correct, 3);

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn limit_questions_restricts_scope() {
    let dir = temp_run_dir("limit");
    let dataset_path = dir.join("dataset.json");
    std::fs::write(&dataset_path, fixture_dataset()).unwrap();
    let out = dir.join("results");
    let cfg = RunConfig {
        out,
        dataset_path,
        conversations_parallel: 1,
        limit_questions: Some(1),
        session_spacing_days: 1,
        mock: true,
        consolidate_max: 50,
        consolidate_pass_limit: 50,
        consolidate_concurrency: 2,
        heartbeat_secs: 0,
    };
    let result = runner::run(cfg).await.expect("limited run");
    // One question -> only its conversation is retained.
    assert_eq!(result.summary.total, 1);
    assert_eq!(result.summary.judged, 1);

    std::fs::remove_dir_all(&dir).ok();
}

/// `compare` must join a run's traces against the committed Hindsight
/// record on the dataset's real question ids. Two real ids are used — one
/// Hindsight answered correctly (`e47becba`), one it failed (`51a45a95`,
/// the coupon question) — and the mock judge is fixed-correct, so the join
/// must land them in the both-correct and rustsight-only-correct buckets.
#[tokio::test]
async fn compare_joins_mock_run_against_committed_record() {
    let session = r#"[{"role":"user","content":"Alice joined Google. Alice works on Search infrastructure."},{"role":"assistant","content":"Great."}]"#;
    let dataset = format!(
        r#"[
  {{"question_id":"e47becba","question_type":"single-session-user","question":"What degree did I graduate with?","answer":"Business Administration","question_date":"2024-05-10","haystack_session_ids":[1],"haystack_sessions":[{session}],"answer_session_ids":[1]}},
  {{"question_id":"51a45a95","question_type":"single-session-user","question":"Where did I redeem a $5 coupon on coffee creamer?","answer":"Target","question_date":"2024-05-10","haystack_session_ids":[1],"haystack_sessions":[{session}],"answer_session_ids":[1]}}
]"#
    );
    let dir = temp_run_dir("compare");
    let dataset_path = dir.join("dataset.json");
    std::fs::write(&dataset_path, dataset).unwrap();
    let out = dir.join("results");
    let cfg = RunConfig {
        out: out.clone(),
        dataset_path,
        conversations_parallel: 1,
        limit_questions: None,
        session_spacing_days: 1,
        mock: true,
        consolidate_max: 50,
        consolidate_pass_limit: 50,
        consolidate_concurrency: 2,
        heartbeat_secs: 0,
    };
    let result = runner::run(cfg).await.expect("mock run");
    assert_eq!(result.summary.total, 2);
    assert_eq!(result.summary.correct, 2);

    let record = record_compare::default_record_path();
    let c = record_compare::compare(&out, &record).expect("compare joins");
    assert_eq!(c.matched, 2);
    assert_eq!(c.unmatched_run_ids.len(), 0);
    assert_eq!(c.record_not_run, 498);
    // Mock judge is fixed-correct: e47becba (Hindsight right) -> both right,
    // 51a45a95 (Hindsight wrong) -> rustsight-only correct.
    assert_eq!(c.both_correct, 1);
    assert_eq!(c.rustsight_only_correct, 1);
    assert_eq!(c.parity_gaps.len(), 0);
    assert_eq!(c.wins.len(), 1);
    assert_eq!(c.wins[0].question_id, "51a45a95");
    let md = c.render_markdown();
    assert!(md.contains("rustsight-only correct — Hindsight wrong (1)"));

    std::fs::remove_dir_all(&dir).ok();
}

/// The committed full-stack smoke fixture (what `scripts/fullstack-smoke`
/// runs against real providers) must stay parseable and drain its bank
/// fully offline too — this is the repeatability guard for the front-to-
/// back validation procedure added after the 21-hour LongMemEval failure.
#[tokio::test]
async fn committed_fullstack_fixture_runs_end_to_end() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fullstack_smoke.json");
    assert!(fixture.exists(), "fixture missing: {}", fixture.display());
    let dir = temp_run_dir("committed-fixture");
    let out = dir.join("results");
    let cfg = RunConfig {
        out: out.clone(),
        dataset_path: fixture,
        conversations_parallel: 1,
        limit_questions: None,
        session_spacing_days: 1,
        mock: true,
        consolidate_max: 50,
        consolidate_pass_limit: 50,
        consolidate_concurrency: 2,
        heartbeat_secs: 0,
    };
    let result = runner::run(cfg).await.expect("fixture run");
    assert_eq!(result.summary.total, 7);
    assert_eq!(result.summary.judged, 7);
    // one bank, all five abilities covered
    assert_eq!(result.summary.by_ability.len(), 5);
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("state.json")).unwrap()).unwrap();
    let conversations = state["conversations"].as_object().unwrap();
    assert_eq!(conversations.len(), 1);
    for (_, c) in conversations {
        assert_eq!(c["retained"], true);
        assert_eq!(c["consolidated"], true);
        assert_eq!(c["partial"], false, "fixture bank must drain fully");
    }

    std::fs::remove_dir_all(&dir).ok();
}
