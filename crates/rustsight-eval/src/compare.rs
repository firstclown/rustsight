//! `compare`: score a rustsight eval run question-by-question against the
//! committed Hindsight LongMemEval-S record
//! (`fixtures/hindsight_longmemeval_s.json`), so parity work targets the
//! same questions upstream fails. The record is the Agent Memory Benchmark
//! run of upstream Hindsight (94.6% overall); ids are the dataset's
//! `question_id`s, so they join directly on run traces.
//!
//! Verdicts come from different judges (ours vs AMB's
//! `gemini-2.5-flash-lite`), so agreement is evidence, not ground truth.

use crate::dataset::ability_of;
use crate::report::QuestionTrace;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The committed record fixture, next to this crate's sources.
pub fn default_record_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/hindsight_longmemeval_s.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordMeta {
    #[serde(default)]
    pub answer_llm: String,
    #[serde(default)]
    pub judge_llm: String,
    #[serde(default)]
    pub total_queries: usize,
    #[serde(default)]
    pub correct: usize,
    #[serde(default)]
    pub accuracy: f64,
    #[serde(default)]
    pub source_repo: String,
    #[serde(default)]
    pub source_path: String,
    #[serde(default)]
    pub fetched: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HindsightResult {
    pub query_id: String,
    #[serde(default)]
    pub question_type: Option<String>,
    pub correct: bool,
    #[serde(default)]
    pub answer: Option<String>,
    #[serde(default)]
    pub judge_reason: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HindsightRecord {
    pub meta: RecordMeta,
    pub results: Vec<HindsightResult>,
}

impl HindsightRecord {
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&content).map_err(|e| format!("record parse failed: {e}"))
    }

    fn by_id(&self) -> BTreeMap<&str, &HindsightResult> {
        self.results
            .iter()
            .map(|r| (r.query_id.as_str(), r))
            .collect()
    }
}

/// One question where the two systems disagree (or both fail).
#[derive(Debug, Clone, Serialize)]
pub struct Divergence {
    pub question_id: String,
    pub ability: String,
    pub question: String,
    pub gold_answer: String,
    pub rustsight_answer: Option<String>,
    pub hindsight_answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustsight_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AbilityComparison {
    pub ability: String,
    pub total: usize,
    pub rustsight_correct: usize,
    pub hindsight_correct: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub run_dir: PathBuf,
    pub record: RecordMeta,
    pub matched: usize,
    /// Run questions absent from the record (e.g. the fullstack-smoke
    /// fixture's `fss_*` ids).
    pub unmatched_run_ids: Vec<String>,
    /// Record questions this run never attempted (partial runs).
    pub record_not_run: usize,
    /// Matched questions with no verdict (answer error or judge skipped);
    /// counted as not-correct above.
    pub rustsight_unjudged: usize,
    pub both_correct: usize,
    pub shared_failures: usize,
    /// Hindsight correct, rustsight not — the parity gaps.
    pub hindsight_only_correct: usize,
    /// rustsight correct, Hindsight not.
    pub rustsight_only_correct: usize,
    pub by_ability: Vec<AbilityComparison>,
    pub parity_gaps: Vec<Divergence>,
    pub wins: Vec<Divergence>,
    pub shared_failures_list: Vec<Divergence>,
}

impl Comparison {
    /// Fraction of matched questions with identical outcomes, whatever the
    /// verdict. A low agreement with a good score means the systems fail on
    /// different questions; a low agreement with a bad score means rustsight
    /// is failing broadly, not divergently.
    pub fn agreement(&self) -> f64 {
        if self.matched == 0 {
            return 0.0;
        }
        (self.both_correct + self.shared_failures) as f64 / self.matched as f64
    }

    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Hindsight record comparison\n\n");
        md.push_str(&format!(
            "Run `{}` against the Hindsight LongMemEval-S record: {}/{} ({:.1}%), answer LLM `{}`, judge `{}` (fetched {} from {}/{}).\n\n",
            self.run_dir.display(),
            self.record.correct,
            self.record.total_queries,
            self.record.accuracy * 100.0,
            self.record.answer_llm,
            self.record.judge_llm,
            self.record.fetched,
            self.record.source_repo,
            self.record.source_path,
        ));
        md.push_str(
            "Verdicts come from different judges, so agreement is evidence, not ground truth.\n\n",
        );

        let rs_correct = self.both_correct + self.rustsight_only_correct;
        let hs_correct = self.both_correct + self.hindsight_only_correct;
        md.push_str(&format!(
            "| | Hindsight correct | Hindsight wrong |\n|---|---|---|\n\
             | **rustsight correct** | {} | {} |\n\
             | **rustsight wrong** | {} | {} |\n\n",
            self.both_correct,
            self.rustsight_only_correct,
            self.hindsight_only_correct,
            self.shared_failures
        ));
        md.push_str(&format!(
            "Matched questions: {} (rustsight {}/{} = {:.1}%, Hindsight {}/{} = {:.1}%, agreement {:.1}%).\n\n",
            self.matched,
            rs_correct,
            self.matched,
            rs_correct as f64 / self.matched.max(1) as f64 * 100.0,
            hs_correct,
            self.matched,
            hs_correct as f64 / self.matched.max(1) as f64 * 100.0,
            self.agreement() * 100.0,
        ));
        if self.rustsight_unjudged > 0 {
            md.push_str(&format!(
                "Unjudged/unanswered rustsight questions counted as wrong: {}.\n\n",
                self.rustsight_unjudged
            ));
        }
        if !self.unmatched_run_ids.is_empty() {
            md.push_str(&format!(
                "Run questions not in the record (ignored above): {} — e.g. {}.\n\n",
                self.unmatched_run_ids.len(),
                self.unmatched_run_ids
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if self.record_not_run > 0 {
            md.push_str(&format!(
                "Record questions not attempted in this run: {} (partial run).\n\n",
                self.record_not_run
            ));
        }

        md.push_str("| ability | n | rustsight | hindsight | delta |\n|---|---|---|---|---|\n");
        for a in &self.by_ability {
            let delta = a.rustsight_correct as f64 - a.hindsight_correct as f64;
            md.push_str(&format!(
                "| {} | {} | {} | {} | {:+.0} q ({:+.1} pp) |\n",
                a.ability,
                a.total,
                a.rustsight_correct,
                a.hindsight_correct,
                delta,
                if a.total > 0 {
                    delta / a.total as f64 * 100.0
                } else {
                    0.0
                }
            ));
        }

        let sections = [
            (
                "Parity gaps — Hindsight correct, rustsight not",
                &self.parity_gaps,
            ),
            ("rustsight-only correct — Hindsight wrong", &self.wins),
            ("Shared failures — both wrong", &self.shared_failures_list),
        ];
        for (title, list) in sections {
            md.push_str(&format!("\n## {title} ({})\n\n", list.len()));
            if list.is_empty() {
                md.push_str("(none)\n");
                continue;
            }
            md.push_str("| id | ability | question | gold | rustsight | hindsight |\n|---|---|---|---|---|---|\n");
            for d in list.iter().take(80) {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    d.question_id,
                    d.ability,
                    escape_table(&d.question, 70),
                    escape_table(&d.gold_answer, 40),
                    match (&d.rustsight_answer, &d.rustsight_error) {
                        (Some(a), _) => escape_table(a, 60),
                        (None, Some(e)) => format!("*(error: {})*", escape_table(e, 50)),
                        (None, None) => "*(no answer)*".into(),
                    },
                    d.hindsight_answer
                        .as_deref()
                        .map(|a| escape_table(a, 60))
                        .unwrap_or_else(|| "*(none)*".into()),
                ));
            }
            if list.len() > 80 {
                md.push_str(&format!(
                    "\n... and {} more (see compare.json).\n",
                    list.len() - 80
                ));
            }
        }
        md
    }
}

fn escape_table(s: &str, max_chars: usize) -> String {
    let truncated: String = s.chars().take(max_chars).collect();
    truncated.replace('|', "\\|").replace('\n', " ")
}

/// Load a run directory's traces, join them against the record, and build
/// the question-by-question comparison.
pub fn compare(run_dir: &Path, record_path: &Path) -> Result<Comparison, String> {
    let record = HindsightRecord::load(record_path)?;
    let by_id = record.by_id();
    let traces = load_traces(run_dir)?;
    if traces.is_empty() {
        return Err(format!(
            "no traces found in {}",
            run_dir.join("traces").display()
        ));
    }
    if !traces
        .iter()
        .any(|t| by_id.contains_key(t.question_id.as_str()))
    {
        return Err(format!(
            "0 of {} run questions matched the Hindsight record — ids like `{}` are not \
             LongMemEval-S question ids (the record covers the dataset's 500 ids, so custom \
             fixtures such as the fullstack smoke do not join)",
            traces.len(),
            traces[0].question_id
        ));
    }

    let mut c = Comparison {
        run_dir: run_dir.to_path_buf(),
        record: record.meta.clone(),
        matched: 0,
        unmatched_run_ids: Vec::new(),
        record_not_run: 0,
        rustsight_unjudged: 0,
        both_correct: 0,
        shared_failures: 0,
        hindsight_only_correct: 0,
        rustsight_only_correct: 0,
        by_ability: Vec::new(),
        parity_gaps: Vec::new(),
        wins: Vec::new(),
        shared_failures_list: Vec::new(),
    };

    let mut ability_order: Vec<String> = Vec::new();
    let mut ability_stats: BTreeMap<String, AbilityComparison> = BTreeMap::new();

    for t in &traces {
        let Some(hs) = by_id.get(t.question_id.as_str()) else {
            c.unmatched_run_ids.push(t.question_id.clone());
            continue;
        };
        c.matched += 1;
        let rs_right = t.correct == Some(true);
        if t.correct.is_none() {
            c.rustsight_unjudged += 1;
        }
        let ability = if t.ability.is_empty() {
            ability_of(&hs.query_id, hs.question_type.as_deref())
                .as_str()
                .to_string()
        } else {
            t.ability.clone()
        };
        let entry = ability_stats
            .entry(ability.clone())
            .or_insert(AbilityComparison {
                ability,
                total: 0,
                rustsight_correct: 0,
                hindsight_correct: 0,
            });
        entry.total += 1;
        if rs_right {
            entry.rustsight_correct += 1;
        }
        if hs.correct {
            entry.hindsight_correct += 1;
        }

        let divergence = Divergence {
            question_id: t.question_id.clone(),
            ability: entry.ability.clone(),
            question: t.question.clone(),
            gold_answer: t.gold_answer.clone(),
            rustsight_answer: t.hypothesis.clone(),
            hindsight_answer: hs.answer.clone(),
            rustsight_error: t.error.clone(),
        };
        match (rs_right, hs.correct) {
            (true, true) => c.both_correct += 1,
            (true, false) => {
                c.rustsight_only_correct += 1;
                c.wins.push(divergence);
            }
            (false, true) => {
                c.hindsight_only_correct += 1;
                c.parity_gaps.push(divergence);
            }
            (false, false) => {
                c.shared_failures += 1;
                c.shared_failures_list.push(divergence);
            }
        }
        if !ability_order.contains(&t.ability) {
            ability_order.push(t.ability.clone());
        }
    }

    c.record_not_run = by_id
        .keys()
        .filter(|id| !traces.iter().any(|t| t.question_id == **id))
        .count();
    c.by_ability = ability_stats.into_values().collect();
    Ok(c)
}

fn load_traces(run_dir: &Path) -> Result<Vec<QuestionTrace>, String> {
    let traces_dir = run_dir.join("traces");
    let entries = std::fs::read_dir(&traces_dir)
        .map_err(|e| format!("cannot read {}: {e}", traces_dir.display()))?;
    let mut traces: Vec<QuestionTrace> = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        if let Ok(t) = serde_json::from_str(&content) {
            traces.push(t);
        }
    }
    traces.sort_by(|a, b| a.question_id.cmp(&b.question_id));
    Ok(traces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(id: &str, ability: &str, correct: Option<bool>, error: Option<&str>) -> QuestionTrace {
        QuestionTrace {
            question_id: id.into(),
            ability: ability.into(),
            question: format!("question {id}"),
            gold_answer: format!("gold {id}"),
            hypothesis: if error.is_none() {
                Some(format!("hyp {id}"))
            } else {
                None
            },
            correct,
            error: error.map(str::to_string),
            ..Default::default()
        }
    }

    fn write_run(dir: &Path, traces: &[QuestionTrace]) {
        let tdir = dir.join("traces");
        std::fs::create_dir_all(&tdir).unwrap();
        for t in traces {
            std::fs::write(
                tdir.join(format!("{}.json", t.question_id)),
                serde_json::to_string(t).unwrap(),
            )
            .unwrap();
        }
    }

    fn record_json(results: &str) -> String {
        format!(
            r#"{{"meta": {{"answer_llm": "test-answer", "judge_llm": "test-judge", "total_queries": 5, "correct": 3, "accuracy": 0.6, "source_repo": "r", "source_path": "p", "fetched": "2026-11-04"}}, "results": [{results}]}}"#
        )
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rs-eval-cmp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn matrix_abilities_and_divergences() {
        let dir = tmp_dir("matrix");
        write_run(
            &dir,
            &[
                trace("q1", "information_extraction", Some(true), None), // both right
                trace("q2", "information_extraction", Some(false), None), // gap
                trace("q3", "temporal_reasoning", Some(false), None),    // both wrong
                trace("q4", "temporal_reasoning", Some(true), None),     // win
                trace("q5_abs", "abstention", None, Some("reflect failed")), // gap (error)
                trace("smoke_only", "abstention", Some(true), None),     // unmatched
            ],
        );
        let rec = dir.join("record.json");
        std::fs::write(
            &rec,
            record_json(
                r#"
                {"query_id": "q1", "question_type": "single-session-user", "correct": true, "answer": "hs q1"},
                {"query_id": "q2", "question_type": "single-session-user", "correct": true, "answer": "hs q2"},
                {"query_id": "q3", "question_type": "temporal-reasoning", "correct": false, "answer": "hs q3"},
                {"query_id": "q4", "question_type": "temporal-reasoning", "correct": false, "answer": "hs q4"},
                {"query_id": "q5_abs", "question_type": "single-session-user", "correct": true, "answer": "hs q5"},
                {"query_id": "q6", "question_type": "multi-session", "correct": false, "answer": "hs q6"}
                "#,
            ),
        )
        .unwrap();

        let c = compare(&dir, &rec).unwrap();
        assert_eq!(c.matched, 5);
        assert_eq!(c.unmatched_run_ids, vec!["smoke_only".to_string()]);
        assert_eq!(c.record_not_run, 1);
        assert_eq!(c.rustsight_unjudged, 1);
        assert_eq!(c.both_correct, 1);
        assert_eq!(c.shared_failures, 1);
        assert_eq!(c.hindsight_only_correct, 2);
        assert_eq!(c.rustsight_only_correct, 1);

        let ie = c
            .by_ability
            .iter()
            .find(|a| a.ability == "information_extraction")
            .unwrap();
        assert_eq!(
            (ie.total, ie.rustsight_correct, ie.hindsight_correct),
            (2, 1, 2)
        );
        let tr = c
            .by_ability
            .iter()
            .find(|a| a.ability == "temporal_reasoning")
            .unwrap();
        assert_eq!(
            (tr.total, tr.rustsight_correct, tr.hindsight_correct),
            (2, 1, 0)
        );

        let gap_ids: Vec<&str> = c
            .parity_gaps
            .iter()
            .map(|d| d.question_id.as_str())
            .collect();
        assert_eq!(gap_ids, vec!["q2", "q5_abs"]);
        assert!(c.parity_gaps[1].rustsight_answer.is_none());
        assert_eq!(
            c.parity_gaps[1].rustsight_error.as_deref(),
            Some("reflect failed")
        );
        assert_eq!(c.wins[0].hindsight_answer.as_deref(), Some("hs q4"));
        assert_eq!(c.shared_failures_list[0].question_id, "q3");

        let md = c.render_markdown();
        assert!(md.contains("## Parity gaps — Hindsight correct, rustsight not (2)"));
        assert!(md.contains("smoke_only"));
        assert!(md.contains("Record questions not attempted in this run: 1"));
        assert!(md.contains("Unjudged/unanswered rustsight questions counted as wrong: 1"));
        assert!(md.contains("| temporal_reasoning | 2 | 1 | 0 | +1 q (+50.0 pp) |"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn record_ability_rollup_matches_dataset_rule() {
        let dir = tmp_dir("rollup");
        // Trace with empty ability forces the record-side rollup.
        write_run(&dir, &[trace("x_abs", "", Some(true), None)]);
        let rec = dir.join("record.json");
        std::fs::write(
            &rec,
            record_json(
                r#"{"query_id": "x_abs", "question_type": "multi-session", "correct": false}"#,
            ),
        )
        .unwrap();
        let c = compare(&dir, &rec).unwrap();
        assert_eq!(c.by_ability[0].ability, "abstention");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_traces_dir_is_an_error() {
        let dir = tmp_dir("missing");
        let rec = dir.join("record.json");
        std::fs::write(&rec, record_json(r#"{"query_id": "q1", "correct": true}"#)).unwrap();
        assert!(compare(&dir, &rec).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_matched_ids_is_a_clear_error() {
        let dir = tmp_dir("zero");
        write_run(&dir, &[trace("fss_q1", "abstention", Some(true), None)]);
        let rec = dir.join("record.json");
        std::fs::write(&rec, record_json(r#"{"query_id": "q1", "correct": true}"#)).unwrap();
        let err = compare(&dir, &rec).unwrap_err();
        assert!(err.contains("0 of 1 run questions matched"));
        assert!(err.contains("fss_q1"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The committed Hindsight record must stay loadable and self-consistent:
    /// this is what `compare` joins real runs against.
    #[test]
    fn committed_record_loads_and_is_consistent() {
        let record = HindsightRecord::load(&default_record_path()).unwrap();
        assert_eq!(record.results.len(), 500);
        assert_eq!(record.meta.total_queries, 500);
        assert_eq!(
            record.results.iter().filter(|r| r.correct).count(),
            record.meta.correct
        );
        assert!((record.meta.accuracy - 0.946).abs() < 1e-9);
        let mut ids: Vec<&str> = record.results.iter().map(|r| r.query_id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 500, "record query ids must be unique");
        assert!(record.results.iter().all(|r| r.question_type.is_some()));
        assert!(record.meta.answer_llm.contains("gemini"));
    }
}
