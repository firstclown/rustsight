//! Per-question traces and the run report (markdown + JSON).

use crate::counting::CountersSnapshot;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One file per question under `<run>/traces/<question_id>.json`. Written by
/// the answer phase, extended by the judge phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuestionTrace {
    pub question_id: String,
    pub bank_id: String,
    pub ability: String,
    pub abstention: bool,
    pub question: String,
    pub gold_answer: String,
    pub evidence_session_ids: Vec<String>,
    pub hypothesis: Option<String>,
    pub citations: Vec<String>,
    pub retrieved_ids: Vec<String>,
    pub retrieved: usize,
    pub rounds: usize,
    pub answer_ms: u64,
    pub error: Option<String>,
    pub judge_raw: Option<String>,
    pub judge_parsed: Option<bool>,
    pub correct: Option<bool>,
    pub judge_ms: Option<u64>,
}

impl QuestionTrace {
    pub fn failed(&self) -> bool {
        self.correct == Some(false) || self.error.is_some()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AbilityStat {
    pub ability: String,
    pub total: usize,
    pub answered: usize,
    pub judged: usize,
    pub correct: usize,
    pub errors: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportSummary {
    pub total: usize,
    pub answered: usize,
    pub judged: usize,
    pub correct: usize,
    pub errors: usize,
    pub parse_failures: usize,
    pub accuracy: f64,
    pub by_ability: Vec<AbilityStat>,
}

pub fn aggregate(traces: &[QuestionTrace]) -> ReportSummary {
    let mut abilities: Vec<&str> = Vec::new();
    for t in traces {
        if !abilities.contains(&t.ability.as_str()) {
            abilities.push(&t.ability);
        }
    }
    abilities.sort_unstable();

    let by_ability = abilities
        .into_iter()
        .map(|a| {
            let group: Vec<&QuestionTrace> = traces.iter().filter(|t| t.ability == a).collect();
            let total = group.len();
            let answered = group.iter().filter(|t| t.hypothesis.is_some()).count();
            let judged = group.iter().filter(|t| t.correct.is_some()).count();
            let correct = group.iter().filter(|t| t.correct == Some(true)).count();
            let errors = group.iter().filter(|t| t.error.is_some()).count();
            AbilityStat {
                ability: a.to_string(),
                total,
                answered,
                judged,
                correct,
                errors,
                accuracy: if judged > 0 {
                    correct as f64 / judged as f64
                } else {
                    0.0
                },
            }
        })
        .collect();

    let total = traces.len();
    let answered = traces.iter().filter(|t| t.hypothesis.is_some()).count();
    let judged = traces.iter().filter(|t| t.correct.is_some()).count();
    let correct = traces.iter().filter(|t| t.correct == Some(true)).count();
    let errors = traces.iter().filter(|t| t.error.is_some()).count();
    let parse_failures = traces
        .iter()
        .filter(|t| t.judge_parsed == Some(false))
        .count();
    ReportSummary {
        total,
        answered,
        judged,
        correct,
        errors,
        parse_failures,
        accuracy: if judged > 0 {
            correct as f64 / judged as f64
        } else {
            0.0
        },
        by_ability,
    }
}

/// Paper Table 3 reference points (arXiv:2512.12818).
pub const PAPER_REFERENCES: &str =
    "full-context OSS-20B 39.0% · Hindsight OSS-20B 83.6% · Hindsight OSS-120B 89.0% · Hindsight + Gemini-3-Pro head 91.4%";

/// Pointer to the per-question Hindsight record and the parity command.
pub const HINDSIGHT_RECORD_NOTE: &str =
    "per-question Hindsight record (AMB run, 94.6%): `cargo run -p rustsight-eval -- compare --run <dir>`";

pub fn render_markdown(
    summary: &ReportSummary,
    counters: &CountersSnapshot,
    run_info: &serde_json::Value,
    elapsed_secs: u64,
) -> String {
    let mut md = String::new();
    md.push_str("# LongMemEval-S run\n\n");
    md.push_str("```json\n");
    md.push_str(&serde_json::to_string_pretty(run_info).unwrap_or_default());
    md.push_str("\n```\n\n");
    md.push_str(&format!(
        "Wall time: {}h {:02}m {:02}s\n\n",
        elapsed_secs / 3600,
        (elapsed_secs % 3600) / 60,
        elapsed_secs % 60
    ));
    md.push_str("Estimated traffic (SimpleTokenEstimator, chars/4):\n\n");
    md.push_str(&format!(
        "| stream | requests | in tokens | out tokens |\n|---|---|---|---|\n| llm | {} | {} | {} |\n| embed | {} | {} | 0 |\n| judge | {} | {} | {} |\n\n",
        counters.llm.requests,
        counters.llm.input_tokens,
        counters.llm.output_tokens,
        counters.embed.requests,
        counters.embed.input_tokens,
        counters.judge.requests,
        counters.judge.input_tokens,
        counters.judge.output_tokens,
    ));
    md.push_str("| ability | judged/total | correct | accuracy |\n|---|---|---|---|\n");
    for a in &summary.by_ability {
        md.push_str(&format!(
            "| {} | {}/{} | {} | {:.1}% |\n",
            a.ability,
            a.judged,
            a.total,
            a.correct,
            a.accuracy * 100.0
        ));
    }
    md.push_str(&format!(
        "| **overall** | {}/{} | {} | **{:.1}%** |\n\n",
        summary.judged,
        summary.total,
        summary.correct,
        summary.accuracy * 100.0
    ));
    if summary.parse_failures > 0 {
        md.push_str(&format!(
            "Judge parse failures: {} (counted incorrect; see traces)\n\n",
            summary.parse_failures
        ));
    }
    if summary.errors > 0 {
        md.push_str(&format!(
            "Answer errors: {} (see traces for messages)\n\n",
            summary.errors
        ));
    }
    md.push_str(&format!("Paper reference: {PAPER_REFERENCES}\n"));
    md.push_str(&format!("Hindsight record: {HINDSIGHT_RECORD_NOTE}\n"));
    md
}

pub fn write_report(
    dir: &Path,
    summary: &ReportSummary,
    counters: &CountersSnapshot,
    run_info: &serde_json::Value,
    elapsed_secs: u64,
) -> Result<(), String> {
    let json = serde_json::json!({
        "summary": summary,
        "counters": counters,
        "run": run_info,
        "elapsed_secs": elapsed_secs,
        "paper_references": PAPER_REFERENCES,
    });
    std::fs::write(
        dir.join("report.json"),
        serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("report.json write failed: {e}"))?;
    std::fs::write(
        dir.join("report.md"),
        render_markdown(summary, counters, run_info, elapsed_secs),
    )
    .map_err(|e| format!("report.md write failed: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(id: &str, ability: &str, correct: Option<bool>, error: Option<&str>) -> QuestionTrace {
        QuestionTrace {
            question_id: id.into(),
            ability: ability.into(),
            hypothesis: if error.is_none() {
                Some("hyp".into())
            } else {
                None
            },
            correct,
            error: error.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn aggregates_by_ability() {
        let traces = vec![
            trace("q1", "information_extraction", Some(true), None),
            trace("q2", "information_extraction", Some(false), None),
            trace("q3", "abstention", Some(true), None),
            trace("q4", "temporal_reasoning", None, Some("reflect failed")),
            trace("q5", "abstention", Some(false), None),
        ];
        let s = aggregate(&traces);
        assert_eq!(s.total, 5);
        assert_eq!(s.answered, 4);
        assert_eq!(s.judged, 4);
        assert_eq!(s.correct, 2);
        assert_eq!(s.accuracy, 0.5);
        assert_eq!(s.errors, 1);
        let ie = s
            .by_ability
            .iter()
            .find(|a| a.ability == "information_extraction")
            .unwrap();
        assert_eq!(ie.accuracy, 0.5);
        let ab = s
            .by_ability
            .iter()
            .find(|a| a.ability == "abstention")
            .unwrap();
        assert_eq!(ab.correct, 1);
        let tr = s
            .by_ability
            .iter()
            .find(|a| a.ability == "temporal_reasoning")
            .unwrap();
        assert_eq!(tr.judged, 0);
        assert_eq!(tr.errors, 1);
    }

    #[test]
    fn markdown_renders_tables() {
        let s = aggregate(&[trace("q1", "abstention", Some(true), None)]);
        let md = render_markdown(
            &s,
            &CountersSnapshot::default(),
            &serde_json::json!({"mock": true}),
            65,
        );
        assert!(md.contains("| ability |"));
        assert!(md.contains("**overall**"));
        assert!(md.contains("0h 01m 05s"));
        assert!(md.contains("83.6%"));
        assert!(md.contains("compare --run"));
    }
}
