//! `analyze`: cluster failures from a run directory so the next iteration
//! (prompt/fusion tweak) is informed by evidence, not vibes.

use crate::report::QuestionTrace;
use std::collections::BTreeMap;
use std::path::Path;

pub fn analyze(run_dir: &Path) -> Result<String, String> {
    let traces_dir = run_dir.join("traces");
    let mut traces: Vec<QuestionTrace> = Vec::new();
    let entries = std::fs::read_dir(&traces_dir)
        .map_err(|e| format!("cannot read {}: {e}", traces_dir.display()))?;
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
    if traces.is_empty() {
        return Err(format!("no traces found in {}", traces_dir.display()));
    }
    traces.sort_by(|a, b| a.question_id.cmp(&b.question_id));

    let mut out = String::new();
    let failed: Vec<&QuestionTrace> = traces.iter().filter(|t| t.failed()).collect();
    out.push_str(&format!(
        "{} traces, {} failed ({:.1}%)\n\n",
        traces.len(),
        failed.len(),
        failed.len() as f64 / traces.len() as f64 * 100.0
    ));

    let mut by_ability: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for t in &traces {
        let entry = by_ability.entry(t.ability.as_str()).or_insert((0, 0));
        entry.0 += 1;
        if t.failed() {
            entry.1 += 1;
        }
    }
    out.push_str("failures by ability:\n");
    for (ability, (total, fail)) in &by_ability {
        out.push_str(&format!(
            "  {ability:>26} {fail}/{total} ({:.0}%)\n",
            *fail as f64 / *total as f64 * 100.0
        ));
    }

    let empty_answer = traces
        .iter()
        .filter(|t| t.hypothesis.as_deref().unwrap_or("").trim().is_empty())
        .count();
    let no_citations = traces
        .iter()
        .filter(|t| t.hypothesis.is_some() && t.citations.is_empty())
        .count();
    let zero_retrieved = traces
        .iter()
        .filter(|t| t.hypothesis.is_some() && t.retrieved_ids.is_empty())
        .count();
    let judge_unparsed = traces
        .iter()
        .filter(|t| t.judge_parsed == Some(false))
        .count();
    out.push_str("\nsignals across all traces:\n");
    out.push_str(&format!("  empty answer:        {empty_answer}\n"));
    out.push_str(&format!("  no citations:        {no_citations}\n"));
    out.push_str(&format!("  nothing retrieved:   {zero_retrieved}\n"));
    out.push_str(&format!("  judge unparseable:   {judge_unparsed}\n"));
    out.push_str(&format!(
        "  answer errors:       {}\n",
        traces.iter().filter(|t| t.error.is_some()).count()
    ));

    out.push_str("\nfailed questions (id | ability | signal | gold => hypothesis):\n");
    for t in failed.iter().take(40) {
        let signal = if let Some(err) = &t.error {
            format!("error: {err}")
        } else if t.judge_parsed == Some(false) {
            "judge-unparsed".to_string()
        } else if t.citations.is_empty() {
            "no-citations".to_string()
        } else {
            "wrong-answer".to_string()
        };
        let hyp: String = t
            .hypothesis
            .as_deref()
            .unwrap_or("(none)")
            .chars()
            .take(90)
            .collect();
        out.push_str(&format!(
            "  {} | {} | {} | {} => {}\n",
            t.question_id, t.ability, signal, t.gold_answer, hyp
        ));
    }
    if failed.len() > 40 {
        out.push_str(&format!("  ... and {} more\n", failed.len() - 40));
    }
    Ok(out)
}
