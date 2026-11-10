//! LongMemEval-S ingestion: parse the HuggingFace JSON, roll question types
//! up into the paper's five abilities, group questions into conversations
//! (one memory bank each), and synthesize per-session timestamps.
//!
//! Dataset facts (verified against the LongMemEval README, repo
//! `xiaowu0162/LongMemEval`; the `xiamen-unity/LongMemEval` repo named in
//! RS-6 is its deprecated mirror):
//! - `longmemeval_s.json`: ~500 questions, ~115k tokens / ~40-50 sessions
//!   per conversation, sessions sorted by timestamp but carrying none.
//! - Fields: `question_id` (an `_abs` suffix marks an abstention question),
//!   `question_type` ∈ {single-session-user, single-session-assistant,
//!   single-session-preference, temporal-reasoning, knowledge-update,
//!   multi-session}, `question`, `answer` (string or number), `question_date`,
//!   `haystack_session_ids`, `haystack_sessions` (list of sessions, each a
//!   list of `{role, content[, has_answer]}` turns), `answer_session_ids`.
//! - Questions sharing a `haystack_session_ids` list share one conversation;
//!   the harness retains each conversation exactly once.

use chrono::{Datelike, Duration, NaiveDate};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;

pub const DEFAULT_VARIANT: &str = "longmemeval-s";
pub const DEFAULT_DATASET_FILE: &str = "longmemeval_s.json";

/// Escapes one raw control byte the way JSON requires.
fn push_escaped(b: u8, out: &mut Vec<u8>) {
    match b {
        b'\n' => out.extend_from_slice(b"\\n"),
        b'\r' => out.extend_from_slice(b"\\r"),
        b'\t' => out.extend_from_slice(b"\\t"),
        0x08 => out.extend_from_slice(b"\\b"),
        0x0C => out.extend_from_slice(b"\\f"),
        _ => {
            out.extend_from_slice(b"\\u00");
            const HEX: &[u8; 16] = b"0123456789abcdef";
            out.push(HEX[(b >> 4) as usize]);
            out.push(HEX[(b & 0xF) as usize]);
        }
    }
}

/// Streaming JSON pre-filter for the real dataset, which embeds raw
/// newlines inside string values (rejected by strict parsers like
/// serde_json). Escapes control bytes that occur *inside string tokens*
/// and passes everything else — including structural newlines — through
/// untouched. UTF-8-safe: multi-byte sequences never contain ASCII bytes,
/// so byte-level quote/backslash tracking is sound.
pub struct SanitizingReader<R: Read> {
    inner: R,
    in_string: bool,
    escaped: bool,
    out: Vec<u8>,
    pos: usize,
}

impl<R: Read> SanitizingReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            in_string: false,
            escaped: false,
            out: Vec::new(),
            pos: 0,
        }
    }

    fn step(&mut self, b: u8) {
        let out = &mut self.out;
        if !self.in_string {
            out.push(b);
            if b == b'"' {
                self.in_string = true;
                self.escaped = false;
            }
            return;
        }
        if self.escaped {
            out.push(b);
            self.escaped = false;
            return;
        }
        match b {
            b'\\' => {
                out.push(b);
                self.escaped = true;
            }
            b'"' => {
                out.push(b);
                self.in_string = false;
            }
            0x00..=0x1F => push_escaped(b, out),
            _ => out.push(b),
        }
    }
}

impl<R: Read> Read for SanitizingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.pos >= self.out.len() {
            if buf.is_empty() {
                return Ok(0);
            }
            let mut chunk = [0u8; 64 * 1024];
            let n = self.inner.read(&mut chunk)?;
            if n == 0 {
                return Ok(0);
            }
            self.out.clear();
            self.pos = 0;
            for &b in &chunk[..n] {
                self.step(b);
            }
        }
        let n = (self.out.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&self.out[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ability {
    InformationExtraction,
    MultiSessionReasoning,
    TemporalReasoning,
    KnowledgeUpdate,
    Abstention,
}

impl Ability {
    pub fn as_str(self) -> &'static str {
        match self {
            Ability::InformationExtraction => "information_extraction",
            Ability::MultiSessionReasoning => "multi_session_reasoning",
            Ability::TemporalReasoning => "temporal_reasoning",
            Ability::KnowledgeUpdate => "knowledge_update",
            Ability::Abstention => "abstention",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawTurn {
    role: String,
    content: String,
    #[serde(default)]
    #[allow(dead_code)]
    has_answer: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawQuestion {
    question_id: String,
    #[serde(default)]
    question_type: Option<String>,
    question: String,
    #[serde(default)]
    answer: Option<Value>,
    #[serde(default)]
    question_date: Option<String>,
    #[serde(default)]
    haystack_session_ids: Option<Vec<Value>>,
    haystack_sessions: Vec<Vec<RawTurn>>,
    #[serde(default)]
    answer_session_ids: Option<Vec<Value>>,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub question_id: String,
    pub bank_id: String,
    pub question: String,
    pub gold_answer: String,
    pub question_date: Option<NaiveDate>,
    pub ability: Ability,
    pub evidence_session_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Conversation {
    pub bank_id: String,
    pub sessions: Vec<String>,
    pub anchor_date: NaiveDate,
    pub questions: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub questions: Vec<Question>,
    pub conversations: Vec<Conversation>,
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn parse_date(raw: &str) -> Option<NaiveDate> {
    let normalized = raw.trim().replace('/', "-");
    NaiveDate::parse_from_str(&normalized, "%Y-%m-%d").ok()
}

/// Roll a raw LongMemEval question into the paper's five abilities. Public
/// so the Hindsight-record comparison (`compare.rs`) maps record entries
/// onto the same rollup the run's traces use.
pub fn ability_of(question_id: &str, question_type: Option<&str>) -> Ability {
    if question_id.ends_with("_abs") {
        return Ability::Abstention;
    }
    match question_type.unwrap_or_default() {
        "multi-session" => Ability::MultiSessionReasoning,
        "temporal-reasoning" => Ability::TemporalReasoning,
        "knowledge-update" => Ability::KnowledgeUpdate,
        _ => Ability::InformationExtraction,
    }
}

fn render_session(turns: &[RawTurn]) -> String {
    turns
        .iter()
        .map(|t| {
            let role = match t.role.as_str() {
                "assistant" => "Assistant",
                _ => "User",
            };
            format!("{role}: {}", t.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn conversation_key(ids: Option<&Vec<Value>>, sessions: &[Vec<RawTurn>]) -> String {
    let parts: Vec<String> = match ids {
        Some(ids) if !ids.is_empty() => ids.iter().map(value_to_string).collect(),
        _ => {
            // Fall back to hashing the sessions themselves so grouping still
            // dedupes when a variant omits the id list.
            sessions
                .iter()
                .map(|s| {
                    let text = render_session(s);
                    let digest = Sha256::digest(text.as_bytes());
                    format!("s:{digest:02x}")
                })
                .collect()
        }
    };
    serde_json::to_string(&parts).unwrap_or_default()
}

fn bank_id_for(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    let prefix = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("sha256 digest has at least 8 bytes"),
    );
    format!("lmes-{prefix:016x}")
}

/// Synthesize one timestamp per session: session `i` of `n` lands
/// `spacing_days * (n - i)` before the anchor date (i.e. sessions are
/// strictly increasing in time and the last one hits the anchor), all at
/// 12:00 UTC.
pub fn session_timestamps(n: usize, anchor: NaiveDate, spacing_days: i64) -> Vec<i64> {
    let spacing = spacing_days.max(0);
    (1..=n)
        .map(|i| {
            let date = anchor - Duration::days(spacing * (n as i64 - i as i64));
            date.and_hms_opt(12, 0, 0)
                .map(|t| t.and_utc().timestamp_millis())
                .unwrap_or_else(|| {
                    anchor
                        .and_hms_opt(12, 0, 0)
                        .map(|t| t.and_utc().timestamp_millis())
                        .unwrap_or(0)
                })
        })
        .collect()
}

/// Midday timestamp for a question date; `None` when the dataset entry
/// carries no usable date.
pub fn question_timestamp(date: NaiveDate) -> i64 {
    date.and_hms_opt(12, 0, 0)
        .map(|t| t.and_utc().timestamp_millis())
        .unwrap_or(0)
}

pub fn date_string(date: NaiveDate) -> String {
    format!("{:04}-{:02}-{:02}", date.year(), date.month(), date.day())
}

pub fn parse_dataset(content: &str) -> Result<Dataset, String> {
    let raw: Vec<RawQuestion> =
        serde_json::from_str(content).map_err(|e| format!("dataset parse failed: {e}"))?;
    build_dataset(raw)
}

pub fn parse_dataset_reader<R: Read>(reader: R) -> Result<Dataset, String> {
    let raw: Vec<RawQuestion> = serde_json::from_reader(SanitizingReader::new(reader))
        .map_err(|e| format!("dataset parse failed: {e}"))?;
    build_dataset(raw)
}

fn build_dataset(raw: Vec<RawQuestion>) -> Result<Dataset, String> {
    if raw.is_empty() {
        return Err("dataset contains no questions".into());
    }
    let mut questions: Vec<Question> = Vec::with_capacity(raw.len());
    let mut conv_index: HashMap<String, usize> = HashMap::new();
    let mut conversations: Vec<Conversation> = Vec::new();

    for (qi, q) in raw.into_iter().enumerate() {
        let key = conversation_key(q.haystack_session_ids.as_ref(), &q.haystack_sessions);
        let bank_id = bank_id_for(&key);
        let date = q.question_date.as_deref().and_then(parse_date);
        let ability = ability_of(&q.question_id, q.question_type.as_deref());
        let idx = *conv_index.entry(key).or_insert_with(|| {
            let anchor =
                date.unwrap_or_else(|| NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"));
            // The real dataset intersperses zero-turn placeholder sessions
            // (1,230 of them across the file); they carry no content, and
            // the engine rejects empty retain payloads, so drop them here.
            let sessions: Vec<String> = q
                .haystack_sessions
                .iter()
                .map(|s| render_session(s))
                .filter(|s| !s.trim().is_empty())
                .collect();
            conversations.push(Conversation {
                bank_id: bank_id.clone(),
                sessions,
                anchor_date: anchor,
                questions: vec![],
            });
            conversations.len() - 1
        });
        // The anchor must not predate any question in the conversation;
        // track the max question date.
        if let Some(d) = date {
            if d > conversations[idx].anchor_date {
                conversations[idx].anchor_date = d;
            }
        }
        conversations[idx].questions.push(qi);
        questions.push(Question {
            question_id: q.question_id,
            bank_id,
            question: q.question,
            gold_answer: q.answer.as_ref().map(value_to_string).unwrap_or_default(),
            question_date: date,
            ability,
            evidence_session_ids: q
                .answer_session_ids
                .unwrap_or_default()
                .iter()
                .map(value_to_string)
                .collect(),
        });
    }

    Ok(Dataset {
        questions,
        conversations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> String {
        let session = r#"[{"role":"user","content":"Alice joined Google. Alice works on Search."},{"role":"assistant","content":"Nice!"}]"#;
        format!(
            r#"[
  {{"question_id":"abc_1","question_type":"single-session-user","question":"Where does Alice work?","answer":"Google","question_date":"2024/05/10","haystack_session_ids":[3,7],"haystack_sessions":[{session},{session}],"answer_session_ids":[7]}},
  {{"question_id":"abc_2_abs","question_type":"single-session-user","question":"What is Alice's salary?","answer":"Not mentioned","question_date":"2024-05-12","haystack_session_ids":[3,7],"haystack_sessions":[{session},{session}],"answer_session_ids":[]}},
  {{"question_id":"def_1","question_type":"temporal-reasoning","question":"What did Bob do last month?","answer":42,"question_date":"2024-06-01","haystack_session_ids":["s1"],"haystack_sessions":[[{{"role":"user","content":"Bob started running. Bob runs marathons."}}]],"answer_session_ids":["s1"]}}
]"#
        )
    }

    #[test]
    fn parses_groups_and_rolls_up() {
        let ds = parse_dataset(&fixture()).unwrap();
        assert_eq!(ds.questions.len(), 3);
        assert_eq!(ds.conversations.len(), 2);

        let q0 = &ds.questions[0];
        assert_eq!(q0.ability, Ability::InformationExtraction);
        assert_eq!(q0.gold_answer, "Google");
        assert_eq!(q0.question_date, parse_date("2024-05-10"));
        assert_eq!(q0.evidence_session_ids, vec!["7".to_string()]);

        let q1 = &ds.questions[1];
        assert_eq!(q1.ability, Ability::Abstention);
        assert_eq!(q1.bank_id, q0.bank_id);

        let q2 = &ds.questions[2];
        assert_eq!(q2.ability, Ability::TemporalReasoning);
        assert_eq!(q2.gold_answer, "42");

        let conv_a = ds
            .conversations
            .iter()
            .find(|c| c.bank_id == q0.bank_id)
            .unwrap();
        assert_eq!(conv_a.sessions.len(), 2);
        assert!(conv_a.sessions[0].starts_with("User: Alice joined Google."));
        assert!(conv_a.sessions[0].contains("\nAssistant: Nice!"));
        assert_eq!(conv_a.questions, vec![0, 1]);
        // anchor = max question date in the conversation
        assert_eq!(
            conv_a.anchor_date,
            parse_date("2024-05-12").expect("valid date")
        );
    }

    #[test]
    fn timestamps_increase_and_end_at_anchor() {
        let anchor = parse_date("2024-05-12").unwrap();
        let ts = session_timestamps(3, anchor, 2);
        assert_eq!(ts.len(), 3);
        assert!(ts.windows(2).all(|w| w[0] < w[1]));
        let last = chrono::DateTime::from_timestamp_millis(ts[2]).unwrap();
        assert_eq!(last.format("%Y-%m-%d").to_string(), "2024-05-12");
        let first = chrono::DateTime::from_timestamp_millis(ts[0]).unwrap();
        assert_eq!(first.format("%Y-%m-%d").to_string(), "2024-05-08");
    }

    #[test]
    fn rejects_empty_dataset() {
        assert!(parse_dataset("[]").is_err());
        assert!(parse_dataset("not json").is_err());
    }

    #[test]
    fn zero_turn_sessions_are_dropped() {
        let json = r#"[
  {"question_id":"q1","question_type":"multi-session","question":"q?","answer":"a","question_date":"2024-05-10","haystack_session_ids":[1,2,3],"haystack_sessions":[[],[{"role":"user","content":"Real content."}],[]],"answer_session_ids":[2]}
]"#;
        let ds = parse_dataset(json).unwrap();
        assert_eq!(ds.conversations.len(), 1);
        assert_eq!(
            ds.conversations[0].sessions,
            vec!["User: Real content.".to_string()]
        );
    }

    #[test]
    fn sanitizing_reader_escapes_control_chars_inside_strings_only() {
        // Raw newline + tab inside a string value; structural newlines and
        // the escaped \n sequence must pass through untouched.
        let dirty = b"{\"a\": \"line1\nline2\tend\\\\nkept\",\n  \"b\": 1\n}";
        let mut clean = Vec::new();
        SanitizingReader::new(std::io::Cursor::new(dirty))
            .read_to_end(&mut clean)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&clean).unwrap();
        assert_eq!(parsed["a"], "line1\nline2\tend\\nkept");
        assert_eq!(parsed["b"], 1);
        // structural newlines survived; each in-string control char gained one byte
        assert_eq!(clean.len(), dirty.len() + 2);
    }

    #[test]
    fn sanitizing_reader_handles_quotes_after_escapes() {
        // A backslash-escaped quote must not close the string, so the raw
        // control char after it still gets escaped.
        let dirty = b"[\"a \\\" quoted\n tail\"]";
        let mut clean = Vec::new();
        SanitizingReader::new(std::io::Cursor::new(dirty))
            .read_to_end(&mut clean)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&clean).unwrap();
        assert_eq!(parsed[0], "a \" quoted\n tail");
    }

    #[test]
    fn parse_dataset_reader_accepts_dirty_json() {
        let dirty = "[{\"question_id\":\"q1\",\"question_type\":\"multi-session\",\"question\":\"a\nb\",\"answer\":\"x\",\"question_date\":\"2024-05-10\",\"haystack_session_ids\":[1],\"haystack_sessions\":[[{\"role\":\"user\",\"content\":\"hi\nthere\"}]],\"answer_session_ids\":[1]}]";
        let ds = parse_dataset_reader(std::io::Cursor::new(dirty.as_bytes().to_vec())).unwrap();
        assert_eq!(ds.questions.len(), 1);
        // the raw newline inside question/content survived as a real \n
        assert_eq!(ds.questions[0].question, "a\nb");
        assert!(ds.conversations[0].sessions[0].contains("hi\nthere"));
    }
}
