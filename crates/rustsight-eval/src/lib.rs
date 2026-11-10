//! rustsight-eval: LongMemEval-S harness for the rustsight engine (RS-6).
//!
//! Runs the paper's protocol end-to-end: retain each conversation's session
//! transcripts into a per-conversation bank, consolidate, answer the 500
//! questions via `reflect`, and score with a binary LLM judge
//! (GPT-OSS-120b at temperature 0, `\boxed{yes}`/`\boxed{no}` verdicts,
//! abstention scored with a dedicated unanswerable-question prompt).
//!
//! Methodology notes:
//! - Dataset: original `longmemeval_s.json` from HuggingFace
//!   `xiaowu0162/longmemeval` (the `xiamen-unity/LongMemEval` repo named in
//!   the ticket is its deprecated mirror). The paper's S setting: ~115k
//!   tokens, ~40-50 sessions per conversation, 500 questions.
//! - Sessions carry no timestamps; the harness synthesizes one per session
//!   (`--session-spacing-days`, default 1) ending at the conversation's
//!   latest `question_date`, and passes it as the retain `timestamp_ms`.
//! - The engine clock is a `FixedClock` anchored per question so temporal
//!   parsing and `mentioned_at` bookkeeping line up with `question_date`.
//! - Judge prompts follow the paper's Appendix A.4 (LongMemEval
//!   convention); reference scores to compare against (paper Table 3):
//!   full-context OSS-20B 39.0%, Hindsight OSS-20B 83.6%, OSS-120B 89.0%,
//!   Gemini-3-Pro head 91.4%.

pub mod analyze;
pub mod compare;
pub mod counting;
pub mod dataset;
pub mod download;
pub mod judge;
pub mod report;
pub mod runner;
pub mod state;
