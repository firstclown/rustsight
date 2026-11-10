//! LLM judge: binary correctness verdicts in the paper's style
//! (arXiv:2512.12818 App. A.4, following the LongMemEval convention):
//! GPT-OSS-120b at temperature 0, final answer rendered as
//! `\boxed{yes}` / `\boxed{no}`. Abstention questions get a dedicated
//! "unanswerable question" prompt and are scored separately in the report.

use crate::counting::Usage;
use rustsight_core::traits::{ChatMessage, LlmProvider, LlmRequest, TokenEstimator};
use rustsight_providers::OpenAiCompat;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct JudgeEnv {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

fn env_required(name: &str) -> Result<String, String> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| format!("{name} is not set"))
}

impl JudgeEnv {
    /// `RUSTSIGHT_EVAL_JUDGE_BASE_URL` / `_API_KEY` / `_MODEL` — all three
    /// required (no silent fallback: a judge silently pointed at the
    /// extraction model would corrupt cross-model comparisons).
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            base_url: env_required("RUSTSIGHT_EVAL_JUDGE_BASE_URL")?,
            api_key: env_required("RUSTSIGHT_EVAL_JUDGE_API_KEY")?,
            model: env_required("RUSTSIGHT_EVAL_JUDGE_MODEL")?,
        })
    }

    pub fn build(&self, usage: Arc<Usage>, est: Arc<dyn TokenEstimator>) -> Result<Judge, String> {
        let client = rustsight_providers::http_client()?;
        let llm = Arc::new(OpenAiCompat {
            client,
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
        });
        Ok(Judge::Real { llm, usage, est })
    }
}

pub enum Judge {
    Real {
        llm: Arc<OpenAiCompat>,
        usage: Arc<Usage>,
        est: Arc<dyn TokenEstimator>,
    },
    /// Offline smoke mode: always returns the given verdict.
    Fixed { correct: bool },
}

#[derive(Debug, Clone)]
pub struct Verdict {
    pub correct: bool,
    pub parsed: bool,
    pub raw: String,
}

fn build_correctness_prompt(question: &str, gold: &str, hypothesis: &str) -> String {
    format!(
        "I will give you a question, a ground-truth answer, and a response from a model. \
Please answer yes if the model's response conveys the same information as the \
ground-truth answer, allowing for differences in wording, units, or formatting. \
Additional correct details do not make a response wrong. Otherwise answer no.

Question:
{question}

Ground-truth answer:
{gold}

Model response:
{hypothesis}

Does the model's response correctly answer the question? You may provide \
reasoning, but you MUST end your response with your final answer in the \
format: \\boxed{{yes}} or \\boxed{{no}}"
    )
}

fn build_abstention_prompt(question: &str, explanation: &str, hypothesis: &str) -> String {
    format!(
        "I will give you an unanswerable question, an explanation, and a response \
from a model. Please answer yes if the model correctly identifies the question \
as unanswerable. The model could say that the information is incomplete, or \
that some other information is given but the asked information is not. A \
response that fabricates the asked-for information must be answered no.

Question:
{question}

Explanation:
{explanation}

Model response:
{hypothesis}

Does the model correctly identify the question as unanswerable? You may \
provide reasoning, but you MUST end your response with your final answer in \
the format: \\boxed{{yes}} or \\boxed{{no}}"
    )
}

/// Parse a judge response: `\boxed{yes}`/`\boxed{no}` wins; otherwise fall
/// back to a trailing yes/no word. `None` means unparseable.
pub fn parse_verdict(raw: &str) -> Option<bool> {
    let lower = raw.to_lowercase();
    if lower.contains("\\boxed{yes}") {
        return Some(true);
    }
    if lower.contains("\\boxed{no}") {
        return Some(false);
    }
    let last = lower.split_whitespace().last()?;
    match last.trim_matches(['.', '!', ',', '"', '\'']) {
        "yes" | "yes." => Some(true),
        "no" | "no." => Some(false),
        _ => None,
    }
}

impl Judge {
    pub async fn judge(
        &self,
        question: &str,
        gold: &str,
        hypothesis: &str,
        abstention: bool,
    ) -> rustsight_core::Result<Verdict> {
        match self {
            Judge::Fixed { correct } => Ok(Verdict {
                correct: *correct,
                parsed: true,
                raw: format!("fixed judge: correct={correct}"),
            }),
            Judge::Real { llm, usage, est } => {
                let prompt = if abstention {
                    build_abstention_prompt(question, gold, hypothesis)
                } else {
                    build_correctness_prompt(question, gold, hypothesis)
                };
                let tokens_in = est.count(&prompt) as u64;
                let req = LlmRequest {
                    system: None,
                    messages: vec![ChatMessage::user(prompt)],
                    temperature: Some(0.0),
                    ..Default::default()
                };
                let resp = llm.complete(&req).await?;
                let raw = resp.content.unwrap_or_default();
                usage.add(tokens_in, est.count(&raw) as u64);
                match parse_verdict(&raw) {
                    Some(correct) => Ok(Verdict {
                        correct,
                        parsed: true,
                        raw,
                    }),
                    None => Ok(Verdict {
                        correct: false,
                        parsed: false,
                        raw,
                    }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustsight_providers::SimpleTokenEstimator;

    fn usage() -> Arc<Usage> {
        Arc::new(Usage::default())
    }

    #[test]
    fn parses_boxed_verdicts() {
        assert_eq!(parse_verdict("thinking... \\boxed{yes}"), Some(true));
        assert_eq!(parse_verdict("$\\boxed{no}$"), Some(false));
        assert_eq!(parse_verdict("\\boxed{YES}"), Some(true));
        assert_eq!(parse_verdict("I believe it is correct. no"), Some(false));
        assert_eq!(parse_verdict("I believe it is correct. yes."), Some(true));
        assert_eq!(parse_verdict("maybe"), None);
        assert_eq!(parse_verdict(""), None);
    }

    #[test]
    fn prompts_name_all_parts() {
        let p = build_correctness_prompt("Q?", "gold", "hyp");
        assert!(p.contains("Q?"));
        assert!(p.contains("gold"));
        assert!(p.contains("hyp"));
        assert!(p.contains("\\boxed{yes}"));
        let a = build_abstention_prompt("Q?", "why", "hyp");
        assert!(a.contains("unanswerable"));
        assert!(a.contains("\\boxed{no}"));
    }

    #[tokio::test]
    async fn fixed_judge_short_circuits() {
        let judge = Judge::Fixed { correct: true };
        let v = judge.judge("q", "g", "h", true).await.unwrap();
        assert!(v.correct);
        assert!(v.parsed);
    }

    #[tokio::test]
    async fn real_judge_parses_boxed_yes() {
        let env = JudgeEnv {
            base_url: "http://127.0.0.1:9".into(),
            api_key: "k".into(),
            model: "m".into(),
        };
        // Point at a closed port with a short timeout: the request fails,
        // proving the error path propagates without panicking.
        let judge = env.build(usage(), Arc::new(SimpleTokenEstimator)).unwrap();
        assert!(judge.judge("q", "g", "h", false).await.is_err());
    }
}
