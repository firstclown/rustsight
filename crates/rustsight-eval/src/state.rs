//! Resumable run state, persisted as `state.json` in the run directory.
//! Units (conversation retained, question answered/judged) are marked only
//! after they complete, so a crashed run resumes where it left off.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunState {
    #[serde(default)]
    pub conversations: BTreeMap<String, ConversationState>,
    #[serde(default)]
    pub questions: BTreeMap<String, QuestionState>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ConversationState {
    pub retained: bool,
    pub consolidated: bool,
    /// True when the harness gave up on consolidation with a residual dirty
    /// queue (stalled passes) but kept the bank — it stays answerable, and
    /// a resume must NOT re-retain it (memories would duplicate).
    #[serde(default)]
    pub partial: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct QuestionState {
    pub answered: bool,
    pub judged: bool,
}

impl RunState {
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(content) if !content.trim().is_empty() => {
                serde_json::from_str(&content).map_err(|e| format!("state file corrupt: {e}"))
            }
            _ => Ok(Self::default()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let tmp = path.with_extension("json.tmp");
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("state serialize failed: {e}"))?;
        std::fs::write(&tmp, content).map_err(|e| format!("state write failed: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("state rename failed: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_missing_file() {
        let dir = std::env::temp_dir().join(format!("rs-eval-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");

        let mut state = RunState::default();
        state.conversations.insert(
            "lmes-x".into(),
            ConversationState {
                retained: true,
                consolidated: true,
                partial: false,
            },
        );
        state.questions.insert(
            "q1".into(),
            QuestionState {
                answered: true,
                judged: false,
            },
        );
        state.save(&path).unwrap();

        let loaded = RunState::load(&path).unwrap();
        assert!(loaded.conversations["lmes-x"].retained);
        assert!(loaded.questions["q1"].answered);
        assert!(!loaded.questions["q1"].judged);
        assert!(!loaded.questions.contains_key("q2"));

        std::fs::remove_file(&path).unwrap();
        assert!(RunState::load(&path).unwrap().questions.is_empty());
    }
}
