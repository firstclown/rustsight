use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("storage error: {0}")]
    Store(String),
    #[error("llm provider error: {0}")]
    Llm(String),
    #[error("llm provider error (HTTP {status}): {message}")]
    LlmStatus { status: u16, message: String },
    #[error("embedding provider error: {0}")]
    Embed(String),
    #[error("embedding provider error (HTTP {status}): {message}")]
    EmbedStatus { status: u16, message: String },
    #[error("rerank provider error: {0}")]
    Rerank(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("invalid input: {0}")]
    BadInput(String),
    #[error("not found: {0}")]
    NotFound(String),
}

impl Error {
    /// Whether this failure is transient and worth retrying: provider calls
    /// that surfaced an HTTP 429 (rate limit) or 5xx (server error) status.
    /// Everything else (4xx, parse errors, storage errors) is permanent for
    /// the purposes of a retry loop.
    pub fn is_transient(&self) -> bool {
        match self {
            Error::LlmStatus { status, .. } | Error::EmbedStatus { status, .. } => {
                *status == 429 || (500..=599).contains(status)
            }
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_statuses() {
        assert!(Error::LlmStatus {
            status: 429,
            message: "rate limited".into()
        }
        .is_transient());
        assert!(Error::LlmStatus {
            status: 503,
            message: "upstream down".into()
        }
        .is_transient());
        assert!(Error::EmbedStatus {
            status: 500,
            message: "boom".into()
        }
        .is_transient());
    }

    #[test]
    fn permanent_errors_are_not_transient() {
        assert!(!Error::LlmStatus {
            status: 400,
            message: "bad request".into()
        }
        .is_transient());
        assert!(!Error::LlmStatus {
            status: 401,
            message: "unauthorized".into()
        }
        .is_transient());
        assert!(!Error::Llm("connection reset".into()).is_transient());
        assert!(!Error::Parse("truncated json".into()).is_transient());
        assert!(!Error::Store("locked".into()).is_transient());
    }
}
