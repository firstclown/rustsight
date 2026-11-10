use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("storage error: {0}")]
    Store(String),
    #[error("llm provider error: {0}")]
    Llm(String),
    #[error("llm provider error (HTTP {status}): {message}")]
    LlmStatus { status: u16, message: String },
    /// Transport-level LLM failure (connection dropped mid-body, timeout,
    /// TLS reset): the request never received a protocol-level answer, so
    /// the full cause chain is preserved in the message for diagnosis.
    #[error("llm transport error: {0}")]
    LlmTransport(String),
    #[error("embedding provider error: {0}")]
    Embed(String),
    #[error("embedding provider error (HTTP {status}): {message}")]
    EmbedStatus { status: u16, message: String },
    /// Embedding twin of [`Error::LlmTransport`].
    #[error("embedding transport error: {0}")]
    EmbedTransport(String),
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
    /// Whether this failure is transient and worth retrying:
    /// - provider calls that surfaced an HTTP 429 (rate limit) or 5xx
    ///   (server error) status, and
    /// - transport-level failures (dropped connection, timeout), where the
    ///   call never got a protocol answer — provider calls are effectively
    ///   idempotent, so re-issuing them is safe.
    ///
    /// Everything else (4xx, parse errors, storage errors) is permanent for
    /// the purposes of a retry loop.
    pub fn is_transient(&self) -> bool {
        match self {
            Error::LlmStatus { status, .. } | Error::EmbedStatus { status, .. } => {
                *status == 429 || (500..=599).contains(status)
            }
            Error::LlmTransport(_) | Error::EmbedTransport(_) => true,
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

    #[test]
    fn transport_failures_are_transient() {
        assert!(Error::LlmTransport(
            "error decoding response body: request or response body error: operation timed out"
                .into()
        )
        .is_transient());
        assert!(Error::EmbedTransport(
            "error decoding response body: end of file before message length reached".into()
        )
        .is_transient());
    }
}
