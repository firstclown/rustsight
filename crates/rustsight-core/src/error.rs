use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("storage error: {0}")]
    Store(String),
    #[error("llm provider error: {0}")]
    Llm(String),
    #[error("embedding provider error: {0}")]
    Embed(String),
    #[error("rerank provider error: {0}")]
    Rerank(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("invalid input: {0}")]
    BadInput(String),
    #[error("not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, Error>;
