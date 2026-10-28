//! Bearer-token authentication for the HTTP API (CR-1).
//!
//! The token comes from `RUSTSIGHT_API_TOKEN` (or a file named by
//! `RUSTSIGHT_API_TOKEN_FILE`) — never a CLI flag, since flags leak via
//! `ps`. Unset/empty means no auth, which is only acceptable on loopback
//! binds (see [`crate::http::serve`], which refuses non-loopback binds
//! without a token unless explicitly overridden).
//!
//! The MCP stdio server deliberately needs no auth: the parent process
//! controls the pipe, so it is not an attack surface here. Don't "fix"
//! that later.

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiAuth {
    None,
    Token(String),
}

impl ApiAuth {
    /// Token from `RUSTSIGHT_API_TOKEN`, falling back to the file named by
    /// `RUSTSIGHT_API_TOKEN_FILE` (whole file, trimmed). Unset or empty
    /// values (both vars) mean no authentication.
    pub fn from_env() -> Self {
        if let Some(t) = token_from_var("RUSTSIGHT_API_TOKEN") {
            return Self::Token(t);
        }
        if let Some(t) = token_from_var("RUSTSIGHT_API_TOKEN_FILE")
            .and_then(|path| std::fs::read_to_string(path.trim()).ok())
            .and_then(token_value)
        {
            return Self::Token(t);
        }
        Self::None
    }
}

fn token_from_var(name: &str) -> Option<String> {
    std::env::var(name).ok().and_then(token_value)
}

fn token_value(v: String) -> Option<String> {
    let t = v.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// Byte-slice equality whose comparison time does not depend on *where*
/// the slices differ (the classic XOR fold). The early length check only
/// reveals lengths, never token content.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Axum middleware: with `ApiAuth::Token`, every request must carry
/// `Authorization: Bearer <token>` or it is rejected with 401. With
/// `ApiAuth::None` (loopback-only setups) requests pass through.
pub async fn require_token(State(auth): State<Arc<ApiAuth>>, req: Request, next: Next) -> Response {
    match &*auth {
        ApiAuth::None => next.run(req).await,
        ApiAuth::Token(expected) => {
            let provided = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(str::as_bytes);
            match provided {
                Some(p) if constant_time_eq(p, expected.as_bytes()) => next.run(req).await,
                _ => StatusCode::UNAUTHORIZED.into_response(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_bytes() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secrets"));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn token_values_are_trimmed_and_empty_rejected() {
        assert_eq!(token_value("  tok\n".into()).as_deref(), Some("tok"));
        assert_eq!(token_value("   ".into()), None);
        assert_eq!(token_value(String::new()), None);
    }
}
