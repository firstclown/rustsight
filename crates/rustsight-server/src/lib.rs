pub mod auth;
pub mod http;
pub mod mcp;

use std::time::Duration;

pub const DEFAULT_CONSOLIDATE_INTERVAL_MS: u64 = 5_000;

/// Background consolidation cadence from `RUSTSIGHT_CONSOLIDATE_INTERVAL_MS`
/// (default 5s). Invalid values fall back to the default with a warning.
pub fn consolidate_interval_from_env() -> Duration {
    let raw = std::env::var("RUSTSIGHT_CONSOLIDATE_INTERVAL_MS").unwrap_or_default();
    if raw.trim().is_empty() {
        return Duration::from_millis(DEFAULT_CONSOLIDATE_INTERVAL_MS);
    }
    match raw.trim().parse::<u64>() {
        Ok(ms) => Duration::from_millis(ms.max(1)),
        Err(_) => {
            eprintln!(
                "rustsight: invalid RUSTSIGHT_CONSOLIDATE_INTERVAL_MS {raw:?}, using {DEFAULT_CONSOLIDATE_INTERVAL_MS}ms"
            );
            Duration::from_millis(DEFAULT_CONSOLIDATE_INTERVAL_MS)
        }
    }
}
