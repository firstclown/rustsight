//! Rerank combined scoring and recall budget resolution — ports of upstream
//! `hindsight_api/engine/search/reranking.py` (`compute_recency_decay`,
//! `_recency_for_unit`, `apply_combined_scoring`) and
//! `hindsight_api/engine/memory_engine.py` (`_resolve_thinking_budget`),
//! tag v1.1.0. The cross-encoder score is the primary relevance signal;
//! recency and proof-count boosts are multiplicative so their influence
//! stays proportional to base relevance regardless of score calibration.

use crate::types::MemoryRef;
use chrono::{Datelike, Utc};

/// Max relative recency adjustment (±10% at the default 0.2).
pub const RECENCY_ALPHA: f32 = 0.2;
/// Max relative temporal-window adjustment. rustsight's temporal arm
/// produces membership, not proximity scores, so the signal stays neutral
/// (0.5) and this boost collapses to 1.0 — kept for formula parity.
pub const TEMPORAL_ALPHA: f32 = 0.2;
/// Max relative proof-count adjustment (±5% at the default 0.1).
pub const PROOF_COUNT_ALPHA: f32 = 0.1;

/// Age→freshness curve shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecencyDecayFunction {
    /// Straight decay to a 0.1 floor over the window (default).
    Linear,
    /// 0.5 at the halflife, exponential either side.
    Exponential,
    /// Recency stays neutral.
    None,
}

impl RecencyDecayFunction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Exponential => "exponential",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScoringConfig {
    pub recency_decay_function: RecencyDecayFunction,
    pub recency_decay_linear_window_days: f64,
    pub recency_decay_halflife_days: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            recency_decay_function: RecencyDecayFunction::Linear,
            recency_decay_linear_window_days: 365.0,
            recency_decay_halflife_days: 90.0,
        }
    }
}

/// Map a memory's age in days to a freshness signal in [0, 1] (neutral 0.5).
/// Future-dated memories clamp to maximum freshness so they are never
/// penalised.
pub fn compute_recency_decay(days_ago: f64, cfg: &ScoringConfig) -> f64 {
    match cfg.recency_decay_function {
        RecencyDecayFunction::None => 0.5,
        RecencyDecayFunction::Exponential => {
            if cfg.recency_decay_halflife_days <= 0.0 {
                0.5
            } else if days_ago <= 0.0 {
                1.0
            } else {
                0.5f64.powf(days_ago / cfg.recency_decay_halflife_days)
            }
        }
        RecencyDecayFunction::Linear => {
            let window = if cfg.recency_decay_linear_window_days > 0.0 {
                cfg.recency_decay_linear_window_days
            } else {
                365.0
            };
            (1.0 - days_ago / window).clamp(0.1, 1.0)
        }
    }
}

const DAY_MS: f64 = 86_400_000.0;
/// How far short of an exact calendar month/year a span may fall and still
/// read as that period (extraction spells period ends three ways: period−1s,
/// period−1day, exclusive next-period start).
const CALENDAR_PERIOD_TOLERANCE_SECS: f64 = 86_400.0;

/// True when `[start_ms, end_ms]` covers exactly one calendar month or year
/// — a *coarse date* ("sometime in 2015") rather than a genuine interval.
fn spans_calendar_period(start_ms: i64, end_ms: i64) -> bool {
    let span_secs = (end_ms - start_ms) as f64 / 1000.0;
    if span_secs <= 0.0 {
        return false;
    }
    let start = chrono::DateTime::from_timestamp_millis(start_ms)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);
    let (year, month) = (start.year(), start.month());
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31.0,
        4 | 6 | 9 | 11 => 30.0,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29.0
            } else {
                28.0
            }
        }
        _ => 30.0,
    };
    let month_seconds = days_in_month * 86_400.0;
    let year_seconds = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
        366.0 * 86_400.0
    } else {
        365.0 * 86_400.0
    };
    [month_seconds, year_seconds]
        .iter()
        .any(|&period| period - CALENDAR_PERIOD_TOLERANCE_SECS <= span_secs && span_secs <= period)
}

/// Freshness signal in [0, 1] (neutral 0.5) for one memory ref.
///
/// A ref carrying a coarse date (a span of exactly one calendar month or
/// year) is scored from the END of the period — the latest the event could
/// have happened — with the signal capped at neutral, because the end of a
/// period is a bound, not an observation. Everything else ages from its
/// effective instant: `occurred_start`, then `mentioned_at`, then
/// `occurred_end` (upstream's COALESCE order).
pub fn recency_for_ref(m: &MemoryRef, now_ms: i64, cfg: &ScoringConfig) -> f64 {
    if let (Some(start), Some(end)) = (m.occurred_start_ms, m.occurred_end_ms) {
        if spans_calendar_period(start, end) {
            let days = (now_ms - end) as f64 / DAY_MS;
            return compute_recency_decay(days, cfg).min(0.5);
        }
    }
    let effective = m
        .occurred_start_ms
        .or(m.mentioned_at_ms)
        .or(m.occurred_end_ms);
    match effective {
        Some(at) => compute_recency_decay((now_ms - at) as f64 / DAY_MS, cfg),
        None => 0.5,
    }
}

/// Log-normalized evidence strength: proof_count=1 → 0.5 (neutral
/// multiplier), proof_count=150 → clamped to 1.0 (max +5% boost); `None` and
/// 0 stay neutral.
pub fn proof_norm(proof_count: Option<usize>) -> f32 {
    match proof_count {
        Some(n) if n >= 1 => (0.5 + (n as f32).ln() / 10.0).clamp(0.0, 1.0),
        _ => 0.5,
    }
}

/// The combined score: normalized cross-encoder relevance modulated by
/// multiplicative recency, temporal, and proof-count boosts. The temporal
/// signal stays neutral (0.5) in rustsight — its temporal arm produces
/// window membership, not proximity — so that boost is carried for formula
/// parity and collapses to 1.0.
pub fn combined_score(
    ce_normalized: f32,
    recency: f64,
    temporal: f32,
    proof: Option<usize>,
) -> f32 {
    let recency = recency as f32;
    let recency_boost = 1.0 + RECENCY_ALPHA * (recency - 0.5);
    let temporal_boost = 1.0 + TEMPORAL_ALPHA * (temporal - 0.5);
    let proof_boost = 1.0 + PROOF_COUNT_ALPHA * (proof_norm(proof) - 0.5);
    ce_normalized * recency_boost * temporal_boost * proof_boost
}

/// Normalize raw reranker scores to [0, 1]: calibrated scores already in
/// [0, 1] (external API rerankers — Cohere, Jina) pass through so absolute
/// confidence is preserved; logits (local models) go through a sigmoid.
/// NaNs sanitize to 0.0.
pub fn normalize_ce_scores(scores: &[f32]) -> Vec<f32> {
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    if scores.iter().all(|s| !s.is_nan() && *s >= 0.0 && *s <= 1.0) {
        scores.to_vec()
    } else {
        scores
            .iter()
            .map(|&s| if s.is_nan() { 0.0 } else { sigmoid(s) })
            .collect()
    }
}

/// Seed normalized CE scores from RRF rank for a passthrough reranker (the
/// slim deployment's no-op cross-encoder): map rank → [0.1, 1.0] so the
/// boosts can nudge adjacent orderings without overpowering RRF.
pub fn seed_ce_from_rrf(n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    let denom = n.saturating_sub(1).max(1) as f32;
    (0..n).map(|rank| 1.0 - 0.9 * rank as f32 / denom).collect()
}

// ---------------------------------------------------------------------------
// Recall budget (upstream `_resolve_thinking_budget`)
// ---------------------------------------------------------------------------

/// Budget levels for recall/reflect operations; `None` falls back to Mid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    Low,
    Mid,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetFunction {
    Fixed,
    Adaptive,
}

/// Recall-pool sizing, mirroring upstream's `recall_budget_*` config keys:
/// "fixed" returns the level's constant directly (the legacy default);
/// "adaptive" scales the caller's token budget by the level's ratio,
/// clamped to [min, max].
#[derive(Debug, Clone, Copy)]
pub struct RecallBudgetConfig {
    pub function: BudgetFunction,
    pub fixed_low: usize,
    pub fixed_mid: usize,
    pub fixed_high: usize,
    pub adaptive_low: f32,
    pub adaptive_mid: f32,
    pub adaptive_high: f32,
    pub min: usize,
    pub max: usize,
}

impl Default for RecallBudgetConfig {
    fn default() -> Self {
        Self {
            function: BudgetFunction::Fixed,
            fixed_low: 100,
            fixed_mid: 300,
            fixed_high: 1000,
            adaptive_low: 0.025,
            adaptive_mid: 0.075,
            adaptive_high: 0.25,
            min: 20,
            max: 2000,
        }
    }
}

impl RecallBudgetConfig {
    /// Map a budget level (None → Mid, preserving the legacy default) to the
    /// per-arm retrieval limit.
    pub fn resolve(&self, budget: Option<Budget>, max_tokens: usize) -> usize {
        let budget = budget.unwrap_or(Budget::Mid);
        match self.function {
            BudgetFunction::Adaptive => {
                let ratio = match budget {
                    Budget::Low => self.adaptive_low,
                    Budget::Mid => self.adaptive_mid,
                    Budget::High => self.adaptive_high,
                };
                let raw = (max_tokens as f32 * ratio).round() as usize;
                raw.clamp(self.min, self.max)
            }
            BudgetFunction::Fixed => match budget {
                Budget::Low => self.fixed_low,
                Budget::Mid => self.fixed_mid,
                Budget::High => self.fixed_high,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_decay_to_floor() {
        let cfg = ScoringConfig::default();
        assert!((compute_recency_decay(0.0, &cfg) - 1.0).abs() < 1e-9);
        let half = compute_recency_decay(182.5, &cfg);
        assert!((half - 0.5).abs() < 1e-9, "{half}");
        // floor is 0.1, not 0
        assert!((compute_recency_decay(1000.0, &cfg) - 0.1).abs() < 1e-9);
        // future-dated clamps to max freshness
        assert!((compute_recency_decay(-10.0, &cfg) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn exponential_and_none_decay() {
        let mut cfg = ScoringConfig {
            recency_decay_function: RecencyDecayFunction::Exponential,
            ..ScoringConfig::default()
        };
        assert!((compute_recency_decay(90.0, &cfg) - 0.5).abs() < 1e-9);
        assert!((compute_recency_decay(0.0, &cfg) - 1.0).abs() < 1e-9);
        cfg.recency_decay_function = RecencyDecayFunction::None;
        assert!((compute_recency_decay(0.0, &cfg) - 0.5).abs() < 1e-9);
    }

    fn day_ms(d: i64) -> i64 {
        d * 86_400_000
    }

    #[test]
    fn recency_effective_instant_order() {
        let cfg = ScoringConfig::default();
        let now = day_ms(400);
        // occurred_start (day 398) is more recent than mentioned_at (day
        // 390) and wins the COALESCE, so the ref reads fresher than the same
        // ref with occurred_start stripped.
        let mut m = MemoryRef {
            id: "m".into(),
            kind: crate::types::MemoryKind::World,
            text: String::new(),
            score: 0.0,
            occurred_start_ms: Some(day_ms(398)),
            occurred_end_ms: None,
            mentioned_at_ms: Some(day_ms(390)),
            entity_ids: vec![],
            proof_count: None,
        };
        let from_start = recency_for_ref(&m, now, &cfg);
        m.occurred_start_ms = None;
        let from_mentioned = recency_for_ref(&m, now, &cfg);
        assert!(from_start > from_mentioned);
        // no dates at all → neutral
        m.mentioned_at_ms = None;
        assert!((recency_for_ref(&m, now, &cfg) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn coarse_period_scores_from_end_capped_at_neutral() {
        let cfg = ScoringConfig::default();
        // Calendar year 2015 (2015-01-01 → 2015-12-31).
        let start = chrono::NaiveDate::from_ymd_opt(2015, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let end = chrono::NaiveDate::from_ymd_opt(2016, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let m = MemoryRef {
            id: "m".into(),
            kind: crate::types::MemoryKind::World,
            text: String::new(),
            score: 0.0,
            occurred_start_ms: Some(start),
            occurred_end_ms: Some(end),
            mentioned_at_ms: None,
            entity_ids: vec![],
            proof_count: None,
        };
        // Asked "now" = mid-2015: the period's end is in the future, but a
        // coarse date never earns a freshness boost — capped at 0.5.
        let mid_2015 = chrono::NaiveDate::from_ymd_opt(2015, 7, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        assert!((recency_for_ref(&m, mid_2015, &cfg) - 0.5).abs() < 1e-9);
        // A decade later the period decays from its end toward the floor.
        let asked_2026 = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        assert!(recency_for_ref(&m, asked_2026, &cfg) < 0.5);
        // A genuine interval (10 days) is NOT a coarse date.
        let interval = MemoryRef {
            occurred_start_ms: Some(start),
            occurred_end_ms: Some(start + day_ms(10)),
            ..m.clone()
        };
        let r = recency_for_ref(&interval, mid_2015, &cfg);
        assert!(r > 0.5, "genuine interval ages from its start: {r}");
    }

    #[test]
    fn proof_norm_curve() {
        assert!((proof_norm(None) - 0.5).abs() < 1e-6);
        assert!((proof_norm(Some(0)) - 0.5).abs() < 1e-6);
        assert!((proof_norm(Some(1)) - 0.5).abs() < 1e-6);
        assert!((proof_norm(Some(150)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn combined_score_boost_ranges() {
        // Neutral signals reproduce the CE score exactly.
        let c = combined_score(0.75, 0.5, 0.5, None);
        assert!((c - 0.75).abs() < 1e-6);
        // Fresh + proven vs old + unproven: ±10%/±5% at most.
        assert!(combined_score(1.0, 1.0, 0.5, Some(150)) <= 1.0 * 1.1 * 1.05 + 1e-6);
        assert!(combined_score(1.0, 0.0, 0.5, None) >= 1.0 * 0.9 - 1e-6);
        // Temporal stays neutral in rustsight; the boost term is carried
        // for formula parity.
        assert!((combined_score(1.0, 0.5, 1.0, None) - 1.1).abs() < 1e-6);
    }

    #[test]
    fn ce_normalization_passthrough_vs_sigmoid() {
        // Calibrated scores in [0,1] pass through untouched.
        assert_eq!(
            normalize_ce_scores(&[0.007, 0.5, 1.0]),
            vec![0.007, 0.5, 1.0]
        );
        // Logits sigmoid to (0,1).
        let n = normalize_ce_scores(&[5.0, 0.0, -5.0]);
        assert!((n[0] - 0.993307).abs() < 1e-6, "{n:?}");
        assert!((n[1] - 0.5).abs() < 1e-6);
        assert!((n[2] - 0.006693).abs() < 1e-6);
        // NaN sanitizes to 0
        assert_eq!(normalize_ce_scores(&[f32::NAN])[0], 0.0);
    }

    #[test]
    fn rrf_seeding_spread() {
        assert_eq!(seed_ce_from_rrf(0), Vec::<f32>::new());
        let s = seed_ce_from_rrf(3);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[2] - 0.1).abs() < 1e-6);
        let one = seed_ce_from_rrf(1);
        assert_eq!(one, vec![1.0]);
    }

    #[test]
    fn budget_resolution_defaults() {
        let cfg = RecallBudgetConfig::default();
        // None → MID; fixed function ignores max_tokens.
        assert_eq!(cfg.resolve(None, 2048), 300);
        assert_eq!(cfg.resolve(Some(Budget::Low), 99999), 100);
        assert_eq!(cfg.resolve(Some(Budget::High), 0), 1000);
        // Adaptive: round(max_tokens * ratio), clamped.
        let adaptive = RecallBudgetConfig {
            function: BudgetFunction::Adaptive,
            ..RecallBudgetConfig::default()
        };
        assert_eq!(adaptive.resolve(Some(Budget::Mid), 2048), 154); // 2048*0.075
        assert_eq!(adaptive.resolve(Some(Budget::High), 100_000), 2000);
        assert_eq!(adaptive.resolve(Some(Budget::Low), 100), 20);
    }
}
