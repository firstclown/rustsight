pub fn temporal_weight(delta_ms: i64, sigma_ms: f64) -> f32 {
    let d = delta_ms.unsigned_abs() as f64;
    (-(d / sigma_ms)).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decays_with_distance() {
        let sigma = 7.0 * 86_400_000.0;
        assert!((temporal_weight(0, sigma) - 1.0).abs() < 1e-6);
        let w7 = temporal_weight(604_800_000, sigma);
        assert!((w7 - std::f32::consts::E.recip()).abs() < 1e-4);
        assert!(temporal_weight(60_000_000_000, sigma) < 0.05);
    }
}
