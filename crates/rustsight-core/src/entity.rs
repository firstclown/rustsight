pub fn normalize_name(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

pub fn string_similarity(a: &str, b: &str) -> f32 {
    let na = normalize_name(a);
    let nb = normalize_name(b);
    if na.is_empty() || nb.is_empty() {
        return 0.0;
    }
    if na == nb {
        return 1.0;
    }
    let max = na.chars().count().max(nb.chars().count()) as f32;
    1.0 - levenshtein(&na, &nb) as f32 / max
}

#[derive(Debug, Clone, Copy)]
pub struct ResolverConfig {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub link_threshold: f32,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6,
            beta: 0.2,
            gamma: 0.2,
            link_threshold: 0.82,
        }
    }
}

pub fn combined_score(str_sim: f32, co_sim: f32, temp_sim: f32, cfg: &ResolverConfig) -> f32 {
    cfg.alpha * str_sim + cfg.beta * co_sim + cfg.gamma * temp_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_punctuation() {
        assert_eq!(normalize_name("  Alice ! Smith "), "alice smith");
        assert_eq!(normalize_name("ACME Corp."), "acme corp");
    }

    #[test]
    fn similarity_identical_and_far() {
        assert!((string_similarity("Alice", "alice") - 1.0).abs() < 1e-6);
        assert!(string_similarity("Alice", "Quantum Electrodynamics") < 0.4);
    }

    #[test]
    fn similarity_near_miss_high() {
        let s = string_similarity("Alice Smith", "Alise Smith");
        assert!(s > 0.7, "{s}");
    }

    #[test]
    fn combined_score_weights() {
        let cfg = ResolverConfig::default();
        let s = combined_score(1.0, 0.0, 0.0, &cfg);
        assert!((s - 0.6).abs() < 1e-6);
    }
}
