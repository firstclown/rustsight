use chrono::{Datelike, Duration, NaiveDate};

const DAY_MS: i64 = 86_400_000;

fn date_ms(d: NaiveDate) -> i64 {
    d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_millis()
}

fn month_index(name: &str) -> Option<u32> {
    let n = name.to_lowercase();
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    for (i, m) in months.iter().enumerate() {
        if n == *m || (n.len() >= 3 && m.starts_with(&n)) {
            return Some(i as u32 + 1);
        }
    }
    None
}

fn parse_date(s: &str, today: &NaiveDate, explicit_year: Option<i32>) -> Option<NaiveDate> {
    let first = s
        .split_whitespace()
        .next()
        .unwrap_or(s)
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
    if let Ok(d) = NaiveDate::parse_from_str(first, "%Y-%m-%d") {
        return Some(d);
    }
    if let Ok(d) = NaiveDate::parse_from_str(first, "%Y/%m/%d") {
        return Some(d);
    }
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() >= 2 {
        if let Some(m) = month_index(parts[0].trim_matches(|c: char| !c.is_alphanumeric())) {
            let day: u32 = parts[1].trim_end_matches(',').parse().ok()?;
            let year = explicit_year.unwrap_or_else(|| {
                let mut y = today.year();
                if m > today.month() {
                    y -= 1;
                }
                y
            });
            return NaiveDate::from_ymd_opt(year, m, day);
        }
    }
    None
}

fn find_year(tokens: &[&str]) -> Option<i32> {
    for t in tokens {
        let t = t.trim_matches(|c: char| !c.is_ascii_digit());
        if let Ok(y) = t.parse::<i32>() {
            if (1900..=2100).contains(&y) {
                return Some(y);
            }
        }
    }
    None
}

pub fn parse_window(query: &str, now_ms: i64) -> Option<(i64, i64)> {
    let now = chrono::DateTime::from_timestamp_millis(now_ms)?;
    let today = now.date_naive();
    let ql = query.to_lowercase();
    let tokens: Vec<&str> = ql.split_whitespace().collect();
    let explicit_year = find_year(&tokens);

    if let Some(pos) = ql.find("between") {
        let rest = &ql[pos + "between".len()..];
        if let Some(mid) = rest.find(" and ") {
            let a = parse_date(rest[..mid].trim(), &today, explicit_year);
            let b = parse_date(rest[mid + 5..].trim(), &today, explicit_year);
            if let (Some(a), Some(b)) = (a, b) {
                let (s, e) = if a <= b { (a, b) } else { (b, a) };
                return Some((date_ms(s), date_ms(e + Duration::days(1))));
            }
        }
    }

    if let Some(pos) = ql.find("since ") {
        let rest = ql[pos + 6..].trim();
        if let Some(d) = parse_date(rest, &today, explicit_year) {
            return Some((date_ms(d), now_ms));
        }
        if let Ok(y) = rest.trim().parse::<i32>() {
            if let Some(d) = NaiveDate::from_ymd_opt(y, 1, 1) {
                return Some((date_ms(d), now_ms));
            }
        }
    }

    if let Some(y) = find_year(&tokens) {
        if let Some(s) = NaiveDate::from_ymd_opt(y, 1, 1) {
            let e = NaiveDate::from_ymd_opt(y + 1, 1, 1)?;
            return Some((date_ms(s), date_ms(e)));
        }
    }

    let relative: &[(&str, i64)] = &[
        ("last week", 7),
        ("last month", 30),
        ("last quarter", 91),
        ("last year", 365),
        ("past week", 7),
        ("past month", 30),
        ("past year", 365),
        ("this week", 7),
    ];
    for (phrase, days) in relative {
        if ql.contains(phrase) {
            return Some((now_ms - days * DAY_MS, now_ms));
        }
    }

    if ql.contains("yesterday") {
        let y = today - Duration::days(1);
        return Some((date_ms(y), date_ms(y + Duration::days(1))));
    }

    if ql.contains("this year") {
        let s = NaiveDate::from_ymd_opt(today.year(), 1, 1)?;
        return Some((date_ms(s), now_ms));
    }

    if ql.contains("this month") {
        let s = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?;
        return Some((date_ms(s), now_ms));
    }

    for t in &tokens {
        let t = t.trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(m) = month_index(t) {
            let mut year = today.year();
            if m > today.month() {
                year -= 1;
            }
            let s = NaiveDate::from_ymd_opt(year, m, 1)?;
            let (ny, nm) = if m == 12 {
                (year + 1, 1)
            } else {
                (year, m + 1)
            };
            let e = NaiveDate::from_ymd_opt(ny, nm, 1)?;
            return Some((date_ms(s), date_ms(e)));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> i64 {
        chrono::NaiveDate::from_ymd_opt(2026, 10, 27)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis()
    }

    #[test]
    fn year_window() {
        let (s, e) = parse_window("what happened in 2023?", now()).unwrap();
        assert_eq!(
            chrono::DateTime::from_timestamp_millis(s).unwrap().year(),
            2023
        );
        assert_eq!(
            chrono::DateTime::from_timestamp_millis(e).unwrap().year(),
            2024
        );
    }

    #[test]
    fn last_month_window() {
        let n = now();
        let (s, e) = parse_window("what did we do last month", n).unwrap();
        assert_eq!(e, n);
        assert_eq!(n - s, 30 * DAY_MS);
    }

    #[test]
    fn between_dates() {
        let (s, e) = parse_window("Between 2023-03-01 and 2023-04-15 anything?", now()).unwrap();
        assert_eq!(
            chrono::DateTime::from_timestamp_millis(s)
                .unwrap()
                .date_naive(),
            chrono::NaiveDate::from_ymd_opt(2023, 3, 1).unwrap()
        );
        assert!(e > s);
    }

    #[test]
    fn month_name_window() {
        let (s, e) = parse_window("what happened in may?", now()).unwrap();
        let sm = chrono::DateTime::from_timestamp_millis(s).unwrap();
        assert_eq!(sm.year(), 2026);
        assert_eq!(sm.month(), 5);
        assert!(e > s);
    }

    #[test]
    fn yesterday() {
        let (s, e) = parse_window("yesterday's meeting", now()).unwrap();
        assert_eq!((e - s) / DAY_MS, 1);
    }

    #[test]
    fn no_window() {
        assert!(parse_window("where does Alice work?", now()).is_none());
    }

    #[test]
    fn since_year() {
        let n = now();
        let (s, e) = parse_window("everything since 2024", n).unwrap();
        assert_eq!(e, n);
        assert_eq!(
            chrono::DateTime::from_timestamp_millis(s).unwrap().year(),
            2024
        );
    }

    #[test]
    fn between_month_day_defaults_to_recent_year() {
        let (s, e) = parse_window("between March 5 and April 10 anything?", now()).unwrap();
        let start = chrono::DateTime::from_timestamp_millis(s)
            .unwrap()
            .date_naive();
        let end = chrono::DateTime::from_timestamp_millis(e)
            .unwrap()
            .date_naive();
        assert_eq!(start, chrono::NaiveDate::from_ymd_opt(2026, 3, 5).unwrap());
        assert_eq!(end, chrono::NaiveDate::from_ymd_opt(2026, 4, 11).unwrap());
    }

    #[test]
    fn between_month_day_spans_year_boundary() {
        let (s, e) = parse_window("between December 5 and January 10", now()).unwrap();
        let start = chrono::DateTime::from_timestamp_millis(s)
            .unwrap()
            .date_naive();
        let end = chrono::DateTime::from_timestamp_millis(e)
            .unwrap()
            .date_naive();
        assert_eq!(start, chrono::NaiveDate::from_ymd_opt(2025, 12, 5).unwrap());
        assert_eq!(end, chrono::NaiveDate::from_ymd_opt(2026, 1, 11).unwrap());
    }

    #[test]
    fn between_month_day_honors_explicit_year() {
        let (s, e) = parse_window("between March 5 2024 and April 10 2024", now()).unwrap();
        let start = chrono::DateTime::from_timestamp_millis(s)
            .unwrap()
            .date_naive();
        let end = chrono::DateTime::from_timestamp_millis(e)
            .unwrap()
            .date_naive();
        assert_eq!(start, chrono::NaiveDate::from_ymd_opt(2024, 3, 5).unwrap());
        assert_eq!(end, chrono::NaiveDate::from_ymd_opt(2024, 4, 11).unwrap());
    }

    #[test]
    fn since_month_day_uses_recent_year() {
        let n = now();
        let (s, e) = parse_window("everything since March 5", n).unwrap();
        let start = chrono::DateTime::from_timestamp_millis(s)
            .unwrap()
            .date_naive();
        assert_eq!(start, chrono::NaiveDate::from_ymd_opt(2026, 3, 5).unwrap());
        assert_eq!(e, n);
    }
}
