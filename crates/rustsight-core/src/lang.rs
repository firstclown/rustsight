//! Script detection and CJK text handling (RS-10e).
//!
//! - [`cjk_segment`] makes CJK text searchable with FTS5's `unicode61`
//!   tokenizer, which treats an unbroken run of ideographs as one token (a
//!   whole Chinese sentence would only match itself). Each CJK run is
//!   rewritten as overlapping character bigrams — a lone character stays a
//!   unigram — the shape of Lucene's CJKAnalyzer. Other text passes through
//!   untouched, so non-CJK rows index exactly as before.
//! - [`detect_script`] routes queries; Han-bearing queries also run
//!   [`parse_chinese_window`], the common subset of upstream's
//!   `chinese_temporal_periods.py` (the full rule set is a recorded
//!   divergence).

use chrono::{Datelike, Duration, NaiveDate};

/// The dominant writing system of a text, by letter count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    /// No letters at all.
    None,
    Latin,
    /// Chinese ideographs (also used by Japanese kanji).
    Han,
    /// Japanese kana present (kana decides Japanese over Chinese).
    Kana,
    Hangul,
    Other,
}

pub fn is_han(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF      // Extension A
        | 0x4E00..=0x9FFF    // Unified Ideographs
        | 0xF900..=0xFAFF    // Compatibility Ideographs
        | 0x20000..=0x3134F  // Extensions B–H
    ) || c == '〇'
}

pub fn is_kana(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x309F      // Hiragana
        | 0x30A0..=0x30FF    // Katakana (incl. ー)
        | 0x31F0..=0x31FF    // Katakana phonetic extensions
        | 0xFF66..=0xFF9F    // Half-width katakana
    )
}

pub fn is_hangul(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF      // Jamo
        | 0x3130..=0x318F    // Compatibility Jamo
        | 0xA960..=0xA97F
        | 0xAC00..=0xD7AF    // Syllables
        | 0xD7B0..=0xD7FF
    )
}

/// Han, kana or Hangul: written without spaces between words.
pub fn is_cjk(c: char) -> bool {
    is_han(c) || is_kana(c) || is_hangul(c)
}

pub fn has_cjk(s: &str) -> bool {
    s.chars().any(is_cjk)
}

pub fn has_han(s: &str) -> bool {
    s.chars().any(is_han)
}

pub fn detect_script(s: &str) -> Script {
    let (mut latin, mut han, mut kana, mut hangul, mut other) = (0, 0, 0, 0, 0);
    for c in s.chars().filter(|c| c.is_alphabetic()) {
        if is_kana(c) {
            kana += 1;
        } else if is_han(c) {
            han += 1;
        } else if is_hangul(c) {
            hangul += 1;
        } else if c.is_ascii_alphabetic() || ('\u{00C0}'..='\u{024F}').contains(&c) {
            latin += 1;
        } else {
            other += 1;
        }
    }
    if kana > 0 {
        return Script::Kana;
    }
    let best = [
        (han, Script::Han),
        (hangul, Script::Hangul),
        (latin, Script::Latin),
        (other, Script::Other),
    ]
    .into_iter()
    .max_by_key(|(n, _)| *n)
    .filter(|(n, _)| *n > 0);
    best.map_or(Script::None, |(_, s)| s)
}

/// Overlapping bigrams of one CJK run; a single character is its own
/// unigram.
fn bigrams(run: &[char]) -> Vec<String> {
    if run.len() == 1 {
        return vec![run[0].to_string()];
    }
    run.windows(2).map(|w| w.iter().collect()).collect()
}

/// Rewrite CJK runs as space-separated bigrams, leaving everything else
/// as is. Identity on text without CJK characters, and idempotent.
pub fn cjk_segment(text: &str) -> String {
    if !has_cjk(text) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() * 2);
    let mut run: Vec<char> = vec![];
    // Separate a finished run from what follows unless that is a space.
    let mut after_run = false;
    for c in text.chars() {
        if is_cjk(c) {
            run.push(c);
            continue;
        }
        if !run.is_empty() {
            if !out.is_empty() && !out.ends_with(char::is_whitespace) {
                out.push(' ');
            }
            out.push_str(&bigrams(&run).join(" "));
            run.clear();
            after_run = true;
        }
        if after_run && !c.is_whitespace() {
            out.push(' ');
        }
        after_run = false;
        out.push(c);
    }
    if !run.is_empty() {
        if !out.is_empty() && !out.ends_with(char::is_whitespace) {
            out.push(' ');
        }
        out.push_str(&bigrams(&run).join(" "));
    }
    out
}

/// Query terms for BM25: lowercased words of two or more bytes, plus the
/// bigrams of every CJK run, deduplicated in order.
pub fn search_terms(query: &str) -> Vec<String> {
    let segmented = cjk_segment(query);
    let mut seen = std::collections::HashSet::new();
    let mut out = vec![];
    for w in segmented.split(|c: char| !c.is_alphanumeric()) {
        let w = w.to_lowercase();
        let keep = w.chars().any(is_cjk) || w.len() >= 2;
        if keep && seen.insert(w.clone()) {
            out.push(w);
        }
    }
    out
}

// ── Chinese relative periods ─────────────────────────────────────────────

const DAY_MS: i64 = 86_400_000;

fn day_ms(d: NaiveDate) -> i64 {
    d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_millis()
}

/// `[first, last]` whole days as a half-open ms window.
fn days(first: NaiveDate, last: NaiveDate) -> (i64, i64) {
    (day_ms(first), day_ms(last) + DAY_MS)
}

/// Arabic digits or Chinese numerals up to 99 (`三`, `十`, `十五`, `二十三`, `两`).
fn parse_number(s: &str) -> Option<i64> {
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    let digit = |c: char| {
        "零一二三四五六七八九"
            .find(c)
            .map(|i| (i / 3) as i64)
            .or((c == '两').then_some(2))
    };
    let chars: Vec<char> = s.chars().collect();
    match chars.as_slice() {
        [] => None,
        [c] if *c == '十' => Some(10),
        [c] => digit(*c),
        ['十', u] => Some(10 + digit(*u)?),
        [t, '十'] => Some(digit(*t)? * 10),
        [t, '十', u] => Some(digit(*t)? * 10 + digit(*u)?),
        _ => None,
    }
}

fn month_start(y: i32, m: u32) -> Option<NaiveDate> {
    NaiveDate::from_ymd_opt(y, m, 1)
}

fn month_end(y: i32, m: u32) -> Option<NaiveDate> {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    Some(month_start(ny, nm)? - Duration::days(1))
}

fn shift_months(d: NaiveDate, months: i32) -> Option<NaiveDate> {
    let total = d.year() * 12 + d.month0() as i32 + months;
    let (y, m) = (total.div_euclid(12), total.rem_euclid(12) as u32 + 1);
    let last = month_end(y, m)?.day();
    NaiveDate::from_ymd_opt(y, m, d.day().min(last))
}

/// Number then unit, e.g. `3天` / `三个月` / `两周`, at the start of `s`.
fn count_unit(s: &str) -> Option<(i64, char, usize)> {
    let num_len: usize = s
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || "零一二两三四五六七八九十".contains(*c))
        .map(|(_, c)| c.len_utf8())
        .sum();
    let n = parse_number(&s[..num_len])?;
    let rest = &s[num_len..];
    let rest = rest.strip_prefix('个').unwrap_or(rest);
    let unit = rest.chars().next()?;
    let unit = match unit {
        '天' | '日' => 'd',
        '周' | '星' | '礼' => 'w',
        '月' => 'm',
        '年' => 'y',
        _ => return None,
    };
    let unit_len = match rest.chars().next()? {
        '星' if rest.starts_with("星期") => "星期".len(),
        '礼' if rest.starts_with("礼拜") => "礼拜".len(),
        '星' | '礼' => return None,
        c => c.len_utf8(),
    };
    Some((n, unit, s.len() - rest.len() + unit_len))
}

fn back(today: NaiveDate, n: i64, unit: char) -> Option<NaiveDate> {
    match unit {
        'd' => Some(today - Duration::days(n)),
        'w' => Some(today - Duration::days(7 * n)),
        'm' => shift_months(today, -(n as i32)),
        _ => shift_months(today, -(12 * n as i32)),
    }
}

/// `YYYY年M月[D日]` / `M月D日` anywhere in the query.
fn explicit_date(q: &str, today: NaiveDate) -> Option<(i64, i64)> {
    let chars: Vec<char> = q.chars().collect();
    let num_at = |i: usize| -> (Option<u32>, usize) {
        let digits: String = chars[i..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        (digits.parse().ok(), digits.chars().count())
    };
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() || (i > 0 && chars[i - 1].is_ascii_digit()) {
            i += 1;
            continue;
        }
        let (n, len) = num_at(i);
        let mut j = i + len;
        let (year, month) = if chars.get(j) == Some(&'年') && len == 4 {
            j += 1;
            let (m, mlen) = num_at(j);
            if m.is_none() || chars.get(j + mlen) != Some(&'月') {
                // `2024年`: the whole year.
                let y = n? as i32;
                return Some(days(
                    NaiveDate::from_ymd_opt(y, 1, 1)?,
                    NaiveDate::from_ymd_opt(y, 12, 31)?,
                ));
            }
            j += mlen + 1;
            (n? as i32, m?)
        } else if chars.get(j) == Some(&'月') {
            j += 1;
            (today.year(), n?)
        } else {
            i += len;
            continue;
        };
        let (d, dlen) = num_at(j);
        if let (Some(d), Some('日' | '号')) = (d, chars.get(j + dlen)) {
            let date = NaiveDate::from_ymd_opt(year, month, d)?;
            return Some(days(date, date));
        }
        if (1..=12).contains(&month) && chars[i..j].contains(&'年') {
            return Some(days(month_start(year, month)?, month_end(year, month)?));
        }
        i = j;
    }
    None
}

/// A time window for the common Chinese relative expressions, in the
/// query's reference frame (`now_ms`, UTC days, weeks start on Monday):
/// 今天/昨天/前天, 本周/这周/上周/下周, 本月/这个月/上个月/下个月,
/// 今年/去年/前年/明年, 最近/过去/近 N 天|周|个月|年 (and 几 = 5), N 天前
/// (that one day), YYYY年[M月[D日]] and M月D日.
pub fn parse_chinese_window(query: &str, now_ms: i64) -> Option<(i64, i64)> {
    if !has_han(query) {
        return None;
    }
    let today = chrono::DateTime::from_timestamp_millis(now_ms)?.date_naive();
    let q: String = query.chars().filter(|c| !c.is_whitespace()).collect();

    // Rolling windows before bare periods: "最近三个月" is not "月".
    for prefix in ["最近", "过去", "近"] {
        if let Some(at) = q.find(prefix) {
            let rest = &q[at + prefix.len()..];
            if let Some(after) = rest.strip_prefix('几') {
                if let Some((_, unit, _)) = count_unit(&format!("5{after}")) {
                    return Some(days(back(today, 5, unit)?, today));
                }
            }
            if let Some((n, unit, _)) = count_unit(rest) {
                return Some(days(back(today, n, unit)?, today));
            }
        }
    }
    // N天前 / N周前 / N个月前 / N年前: that single day.
    for (at, _) in q.match_indices('前') {
        let head = &q[..at];
        for (start, _) in head.char_indices() {
            if let Some((n, unit, len)) = count_unit(&head[start..]) {
                if start + len == head.len() {
                    let d = back(today, n, unit)?;
                    return Some(days(d, d));
                }
            }
        }
    }
    if let Some(window) = explicit_date(&q, today) {
        return Some(window);
    }

    let day = |offset: i64| {
        let d = today + Duration::days(offset);
        days(d, d)
    };
    // Longest words first: 前天 before 天, 大前天 before 前天.
    for (word, offset) in [
        ("大前天", -3),
        ("前天", -2),
        ("昨天", -1),
        ("昨日", -1),
        ("今天", 0),
        ("今日", 0),
        ("明天", 1),
    ] {
        if q.contains(word) {
            return Some(day(offset));
        }
    }
    let week_start = today - Duration::days(today.weekday().num_days_from_monday() as i64);
    for (word, weeks) in [
        ("上上周", -2),
        ("上周", -1),
        ("上个星期", -1),
        ("上星期", -1),
        ("本周", 0),
        ("这周", 0),
        ("这个星期", 0),
        ("下周", 1),
        ("下个星期", 1),
    ] {
        if q.contains(word) {
            let start = week_start + Duration::days(7 * weeks);
            return Some(days(start, start + Duration::days(6)));
        }
    }
    for (word, months) in [
        ("上上个月", -2),
        ("上个月", -1),
        ("上月", -1),
        ("本月", 0),
        ("这个月", 0),
        ("下个月", 1),
        ("下月", 1),
    ] {
        if q.contains(word) {
            let m = shift_months(month_start(today.year(), today.month())?, months)?;
            return Some(days(m, month_end(m.year(), m.month())?));
        }
    }
    for (word, years) in [
        ("大前年", -3),
        ("前年", -2),
        ("去年", -1),
        ("今年", 0),
        ("明年", 1),
    ] {
        if q.contains(word) {
            let y = today.year() + years;
            return Some(days(
                NaiveDate::from_ymd_opt(y, 1, 1)?,
                NaiveDate::from_ymd_opt(y, 12, 31)?,
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts_are_detected_by_letters() {
        assert_eq!(detect_script("爱丽丝加入了谷歌"), Script::Han);
        assert_eq!(detect_script("東京に住んでいます"), Script::Kana);
        assert_eq!(detect_script("앨리스는 구글에 입사했다"), Script::Hangul);
        assert_eq!(detect_script("Alice joined Google 谷歌"), Script::Latin);
        assert_eq!(detect_script("Москва"), Script::Other);
        assert_eq!(detect_script("2024 !"), Script::None);
        assert!(has_han("在Google工作") && !has_han("グーグル"));
    }

    #[test]
    fn cjk_runs_become_bigrams_and_the_rest_is_untouched() {
        assert_eq!(cjk_segment("Alice joined Google."), "Alice joined Google.");
        assert_eq!(
            cjk_segment("爱丽丝加入了谷歌。"),
            "爱丽 丽丝 丝加 加入 入了 了谷 谷歌 。"
        );
        assert_eq!(cjk_segment("在Google工作"), "在 Google 工作");
        assert_eq!(cjk_segment("东京タワー"), "东京 京タ タワ ワー");
        let once = cjk_segment("她在北京的谷歌公司工作");
        assert_eq!(cjk_segment(&once), once, "idempotent");
    }

    #[test]
    fn search_terms_cover_words_and_bigrams() {
        assert_eq!(
            search_terms("谷歌工程师"),
            vec!["谷歌", "歌工", "工程", "程师"]
        );
        assert_eq!(search_terms("Where is 谷歌?"), vec!["where", "is", "谷歌"]);
        assert_eq!(
            search_terms("在 a"),
            vec!["在"],
            "single CJK chars count, single latin bytes do not"
        );
    }

    fn at(y: i32, m: u32, d: u32) -> i64 {
        day_ms(NaiveDate::from_ymd_opt(y, m, d).unwrap()) + 12 * 3_600_000
    }

    fn window(q: &str) -> Option<(NaiveDate, NaiveDate)> {
        // Wednesday 2026-11-18.
        parse_chinese_window(q, at(2026, 11, 18)).map(|(s, e)| {
            let d = |ms: i64| {
                chrono::DateTime::from_timestamp_millis(ms)
                    .unwrap()
                    .date_naive()
            };
            (d(s), d(e - DAY_MS))
        })
    }

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn chinese_periods() {
        let cases: &[(&str, (NaiveDate, NaiveDate))] = &[
            ("今天开会了吗", (ymd(2026, 11, 18), ymd(2026, 11, 18))),
            ("昨天发生了什么", (ymd(2026, 11, 17), ymd(2026, 11, 17))),
            ("前天", (ymd(2026, 11, 16), ymd(2026, 11, 16))),
            ("上周做了什么", (ymd(2026, 11, 9), ymd(2026, 11, 15))),
            ("本周", (ymd(2026, 11, 16), ymd(2026, 11, 22))),
            ("下周", (ymd(2026, 11, 23), ymd(2026, 11, 29))),
            ("上个月的会议", (ymd(2026, 10, 1), ymd(2026, 10, 31))),
            ("本月", (ymd(2026, 11, 1), ymd(2026, 11, 30))),
            ("去年", (ymd(2025, 1, 1), ymd(2025, 12, 31))),
            ("前年", (ymd(2024, 1, 1), ymd(2024, 12, 31))),
            ("今年", (ymd(2026, 1, 1), ymd(2026, 12, 31))),
            ("最近三天", (ymd(2026, 11, 15), ymd(2026, 11, 18))),
            ("过去两周", (ymd(2026, 11, 4), ymd(2026, 11, 18))),
            ("最近几个月", (ymd(2026, 6, 18), ymd(2026, 11, 18))),
            ("近1年", (ymd(2025, 11, 18), ymd(2026, 11, 18))),
            ("十天前", (ymd(2026, 11, 8), ymd(2026, 11, 8))),
            ("3个月前", (ymd(2026, 8, 18), ymd(2026, 8, 18))),
            ("2024年3月5日", (ymd(2024, 3, 5), ymd(2024, 3, 5))),
            ("2024年3月", (ymd(2024, 3, 1), ymd(2024, 3, 31))),
            ("2023年", (ymd(2023, 1, 1), ymd(2023, 12, 31))),
            ("6月1日", (ymd(2026, 6, 1), ymd(2026, 6, 1))),
        ];
        for (q, want) in cases {
            assert_eq!(window(q), Some(*want), "{q}");
        }
        assert_eq!(window("爱丽丝在哪里工作"), None);
        assert_eq!(
            window("last week"),
            None,
            "non-Han queries are not parsed here"
        );
    }

    #[test]
    fn chinese_numbers() {
        for (s, n) in [
            ("三", 3),
            ("十", 10),
            ("十五", 15),
            ("二十", 20),
            ("二十三", 23),
            ("两", 2),
            ("12", 12),
        ] {
            assert_eq!(parse_number(s), Some(n), "{s}");
        }
        assert_eq!(parse_number("百"), None);
    }
}
