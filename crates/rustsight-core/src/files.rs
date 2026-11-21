//! File ingest (RS-10c): turn uploaded files into retainable text.
//!
//! A [`FileParser`] converts bytes to text; the [`ParserRegistry`] runs an
//! ordered fallback chain (upstream `FileParserRegistry.convert_with_fallback`):
//! each parser is tried in turn until one succeeds. Built-ins:
//!
//! - `text` — UTF-8 text: plain, markdown, CSV/TSV, JSON, YAML, XML, source
//!   code. Declines HTML (so `html` gets it), binaries (NUL bytes) and
//!   anything that is not valid UTF-8.
//! - `html` — tag-stripped visible text: `script`/`style`/`head`-like
//!   blocks dropped, block elements become line breaks, entities decoded.
//!
//! Heavier converters (PDF, DOCX, OCR) implement the same trait and are
//! registered with [`ParserRegistry::register`].

use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Upstream `SHORT_ID_LENGTH`: an attachment id is this many leading hex
/// characters of the file's sha256.
pub const ATTACHMENT_ID_LEN: usize = 12;

/// A stored original file (served back from `/attachments/{id}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    /// Content-addressed: `sha256[..12]`.
    pub id: String,
    pub bank_id: String,
    /// The document the file was retained into.
    pub document_id: String,
    pub filename: String,
    pub content_type: Option<String>,
    pub sha256: String,
    pub bytes: Vec<u8>,
    pub created_at_ms: i64,
}

impl Attachment {
    pub fn new(
        bank_id: &str,
        document_id: &str,
        filename: &str,
        content_type: Option<&str>,
        bytes: Vec<u8>,
        now_ms: i64,
    ) -> Self {
        let sha256: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Self {
            id: sha256[..ATTACHMENT_ID_LEN].to_string(),
            bank_id: bank_id.to_string(),
            document_id: document_id.to_string(),
            filename: filename.to_string(),
            content_type: content_type
                .map(str::to_string)
                .filter(|c| !c.trim().is_empty()),
            sha256,
            bytes,
            created_at_ms: now_ms,
        }
    }
}

/// One file handed to a parser.
#[derive(Debug, Clone, Copy)]
pub struct FileInput<'a> {
    pub bytes: &'a [u8],
    pub filename: &'a str,
    pub content_type: Option<&'a str>,
}

impl FileInput<'_> {
    fn extension(&self) -> String {
        self.filename
            .rsplit_once('.')
            .map(|(_, ext)| ext.to_ascii_lowercase())
            .unwrap_or_default()
    }

    /// The media type without parameters, lowercased.
    fn mime(&self) -> String {
        self.content_type
            .unwrap_or_default()
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
    }

    fn looks_html(&self) -> bool {
        matches!(self.mime().as_str(), "text/html" | "application/xhtml+xml")
            || matches!(self.extension().as_str(), "html" | "htm" | "xhtml")
    }
}

/// Converts one file to text. Return `Err` to let the next parser in the
/// chain try.
pub trait FileParser: Send + Sync {
    fn name(&self) -> &str;
    fn convert(&self, file: &FileInput<'_>) -> Result<String>;
}

/// What a successful chain produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    pub text: String,
    /// The parser that succeeded.
    pub parser: String,
}

/// Named parsers, tried in caller-chosen order.
#[derive(Clone, Default)]
pub struct ParserRegistry {
    parsers: BTreeMap<String, Arc<dyn FileParser>>,
}

impl ParserRegistry {
    /// `text` and `html`.
    pub fn builtin() -> Self {
        let mut r = Self::default();
        r.register(Arc::new(TextParser));
        r.register(Arc::new(HtmlParser));
        r
    }

    /// Add (or replace) a parser under its `name()`.
    pub fn register(&mut self, parser: Arc<dyn FileParser>) {
        self.parsers.insert(parser.name().to_string(), parser);
    }

    /// Registered names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.parsers.keys().cloned().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.parsers.contains_key(name)
    }

    /// Try `chain` in order; the first success wins. Fails with every
    /// parser's reason when none succeeds.
    pub fn convert_with_fallback(
        &self,
        chain: &[String],
        file: &FileInput<'_>,
    ) -> Result<Converted> {
        if chain.is_empty() {
            return Err(Error::BadInput("no parser chain given".into()));
        }
        let mut reasons = vec![];
        for name in chain {
            let Some(parser) = self.parsers.get(name) else {
                reasons.push(format!("{name}: not registered"));
                continue;
            };
            match parser.convert(file) {
                Ok(text) => {
                    return Ok(Converted {
                        text,
                        parser: name.clone(),
                    })
                }
                Err(e) => reasons.push(format!("{name}: {}", strip_bad_input(&e))),
            }
        }
        Err(Error::BadInput(format!(
            "no parser could convert '{}' ({})",
            file.filename,
            reasons.join("; ")
        )))
    }
}

fn strip_bad_input(e: &Error) -> String {
    match e {
        Error::BadInput(m) => m.clone(),
        other => other.to_string(),
    }
}

/// `text`: valid UTF-8 without NUL bytes, HTML excluded.
pub struct TextParser;

impl FileParser for TextParser {
    fn name(&self) -> &str {
        "text"
    }

    fn convert(&self, file: &FileInput<'_>) -> Result<String> {
        if file.looks_html() {
            return Err(Error::BadInput("HTML is handled by the html parser".into()));
        }
        decode_text(file.bytes)
    }
}

fn decode_text(bytes: &[u8]) -> Result<String> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    if bytes.contains(&0) {
        return Err(Error::BadInput("binary content (NUL bytes)".into()));
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| Error::BadInput("not valid UTF-8 text".into()))?;
    Ok(text.replace("\r\n", "\n"))
}

/// `html`: the document's visible text.
pub struct HtmlParser;

impl FileParser for HtmlParser {
    fn name(&self) -> &str {
        "html"
    }

    fn convert(&self, file: &FileInput<'_>) -> Result<String> {
        let source = decode_text(file.bytes)?;
        if !file.looks_html() && !sniff_html(&source) {
            return Err(Error::BadInput("not an HTML document".into()));
        }
        Ok(html_to_text(&source))
    }
}

fn sniff_html(s: &str) -> bool {
    let head: String = s
        .trim_start()
        .chars()
        .take(256)
        .collect::<String>()
        .to_ascii_lowercase();
    head.starts_with("<!doctype html") || head.starts_with("<html") || head.contains("<body")
}

/// Elements whose content is never visible text.
const HIDDEN: &[&str] = &[
    "script", "style", "head", "noscript", "template", "svg", "iframe", "object",
];

/// Elements that break the text flow.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
    "title",
];

/// Strip tags, drop hidden elements, decode entities, and normalize
/// whitespace: runs of spaces collapse and every block boundary becomes
/// one line break (blank lines are dropped).
pub fn html_to_text(html: &str) -> String {
    let mut out = String::new();
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        out.push_str(&decode_entities(&rest[..lt]));
        rest = &rest[lt..];
        if let Some(after) = rest.strip_prefix("<!--") {
            rest = after.find("-->").map_or("", |end| &after[end + 3..]);
            continue;
        }
        let Some(gt) = tag_end(rest) else {
            // A stray '<' with no closing '>': keep it as text.
            out.push('<');
            rest = &rest[1..];
            continue;
        };
        let tag = &rest[1..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if !closing && HIDDEN.contains(&name.as_str()) && !tag.ends_with('/') {
            rest = skip_element(rest, &name);
            out.push('\n');
            continue;
        }
        if BLOCKS.contains(&name.as_str()) {
            out.push('\n');
        } else if matches!(name.as_str(), "td" | "th") {
            out.push(' ');
        }
    }
    out.push_str(&decode_entities(rest));
    normalize_whitespace(&out)
}

/// Index of the `>` closing the tag at the start of `s`, honoring quoted
/// attribute values.
fn tag_end(s: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (i, c) in s.char_indices().skip(1) {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(c),
            (None, '>') => return Some(i),
            (None, '<') if i == 1 => return None,
            _ => {}
        }
    }
    None
}

/// Skip to just past `</name>` (case-insensitive); the rest of the input
/// if it never closes.
fn skip_element<'a>(s: &'a str, name: &str) -> &'a str {
    let lower = s.to_ascii_lowercase();
    let needle = format!("</{name}");
    match lower.find(&needle) {
        Some(at) => {
            let after = &s[at..];
            after.find('>').map_or("", |gt| &after[gt + 1..])
        }
        None => "",
    }
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest[1..]
            .find(';')
            .filter(|&end| end <= 10)
            .and_then(|end| {
                let entity = &rest[1..1 + end];
                decode_entity(entity).map(|c| (c, end + 2))
            });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn decode_entity(entity: &str) -> Option<char> {
    if let Some(num) = entity.strip_prefix('#') {
        let code = match num.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => num.parse().ok()?,
        };
        return char::from_u32(code);
    }
    Some(match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "ndash" => '–',
        "mdash" => '—',
        "hellip" => '…',
        "lsquo" => '‘',
        "rsquo" => '’',
        "ldquo" => '“',
        "rdquo" => '”',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "euro" => '€',
        "middot" => '·',
        "bull" => '•',
        _ => return None,
    })
}

fn normalize_whitespace(s: &str) -> String {
    s.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file<'a>(bytes: &'a [u8], filename: &'a str, ct: Option<&'a str>) -> FileInput<'a> {
        FileInput {
            bytes,
            filename,
            content_type: ct,
        }
    }

    fn chain(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn text_decodes_utf8_and_strips_bom() {
        let r = ParserRegistry::builtin();
        let out = r
            .convert_with_fallback(
                &chain(&["text"]),
                &file(
                    "\u{feff}Alice joined Google.\r\nCafé ☕".as_bytes(),
                    "a.md",
                    None,
                ),
            )
            .unwrap();
        assert_eq!(out.text, "Alice joined Google.\nCafé ☕");
        assert_eq!(out.parser, "text");
        for (bytes, why) in [
            (&b"PK\x03\x04\0\0"[..], "binary"),
            (&b"\xff\xfe\xfd"[..], "latin"),
        ] {
            let err = r
                .convert_with_fallback(&chain(&["text"]), &file(bytes, "x.bin", None))
                .unwrap_err();
            assert!(matches!(err, Error::BadInput(_)), "{why}");
        }
    }

    #[test]
    fn html_strips_tags_scripts_and_styles() {
        let html = r#"<!DOCTYPE html><html><head><title>T</title><style>p{color:red}</style>
            <script>var x = "<p>not text</p>";</script></head>
            <body><h1>Notes</h1><p>Alice &amp; Bob joined <b>Google</b>&nbsp;in&#32;2024.</p>
            <!-- hidden <p>comment</p> --><ul><li>one</li><li>two &lt;3</li></ul>
            <a href="x>y">link</a><SCRIPT type="x">alert(1)</SCRIPT>tail</body></html>"#;
        assert_eq!(
            html_to_text(html),
            "Notes\nAlice & Bob joined Google in 2024.\none\ntwo <3\nlink\ntail"
        );
    }

    #[test]
    fn chain_falls_back_in_order() {
        let r = ParserRegistry::builtin();
        let page = b"<html><body><p>Hi there</p></body></html>";
        // `text` declines HTML, so the default chain lands on `html`.
        let out = r
            .convert_with_fallback(&chain(&["text", "html"]), &file(page, "p.html", None))
            .unwrap();
        assert_eq!(
            (out.text.as_str(), out.parser.as_str()),
            ("Hi there", "html")
        );
        // A sniffed HTML body without a telling name or type still converts.
        let out = r
            .convert_with_fallback(&chain(&["html", "text"]), &file(page, "upload", None))
            .unwrap();
        assert_eq!(out.parser, "html");
        // Plain text is not HTML: `html` declines, `text` takes it.
        let out = r
            .convert_with_fallback(&chain(&["html", "text"]), &file(b"plain", "n.txt", None))
            .unwrap();
        assert_eq!(out.parser, "text");
    }

    #[test]
    fn unsupported_files_report_every_reason() {
        let r = ParserRegistry::builtin();
        let err = r
            .convert_with_fallback(
                &chain(&["text", "html", "pdf"]),
                &file(b"%PDF-1.7\0\x01", "r.pdf", Some("application/pdf")),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("no parser could convert 'r.pdf'"), "{err}");
        assert!(err.contains("text: binary content"), "{err}");
        assert!(err.contains("pdf: not registered"), "{err}");
        assert_eq!(r.names(), vec!["html", "text"]);
    }

    #[test]
    fn attachment_ids_are_content_addressed() {
        let a = Attachment::new("b", "d", "f.txt", Some(""), b"abc".to_vec(), 1);
        assert_eq!(
            a.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(a.id, "ba7816bf8f01");
        assert_eq!(a.content_type, None);
    }
}
