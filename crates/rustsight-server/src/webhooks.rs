//! Webhooks (RS-10b): event fan-out, durable delivery, and the
//! Hindsight-compatible management API.
//!
//! Behavior follows upstream hindsight-api 1.2.1 `webhooks/manager.py`,
//! `webhooks/url_guard.py` and `MemoryEngine._handle_webhook_delivery`:
//!
//! - [`WebhookSink`] (an [`EventSink`]) writes one delivery row per enabled
//!   webhook subscribed to the event — the bank's own plus the global one
//!   from `RUSTSIGHT_WEBHOOK_URL` — at the moment the event fires, so a
//!   restart resumes pending deliveries (transactional-outbox shape).
//! - [`spawn_delivery_worker`] sends them: upstream headers and HMAC
//!   signatures, GET/POST, custom headers/params, a per-webhook timeout,
//!   and the retry schedule 5s / 5m / 30m / 2h / 5h (six attempts).
//! - The SSRF guard refuses loopback/private/link-local/CGNAT/reserved
//!   destinations unless `RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS` permits them —
//!   at registration for IP literals, at delivery for every resolved
//!   address (the connection is pinned to the addresses checked, and
//!   redirects are not followed).

use crate::compat::{iso, ApiError};
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, patch};
use axum::{Json, Router};
use rustsight_core::events::{EngineEvent, EventSink, EventType};
use rustsight_core::types::*;
use rustsight_core::{Clock, Engine, MemoryStore};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// Retry delays in seconds after each failed attempt (upstream
/// `RETRY_DELAYS`); a delivery gets one first attempt plus these.
pub const RETRY_DELAYS_SECS: [i64; 5] = [5, 300, 1800, 7200, 18000];
pub const MAX_ATTEMPTS: u32 = RETRY_DELAYS_SECS.len() as u32 + 1;

/// Stored response bodies are truncated to this many bytes.
const MAX_STORED_BODY: usize = 4096;

// ── Settings ─────────────────────────────────────────────────────────────

/// Operator exceptions to the private-range block: exact hostnames and
/// IP/CIDR networks (upstream `Allowlist`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Allowlist {
    hostnames: Vec<String>,
    networks: Vec<(IpAddr, u8)>,
}

impl Allowlist {
    pub fn parse(entries: &str) -> Self {
        let mut out = Allowlist::default();
        for raw in entries.split(',') {
            let entry = raw.trim();
            if entry.is_empty() {
                continue;
            }
            match parse_network(entry) {
                Some(net) => out.networks.push(net),
                None => out.hostnames.push(entry.to_lowercase()),
            }
        }
        out
    }

    fn allows_host(&self, host: &str) -> bool {
        self.hostnames.iter().any(|h| h == &host.to_lowercase())
    }

    fn allows_ip(&self, ip: IpAddr) -> bool {
        self.networks
            .iter()
            .any(|(net, len)| in_network(ip, *net, *len))
    }
}

fn parse_network(entry: &str) -> Option<(IpAddr, u8)> {
    let (addr, len) = match entry.split_once('/') {
        Some((a, l)) => (a, Some(l.parse::<u8>().ok()?)),
        None => (entry, None),
    };
    let ip: IpAddr = addr.trim_matches(['[', ']']).parse().ok()?;
    let max = if ip.is_ipv4() { 32 } else { 128 };
    let len = len.unwrap_or(max);
    (len <= max).then_some((ip, len))
}

fn in_network(ip: IpAddr, net: IpAddr, len: u8) -> bool {
    match (ip, net) {
        (IpAddr::V4(a), IpAddr::V4(n)) => {
            let mask = if len == 0 { 0 } else { u32::MAX << (32 - len) };
            u32::from(a) & mask == u32::from(n) & mask
        }
        (IpAddr::V6(a), IpAddr::V6(n)) => {
            let mask = if len == 0 {
                0
            } else {
                u128::MAX << (128 - len)
            };
            u128::from(a) & mask == u128::from(n) & mask
        }
        _ => false,
    }
}

/// Ranges unsafe for caller-supplied fetches (upstream `_ip_is_blocked`:
/// Python's private / loopback / link-local / reserved / multicast /
/// unspecified classification plus CGNAT).
pub fn ip_is_blocked(ip: IpAddr) -> bool {
    let v4_blocked = |a: Ipv4Addr| {
        let o = a.octets();
        a.is_private()
            || a.is_loopback()
            || a.is_link_local()
            || a.is_multicast()
            || a.is_broadcast()
            || a.is_unspecified()
            || a.is_documentation()
            || o[0] == 0
            || (o[0] == 100 && (o[1] & 0xC0) == 64) // 100.64.0.0/10 CGNAT
            || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24
            || (o[0] == 198 && (o[1] & 0xFE) == 18) // 198.18.0.0/15
            || o[0] >= 240 // 240.0.0.0/4 reserved
    };
    match ip {
        IpAddr::V4(a) => v4_blocked(a),
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return v4_blocked(v4);
            }
            let seg = a.segments();
            a.is_loopback()
                || a.is_unspecified()
                || a.is_multicast()
                || (seg[0] & 0xFE00) == 0xFC00 // fc00::/7 unique local
                || (seg[0] & 0xFFC0) == 0xFE80 // fe80::/10 link-local
                || (seg[0] & 0xFFC0) == 0xFEC0 // fec0::/10 site-local
                || (seg[0] == 0x2001 && seg[1] == 0x0DB8) // documentation
                || (seg[0] == 0x2001 && seg[1] < 0x0200) // 2001::/23 IETF
                || (seg[0] == 0x0100 && seg[1..4] == [0, 0, 0]) // 100::/64 discard
        }
    }
}

/// Webhook settings read from the environment.
#[derive(Debug, Clone, Default)]
pub struct WebhookSettings {
    /// The env-configured global webhook (`RUSTSIGHT_WEBHOOK_URL`).
    pub global: Option<Webhook>,
    pub allowlist: Allowlist,
    /// Return stored receiver response bodies from the deliveries API
    /// (withheld by default: a response-exfiltration primitive).
    pub expose_response_body: bool,
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(format!("RUSTSIGHT_{name}"))
        .or_else(|_| std::env::var(format!("HINDSIGHT_API_{name}")))
        .ok()
        .filter(|v| !v.trim().is_empty())
}

impl WebhookSettings {
    /// `RUSTSIGHT_WEBHOOK_URL` / `_SECRET` / `_EVENT_TYPES` (comma list,
    /// default `consolidation.completed`), `RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS`,
    /// `RUSTSIGHT_WEBHOOK_EXPOSE_RESPONSE_BODY`; each also answers to its
    /// `HINDSIGHT_API_*` spelling.
    pub fn from_env() -> Self {
        let event_types = env_var("WEBHOOK_EVENT_TYPES")
            .unwrap_or_else(|| "consolidation.completed".into())
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        let global = env_var("WEBHOOK_URL").map(|url| Webhook {
            id: String::new(),
            bank_id: None,
            url,
            secret: env_var("WEBHOOK_SECRET"),
            event_types,
            enabled: true,
            http_config: WebhookHttpConfig::default(),
            created_at_ms: 0,
            updated_at_ms: 0,
        });
        let truthy = |v: Option<String>| {
            v.is_some_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        };
        Self {
            global,
            allowlist: Allowlist::parse(&env_var("WEBHOOK_ALLOWED_HOSTS").unwrap_or_default()),
            expose_response_body: truthy(env_var("WEBHOOK_EXPOSE_RESPONSE_BODY")),
        }
    }
}

// ── URL guard ────────────────────────────────────────────────────────────

/// Registration-time check, no DNS (upstream `validate_url_syntax`):
/// http(s) only, a host, no control characters, and no IP-literal host in
/// a blocked range. The `url` crate normalizes legacy numeric IPv4
/// spellings (`127.1`, `2130706433`, `0x7f.1`) to addresses, so they are
/// judged as the addresses they are.
pub fn validate_url_syntax(raw: &str, allowlist: &Allowlist) -> Result<reqwest::Url, String> {
    if raw.chars().any(|c| (c as u32) < 0x20 || c as u32 == 0x7F) {
        return Err("Invalid webhook URL: contains non-printable characters".into());
    }
    let url = reqwest::Url::parse(raw).map_err(|e| format!("Invalid webhook URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!(
            "Webhook URL scheme must be http or https, got '{}'",
            url.scheme()
        ));
    }
    let host = url.host_str().unwrap_or_default().to_string();
    if host.is_empty() {
        return Err("Webhook URL must include a host".into());
    }
    if allowlist.allows_host(&host) {
        return Ok(url);
    }
    if let Some(ip) = literal_ip(&url) {
        if ip_is_blocked(ip) && !allowlist.allows_ip(ip) {
            return Err(format!(
                "Webhook URL host '{host}' targets a private/loopback/link-local address, \
                 which is not an allowed destination"
            ));
        }
    }
    Ok(url)
}

fn literal_ip(url: &reqwest::Url) -> Option<IpAddr> {
    match url.host()? {
        url::Host::Ipv4(a) => Some(IpAddr::V4(a)),
        url::Host::Ipv6(a) => Some(IpAddr::V6(a)),
        url::Host::Domain(_) => None,
    }
}

/// Delivery-time check (upstream `resolve_and_validate`): the addresses a
/// connection may use. A DNS answer is rejected whole if any address is
/// blocked, so a split-horizon or rebound name cannot slip one through.
async fn resolve_and_validate(
    url: &reqwest::Url,
    allowlist: &Allowlist,
) -> Result<Vec<SocketAddr>, String> {
    let host = url.host_str().unwrap_or_default();
    let port = url.port_or_known_default().unwrap_or(80);
    if let Some(ip) = literal_ip(url) {
        if ip_is_blocked(ip) && !allowlist.allows_ip(ip) && !allowlist.allows_host(host) {
            return Err(format!(
                "Webhook host '{host}' targets a disallowed address"
            ));
        }
        return Ok(vec![SocketAddr::new(ip, port)]);
    }
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("Webhook host '{host}' did not resolve: {e}"))?
        .collect();
    if addrs.is_empty() {
        return Err(format!(
            "Webhook host '{host}' did not resolve to any address"
        ));
    }
    if !allowlist.allows_host(host) {
        for a in &addrs {
            if ip_is_blocked(a.ip()) && !allowlist.allows_ip(a.ip()) {
                return Err(format!(
                    "Webhook host '{host}' resolved to disallowed address {}",
                    a.ip()
                ));
            }
        }
    }
    Ok(addrs)
}

// ── Signing ──────────────────────────────────────────────────────────────

/// HMAC-SHA256 (RFC 2104) over `msg`, hex-encoded.
pub fn hmac_sha256_hex(key: &[u8], msg: &[u8]) -> String {
    const BLOCK: usize = 64;
    let mut k = if key.len() > BLOCK {
        Sha256::digest(key).to_vec()
    } else {
        key.to_vec()
    };
    k.resize(BLOCK, 0);
    let pad = |byte: u8| k.iter().map(|b| b ^ byte).collect::<Vec<u8>>();
    let inner = Sha256::new()
        .chain_update(pad(0x36))
        .chain_update(msg)
        .finalize();
    let outer = Sha256::new()
        .chain_update(pad(0x5c))
        .chain_update(inner)
        .finalize();
    outer.iter().map(|b| format!("{b:02x}")).collect()
}

/// `sha256=<hex>` over the exact body bytes (upstream `_sign_payload`;
/// GitHub's construction, sent as both `X-Hindsight-Signature` and
/// `X-Hub-Signature-256`).
pub fn sign_payload(secret: &str, body: &[u8]) -> String {
    format!("sha256={}", hmac_sha256_hex(secret.as_bytes(), body))
}

/// `t=<unix seconds>,v1=<hex>` over `<t>.<body>` (upstream
/// `_sign_payload_v2`): the timestamp is signed, so receivers can reject
/// replays outside a tolerance window.
pub fn sign_payload_v2(secret: &str, body: &[u8], unix_secs: i64) -> String {
    let mut signed = format!("{unix_secs}.").into_bytes();
    signed.extend_from_slice(body);
    format!(
        "t={unix_secs},v1={}",
        hmac_sha256_hex(secret.as_bytes(), &signed)
    )
}

// ── Fan-out (event sink) ─────────────────────────────────────────────────

/// Turns engine events into delivery rows and wakes the delivery worker.
pub struct WebhookSink {
    store: Arc<dyn MemoryStore>,
    clock: Arc<dyn Clock>,
    settings: Arc<WebhookSettings>,
    wakeup: Arc<Notify>,
}

impl WebhookSink {
    pub fn new(
        store: Arc<dyn MemoryStore>,
        clock: Arc<dyn Clock>,
        settings: Arc<WebhookSettings>,
        wakeup: Arc<Notify>,
    ) -> Self {
        Self {
            store,
            clock,
            settings,
            wakeup,
        }
    }

    fn queue(&self, event: &EngineEvent) -> rustsight_core::Result<usize> {
        let (bank_hooks, _) = self
            .store
            .list_webhooks(&event.bank_id, usize::MAX >> 1, 0)?;
        let payload = event.wire_json();
        let now = self.clock.now_ms();
        let mut queued = 0;
        for hook in self.settings.global.iter().chain(bank_hooks.iter()) {
            if !hook.enabled || !hook.event_types.iter().any(|t| t == event.event.as_str()) {
                continue;
            }
            self.store.upsert_webhook_delivery(&WebhookDelivery {
                id: uuid::Uuid::new_v4().to_string(),
                webhook_id: Some(hook.id.clone()).filter(|id| !id.is_empty()),
                bank_id: event.bank_id.clone(),
                event_id: event.event_id.clone(),
                event_type: event.event.as_str().to_string(),
                url: hook.url.clone(),
                secret: hook.secret.clone(),
                http_config: hook.http_config.clone(),
                payload: payload.clone(),
                status: "pending".into(),
                retry_count: 0,
                next_attempt_at_ms: now,
                last_error: None,
                last_response_status: None,
                last_response_body: None,
                last_attempt_at_ms: None,
                created_at_ms: now,
                updated_at_ms: now,
            })?;
            queued += 1;
        }
        Ok(queued)
    }
}

impl EventSink for WebhookSink {
    fn emit(&self, event: &EngineEvent) {
        match self.queue(event) {
            Ok(0) => {}
            Ok(_) => self.wakeup.notify_one(),
            Err(err) => eprintln!(
                "rustsight: queueing webhook deliveries for {} (bank {}) failed: {err}",
                event.event.as_str(),
                event.bank_id
            ),
        }
    }
}

// ── Delivery ─────────────────────────────────────────────────────────────

enum Outcome {
    Delivered(u16, String),
    /// Non-2xx response or transport failure: retry on schedule.
    Retryable(String, Option<u16>, Option<String>),
    /// The destination is not allowed: never becomes valid on retry.
    Blocked(String),
}

fn truncate_body(mut body: String) -> String {
    if body.len() > MAX_STORED_BODY {
        let mut cut = MAX_STORED_BODY;
        while !body.is_char_boundary(cut) {
            cut -= 1;
        }
        body.truncate(cut);
    }
    body
}

async fn send(d: &WebhookDelivery, allowlist: &Allowlist, unix_secs: i64) -> Outcome {
    let url = match validate_url_syntax(&d.url, allowlist) {
        Ok(u) => u,
        Err(e) => return Outcome::Blocked(e),
    };
    let addrs = match resolve_and_validate(&url, allowlist).await {
        Ok(a) => a,
        // Resolution failures are transient; a disallowed answer is not.
        Err(e) if e.contains("did not resolve") => return Outcome::Retryable(e, None, None),
        Err(e) => return Outcome::Blocked(e),
    };
    let host = url.host_str().unwrap_or_default().to_string();
    let timeout = Duration::from_secs(d.http_config.timeout_seconds.clamp(1, 300));
    let client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .resolve_to_addrs(&host, &addrs)
        .timeout(timeout)
        .build()
    {
        Ok(c) => c,
        Err(e) => return Outcome::Retryable(format!("http client: {e}"), None, None),
    };
    let body = d.payload.as_bytes();
    let mut headers = reqwest::header::HeaderMap::new();
    let mut put = |name: &str, value: &str| {
        if let (Ok(n), Ok(v)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            headers.insert(n, v);
        }
    };
    // Custom headers first so the Hindsight-controlled ones always win;
    // Content-Type stays overridable.
    put("Content-Type", "application/json");
    for (k, v) in &d.http_config.headers {
        put(k, v);
    }
    put("X-Hindsight-Event", &d.event_type);
    put("X-Hindsight-Event-Id", &d.event_id);
    put("X-Hindsight-Delivery-Id", &d.id);
    if let Some(secret) = d.secret.as_deref().filter(|s| !s.is_empty()) {
        let sig = sign_payload(secret, body);
        put("X-Hindsight-Signature", &sig);
        put("X-Hub-Signature-256", &sig);
        put(
            "X-Hindsight-Signature-V2",
            &sign_payload_v2(secret, body, unix_secs),
        );
    }
    let is_get = d.http_config.method.eq_ignore_ascii_case("GET");
    let mut req = if is_get {
        client.get(url)
    } else {
        client.post(url).body(d.payload.clone())
    };
    req = req.headers(headers).query(&d.http_config.params);
    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let text = truncate_body(resp.text().await.unwrap_or_default());
            if (200..300).contains(&status) {
                Outcome::Delivered(status, text)
            } else {
                Outcome::Retryable(format!("HTTP {status}"), Some(status), Some(text))
            }
        }
        Err(e) => Outcome::Retryable(e.to_string(), None, None),
    }
}

/// Attempt one delivery and record the result: completed, a scheduled
/// retry, or failed (destination blocked, or attempts exhausted).
pub async fn deliver_once(
    store: &dyn MemoryStore,
    clock: &dyn Clock,
    allowlist: &Allowlist,
    mut d: WebhookDelivery,
) -> rustsight_core::Result<WebhookDelivery> {
    let now = clock.now_ms();
    let outcome = send(&d, allowlist, now / 1000).await;
    d.last_attempt_at_ms = Some(now);
    d.updated_at_ms = now;
    match outcome {
        Outcome::Delivered(status, body) => {
            d.status = "completed".into();
            d.last_error = None;
            d.last_response_status = Some(status);
            d.last_response_body = Some(body);
        }
        Outcome::Blocked(err) => {
            eprintln!("rustsight: webhook delivery {} blocked: {err}", d.id);
            d.status = "failed".into();
            d.last_error = Some(err);
            d.last_response_status = None;
            d.last_response_body = None;
        }
        Outcome::Retryable(err, status, body) => {
            d.last_error = Some(err);
            d.last_response_status = status;
            d.last_response_body = body;
            if d.retry_count + 1 >= MAX_ATTEMPTS {
                eprintln!(
                    "rustsight: webhook delivery {} to {} failed permanently after {} attempts",
                    d.id,
                    d.url,
                    d.retry_count + 1
                );
                d.status = "failed".into();
            } else {
                let delay = RETRY_DELAYS_SECS[d.retry_count as usize];
                d.retry_count += 1;
                d.next_attempt_at_ms = now + delay * 1000;
            }
        }
    }
    store.upsert_webhook_delivery(&d)?;
    Ok(d)
}

/// Run due deliveries until the task is aborted: up to 8 at once, then
/// sleep until the next one is due (at most 30s) or a new event wakes it.
pub fn spawn_delivery_worker(
    store: Arc<dyn MemoryStore>,
    clock: Arc<dyn Clock>,
    settings: Arc<WebhookSettings>,
    wakeup: Arc<Notify>,
) -> JoinHandle<()> {
    use futures::StreamExt;
    tokio::spawn(async move {
        loop {
            let now = clock.now_ms();
            let due = store.due_webhook_deliveries(now, 32).unwrap_or_else(|err| {
                eprintln!("rustsight: listing webhook deliveries failed: {err}");
                vec![]
            });
            if !due.is_empty() {
                futures::stream::iter(due)
                    .for_each_concurrent(8, |d| {
                        let (store, clock, settings) = (&store, &clock, &settings);
                        async move {
                            if let Err(err) =
                                deliver_once(store.as_ref(), clock.as_ref(), &settings.allowlist, d)
                                    .await
                            {
                                eprintln!("rustsight: recording webhook delivery failed: {err}");
                            }
                        }
                    })
                    .await;
                continue;
            }
            let wait_ms = match store.next_webhook_delivery_at() {
                Ok(Some(at)) => (at - clock.now_ms()).clamp(50, 30_000),
                _ => 30_000,
            };
            tokio::select! {
                _ = wakeup.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(wait_ms as u64)) => {}
            }
        }
    })
}

/// Start webhooks for a long-running server: install the fan-out sink on
/// `engine` and spawn the delivery worker (abort the handle on shutdown).
/// An engine that already has an event sink (an embedder's own) keeps it,
/// and only the worker for previously queued deliveries runs.
pub fn start(engine: &Engine, settings: Arc<WebhookSettings>) -> JoinHandle<()> {
    let wakeup = Arc::new(Notify::new());
    let sink = WebhookSink::new(
        Arc::clone(&engine.store),
        Arc::clone(&engine.clock),
        Arc::clone(&settings),
        Arc::clone(&wakeup),
    );
    if engine.set_event_sink(Arc::new(sink)).is_err() {
        eprintln!("rustsight: engine already has an event sink; webhooks will not be queued");
    }
    spawn_delivery_worker(
        Arc::clone(&engine.store),
        Arc::clone(&engine.clock),
        settings,
        wakeup,
    )
}

// ── HTTP API (Hindsight-compatible) ──────────────────────────────────────

/// Settings the HTTP handlers consult (registration allowlist, body
/// exposure); set once per process by `serve` / the MCP server. Anything
/// that skips that gets env-derived settings on first use.
static SETTINGS: std::sync::OnceLock<Arc<WebhookSettings>> = std::sync::OnceLock::new();

pub fn set_settings(settings: Arc<WebhookSettings>) {
    let _ = SETTINGS.set(settings);
}

fn settings() -> Arc<WebhookSettings> {
    Arc::clone(SETTINGS.get_or_init(|| Arc::new(WebhookSettings::from_env())))
}

pub(crate) fn routes() -> Router<Arc<Engine>> {
    let base = "/v1/default/banks/{bank_id}/webhooks";
    Router::new()
        .route(base, get(list_webhooks).post(create_webhook))
        .route(
            &format!("{base}/{{webhook_id}}"),
            patch(update_webhook).delete(delete_webhook),
        )
        .route(
            &format!("{base}/{{webhook_id}}/deliveries"),
            get(list_deliveries),
        )
}

type ApiResult = Result<Json<Value>, ApiError>;

fn webhook_json(w: &Webhook) -> Value {
    json!({
        "id": w.id,
        "bank_id": w.bank_id,
        "url": w.url,
        // Never returned (upstream redacts it in every response).
        "secret": Value::Null,
        "event_types": w.event_types,
        "enabled": w.enabled,
        "http_config": w.http_config,
        "created_at": iso(w.created_at_ms),
        "updated_at": iso(w.updated_at_ms),
    })
}

fn check_event_types(types: &[String]) -> Result<(), ApiError> {
    for t in types {
        if EventType::parse(t).is_none() {
            let supported: Vec<&str> = EventType::ALL.iter().map(|t| t.as_str()).collect();
            return Err(ApiError::unprocessable(format!(
                "Unsupported event type {t:?}; supported: {}",
                supported.join(", ")
            )));
        }
    }
    Ok(())
}

fn check_url(url: &str) -> Result<(), ApiError> {
    validate_url_syntax(url, &settings().allowlist)
        .map(|_| ())
        .map_err(ApiError::unprocessable)
}

fn http_config(v: &Value) -> Result<WebhookHttpConfig, ApiError> {
    serde_json::from_value(v.clone())
        .map_err(|e| ApiError::unprocessable(format!("http_config: {e}")))
}

#[derive(Deserialize)]
struct CreateWebhook {
    url: String,
    #[serde(default)]
    secret: Option<String>,
    #[serde(default = "default_event_types")]
    event_types: Vec<String>,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    http_config: Option<Value>,
}

fn default_event_types() -> Vec<String> {
    vec!["consolidation.completed".into()]
}

fn default_true() -> bool {
    true
}

async fn create_webhook(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    body: Result<Json<CreateWebhook>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let Json(body) = body?;
    check_url(&body.url)?;
    check_event_types(&body.event_types)?;
    let http_config = match &body.http_config {
        Some(v) => http_config(v)?,
        None => WebhookHttpConfig::default(),
    };
    // Upstream creates the bank row lazily with the webhook.
    let _guard = engine.bank_read(&bank_id).await;
    if engine.store.get_bank(&bank_id)?.is_none() {
        engine.store.ensure_bank(&Bank {
            id: bank_id.clone(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })?;
    }
    let now = engine.clock.now_ms();
    let hook = Webhook {
        id: uuid::Uuid::new_v4().to_string(),
        bank_id: Some(bank_id),
        url: body.url,
        secret: body.secret.filter(|s| !s.is_empty()),
        event_types: body.event_types,
        enabled: body.enabled,
        http_config,
        created_at_ms: now,
        updated_at_ms: now,
    };
    engine.store.upsert_webhook(&hook)?;
    Ok((StatusCode::CREATED, Json(webhook_json(&hook))))
}

#[derive(Deserialize)]
struct Paging {
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_webhooks(
    State(engine): State<Arc<Engine>>,
    Path(bank_id): Path<String>,
    Query(paging): Query<Paging>,
) -> ApiResult {
    let limit = paging.limit.unwrap_or(100);
    if limit > 1000 {
        return Err(ApiError::unprocessable("limit must be between 0 and 1000"));
    }
    let offset = paging.offset.unwrap_or(0);
    if engine.store.get_bank(&bank_id)?.is_none() {
        return Err(ApiError::not_found(format!("Bank '{bank_id}' not found")));
    }
    let (items, total) = engine.store.list_webhooks(&bank_id, limit, offset)?;
    Ok(Json(json!({
        "items": items.iter().map(webhook_json).collect::<Vec<_>>(),
        "total": total,
        "limit": limit,
        "offset": offset,
    })))
}

fn parse_webhook_id(raw: &str) -> Result<String, ApiError> {
    uuid::Uuid::parse_str(raw)
        .map(|u| u.to_string())
        .map_err(|_| ApiError::unprocessable("Invalid webhook ID"))
}

async fn update_webhook(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, webhook_id)): Path<(String, String)>,
    body: Result<Json<Map<String, Value>>, JsonRejection>,
) -> ApiResult {
    let Json(body) = body?;
    let webhook_id = parse_webhook_id(&webhook_id)?;
    let _guard = engine.bank_read(&bank_id).await;
    let mut hook = engine
        .store
        .get_webhook(&bank_id, &webhook_id)?
        .ok_or_else(|| ApiError::not_found("Webhook not found"))?;
    let mut changed = false;
    // PATCH: only fields present are applied; `secret: null` clears it.
    if let Some(v) = body.get("url").filter(|v| !v.is_null()) {
        let url = v
            .as_str()
            .ok_or_else(|| ApiError::unprocessable("url must be a string"))?;
        check_url(url)?;
        hook.url = url.to_string();
        changed = true;
    }
    if let Some(v) = body.get("secret") {
        hook.secret = match v {
            Value::Null => None,
            Value::String(s) => Some(s.clone()).filter(|s| !s.is_empty()),
            _ => return Err(ApiError::unprocessable("secret must be a string or null")),
        };
        changed = true;
    }
    if let Some(v) = body.get("event_types").filter(|v| !v.is_null()) {
        let types: Vec<String> = serde_json::from_value(v.clone())
            .map_err(|_| ApiError::unprocessable("event_types must be a list of strings"))?;
        check_event_types(&types)?;
        hook.event_types = types;
        changed = true;
    }
    if let Some(v) = body.get("enabled").filter(|v| !v.is_null()) {
        hook.enabled = v
            .as_bool()
            .ok_or_else(|| ApiError::unprocessable("enabled must be a boolean"))?;
        changed = true;
    }
    if let Some(v) = body.get("http_config").filter(|v| !v.is_null()) {
        hook.http_config = http_config(v)?;
        changed = true;
    }
    if !changed {
        return Err(ApiError::unprocessable("No fields provided to update"));
    }
    hook.updated_at_ms = engine.clock.now_ms();
    engine.store.upsert_webhook(&hook)?;
    Ok(Json(webhook_json(&hook)))
}

async fn delete_webhook(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, webhook_id)): Path<(String, String)>,
) -> ApiResult {
    let webhook_id = parse_webhook_id(&webhook_id)?;
    let _guard = engine.bank_read(&bank_id).await;
    if !engine.store.delete_webhook(&bank_id, &webhook_id)? {
        return Err(ApiError::not_found("Webhook not found"));
    }
    Ok(Json(json!({ "success": true, "message": Value::Null })))
}

#[derive(Deserialize)]
struct DeliveriesQuery {
    limit: Option<usize>,
    cursor: Option<String>,
}

fn delivery_json(d: &WebhookDelivery, expose_body: bool) -> Value {
    let retry_pending = d.status == "pending" && d.retry_count > 0;
    json!({
        "id": d.id,
        "event_id": d.event_id,
        "webhook_id": d.webhook_id,
        "url": d.url,
        "event_type": d.event_type,
        "status": d.status,
        "attempts": d.retry_count + 1,
        "next_retry_at": if retry_pending { iso(d.next_attempt_at_ms) } else { None },
        "last_error": d.last_error,
        "last_response_status": d.last_response_status,
        "last_response_body": if expose_body { d.last_response_body.clone() } else { None },
        "last_attempt_at": d.last_attempt_at_ms.and_then(iso),
        "created_at": iso(d.created_at_ms),
        "updated_at": iso(d.updated_at_ms),
    })
}

fn cursor_of(d: &WebhookDelivery) -> String {
    format!("{}:{}", d.created_at_ms, d.id)
}

fn parse_cursor(raw: &str) -> Result<(i64, String), ApiError> {
    raw.split_once(':')
        .and_then(|(ms, id)| Some((ms.parse().ok()?, id.to_string())))
        .ok_or_else(|| ApiError::unprocessable("Invalid cursor"))
}

async fn list_deliveries(
    State(engine): State<Arc<Engine>>,
    Path((bank_id, webhook_id)): Path<(String, String)>,
    Query(q): Query<DeliveriesQuery>,
) -> ApiResult {
    let webhook_id = parse_webhook_id(&webhook_id)?;
    let limit = q.limit.unwrap_or(50);
    if limit > 200 {
        return Err(ApiError::unprocessable("limit must be at most 200"));
    }
    if engine.store.get_webhook(&bank_id, &webhook_id)?.is_none() {
        return Err(ApiError::not_found("Webhook not found"));
    }
    let before = q.cursor.as_deref().map(parse_cursor).transpose()?;
    let mut rows =
        engine
            .store
            .list_webhook_deliveries(&bank_id, &webhook_id, limit + 1, before)?;
    let has_more = rows.len() > limit;
    rows.truncate(limit);
    let next_cursor = if has_more {
        rows.last().map(cursor_of)
    } else {
        None
    };
    let expose = settings().expose_response_body;
    Ok(Json(json!({
        "items": rows.iter().map(|d| delivery_json(d, expose)).collect::<Vec<_>>(),
        "next_cursor": next_cursor,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc4231_case_2() {
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Keys longer than a block are hashed first (RFC 4231 case 6).
        assert_eq!(
            hmac_sha256_hex(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            ),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn signatures_use_upstream_formats() {
        assert!(sign_payload("s", b"{}").starts_with("sha256="));
        let v2 = sign_payload_v2("s", b"{}", 1700000000);
        assert_eq!(
            v2,
            format!(
                "t=1700000000,v1={}",
                hmac_sha256_hex(b"s", b"1700000000.{}")
            )
        );
    }

    #[test]
    fn url_guard_blocks_internal_targets_unless_allowlisted() {
        let none = Allowlist::default();
        for bad in [
            "http://127.0.0.1/x",
            "http://127.1/x",
            "http://2130706433/x",
            "http://0x7f.1/x",
            "http://10.1.2.3/",
            "http://169.254.169.254/latest/meta-data",
            "http://100.64.0.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[fd00::1]/",
            "http://0.0.0.0/",
        ] {
            assert!(validate_url_syntax(bad, &none).is_err(), "{bad} allowed");
        }
        for bad in ["ftp://example.com/", "http://", "http://exa\nmple.com/"] {
            assert!(validate_url_syntax(bad, &none).is_err(), "{bad} allowed");
        }
        // DNS names pass registration; they are checked when resolved.
        assert!(validate_url_syntax("https://hooks.example.com/x", &none).is_ok());
        assert!(validate_url_syntax("http://93.184.216.34/", &none).is_ok());
        let allow = Allowlist::parse("127.0.0.1, 10.0.0.0/8, internal.local");
        assert!(validate_url_syntax("http://127.0.0.1:9000/hook", &allow).is_ok());
        assert!(validate_url_syntax("http://10.9.9.9/", &allow).is_ok());
        assert!(validate_url_syntax("http://internal.local/", &allow).is_ok());
        assert!(validate_url_syntax("http://192.168.1.1/", &allow).is_err());
    }

    #[tokio::test]
    async fn resolved_names_are_checked_at_delivery() {
        let none = Allowlist::default();
        let url = reqwest::Url::parse("http://localhost:9/").unwrap();
        let err = resolve_and_validate(&url, &none).await.unwrap_err();
        assert!(err.contains("disallowed address"), "{err}");
        let allow = Allowlist::parse("localhost");
        assert!(resolve_and_validate(&url, &allow).await.is_ok());
    }

    // ── HTTP API + delivery ────────────────────────────────────────────

    use crate::auth::ApiAuth;
    use axum::body::{Body, Bytes};
    use axum::http::{header, HeaderMap, Method, Request};
    use rustsight_core::{FixedClock, RetainRequest};
    use rustsight_providers::{MockEmbed, MockLlm, MockRerank, SimpleTokenEstimator};
    use rustsight_storage::SqliteStore;
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::sync::Mutex;
    use tower::ServiceExt;

    const NOW: i64 = 1_800_000_000_000;

    fn engine() -> Arc<Engine> {
        Arc::new(Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            Arc::new(MockLlm::demo()),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(NOW)),
        ))
    }

    async fn call(
        app: &Router,
        method: Method,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(v) => {
                req = req.header(header::CONTENT_TYPE, "application/json");
                Body::from(v.to_string())
            }
            None => Body::empty(),
        };
        let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn webhook_crud_validates_like_upstream() {
        let engine = engine();
        let app = crate::http::router(Arc::clone(&engine), Arc::new(ApiAuth::None));
        let base = "/v1/default/banks/wb/webhooks";
        let (status, _) = call(&app, Method::GET, base, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "unknown bank");
        for (body, why) in [
            (json!({"url": "http://127.0.0.1:9/x"}), "loopback"),
            (json!({"url": "http://169.254.169.254/"}), "metadata"),
            (json!({"url": "ftp://example.com/"}), "scheme"),
            (
                json!({"url": "https://example.com/", "event_types": ["retain.started"]}),
                "event",
            ),
        ] {
            let (status, resp) = call(&app, Method::POST, base, Some(body)).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{why}: {resp}");
        }
        let (status, created) = call(
            &app,
            Method::POST,
            base,
            Some(
                json!({"url": "https://hooks.example.com/rs", "secret": "s3cret",
                        "http_config": {"method": "GET", "headers": {"X-A": "1"}}}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{created}");
        assert!(created["secret"].is_null(), "secret must never be returned");
        assert_eq!(created["event_types"], json!(["consolidation.completed"]));
        assert_eq!(created["http_config"]["method"], "GET");
        assert_eq!(created["http_config"]["timeout_seconds"], 30);
        let id = created["id"].as_str().unwrap().to_string();
        let stored = engine.store.get_webhook("wb", &id).unwrap().unwrap();
        assert_eq!(stored.secret.as_deref(), Some("s3cret"));

        let (_, list) = call(&app, Method::GET, base, None).await;
        assert_eq!(
            (list["total"].clone(), list["limit"].clone()),
            (json!(1), json!(100))
        );
        let one = format!("{base}/{id}");
        let (status, _) = call(&app, Method::PATCH, &one, Some(json!({}))).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (status, upd) = call(
            &app,
            Method::PATCH,
            &one,
            Some(json!({"enabled": false, "secret": null, "event_types": ["retain.completed"]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{upd}");
        assert_eq!(upd["enabled"], false);
        let stored = engine.store.get_webhook("wb", &id).unwrap().unwrap();
        assert!(stored.secret.is_none() && !stored.enabled);
        let (status, _) = call(
            &app,
            Method::PATCH,
            &format!("{base}/nope"),
            Some(json!({"enabled": true})),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "bad id");
        let (status, deliveries) =
            call(&app, Method::GET, &format!("{one}/deliveries"), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deliveries, json!({"items": [], "next_cursor": null}));
        let (status, del) = call(&app, Method::DELETE, &one, None).await;
        assert_eq!(
            (status, del["success"].clone()),
            (StatusCode::OK, json!(true))
        );
        let (status, _) = call(&app, Method::DELETE, &one, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = call(&app, Method::GET, &format!("{one}/deliveries"), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[derive(Default)]
    struct Received {
        requests: Mutex<Vec<(HeaderMap, Bytes)>>,
        status: AtomicU16,
    }

    /// A local receiver answering every request with `status`.
    async fn receiver() -> (String, Arc<Received>) {
        let got = Arc::new(Received::default());
        got.status.store(200, Ordering::SeqCst);
        let state = Arc::clone(&got);
        let app = Router::new().route(
            "/hook",
            axum::routing::any(move |headers: HeaderMap, body: Bytes| {
                let state = Arc::clone(&state);
                async move {
                    state.requests.lock().unwrap().push((headers, body));
                    StatusCode::from_u16(state.status.load(Ordering::SeqCst)).unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://127.0.0.1:{}/hook", addr.port()), got)
    }

    /// An engine whose retains queue deliveries for one registered webhook.
    fn wired(url: &str, secret: Option<&str>) -> (Arc<Engine>, Webhook) {
        let engine = engine();
        let sink = WebhookSink::new(
            Arc::clone(&engine.store),
            Arc::clone(&engine.clock),
            Arc::new(WebhookSettings::default()),
            Arc::new(Notify::new()),
        );
        engine.set_event_sink(Arc::new(sink)).ok().unwrap();
        let hook = Webhook {
            id: uuid::Uuid::new_v4().to_string(),
            bank_id: Some("wd".into()),
            url: url.into(),
            secret: secret.map(str::to_string),
            event_types: vec!["retain.completed".into()],
            enabled: true,
            http_config: WebhookHttpConfig {
                headers: [
                    ("X-Custom".to_string(), "yes".to_string()),
                    ("X-Hindsight-Event".to_string(), "forged".to_string()),
                ]
                .into(),
                ..Default::default()
            },
            created_at_ms: NOW,
            updated_at_ms: NOW,
        };
        engine
            .store
            .ensure_bank(&Bank {
                id: "wd".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            })
            .unwrap();
        engine.store.upsert_webhook(&hook).unwrap();
        (engine, hook)
    }

    async fn retain_and_take_delivery(engine: &Engine) -> WebhookDelivery {
        engine
            .retain(RetainRequest {
                bank_id: "wd".into(),
                content: "Alice joined Google.".into(),
                document_id: Some("d1".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let mut due = engine.store.due_webhook_deliveries(NOW, 10).unwrap();
        assert_eq!(due.len(), 1, "one delivery per subscribed webhook");
        due.remove(0)
    }

    #[tokio::test]
    async fn signed_delivery_reaches_the_receiver() {
        let (url, got) = receiver().await;
        let (engine, hook) = wired(&url, Some("s3cret"));
        let d = retain_and_take_delivery(&engine).await;
        let allow = Allowlist::parse("127.0.0.1");
        let done = deliver_once(
            engine.store.as_ref(),
            engine.clock.as_ref(),
            &allow,
            d.clone(),
        )
        .await
        .unwrap();
        assert_eq!(done.status, "completed");
        assert_eq!(done.last_response_status, Some(200));

        {
            let requests = got.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            let (headers, body) = &requests[0];
            let h = |name: &str| headers.get(name).map(|v| v.to_str().unwrap().to_string());
            assert_eq!(body.as_ref(), d.payload.as_bytes());
            let payload: Value = serde_json::from_slice(body).unwrap();
            assert_eq!(payload["event"], "retain.completed");
            assert_eq!(payload["data"]["document_id"], "d1");
            assert_eq!(
                h("x-hindsight-event").as_deref(),
                Some("retain.completed"),
                "custom header must not forge it"
            );
            assert_eq!(h("x-custom").as_deref(), Some("yes"));
            assert_eq!(
                h("x-hindsight-event-id"),
                payload["event_id"].as_str().map(str::to_string)
            );
            assert_eq!(h("x-hindsight-delivery-id"), Some(d.id.clone()));
            let sig = sign_payload("s3cret", body);
            assert_eq!(h("x-hindsight-signature"), Some(sig.clone()));
            assert_eq!(h("x-hub-signature-256"), Some(sig));
            assert_eq!(
                h("x-hindsight-signature-v2"),
                Some(sign_payload_v2("s3cret", body, NOW / 1000))
            );
        }

        let app = crate::http::router(Arc::clone(&engine), Arc::new(ApiAuth::None));
        let uri = format!("/v1/default/banks/wd/webhooks/{}/deliveries", hook.id);
        let (status, list) = call(&app, Method::GET, &uri, None).await;
        assert_eq!(status, StatusCode::OK, "{list}");
        let item = &list["items"][0];
        assert_eq!(item["status"], "completed");
        assert_eq!(item["attempts"], 1);
        assert_eq!(item["last_response_status"], 200);
        assert!(item["last_response_body"].is_null(), "withheld by default");
        assert_eq!(item["event_type"], "retain.completed");

        // Webhooks and their history go with the bank.
        engine.delete_bank("wd").await.unwrap();
        assert!(engine.store.get_webhook("wd", &hook.id).unwrap().is_none());
        assert!(engine
            .store
            .list_webhook_deliveries("wd", &hook.id, 10, None)
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn failed_deliveries_retry_on_schedule_then_give_up() {
        let (url, got) = receiver().await;
        got.status.store(500, Ordering::SeqCst);
        let (engine, _) = wired(&url, None);
        let d = retain_and_take_delivery(&engine).await;
        let allow = Allowlist::parse("127.0.0.1");
        let after = deliver_once(engine.store.as_ref(), engine.clock.as_ref(), &allow, d)
            .await
            .unwrap();
        assert_eq!(after.status, "pending");
        assert_eq!(after.retry_count, 1);
        assert_eq!(after.next_attempt_at_ms, NOW + 5_000);
        assert_eq!(after.last_response_status, Some(500));
        assert_eq!(after.last_error.as_deref(), Some("HTTP 500"));
        assert!(engine
            .store
            .due_webhook_deliveries(NOW, 10)
            .unwrap()
            .is_empty());
        assert_eq!(
            engine.store.next_webhook_delivery_at().unwrap(),
            Some(NOW + 5_000)
        );
        assert!(got.requests.lock().unwrap()[0]
            .0
            .get("x-hindsight-signature")
            .is_none());

        // The last allowed attempt fails for good.
        let last = WebhookDelivery {
            retry_count: MAX_ATTEMPTS - 1,
            ..after
        };
        let gone = deliver_once(engine.store.as_ref(), engine.clock.as_ref(), &allow, last)
            .await
            .unwrap();
        assert_eq!(gone.status, "failed");
    }

    #[tokio::test]
    async fn blocked_destinations_fail_without_retrying() {
        let (url, got) = receiver().await;
        let (engine, _) = wired(&url, None);
        let d = retain_and_take_delivery(&engine).await;
        let done = deliver_once(
            engine.store.as_ref(),
            engine.clock.as_ref(),
            &Allowlist::default(),
            d,
        )
        .await
        .unwrap();
        assert_eq!(done.status, "failed");
        assert!(done
            .last_error
            .unwrap()
            .contains("not an allowed destination"));
        assert!(
            got.requests.lock().unwrap().is_empty(),
            "nothing may be sent"
        );
    }

    #[tokio::test]
    async fn worker_delivers_queued_events() {
        let (url, got) = receiver().await;
        let engine = engine();
        engine
            .store
            .ensure_bank(&Bank {
                id: "ww".into(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            })
            .unwrap();
        // The global webhook, allowlisted for the local receiver.
        let settings = WebhookSettings {
            global: Some(Webhook {
                id: String::new(),
                bank_id: None,
                url,
                secret: None,
                event_types: vec!["retain.completed".into()],
                enabled: true,
                http_config: WebhookHttpConfig::default(),
                created_at_ms: 0,
                updated_at_ms: 0,
            }),
            allowlist: Allowlist::parse("127.0.0.1"),
            expose_response_body: false,
        };
        let worker = start(&engine, Arc::new(settings));
        engine
            .retain(RetainRequest {
                bank_id: "ww".into(),
                content: "Alice joined Google.".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while got.requests.lock().unwrap().is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "worker never delivered"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        worker.abort();
        let _ = worker.await;
    }
}
