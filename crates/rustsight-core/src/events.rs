//! Engine lifecycle events (RS-10b).
//!
//! The engine reports what it did through an optional [`EventSink`]
//! (`Engine::with_event_sink`). The server's sink turns events into
//! webhook deliveries; embedders (a desktop shell) can plug their own.
//!
//! Event names, statuses and payload fields follow upstream hindsight-api
//! 1.2.1 `webhooks/models.py` (`WebhookEvent`, `RetainEventData`,
//! `ConsolidationEventData`, `MemoryDefenseEventData`).

use crate::defense::DefenseHit;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;

/// Upstream `WebhookEventType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    RetainCompleted,
    ConsolidationCompleted,
    MemoryDefenseTriggered,
}

impl EventType {
    pub const ALL: [EventType; 3] = [
        EventType::RetainCompleted,
        EventType::ConsolidationCompleted,
        EventType::MemoryDefenseTriggered,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            EventType::RetainCompleted => "retain.completed",
            EventType::ConsolidationCompleted => "consolidation.completed",
            EventType::MemoryDefenseTriggered => "memory_defense.triggered",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.as_str() == s)
    }
}

/// Event-specific payload. `None` fields are left off the wire (upstream
/// serializes with `exclude_none`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EventData {
    Retain {
        #[serde(skip_serializing_if = "Option::is_none")]
        document_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tags: Option<Vec<String>>,
        /// Memories the document owns after this retain; 0 means extraction
        /// returned nothing.
        #[serde(skip_serializing_if = "Option::is_none")]
        memory_unit_count: Option<usize>,
    },
    Consolidation {
        #[serde(skip_serializing_if = "Option::is_none")]
        observations_created: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        observations_updated: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        observations_deleted: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
    },
    MemoryDefense {
        /// `redact` or `block`.
        action: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detector: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        document_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        matched_types: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        /// Fingerprinted previews — never raw values.
        #[serde(skip_serializing_if = "Option::is_none")]
        hits: Option<Vec<DefenseHit>>,
    },
}

/// One logical event. `event_id` is shared by every delivery it fans out to
/// and stable across retries, so receivers dedupe on it.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineEvent {
    pub event_id: String,
    pub event: EventType,
    pub bank_id: String,
    pub operation_id: String,
    /// `completed` / `failed` for retain and consolidation; the action
    /// (`redact` / `block`) for memory defense.
    pub status: String,
    pub timestamp_ms: i64,
    pub data: EventData,
}

#[derive(Serialize)]
struct Wire<'a> {
    event_id: &'a str,
    event: &'a str,
    bank_id: &'a str,
    operation_id: &'a str,
    status: &'a str,
    timestamp: String,
    data: &'a EventData,
}

impl EngineEvent {
    pub fn new(
        event: EventType,
        bank_id: &str,
        operation_id: &str,
        status: &str,
        timestamp_ms: i64,
        data: EventData,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            event,
            bank_id: bank_id.to_string(),
            operation_id: operation_id.to_string(),
            status: status.to_string(),
            timestamp_ms,
            data,
        }
    }

    /// The webhook body: upstream's `WebhookEvent` JSON, field order
    /// included, with an ISO-8601 UTC timestamp.
    pub fn wire_json(&self) -> String {
        let timestamp = DateTime::<Utc>::from_timestamp_millis(self.timestamp_ms)
            .unwrap_or_default()
            .to_rfc3339_opts(SecondsFormat::Micros, true);
        serde_json::to_string(&Wire {
            event_id: &self.event_id,
            event: self.event.as_str(),
            bank_id: &self.bank_id,
            operation_id: &self.operation_id,
            status: &self.status,
            timestamp,
            data: &self.data,
        })
        .expect("event serializes")
    }
}

/// Receives engine events. Called synchronously at the point the event
/// happens (after the work it reports is durable), so implementations must
/// be quick and must not block on the network — queue, then return.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: &EngineEvent);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_json_matches_upstream_shape_and_drops_none() {
        let mut e = EngineEvent::new(
            EventType::ConsolidationCompleted,
            "b",
            "op",
            "completed",
            1_800_000_000_123,
            EventData::Consolidation {
                observations_created: Some(2),
                observations_updated: Some(0),
                observations_deleted: None,
                error_message: None,
            },
        );
        e.event_id = "ev".into();
        assert_eq!(
            e.wire_json(),
            r#"{"event_id":"ev","event":"consolidation.completed","bank_id":"b","operation_id":"op","status":"completed","timestamp":"2027-01-15T08:00:00.123000Z","data":{"observations_created":2,"observations_updated":0}}"#
        );
    }

    #[test]
    fn event_types_round_trip() {
        for t in EventType::ALL {
            assert_eq!(EventType::parse(t.as_str()), Some(t));
        }
        assert_eq!(EventType::parse("retain.started"), None);
    }
}
