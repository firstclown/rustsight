use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub type MemoryId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    World,
    Experience,
    Observation,
    Opinion,
}

impl MemoryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::World => "world",
            MemoryKind::Experience => "experience",
            MemoryKind::Observation => "observation",
            MemoryKind::Opinion => "opinion",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "world" => Some(MemoryKind::World),
            "experience" => Some(MemoryKind::Experience),
            "observation" => Some(MemoryKind::Observation),
            "opinion" => Some(MemoryKind::Opinion),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Disposition {
    pub skepticism: u8,
    pub literalism: u8,
    pub empathy: u8,
    pub bias_strength: f32,
}

impl Default for Disposition {
    fn default() -> Self {
        Self {
            skepticism: 3,
            literalism: 3,
            empathy: 3,
            bias_strength: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bank {
    pub id: String,
    pub mission: Option<String>,
    pub directives: Option<String>,
    pub background: Option<String>,
    #[serde(default)]
    pub disposition: Disposition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: MemoryId,
    pub bank_id: String,
    pub kind: MemoryKind,
    pub text: String,
    pub confidence: Option<f32>,
    pub occurred_start_ms: Option<i64>,
    pub occurred_end_ms: Option<i64>,
    pub mentioned_at_ms: i64,
    pub source: Option<String>,
    pub context: Option<String>,
    /// Document this memory was extracted from (Hindsight `document_id`).
    #[serde(default)]
    pub document_id: Option<String>,
    /// Visibility-scope tags (Hindsight `tags`); empty = untagged.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Caller-supplied key/value metadata, returned with recalls.
    #[serde(default)]
    pub metadata: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct NewMemoryRecord {
    pub kind: MemoryKind,
    pub text: String,
    pub confidence: Option<f32>,
    pub occurred_start_ms: Option<i64>,
    pub occurred_end_ms: Option<i64>,
    pub mentioned_at_ms: i64,
    pub embedding: Vec<f32>,
    pub source: Option<String>,
    pub context: Option<String>,
    pub document_id: Option<String>,
    pub tags: Vec<String>,
    pub metadata: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Product,
    Concept,
    Other,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Person => "person",
            EntityType::Organization => "organization",
            EntityType::Location => "location",
            EntityType::Product => "product",
            EntityType::Concept => "concept",
            EntityType::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "person" => EntityType::Person,
            "organization" | "organisation" | "org" => EntityType::Organization,
            "location" | "place" => EntityType::Location,
            "product" => EntityType::Product,
            "concept" => EntityType::Concept,
            _ => EntityType::Other,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub bank_id: String,
    pub name: String,
    pub etype: EntityType,
    pub last_mentioned_ms: i64,
}

#[derive(Debug, Clone)]
pub struct NewEntity {
    pub bank_id: String,
    pub name: String,
    pub etype: EntityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeKind {
    Entity,
    Temporal,
    Semantic,
    Causal,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeKind::Entity => "entity",
            EdgeKind::Temporal => "temporal",
            EdgeKind::Semantic => "semantic",
            EdgeKind::Causal => "causal",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "entity" => Some(EdgeKind::Entity),
            "temporal" => Some(EdgeKind::Temporal),
            "semantic" => Some(EdgeKind::Semantic),
            "causal" => Some(EdgeKind::Causal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub bank_id: String,
    pub a_id: MemoryId,
    pub b_id: MemoryId,
    pub kind: EdgeKind,
    pub label: Option<String>,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub memory_id: MemoryId,
    pub quote: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub bank_id: String,
    pub entity_id: String,
    pub text: String,
    pub evidence: Vec<Evidence>,
    pub proof_count: usize,
    pub history: Vec<String>,
    pub updated_at_ms: i64,
    pub created_at_ms: i64,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidencePoint {
    pub confidence: f32,
    pub at_ms: i64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Opinion {
    pub id: String,
    pub bank_id: String,
    pub text: String,
    pub confidence: f32,
    pub evidence: Vec<Evidence>,
    pub history: Vec<ConfidencePoint>,
    pub entity_ids: Vec<String>,
    pub updated_at_ms: i64,
    pub created_at_ms: i64,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalModel {
    pub id: String,
    pub bank_id: String,
    pub question: String,
    pub path: Option<String>,
    pub content: String,
    pub last_refreshed_ms: Option<i64>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRef {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub text: String,
    pub score: f32,
    pub occurred_start_ms: Option<i64>,
    pub occurred_end_ms: Option<i64>,
    pub mentioned_at_ms: Option<i64>,
    pub entity_ids: Vec<String>,
    /// Observation evidence strength, `None` for non-observation kinds —
    /// feeds the proof-count boost in rerank combined scoring.
    pub proof_count: Option<usize>,
    /// Retain-time context label (raw facts only).
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub document_id: Option<String>,
    /// Scope tags. For observations: the union of the evidence facts' tags.
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Option<BTreeMap<String, String>>,
}

/// Co-occurrence facts the entity-resolution scoring pass needs, built once
/// per retain batch from the bank's per-fact entity links (upstream
/// `_CooccurrenceIndex`, backed there by the persisted
/// `entity_cooccurrences` table).
#[derive(Debug, Default, Clone)]
pub struct CooccurrenceIndex {
    /// entity id → lowercased canonical names of the entities it has
    /// co-occurred with on a stored fact.
    pub partners_by_entity: HashMap<String, HashSet<String>>,
    /// lowercased entity name → how many distinct entities that entity has
    /// co-occurred with (its degree; hubs get damped votes).
    pub degree_by_name: HashMap<String, usize>,
}

/// How a retain treats an existing document with the same `document_id`
/// (Hindsight `update_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UpdateMode {
    /// Delete the old document's memories and reprocess (upstream default).
    #[default]
    Replace,
    /// Concatenate the new content onto the stored body; existing memories
    /// stay, only the new tail is extracted.
    Append,
}

#[derive(Debug, Clone, Default)]
pub struct RetainRequest {
    pub bank_id: String,
    pub content: String,
    pub context: Option<String>,
    pub timestamp_ms: Option<i64>,
    /// Hindsight `timestamp: "unset"`: timeless content — the extractor
    /// gets no event date (memories are still stamped with the retain time).
    pub timestamp_unset: bool,
    /// Logical document id. `None` = a fresh generated document per retain.
    pub document_id: Option<String>,
    pub update_mode: UpdateMode,
    pub tags: Vec<String>,
    pub metadata: Option<BTreeMap<String, String>>,
    /// Caller-supplied entities, attached to every fact extracted from this
    /// content alongside the extractor's own (upstream `entities`).
    pub entities: Vec<ExtractedEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetainResponse {
    pub memories_stored: usize,
    pub entities: Vec<String>,
    pub edges_created: usize,
    /// How many chunk extractions had to be re-asked at half size after a
    /// parse failure (0 when every first parse succeeded). Surfaced so
    /// benchmark drivers can attribute retain latency.
    #[serde(default)]
    pub reasks: usize,
    /// The document the memories were filed under (given or generated).
    #[serde(default)]
    pub document_id: Option<String>,
}

/// Hindsight `tags_match`: how a tag filter compares against a memory's tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TagsMatch {
    /// OR match; untagged memories are included.
    #[default]
    Any,
    /// AND match; untagged memories are included.
    All,
    /// OR match; untagged memories are excluded.
    AnyStrict,
    /// AND match; untagged memories are excluded.
    AllStrict,
    /// Set equality; with no tags, selects untagged memories only.
    Exact,
}

/// One node of a compound tag filter (upstream `tag_groups`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagGroup {
    Leaf { tags: Vec<String>, mode: TagsMatch },
    And(Vec<TagGroup>),
    Or(Vec<TagGroup>),
    Not(Box<TagGroup>),
}

impl TagGroup {
    pub fn matches(&self, item_tags: &[String]) -> bool {
        match self {
            TagGroup::Leaf { tags, mode } => tags_match(tags, *mode, item_tags),
            TagGroup::And(groups) => groups.iter().all(|g| g.matches(item_tags)),
            TagGroup::Or(groups) => groups.iter().any(|g| g.matches(item_tags)),
            TagGroup::Not(group) => !group.matches(item_tags),
        }
    }
}

/// Upstream `build_tags_where_clause` semantics for one tag list.
fn tags_match(tags: &[String], mode: TagsMatch, item_tags: &[String]) -> bool {
    if tags.is_empty() {
        // Empty scope: `exact` selects untagged rows only; every other
        // mode applies no filtering at all.
        return mode != TagsMatch::Exact || item_tags.is_empty();
    }
    let any = || tags.iter().any(|t| item_tags.contains(t));
    let all = || tags.iter().all(|t| item_tags.contains(t));
    match mode {
        TagsMatch::Any => item_tags.is_empty() || any(),
        TagsMatch::All => item_tags.is_empty() || all(),
        TagsMatch::AnyStrict => !item_tags.is_empty() && any(),
        TagsMatch::AllStrict => !item_tags.is_empty() && all(),
        TagsMatch::Exact => all() && item_tags.iter().all(|t| tags.contains(t)),
    }
}

/// A recall/reflect tag scope: the flat `tags` + `tags_match` pair, or
/// (mutually exclusive upstream) a list of AND-ed `tag_groups`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TagFilter {
    pub tags: Vec<String>,
    pub mode: TagsMatch,
    pub groups: Vec<TagGroup>,
}

impl TagFilter {
    /// True when the filter restricts nothing.
    pub fn is_noop(&self) -> bool {
        self.tags.is_empty() && self.mode != TagsMatch::Exact && self.groups.is_empty()
    }

    pub fn matches(&self, item_tags: &[String]) -> bool {
        tags_match(&self.tags, self.mode, item_tags)
            && self.groups.iter().all(|g| g.matches(item_tags))
    }
}

#[derive(Debug, Clone, Default)]
pub struct RecallRequest {
    pub bank_id: String,
    pub query: String,
    pub token_budget: usize,
    pub top_k_per_arm: usize,
    pub kinds: Option<Vec<MemoryKind>>,
    /// Tag scope; `None` = unfiltered.
    pub tags: Option<TagFilter>,
    /// Anchor for relative temporal expressions and recency scoring
    /// (Hindsight `query_timestamp`); `None` = the engine clock.
    pub now_ms: Option<i64>,
}

/// A retained source document (Hindsight documents API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub bank_id: String,
    pub original_text: String,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub metadata: Option<BTreeMap<String, String>>,
    /// Retain parameters captured for the record (`context`, `event_date`).
    pub retain_params: serde_json::Value,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// Lifecycle record of an async retain (Hindsight operations API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: String,
    pub bank_id: String,
    pub kind: String,
    /// `pending` | `completed` | `failed`.
    pub status: String,
    pub error: Option<String>,
    /// The queued request, so a restart can resume unfinished work.
    pub payload: serde_json::Value,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// Per-request reflect options (Hindsight `ReflectRequest`). The default
/// reproduces the plain `Engine::reflect(bank, query)` behavior.
#[derive(Debug, Clone, Default)]
pub struct ReflectOptions {
    /// Deprecated upstream field: appended to the query as
    /// `"\n\nAdditional context: …"`.
    pub context: Option<String>,
    /// Tool-loop round cap for this request; `None` = `reflect_max_rounds`.
    pub max_rounds: Option<usize>,
    /// Visible answer-length target; `None` = `reflect_answer_max_tokens`.
    pub max_tokens: Option<usize>,
    /// Tag scope applied to every retrieval the loop makes.
    pub tags: Option<TagFilter>,
    /// Restrict retrieval to these kinds (upstream `fact_types`).
    pub kinds: Option<Vec<MemoryKind>>,
    /// JSON Schema for a structured rendering of the answer.
    pub response_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResponse {
    pub memories: Vec<MemoryRef>,
    pub window: Option<(i64, i64)>,
    pub strategies_used: Vec<String>,
}

/// Which call wrote the reflect answer. Upstream precedence (verified in
/// hindsight-api 1.1.0 `agent.py::_finish`): the loop's own closing `done`
/// call IS the answer; the standalone synthesis prompt only runs when the
/// loop produced nothing usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum AnswerPath {
    /// The loop model's volunteered `finish` tool call (or a finish-shaped
    /// plain-content envelope) wrote the answer.
    #[default]
    Loop,
    /// The forced-done round (asked-for finish on the existing
    /// conversation) wrote the answer.
    ForcedDone,
    /// The standalone final-synthesis call over the packed tool history
    /// wrote the answer — the fallback for loops that produced nothing.
    Synthesis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectResponse {
    pub answer: String,
    pub citations: Vec<MemoryId>,
    pub rounds: usize,
    pub retrieved: usize,
    /// Every memory id surfaced to the model across all rounds (deduped,
    /// sorted) — the superset citations are validated against.
    pub retrieved_ids: Vec<MemoryId>,
    /// Which call produced the answer (upstream's done-first precedence).
    #[serde(default)]
    pub answer_path: AnswerPath,
    /// Compatibility view of `answer_path`: true only when the standalone
    /// final-synthesis call wrote the answer.
    pub synthesized: bool,
    /// Wall time of the tool loop (all rounds + forced-done), milliseconds.
    #[serde(default)]
    pub loop_ms: u64,
    /// Wall time of the standalone final-synthesis call, when it ran.
    #[serde(default)]
    pub synthesis_ms: Option<u64>,
    /// The answer rendered against the request's `response_schema`.
    #[serde(default)]
    pub structured_output: Option<serde_json::Value>,
    /// Why `structured_output` could not be produced, when it was asked for.
    #[serde(default)]
    pub structured_output_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedEntity {
    pub name: String,
    pub etype: EntityType,
}

#[derive(Debug, Clone)]
pub struct CausalRef {
    pub fact_index: usize,
    pub relation: String,
}

#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub text: String,
    pub fact_type: MemoryKind,
    pub entities: Vec<ExtractedEntity>,
    pub occurred_start_ms: Option<i64>,
    pub occurred_end_ms: Option<i64>,
    pub causal: Vec<CausalRef>,
}

#[cfg(test)]
mod tag_tests {
    use super::*;

    fn t(tags: &[&str]) -> Vec<String> {
        tags.iter().map(|s| s.to_string()).collect()
    }

    fn filter(tags: &[&str], mode: TagsMatch) -> TagFilter {
        TagFilter {
            tags: t(tags),
            mode,
            groups: vec![],
        }
    }

    #[test]
    fn tags_match_modes_follow_upstream() {
        let untagged: Vec<String> = vec![];
        let a = t(&["a"]);
        let ab = t(&["a", "b"]);
        // any: OR, untagged included
        assert!(filter(&["a", "x"], TagsMatch::Any).matches(&a));
        assert!(filter(&["a"], TagsMatch::Any).matches(&untagged));
        assert!(!filter(&["x"], TagsMatch::Any).matches(&a));
        // all: AND, untagged included
        assert!(filter(&["a", "b"], TagsMatch::All).matches(&ab));
        assert!(!filter(&["a", "b"], TagsMatch::All).matches(&a));
        assert!(filter(&["a", "b"], TagsMatch::All).matches(&untagged));
        // strict variants exclude untagged
        assert!(!filter(&["a"], TagsMatch::AnyStrict).matches(&untagged));
        assert!(filter(&["a"], TagsMatch::AnyStrict).matches(&ab));
        assert!(!filter(&["a", "b"], TagsMatch::AllStrict).matches(&a));
        // exact: set equality; empty scope selects untagged only
        assert!(filter(&["b", "a"], TagsMatch::Exact).matches(&ab));
        assert!(!filter(&["a"], TagsMatch::Exact).matches(&ab));
        assert!(filter(&[], TagsMatch::Exact).matches(&untagged));
        assert!(!filter(&[], TagsMatch::Exact).matches(&a));
        // no tags + non-exact = no filtering
        assert!(filter(&[], TagsMatch::Any).is_noop());
        assert!(!filter(&[], TagsMatch::Exact).is_noop());
    }

    #[test]
    fn tag_groups_compose() {
        let leaf = |tags: &[&str]| TagGroup::Leaf {
            tags: t(tags),
            mode: TagsMatch::AnyStrict,
        };
        let f = TagFilter {
            groups: vec![
                TagGroup::Or(vec![leaf(&["a"]), leaf(&["b"])]),
                TagGroup::Not(Box::new(leaf(&["secret"]))),
            ],
            ..Default::default()
        };
        assert!(!f.is_noop());
        assert!(f.matches(&t(&["a"])));
        assert!(f.matches(&t(&["b", "z"])));
        assert!(!f.matches(&t(&["a", "secret"])));
        assert!(!f.matches(&t(&["z"])));
        assert!(!f.matches(&[]));
    }
}
