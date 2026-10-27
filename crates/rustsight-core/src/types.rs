use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone)]
pub struct RetainRequest {
    pub bank_id: String,
    pub content: String,
    pub context: Option<String>,
    pub timestamp_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetainResponse {
    pub memories_stored: usize,
    pub entities: Vec<String>,
    pub edges_created: usize,
}

#[derive(Debug, Clone)]
pub struct RecallRequest {
    pub bank_id: String,
    pub query: String,
    pub token_budget: usize,
    pub top_k_per_arm: usize,
    pub kinds: Option<Vec<MemoryKind>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResponse {
    pub memories: Vec<MemoryRef>,
    pub window: Option<(i64, i64)>,
    pub strategies_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectResponse {
    pub answer: String,
    pub citations: Vec<MemoryId>,
    pub rounds: usize,
    pub retrieved: usize,
    pub opinions_formed: usize,
}

#[derive(Debug, Clone)]
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
