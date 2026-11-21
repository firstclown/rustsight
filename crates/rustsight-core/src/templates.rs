//! Bank templates (RS-10d): a manifest of bank settings, mental models and
//! directives that can be exported from one bank and imported into another,
//! or applied to every new bank (`Engine::with_default_template`).
//!
//! The manifest is upstream hindsight-api 1.2.1's `BankTemplateManifest`
//! (`engine/bank_templates.py`): `{version: "1", bank, mental_models,
//! directives}`. Mental models match by id and directives by name on
//! re-import; unchanged ones are left alone. `bank` takes the settings in
//! [`crate::bank_config::CONFIG_KEYS`]; any other upstream setting with a
//! non-null value is refused by name rather than silently dropped.

use crate::bank_config::{apply_update, is_config_key, CONFIG_KEYS};
use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::prompts::upstream::reflect::Directive;
use crate::types::{Bank, BankDirective, Disposition, MentalModel, TagFilter};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

pub const CURRENT_VERSION: &str = "1";

/// Every field of upstream's `BankTemplateConfig`. Those outside
/// [`CONFIG_KEYS`] are refused when non-null; names upstream does not know
/// either are ignored, as upstream ignores them.
const UPSTREAM_BANK_FIELDS: &[&str] = &[
    "reflect_mission",
    "retain_mission",
    "retain_extraction_mode",
    "retain_custom_instructions",
    "retain_chunk_size",
    "retain_structured_chunk_size",
    "enable_observations",
    "observations_mission",
    "enable_text_search",
    "enable_temporal_retrieval",
    "enable_graph_retrieval",
    "enable_reranking",
    "reflect_use_decision_model",
    "mental_model_refresh_use_decision_model",
    "mental_model_delta_use_decision_model",
    "disposition_skepticism",
    "disposition_literalism",
    "disposition_empathy",
    "entity_labels",
    "entities_allow_free_form",
    "retain_default_strategy",
    "retain_strategies",
    "retain_chunk_batch_size",
    "retain_max_attachments_per_chunk",
    "mcp_enabled_tools",
    "consolidation_llm_batch_size",
    "consolidation_source_facts_max_tokens",
    "consolidation_source_facts_max_tokens_per_observation",
    "max_observations_per_scope",
    "observation_scope_limits",
    "default_observation_scopes",
    "consolidation_strategies",
    "reflect_source_facts_max_tokens",
    "page_default_trigger",
    "knowledge_page_default_trigger",
    "reflect_default_options",
    "mental_model_min_refresh_interval_seconds",
    "llm_gemini_safety_settings",
    "recall_budget_function",
    "recall_budget_fixed_low",
    "recall_budget_fixed_mid",
    "recall_budget_fixed_high",
    "recall_budget_adaptive_low",
    "recall_budget_adaptive_mid",
    "recall_budget_adaptive_high",
    "recall_budget_min",
    "recall_budget_max",
    "audit_log_enabled",
    "store_document_text",
    "default_timezone",
    "enable_auto_consolidation",
    "consolidation_use_decision_model",
    "consolidation_max_memories_per_round",
    "consolidation_llm_parallelism",
    "recall_include_chunks",
    "recall_prefer_observations",
    "recall_decision_model_prune",
    "recall_decision_model_dedup",
    "recall_max_tokens",
    "recall_chunks_max_tokens",
    "memory_defense",
];

const MM_MAX_TOKENS_DEFAULT: u32 = 2048;
const MM_MAX_TOKENS_RANGE: std::ops::RangeInclusive<i64> = 256..=8192;

#[derive(Debug, Clone, PartialEq)]
pub struct TemplateMentalModel {
    pub id: String,
    pub name: String,
    pub source_query: String,
    pub tags: Vec<String>,
    pub max_tokens: u32,
    pub trigger: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemplateDirective {
    pub name: String,
    pub content: String,
    pub priority: i32,
    pub is_active: bool,
    pub tags: Vec<String>,
}

/// A validated manifest.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Manifest {
    pub version: String,
    /// Supported settings with non-null values, keyed by name.
    pub bank: Option<Map<String, Value>>,
    pub mental_models: Option<Vec<TemplateMentalModel>>,
    pub directives: Option<Vec<TemplateDirective>>,
}

/// What an import did (upstream `BankTemplateImportResponse`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImportReport {
    pub bank_id: String,
    pub config_applied: bool,
    pub mental_models_created: Vec<String>,
    pub mental_models_updated: Vec<String>,
    pub directives_created: Vec<String>,
    pub directives_updated: Vec<String>,
    /// Content-generation operations; empty here — imported models fill
    /// on their next refresh.
    pub operation_ids: Vec<String>,
    pub dry_run: bool,
}

/// Collects schema errors as `loc: message`, upstream's pydantic format.
#[derive(Default)]
struct Errors(Vec<String>);

impl Errors {
    fn push(&mut self, loc: &str, msg: impl AsRef<str>) {
        self.0.push(format!("{loc}: {}", msg.as_ref()));
    }
}

fn string_field(obj: &Map<String, Value>, loc: &str, key: &str, errs: &mut Errors) -> String {
    match obj.get(key) {
        Some(Value::String(s)) => s.clone(),
        None | Some(Value::Null) => {
            errs.push(&format!("{loc}.{key}"), "Field required");
            String::new()
        }
        Some(_) => {
            errs.push(&format!("{loc}.{key}"), "Input should be a valid string");
            String::new()
        }
    }
}

fn tags_field(obj: &Map<String, Value>, loc: &str, errs: &mut Errors) -> Vec<String> {
    match obj.get("tags") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|t| match t {
                Value::String(s) => Some(s.clone()),
                _ => {
                    errs.push(&format!("{loc}.tags"), "Input should be a list of strings");
                    None
                }
            })
            .collect(),
        Some(_) => {
            errs.push(&format!("{loc}.tags"), "Input should be a valid list");
            vec![]
        }
    }
}

/// Upstream `check_mental_model_id`: bank-id rules.
fn check_mental_model_id(id: &str) -> std::result::Result<(), &'static str> {
    if id.is_empty() {
        Err("must not be empty")
    } else if id.len() > 192 {
        Err("must be at most 192 bytes of UTF-8")
    } else if id.contains('/') {
        Err("must not contain '/'")
    } else if id.chars().any(char::is_control) {
        Err("must not contain control characters")
    } else {
        Ok(())
    }
}

fn duplicates<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut dup: Vec<String> = names
        .filter(|n| !seen.insert(*n))
        .map(str::to_string)
        .collect();
    dup.sort();
    dup.dedup();
    dup
}

fn py_list(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|i| format!("'{i}'")).collect();
    format!("[{}]", quoted.join(", "))
}

/// Parse and validate a manifest. Structural problems fail with
/// `Template schema validation failed: ...`, semantic ones (empty names,
/// duplicates) with `Template validation failed: ...` — upstream's two
/// 400 messages.
pub fn parse_manifest(raw: &Value) -> Result<Manifest> {
    let mut errs = Errors::default();
    let Some(obj) = raw.as_object() else {
        return Err(Error::BadInput(
            "Template schema validation failed: Input should be an object".into(),
        ));
    };
    let version = match obj.get("version") {
        Some(Value::String(v)) => match v.parse::<i64>() {
            Err(_) => {
                errs.push(
                    "version",
                    format!("Value error, version must be a numeric string, got '{v}'"),
                );
                v.clone()
            }
            Ok(n) if n < 1 => {
                errs.push("version", "Value error, version must be >= 1");
                v.clone()
            }
            Ok(n) if n > CURRENT_VERSION.parse::<i64>().unwrap_or(1) => {
                errs.push(
                    "version",
                    format!(
                        "Value error, version '{v}' is not supported by this server (max supported: {CURRENT_VERSION})"
                    ),
                );
                v.clone()
            }
            Ok(_) => v.clone(),
        },
        None | Some(Value::Null) => {
            errs.push("version", "Field required");
            String::new()
        }
        Some(_) => {
            errs.push("version", "Input should be a valid string");
            String::new()
        }
    };

    let bank = match obj.get("bank") {
        None | Some(Value::Null) => None,
        Some(Value::Object(fields)) => {
            let mut scratch = Bank {
                id: String::new(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            };
            let mut supported = Map::new();
            for (key, value) in fields {
                if value.is_null() {
                    continue;
                }
                if is_config_key(key) {
                    match apply_update(&mut scratch, key, value) {
                        Ok(()) => {
                            supported.insert(key.clone(), value.clone());
                        }
                        Err(e) => errs.push(&format!("bank.{key}"), plain(e)),
                    }
                } else if UPSTREAM_BANK_FIELDS.contains(&key.as_str()) {
                    errs.push(
                        &format!("bank.{key}"),
                        format!(
                            "not supported by this server (supported: {})",
                            CONFIG_KEYS.join(", ")
                        ),
                    );
                }
            }
            Some(supported)
        }
        Some(_) => {
            errs.push("bank", "Input should be an object");
            None
        }
    };

    let list = |key: &str, errs: &mut Errors| -> Option<Vec<Map<String, Value>>> {
        match obj.get(key) {
            None | Some(Value::Null) => None,
            Some(Value::Array(items)) => Some(
                items
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| match v {
                        Value::Object(m) => Some(m.clone()),
                        _ => {
                            errs.push(&format!("{key}.{i}"), "Input should be an object");
                            None
                        }
                    })
                    .collect(),
            ),
            Some(_) => {
                errs.push(key, "Input should be a valid list");
                None
            }
        }
    };

    let mental_models = list("mental_models", &mut errs).map(|items| {
        items
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let loc = format!("mental_models.{i}");
                let id = string_field(m, &loc, "id", &mut errs);
                if !id.is_empty() {
                    if let Err(why) = check_mental_model_id(&id) {
                        errs.push(
                            &format!("{loc}.id"),
                            format!("Value error, mental model id {why}"),
                        );
                    }
                }
                let max_tokens = match m.get("max_tokens") {
                    None | Some(Value::Null) => MM_MAX_TOKENS_DEFAULT,
                    Some(v) => match v.as_i64() {
                        Some(n) if MM_MAX_TOKENS_RANGE.contains(&n) => n as u32,
                        _ => {
                            errs.push(
                                &format!("{loc}.max_tokens"),
                                "Input should be an integer between 256 and 8192",
                            );
                            MM_MAX_TOKENS_DEFAULT
                        }
                    },
                };
                let trigger = match m.get("trigger") {
                    None | Some(Value::Null) => json!({}),
                    Some(t @ Value::Object(_)) => t.clone(),
                    Some(_) => {
                        errs.push(&format!("{loc}.trigger"), "Input should be an object");
                        json!({})
                    }
                };
                TemplateMentalModel {
                    name: string_field(m, &loc, "name", &mut errs),
                    source_query: string_field(m, &loc, "source_query", &mut errs),
                    tags: tags_field(m, &loc, &mut errs),
                    id,
                    max_tokens,
                    trigger,
                }
            })
            .collect::<Vec<_>>()
    });
    if let Some(models) = &mental_models {
        let dup = duplicates(models.iter().map(|m| m.id.as_str()));
        if !dup.is_empty() {
            errs.push(
                "mental_models",
                format!("Value error, Duplicate mental model ids: {}", py_list(&dup)),
            );
        }
    }

    let directives = list("directives", &mut errs).map(|items| {
        items
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let loc = format!("directives.{i}");
                let priority = match d.get("priority") {
                    None | Some(Value::Null) => 0,
                    Some(v) => match v.as_i64().and_then(|n| i32::try_from(n).ok()) {
                        Some(n) => n,
                        None => {
                            errs.push(
                                &format!("{loc}.priority"),
                                "Input should be a 32-bit integer",
                            );
                            0
                        }
                    },
                };
                let is_active = match d.get("is_active") {
                    None | Some(Value::Null) => true,
                    Some(Value::Bool(b)) => *b,
                    Some(_) => {
                        errs.push(
                            &format!("{loc}.is_active"),
                            "Input should be a valid boolean",
                        );
                        true
                    }
                };
                TemplateDirective {
                    name: string_field(d, &loc, "name", &mut errs),
                    content: string_field(d, &loc, "content", &mut errs),
                    tags: tags_field(d, &loc, &mut errs),
                    priority,
                    is_active,
                }
            })
            .collect::<Vec<_>>()
    });
    if let Some(list) = &directives {
        let dup = duplicates(list.iter().map(|d| d.name.as_str()));
        if !dup.is_empty() {
            errs.push(
                "directives",
                format!("Value error, Duplicate directive names: {}", py_list(&dup)),
            );
        }
    }

    if !errs.0.is_empty() {
        return Err(Error::BadInput(format!(
            "Template schema validation failed: {}",
            errs.0.join("; ")
        )));
    }

    // Semantic checks (upstream `validate_bank_template`).
    let mut semantic = vec![];
    for (i, m) in mental_models.iter().flatten().enumerate() {
        if m.name.trim().is_empty() {
            semantic.push(format!("mental_models[{i}].name: must not be empty"));
        }
        if m.source_query.trim().is_empty() {
            semantic.push(format!(
                "mental_models[{i}].source_query: must not be empty"
            ));
        }
    }
    for (i, d) in directives.iter().flatten().enumerate() {
        if d.name.trim().is_empty() {
            semantic.push(format!("directives[{i}].name: must not be empty"));
        }
        if d.content.trim().is_empty() {
            semantic.push(format!("directives[{i}].content: must not be empty"));
        }
    }
    if !semantic.is_empty() {
        return Err(Error::BadInput(format!(
            "Template validation failed: {}",
            semantic.join("; ")
        )));
    }

    Ok(Manifest {
        version,
        bank,
        mental_models,
        directives,
    })
}

fn plain(e: Error) -> String {
    match e {
        Error::BadInput(m) => m,
        other => other.to_string(),
    }
}

/// Stored id of a template mental model: template ids are bank-local, the
/// store's ids are global, so they are namespaced by bank (template ids
/// cannot contain '/').
fn stored_model_id(bank_id: &str, id: &str) -> String {
    format!("{bank_id}/{id}")
}

fn template_model_id(bank_id: &str, stored: &str) -> String {
    stored
        .strip_prefix(&format!("{bank_id}/"))
        .unwrap_or(stored)
        .to_string()
}

fn same_tags(stored: &[String], wanted: &[String]) -> bool {
    let a: HashSet<&String> = stored.iter().collect();
    let b: HashSet<&String> = wanted.iter().collect();
    a == b
}

/// Upstream `_mental_model_matches_template`: re-applying would change
/// nothing (empty manifest tags leave stored tags alone).
fn model_matches(stored: &MentalModel, mm: &TemplateMentalModel) -> bool {
    stored.name.as_deref() == Some(mm.name.as_str())
        && stored.question == mm.source_query
        && stored.max_tokens.unwrap_or(MM_MAX_TOKENS_DEFAULT) == mm.max_tokens
        && (mm.tags.is_empty() || same_tags(&stored.tags, &mm.tags))
        && stored.trigger.clone().unwrap_or_else(|| json!({})) == mm.trigger
}

/// Upstream `_directive_matches_template`.
fn directive_matches(stored: &BankDirective, d: &TemplateDirective) -> bool {
    stored.content == d.content
        && stored.priority == d.priority
        && stored.is_active == d.is_active
        && (d.tags.is_empty() || same_tags(&stored.tags, &d.tags))
}

/// A JSON Schema of the manifest (`GET /v1/bank-template-schema`).
pub fn manifest_schema() -> Value {
    let strings = json!({ "type": "array", "items": { "type": "string" } });
    let nullable = |t: &str| json!({ "type": [t, "null"] });
    let trait_schema = json!({ "type": ["integer", "null"], "minimum": 1, "maximum": 5 });
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "BankTemplateManifest",
        "description": "A bank template manifest for import/export.",
        "type": "object",
        "required": ["version"],
        "properties": {
            "version": { "type": "string", "description": "Manifest schema version (currently '1')" },
            "bank": {
                "type": ["object", "null"],
                "description": "Bank settings to apply; other Hindsight settings are refused when set.",
                "properties": {
                    "reflect_mission": nullable("string"),
                    "retain_mission": nullable("string"),
                    "observations_mission": nullable("string"),
                    "enable_observations": nullable("boolean"),
                    "entity_labels": { "type": ["array", "object", "null"] },
                    "entities_allow_free_form": nullable("boolean"),
                    "disposition_skepticism": trait_schema,
                    "disposition_literalism": trait_schema,
                    "disposition_empathy": trait_schema,
                    "memory_defense": nullable("object"),
                },
            },
            "mental_models": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "required": ["id", "name", "source_query"],
                    "properties": {
                        "id": { "type": "string", "maxLength": 192 },
                        "name": { "type": "string" },
                        "source_query": { "type": "string" },
                        "tags": strings,
                        "max_tokens": { "type": "integer", "minimum": 256, "maximum": 8192, "default": 2048 },
                        "trigger": { "type": "object" },
                    },
                },
            },
            "directives": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "required": ["name", "content"],
                    "properties": {
                        "name": { "type": "string" },
                        "content": { "type": "string" },
                        "priority": { "type": "integer", "default": 0, "minimum": i32::MIN, "maximum": i32::MAX },
                        "is_active": { "type": "boolean", "default": true },
                        "tags": strings,
                    },
                },
            },
        },
    })
}

impl Engine {
    /// Import a manifest into a bank, creating the bank if needed. A dry
    /// run reports what the manifest holds and changes nothing.
    pub fn import_template(
        &self,
        bank_id: &str,
        manifest: &Manifest,
        dry_run: bool,
    ) -> Result<ImportReport> {
        if dry_run {
            return Ok(ImportReport {
                bank_id: bank_id.to_string(),
                config_applied: manifest.bank.is_some(),
                mental_models_created: manifest
                    .mental_models
                    .iter()
                    .flatten()
                    .map(|m| m.id.clone())
                    .collect(),
                mental_models_updated: vec![],
                directives_created: manifest
                    .directives
                    .iter()
                    .flatten()
                    .map(|d| d.name.clone())
                    .collect(),
                directives_updated: vec![],
                operation_ids: vec![],
                dry_run: true,
            });
        }
        self.provision_bank(bank_id)?;
        self.apply_template(bank_id, manifest, true)
    }

    /// Write a manifest's settings and resources. The config is always
    /// applied; `report_config` only says whether the report claims it
    /// (a default template's settings are part of provisioning).
    pub(crate) fn apply_template(
        &self,
        bank_id: &str,
        manifest: &Manifest,
        report_config: bool,
    ) -> Result<ImportReport> {
        let config = manifest.bank.clone().unwrap_or_default();
        if !config.is_empty() {
            self.update_bank_config(bank_id, &config)?;
        }
        let now = self.clock.now_ms();
        let mut report = ImportReport {
            bank_id: bank_id.to_string(),
            config_applied: report_config && !config.is_empty(),
            mental_models_created: vec![],
            mental_models_updated: vec![],
            directives_created: vec![],
            directives_updated: vec![],
            operation_ids: vec![],
            dry_run: false,
        };

        if let Some(models) = &manifest.mental_models {
            let existing = self.store.list_mental_models(bank_id)?;
            for mm in models {
                let stored_id = stored_model_id(bank_id, &mm.id);
                let current = existing
                    .iter()
                    .find(|m| m.id == stored_id || m.id == mm.id)
                    .cloned();
                match current {
                    Some(m) if model_matches(&m, mm) => {}
                    Some(mut m) => {
                        m.name = Some(mm.name.clone());
                        m.question = mm.source_query.clone();
                        m.max_tokens = Some(mm.max_tokens);
                        if !mm.tags.is_empty() {
                            m.tags = mm.tags.clone();
                        }
                        m.trigger = Some(mm.trigger.clone());
                        self.store.upsert_mental_model(&m)?;
                        report.mental_models_updated.push(mm.id.clone());
                    }
                    None => {
                        self.store.upsert_mental_model(&MentalModel {
                            id: stored_id,
                            bank_id: bank_id.to_string(),
                            question: mm.source_query.clone(),
                            path: None,
                            content: String::new(),
                            last_refreshed_ms: None,
                            created_at_ms: now,
                            name: Some(mm.name.clone()),
                            tags: mm.tags.clone(),
                            max_tokens: Some(mm.max_tokens),
                            trigger: Some(mm.trigger.clone()),
                        })?;
                        report.mental_models_created.push(mm.id.clone());
                    }
                }
            }
        }

        if let Some(directives) = &manifest.directives {
            let existing = self.store.list_directives(bank_id)?;
            for d in directives {
                match existing.iter().find(|e| e.name == d.name) {
                    Some(e) if directive_matches(e, d) => {}
                    Some(e) => {
                        let mut updated = e.clone();
                        updated.content = d.content.clone();
                        updated.priority = d.priority;
                        updated.is_active = d.is_active;
                        if !d.tags.is_empty() {
                            updated.tags = d.tags.clone();
                        }
                        updated.updated_at_ms = now;
                        self.store.upsert_directive(&updated)?;
                        report.directives_updated.push(d.name.clone());
                    }
                    None => {
                        self.store.upsert_directive(&BankDirective {
                            id: uuid::Uuid::new_v4().to_string(),
                            bank_id: bank_id.to_string(),
                            name: d.name.clone(),
                            content: d.content.clone(),
                            priority: d.priority,
                            is_active: d.is_active,
                            tags: d.tags.clone(),
                            created_at_ms: now,
                            updated_at_ms: now,
                        })?;
                        report.directives_created.push(d.name.clone());
                    }
                }
            }
        }
        Ok(report)
    }

    /// A bank's settings, mental models and directives as a manifest;
    /// `None` for an unknown bank. `bank` holds the explicitly set
    /// settings only, so importing it elsewhere copies choices, not
    /// defaults.
    pub fn export_template(&self, bank_id: &str) -> Result<Option<Value>> {
        let Some(bank) = self.store.get_bank(bank_id)? else {
            return Ok(None);
        };
        let stored = self.bank_config_map(bank_id)?;
        let mut settings: Map<String, Value> = stored
            .into_iter()
            .filter(|(key, value)| is_config_key(key) && !value.is_null())
            .collect();
        // Settings that live on the bank row (set through PUT /banks too).
        if let Some(m) = bank.mission.as_ref().filter(|m| !m.trim().is_empty()) {
            settings
                .entry("reflect_mission")
                .or_insert_with(|| json!(m));
        }
        let defaults = Disposition::default();
        for (key, value, default) in [
            (
                "disposition_skepticism",
                bank.disposition.skepticism,
                defaults.skepticism,
            ),
            (
                "disposition_literalism",
                bank.disposition.literalism,
                defaults.literalism,
            ),
            (
                "disposition_empathy",
                bank.disposition.empathy,
                defaults.empathy,
            ),
        ] {
            if value != default {
                settings.entry(key).or_insert_with(|| json!(value));
            }
        }
        let models: Vec<Value> = self
            .store
            .list_mental_models(bank_id)?
            .iter()
            .map(|m| {
                json!({
                    "id": template_model_id(bank_id, &m.id),
                    "name": m.name.clone().unwrap_or_else(|| m.question.clone()),
                    "source_query": m.question,
                    "tags": m.tags,
                    "max_tokens": m.max_tokens.unwrap_or(MM_MAX_TOKENS_DEFAULT),
                    "trigger": m.trigger.clone().unwrap_or_else(|| json!({})),
                })
            })
            .collect();
        let directives: Vec<Value> = self
            .store
            .list_directives(bank_id)?
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "content": d.content,
                    "priority": d.priority,
                    "is_active": d.is_active,
                    "tags": d.tags,
                })
            })
            .collect();
        let or_null = |items: Vec<Value>| {
            if items.is_empty() {
                Value::Null
            } else {
                Value::Array(items)
            }
        };
        Ok(Some(json!({
            "version": CURRENT_VERSION,
            "bank": if settings.is_empty() { Value::Null } else { Value::Object(settings) },
            "mental_models": or_null(models),
            "directives": or_null(directives),
        })))
    }

    /// The directives a reflect follows, highest priority first: the
    /// legacy free-text `Bank.directives` (when set), then every active
    /// stored directive that is untagged or matched by the reflect's tag
    /// filter (upstream's untagged-OR rule; with no filter only untagged
    /// ones apply).
    pub fn reflect_directives(
        &self,
        bank: &Bank,
        tags: Option<&TagFilter>,
    ) -> Result<Vec<Directive>> {
        let mut out: Vec<Directive> = bank
            .directives
            .as_deref()
            .filter(|d| !d.trim().is_empty())
            .map(|d| vec![Directive::new("", d)])
            .unwrap_or_default();
        let filter = tags.filter(|f| !f.is_noop());
        for d in self.store.list_directives(&bank.id)? {
            let applies = d.tags.is_empty() || filter.is_some_and(|f| f.matches(&d.tags));
            if d.is_active && applies {
                out.push(Directive::new(&d.name, &d.content));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(v: Value) -> String {
        parse_manifest(&v).unwrap_err().to_string()
    }

    #[test]
    fn parses_a_full_manifest_with_defaults() {
        let m = parse_manifest(&json!({
            "version": "1",
            "bank": {"reflect_mission": "Help support.", "disposition_empathy": 5,
                     "enable_reranking": null, "not_an_upstream_field": 7},
            "mental_models": [{"id": "sentiment", "name": "Sentiment", "source_query": "Trend?"}],
            "directives": [{"name": "Empathy", "content": "Be kind.", "priority": 10}],
        }))
        .unwrap();
        let bank = m.bank.unwrap();
        assert_eq!(
            bank.len(),
            2,
            "nulls and unknown names are ignored: {bank:?}"
        );
        let mm = &m.mental_models.unwrap()[0];
        assert_eq!((mm.max_tokens, mm.trigger.clone()), (2048, json!({})));
        let d = &m.directives.unwrap()[0];
        assert!(d.is_active && d.tags.is_empty() && d.priority == 10);
    }

    #[test]
    fn versions_are_checked() {
        assert!(err(json!({})).contains("version: Field required"));
        assert!(err(json!({"version": "x"})).contains("version must be a numeric string, got 'x'"));
        assert!(err(json!({"version": "0"})).contains("version must be >= 1"));
        assert!(err(json!({"version": "2"})).contains("max supported: 1"));
        assert!(err(json!({"version": 1})).contains("valid string"));
    }

    #[test]
    fn duplicates_and_bad_fields_are_refused() {
        let e = err(json!({"version": "1", "mental_models": [
            {"id": "a", "name": "A", "source_query": "q"},
            {"id": "a", "name": "B", "source_query": "q"},
            {"id": "x/y", "name": "C", "source_query": "q", "max_tokens": 100}
        ]}));
        assert!(
            e.starts_with("invalid input: Template schema validation failed: "),
            "{e}"
        );
        assert!(e.contains("Duplicate mental model ids: ['a']"), "{e}");
        assert!(
            e.contains("mental_models.2.id: Value error, mental model id must not contain '/'"),
            "{e}"
        );
        assert!(e.contains("mental_models.2.max_tokens"), "{e}");
        let e = err(json!({"version": "1", "directives": [
            {"name": "n", "content": "c"}, {"name": "n", "content": "d"}, {"content": "e"}
        ]}));
        assert!(e.contains("Duplicate directive names: ['n']"), "{e}");
        assert!(e.contains("directives.2.name: Field required"), "{e}");
    }

    #[test]
    fn unsupported_settings_are_refused_by_name() {
        let e = err(json!({"version": "1", "bank": {
            "retain_extraction_mode": "verbose", "disposition_empathy": 9
        }}));
        assert!(
            e.contains("bank.retain_extraction_mode: not supported by this server"),
            "{e}"
        );
        assert!(
            e.contains("bank.disposition_empathy: disposition.empathy must be between 1 and 5"),
            "{e}"
        );
    }

    #[test]
    fn empty_names_fail_semantic_validation() {
        let e = err(json!({"version": "1",
            "mental_models": [{"id": "a", "name": " ", "source_query": "q"}],
            "directives": [{"name": "n", "content": ""}]}));
        assert_eq!(
            e,
            "invalid input: Template validation failed: mental_models[0].name: must not be empty; directives[0].content: must not be empty"
        );
    }

    #[test]
    fn template_model_ids_are_bank_local() {
        assert_eq!(stored_model_id("b", "m"), "b/m");
        assert_eq!(template_model_id("b", "b/m"), "m");
        assert_eq!(template_model_id("b", "uuid-1"), "uuid-1");
    }
}
