//! Hierarchical configuration (RS-10f): global → tenant → bank.
//!
//! Upstream's `config_resolver.py` resolves settings per request from the
//! server's environment, then a tenant layer, then the bank's own
//! overrides. Here a tenant is a named config layer (the `tenants` table);
//! a bank joins one through its `tenant` setting, or the engine's default
//! tenant (`RUSTSIGHT_DEFAULT_TENANT`). Tenants scope configuration only —
//! access control stays the single-token model.
//!
//! [`KEYS`] is the one registry of layered settings: engine knobs whose
//! global value comes from [`EngineConfig`], and bank settings whose global
//! value is the built-in default. Values are type-checked on write at
//! either layer ([`validate`]); [`resolve`] merges the layers and records
//! which one each value came from.

use crate::engine::{Engine, EngineConfig};
use crate::error::{Error, Result};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// What a key's values must look like.
#[derive(Debug, Clone, Copy)]
pub enum Kind {
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
    },
    Bool,
    Str,
    /// An entity-label configuration (`labels::BankLabels`).
    Labels,
    /// A Memory Defense policy (`defense::parse_policy`).
    Defense,
}

#[derive(Debug, Clone, Copy)]
pub struct KeySpec {
    pub name: &'static str,
    pub kind: Kind,
    /// Engine knob (global value from `EngineConfig`) vs bank setting
    /// (global value is the built-in default).
    pub engine: bool,
}

const fn knob(name: &'static str, kind: Kind) -> KeySpec {
    KeySpec {
        name,
        kind,
        engine: true,
    }
}

const fn setting(name: &'static str, kind: Kind) -> KeySpec {
    KeySpec {
        name,
        kind,
        engine: false,
    }
}

const fn int(min: i64, max: i64) -> Kind {
    Kind::Int { min, max }
}

/// Every setting a tenant or bank can override.
pub const KEYS: &[KeySpec] = &[
    knob("chunk_size_tokens", int(16, 1_000_000)),
    knob("chunk_overlap_tokens", int(0, 100_000)),
    knob("reflect_max_rounds", int(1, 100)),
    knob("reflect_answer_max_tokens", int(1, 1_000_000)),
    knob("recall_budget_low", int(1, 100_000)),
    knob("recall_budget_mid", int(1, 100_000)),
    knob("recall_budget_high", int(1, 100_000)),
    knob("reranker_max_candidates", int(0, 1_000_000)),
    knob("consolidate_max_facts_per_entity", int(1, 1_000_000)),
    knob(
        "consolidate_dupe_cosine",
        Kind::Float { min: 0.0, max: 1.0 },
    ),
    knob("output_language", Kind::Str),
    setting("retain_mission", Kind::Str),
    setting("observations_mission", Kind::Str),
    setting("enable_observations", Kind::Bool),
    setting("entity_labels", Kind::Labels),
    setting("entities_allow_free_form", Kind::Bool),
    setting("memory_defense", Kind::Defense),
];

pub fn spec(key: &str) -> Option<&'static KeySpec> {
    KEYS.iter().find(|k| k.name == key)
}

pub fn is_engine_knob(key: &str) -> bool {
    spec(key).is_some_and(|k| k.engine)
}

fn unknown_key(key: &str) -> Error {
    let names: Vec<&str> = KEYS.iter().map(|k| k.name).collect();
    Error::BadInput(format!(
        "unknown config key {key:?}; configurable keys: {}",
        names.join(", ")
    ))
}

/// Type-check one value for `key`. `null` is always accepted: it clears
/// the override so the layer below shows through.
pub fn validate(key: &str, value: &Value) -> Result<()> {
    let spec = spec(key).ok_or_else(|| unknown_key(key))?;
    if value.is_null() {
        return Ok(());
    }
    let bad = |expected: String| Err(Error::BadInput(format!("{key} must be {expected}")));
    match spec.kind {
        Kind::Int { min, max } => match value.as_i64() {
            Some(n) if (min..=max).contains(&n) => Ok(()),
            _ => bad(format!("an integer between {min} and {max}")),
        },
        Kind::Float { min, max } => match value.as_f64() {
            Some(n) if (min..=max).contains(&n) => Ok(()),
            _ => bad(format!("a number between {min} and {max}")),
        },
        Kind::Bool if value.is_boolean() => Ok(()),
        Kind::Bool => bad("a boolean".into()),
        Kind::Str if value.is_string() => Ok(()),
        Kind::Str => bad("a string".into()),
        Kind::Labels => crate::labels::BankLabels::parse(Some(value), true).map(|_| ()),
        Kind::Defense => crate::defense::parse_policy(Some(value)).map(|_| ()),
    }
}

/// Which layer a resolved value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    /// A bank setting nobody set: the built-in default applies.
    Default,
    /// The server's own configuration (`EngineConfig`, from env/flags).
    Global,
    Tenant,
    Bank,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Entry {
    pub value: Value,
    pub source: Layer,
}

/// Every registered key's effective value for one bank.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedConfig {
    /// The tenant the bank resolved through, if any.
    pub tenant: Option<String>,
    pub config: BTreeMap<String, Entry>,
}

impl ResolvedConfig {
    /// The effective value (`None` when unset or null).
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.config
            .get(key)
            .map(|e| &e.value)
            .filter(|v| !v.is_null())
    }

    pub fn source(&self, key: &str) -> Option<Layer> {
        self.config.get(key).map(|e| e.source)
    }

    pub fn usize(&self, key: &str) -> Option<usize> {
        self.get(key)?.as_u64().map(|n| n as usize)
    }

    pub fn f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f64().map(|n| n as f32)
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    /// A non-blank string value, trimmed.
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key)?
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
}

/// The global layer: engine knobs from `EngineConfig`.
pub fn global_layer(c: &EngineConfig) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("chunk_size_tokens".into(), json!(c.chunk_size_tokens));
    m.insert("chunk_overlap_tokens".into(), json!(c.chunk_overlap_tokens));
    m.insert("reflect_max_rounds".into(), json!(c.reflect_max_rounds));
    m.insert(
        "reflect_answer_max_tokens".into(),
        json!(c.reflect_answer_max_tokens),
    );
    m.insert("recall_budget_low".into(), json!(c.recall_budget.fixed_low));
    m.insert("recall_budget_mid".into(), json!(c.recall_budget.fixed_mid));
    m.insert(
        "recall_budget_high".into(),
        json!(c.recall_budget.fixed_high),
    );
    m.insert(
        "reranker_max_candidates".into(),
        json!(c.reranker_max_candidates),
    );
    m.insert(
        "consolidate_max_facts_per_entity".into(),
        json!(c.consolidate_max_facts_per_entity),
    );
    m.insert(
        "consolidate_dupe_cosine".into(),
        // f32 → shortest decimal (0.92, not 0.9200000166893005).
        json!(c
            .consolidate_dupe_cosine
            .to_string()
            .parse::<f64>()
            .unwrap_or_default()),
    );
    m.insert("output_language".into(), json!(c.output_language));
    m
}

/// Merge the layers: the bank's value beats the tenant's, which beats the
/// global one. Keys outside the registry in either map are ignored (bank
/// config JSON also holds unlayered settings such as `mission_scope`).
pub fn resolve(
    global: &Map<String, Value>,
    tenant: Option<(&str, &Map<String, Value>)>,
    bank: &Map<String, Value>,
) -> ResolvedConfig {
    let mut config = BTreeMap::new();
    for spec in KEYS {
        let key = spec.name;
        let set = |m: &Map<String, Value>| m.get(key).filter(|v| !v.is_null()).cloned();
        let entry = if let Some(v) = set(bank) {
            Entry {
                value: v,
                source: Layer::Bank,
            }
        } else if let Some(v) = tenant.and_then(|(_, t)| set(t)) {
            Entry {
                value: v,
                source: Layer::Tenant,
            }
        } else if spec.engine {
            Entry {
                value: global.get(key).cloned().unwrap_or(Value::Null),
                source: Layer::Global,
            }
        } else {
            Entry {
                value: Value::Null,
                source: Layer::Default,
            }
        };
        config.insert(key.to_string(), entry);
    }
    ResolvedConfig {
        tenant: tenant.map(|(id, _)| id.to_string()),
        config,
    }
}

fn as_map(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

impl Engine {
    /// The tenant a bank resolves through: its own `tenant` setting, else
    /// the engine's default tenant.
    pub fn bank_tenant(&self, bank_config: &Map<String, Value>) -> Option<String> {
        bank_config
            .get("tenant")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.default_tenant.clone())
    }

    /// Every layered setting's effective value for `bank_id`, with the
    /// layer it came from. A bank that does not exist yet resolves as an
    /// empty bank layer.
    pub fn resolved_config(&self, bank_id: &str) -> Result<ResolvedConfig> {
        let bank = as_map(self.store.bank_config(bank_id)?);
        let tenant = match self.bank_tenant(&bank) {
            Some(id) => {
                let layer = self
                    .store
                    .tenant_config(&id)?
                    .map(as_map)
                    .unwrap_or_default();
                Some((id, layer))
            }
            None => None,
        };
        Ok(resolve(
            &global_layer(&self.config),
            tenant.as_ref().map(|(id, m)| (id.as_str(), m)),
            &bank,
        ))
    }

    /// The recall budget levels for a bank (tenant/bank overrides of the
    /// fixed low/mid/high widths applied over the global config).
    pub fn recall_budget_for(&self, r: &ResolvedConfig) -> crate::engine::RecallBudgetConfig {
        let mut b = self.config.recall_budget;
        if let Some(n) = r.usize("recall_budget_low") {
            b.fixed_low = n;
        }
        if let Some(n) = r.usize("recall_budget_mid") {
            b.fixed_mid = n;
        }
        if let Some(n) = r.usize("recall_budget_high") {
            b.fixed_high = n;
        }
        b
    }

    /// A tenant's stored overrides (empty for an unknown tenant).
    pub fn tenant_config(&self, tenant_id: &str) -> Result<Map<String, Value>> {
        Ok(self
            .store
            .tenant_config(tenant_id)?
            .map(as_map)
            .unwrap_or_default())
    }

    /// Apply overrides to a tenant (`null` clears one), creating it on
    /// first write. Every update is validated before anything is stored.
    pub fn update_tenant_config(
        &self,
        tenant_id: &str,
        updates: &Map<String, Value>,
    ) -> Result<Map<String, Value>> {
        let tenant_id = tenant_id.trim();
        if tenant_id.is_empty() {
            return Err(Error::BadInput("tenant id is empty".into()));
        }
        for (key, value) in updates {
            validate(key, value)?;
        }
        let mut config = self.tenant_config(tenant_id)?;
        for (key, value) in updates {
            if value.is_null() {
                config.remove(key);
            } else {
                config.insert(key.clone(), value.clone());
            }
        }
        self.store
            .set_tenant_config(tenant_id, &Value::Object(config.clone()))?;
        // A tenant may have just switched observations back on.
        self.wake_worker();
        Ok(config)
    }

    pub fn list_tenants(&self) -> Result<Vec<String>> {
        self.store.list_tenants()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(v: Value) -> Map<String, Value> {
        as_map(v)
    }

    #[test]
    fn bank_beats_tenant_beats_global() {
        let global = global_layer(&EngineConfig::default());
        let tenant = map(json!({"chunk_size_tokens": 500, "reflect_max_rounds": 3,
                                "retain_mission": "tenant mission"}));
        let bank = map(json!({"chunk_size_tokens": 800, "mission_scope": "reflect"}));
        let r = resolve(&global, Some(("acme", &tenant)), &bank);
        assert_eq!(r.tenant.as_deref(), Some("acme"));
        assert_eq!(r.usize("chunk_size_tokens"), Some(800));
        assert_eq!(r.source("chunk_size_tokens"), Some(Layer::Bank));
        assert_eq!(r.usize("reflect_max_rounds"), Some(3));
        assert_eq!(r.source("reflect_max_rounds"), Some(Layer::Tenant));
        assert_eq!(r.str("retain_mission"), Some("tenant mission"));
        assert_eq!(r.usize("chunk_overlap_tokens"), Some(200));
        assert_eq!(r.source("chunk_overlap_tokens"), Some(Layer::Global));
        assert_eq!(r.get("enable_observations"), None);
        assert_eq!(r.source("enable_observations"), Some(Layer::Default));
        assert!(
            !r.config.contains_key("mission_scope"),
            "unlayered keys stay out"
        );

        let none = resolve(&global, None, &Map::new());
        assert_eq!(none.tenant, None);
        assert_eq!(none.usize("chunk_size_tokens"), Some(3000));
        assert_eq!(none.get("output_language"), None, "null global = unset");
    }

    #[test]
    fn values_are_type_checked() {
        assert!(validate("chunk_size_tokens", &json!(512)).is_ok());
        assert!(validate("chunk_size_tokens", &Value::Null).is_ok());
        for (key, value) in [
            ("chunk_size_tokens", json!("512")),
            ("chunk_size_tokens", json!(0)),
            ("chunk_size_tokens", json!(1.5)),
            ("consolidate_dupe_cosine", json!(1.5)),
            ("enable_observations", json!("yes")),
            ("output_language", json!(3)),
            (
                "memory_defense",
                json!({"rules": [{"on": "secrets", "action": "explode"}]}),
            ),
        ] {
            let err = validate(key, &value).unwrap_err().to_string();
            assert!(err.contains(key) || err.contains("action"), "{key}: {err}");
        }
        assert!(validate("consolidate_dupe_cosine", &json!(0.9)).is_ok());
        assert!(validate("entity_labels", &json!([])).is_ok());
    }

    #[test]
    fn unknown_keys_are_refused() {
        let err = validate("chunk_size", &json!(1)).unwrap_err().to_string();
        assert!(err.contains("unknown config key \"chunk_size\""), "{err}");
        // Bank-row and membership settings are not tenant-layered.
        assert!(validate("reflect_mission", &json!("x")).is_err());
        assert!(validate("tenant", &json!("acme")).is_err());
    }
}
