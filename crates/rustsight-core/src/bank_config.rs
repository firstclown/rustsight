//! Hindsight-named per-bank settings: the keys `PATCH /config` and bank
//! templates accept, their validation, and bank provisioning.
//!
//! A setting either lands on the [`Bank`] row (`reflect_mission`, the
//! disposition traits) or in the bank's config JSON (missions for retain
//! and consolidation, observation toggle, entity labels, Memory Defense).
//! Every key also stays recorded in the config JSON as an override, so
//! reads can tell an explicit value from a default.

use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::labels::BankLabels;
use crate::types::{Bank, Disposition};
use serde_json::{json, Map, Value};

/// The per-bank settings this server acts on. Anything else is refused:
/// accepting a setting that changes nothing would misreport what ran.
pub const CONFIG_KEYS: [&str; 10] = [
    "retain_mission",
    "observations_mission",
    "reflect_mission",
    "enable_observations",
    "entity_labels",
    "entities_allow_free_form",
    "disposition_skepticism",
    "disposition_literalism",
    "disposition_empathy",
    "memory_defense",
];

pub fn is_config_key(key: &str) -> bool {
    CONFIG_KEYS.contains(&key)
}

/// Everything a bank's config can override: the Hindsight settings, the
/// layered engine knobs (RS-10f) and the bank's `tenant` membership.
pub fn is_bank_key(key: &str) -> bool {
    is_config_key(key) || crate::config_resolver::is_engine_knob(key) || key == "tenant"
}

fn supported_keys() -> String {
    let mut keys: Vec<&str> = CONFIG_KEYS.to_vec();
    keys.extend(
        crate::config_resolver::KEYS
            .iter()
            .filter(|k| k.engine)
            .map(|k| k.name),
    );
    keys.push("tenant");
    keys.join(", ")
}

/// Upstream accepts both `retain_mission` and `HINDSIGHT_API_RETAIN_MISSION`.
pub fn normalize_key(raw: &str) -> String {
    let lower = raw.to_lowercase();
    lower
        .strip_prefix("hindsight_api_")
        .unwrap_or(&lower)
        .to_string()
}

/// A disposition trait value, 1..=5.
pub fn trait_value(name: &str, v: i64) -> Result<u8> {
    if (1..=5).contains(&v) {
        Ok(v as u8)
    } else {
        Err(Error::BadInput(format!(
            "disposition.{name} must be between 1 and 5"
        )))
    }
}

/// Validate one setting and apply its effect on the bank row (`null`
/// restores the default). Config-JSON settings are only validated here;
/// the caller records them.
pub fn apply_update(bank: &mut Bank, key: &str, value: &Value) -> Result<()> {
    let wrong_type = |expected: &str| Error::BadInput(format!("{key} must be {expected}"));
    match key {
        "retain_mission" | "observations_mission" | "reflect_mission" => {
            if !(value.is_string() || value.is_null()) {
                return Err(wrong_type("a string or null"));
            }
            if key == "reflect_mission" {
                bank.mission = value
                    .as_str()
                    .filter(|m| !m.trim().is_empty())
                    .map(str::to_string);
            }
        }
        "enable_observations" | "entities_allow_free_form" => {
            if !(value.is_boolean() || value.is_null()) {
                return Err(wrong_type("a boolean or null"));
            }
        }
        "entity_labels" => {
            BankLabels::parse(Some(value), true)?;
        }
        "memory_defense" => {
            // Validated on write so a bad policy never reaches retain, which
            // refuses to run under one (screening must not fail open).
            crate::defense::parse_policy(Some(value))?;
        }
        "disposition_skepticism" | "disposition_literalism" | "disposition_empathy" => {
            let name = key.trim_start_matches("disposition_");
            let defaults = Disposition::default();
            let (slot, default) = match name {
                "skepticism" => (&mut bank.disposition.skepticism, defaults.skepticism),
                "literalism" => (&mut bank.disposition.literalism, defaults.literalism),
                _ => (&mut bank.disposition.empathy, defaults.empathy),
            };
            *slot = match value {
                Value::Null => default,
                other => trait_value(
                    name,
                    other.as_i64().ok_or_else(|| wrong_type("an integer"))?,
                )?,
            };
        }
        "tenant" => {
            if !(value.is_string() || value.is_null()) {
                return Err(wrong_type("a string or null"));
            }
        }
        knob if crate::config_resolver::is_engine_knob(knob) => {
            crate::config_resolver::validate(knob, value)?;
        }
        other => {
            return Err(Error::BadInput(format!(
                "Unsupported bank config key {other:?}; supported keys: {}",
                supported_keys()
            )))
        }
    }
    Ok(())
}

fn default_bank(bank_id: &str) -> Bank {
    Bank {
        id: bank_id.to_string(),
        mission: None,
        directives: None,
        background: None,
        disposition: Disposition::default(),
    }
}

impl Engine {
    /// The bank, created if missing. A newly created bank gets the
    /// engine's default template (`with_default_template`) before anything
    /// else touches it, so a caller's own settings land on top of it.
    pub fn provision_bank(&self, bank_id: &str) -> Result<Bank> {
        if let Some(bank) = self.store.get_bank(bank_id)? {
            return Ok(bank);
        }
        let bank = default_bank(bank_id);
        self.store.ensure_bank(&bank)?;
        if let Some(template) = self.default_template.as_deref() {
            // Best effort, as upstream: the template was validated at startup,
            // so a failure here is a store error worth reporting, not fatal.
            if let Err(err) = self.apply_template(bank_id, template, false) {
                eprintln!("rustsight: default bank template on {bank_id} failed: {err}");
            }
        }
        Ok(self.store.get_bank(bank_id)?.unwrap_or(bank))
    }

    /// The bank's config JSON as a map (empty when unset).
    pub fn bank_config_map(&self, bank_id: &str) -> Result<Map<String, Value>> {
        Ok(match self.store.bank_config(bank_id)? {
            Value::Object(map) => map,
            _ => Map::new(),
        })
    }

    /// Apply Hindsight-named settings (`null` clears one). Every update is
    /// validated before anything is stored; an unsupported key is refused.
    pub fn update_bank_config(&self, bank_id: &str, updates: &Map<String, Value>) -> Result<Bank> {
        let mut bank = self.provision_bank(bank_id)?;
        let mut config = self.bank_config_map(bank_id)?;
        for (raw_key, value) in updates {
            let key = normalize_key(raw_key);
            if !is_bank_key(&key) {
                return Err(Error::BadInput(format!(
                    "Unsupported bank config key {raw_key:?}; supported keys: {}",
                    supported_keys()
                )));
            }
            apply_update(&mut bank, &key, value)?;
            if value.is_null() {
                config.remove(&key);
            } else {
                config.insert(key, value.clone());
            }
        }
        self.store_bank_settings(&bank, config)?;
        Ok(bank)
    }

    /// Drop every override, restoring defaults.
    pub fn reset_bank_config(&self, bank_id: &str) -> Result<Bank> {
        let mut bank = self.provision_bank(bank_id)?;
        let mut config = self.bank_config_map(bank_id)?;
        let overridden: Vec<String> = config
            .keys()
            .filter(|key| is_bank_key(key))
            .cloned()
            .collect();
        for key in overridden {
            apply_update(&mut bank, &key, &Value::Null)?;
            config.remove(&key);
        }
        self.store_bank_settings(&bank, config)?;
        Ok(bank)
    }

    fn store_bank_settings(&self, bank: &Bank, mut config: Map<String, Value>) -> Result<()> {
        // A bank configured with Hindsight's names follows Hindsight's
        // mission split: `reflect_mission` steers reflect only.
        config.insert("mission_scope".into(), json!("reflect"));
        self.store.ensure_bank(bank)?;
        self.store.update_bank(bank)?;
        self.store
            .set_bank_config(&bank.id, &Value::Object(config))?;
        // Observations may have just been switched back on.
        self.wake_worker();
        Ok(())
    }
}
