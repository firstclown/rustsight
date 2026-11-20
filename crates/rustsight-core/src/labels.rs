//! Per-bank entity labels (upstream `engine/retain/entity_labels.py`).
//!
//! A bank can configure a controlled vocabulary the extractor classifies
//! every fact against. Chosen values become `key:value` entities, and groups
//! flagged `tag: true` also land in the fact's tags, where recall can filter
//! on them.

use crate::prompts::upstream::extraction::{
    LabelGroup, LabelGroupKind, LabelValue, MapField, MapFieldKind,
};
use crate::types::{EntityType, ExtractedEntity, ExtractedFact};
use crate::{Error, Result};
use serde_json::{Map, Value};
use std::collections::HashSet;

/// A bank's parsed label configuration.
#[derive(Debug, Clone)]
pub struct BankLabels {
    pub groups: Vec<LabelGroup>,
    /// Whether ordinary named entities are kept next to label entities.
    pub free_form: bool,
    /// Lowercased keys of the `tag: true` groups.
    tag_keys: HashSet<String>,
    /// Lowercased `key:value` strings of the closed vocabularies.
    lookup: HashSet<String>,
}

fn bad(message: impl Into<String>) -> Error {
    Error::BadInput(format!("entity_labels: {}", message.into()))
}

fn text_field(obj: &Map<String, Value>, key: &str) -> Result<String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(bad(format!("{key} must be a string"))),
    }
}

fn parse_values(obj: &Map<String, Value>) -> Result<Vec<LabelValue>> {
    let Some(values) = obj.get("values").filter(|v| !v.is_null()) else {
        return Ok(vec![]);
    };
    values
        .as_array()
        .ok_or_else(|| bad("values must be a list"))?
        .iter()
        .map(|v| {
            let v = v
                .as_object()
                .ok_or_else(|| bad("each value must be an object"))?;
            let value = v
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("each value needs a string 'value'"))?;
            Ok(LabelValue::new(value, &text_field(v, "description")?))
        })
        .collect()
}

fn parse_fields(obj: &Map<String, Value>) -> Result<Vec<(String, MapField)>> {
    let Some(fields) = obj.get("fields").filter(|v| !v.is_null()) else {
        return Ok(vec![]);
    };
    fields
        .as_object()
        .ok_or_else(|| bad("fields must be an object"))?
        .iter()
        .map(|(name, field)| {
            let field = field
                .as_object()
                .ok_or_else(|| bad(format!("field {name} must be an object")))?;
            let kind = match field.get("type").and_then(Value::as_str).unwrap_or("text") {
                "text" => MapFieldKind::Text,
                "multi-text" => MapFieldKind::MultiText,
                "value" => MapFieldKind::Value,
                "multi-values" => MapFieldKind::MultiValues,
                "map" => MapFieldKind::Map,
                other => return Err(bad(format!("unknown field type {other:?}"))),
            };
            Ok((
                name.clone(),
                MapField {
                    kind,
                    description: text_field(field, "description")?,
                    values: parse_values(field)?,
                    fields: parse_fields(field)?,
                },
            ))
        })
        .collect()
}

fn parse_kind(obj: &Map<String, Value>) -> Result<LabelGroupKind> {
    let flag = |key: &str| obj.get(key).and_then(Value::as_bool).unwrap_or(false);
    Ok(match obj.get("type").and_then(Value::as_str) {
        Some("value") => LabelGroupKind::Value,
        Some("multi-values") => LabelGroupKind::MultiValues,
        Some("text") => LabelGroupKind::Text,
        Some("multi-text") => LabelGroupKind::MultiText,
        Some("map") => LabelGroupKind::Map,
        Some(other) => return Err(bad(format!("unknown group type {other:?}"))),
        // Upstream's legacy spelling.
        None if flag("free_values") => LabelGroupKind::Text,
        None if flag("multi_value") => LabelGroupKind::MultiValues,
        None => LabelGroupKind::Value,
    })
}

impl BankLabels {
    /// Parse the `entity_labels` config value: a list of groups or
    /// `{"attributes": [...]}`. `None` when it configures nothing and
    /// free-form entities stay on.
    pub fn parse(raw: Option<&Value>, free_form: bool) -> Result<Option<Self>> {
        let groups_raw: &[Value] = match raw {
            None | Some(Value::Null) => &[],
            Some(Value::Array(items)) => items,
            Some(Value::Object(obj)) => match obj.get("attributes") {
                None | Some(Value::Null) => &[],
                Some(Value::Array(items)) => items,
                Some(_) => return Err(bad("attributes must be a list")),
            },
            Some(_) => return Err(bad("must be a list of label groups")),
        };
        let mut groups = Vec::with_capacity(groups_raw.len());
        let mut tag_keys = HashSet::new();
        let mut lookup = HashSet::new();
        for raw_group in groups_raw {
            let obj = raw_group
                .as_object()
                .ok_or_else(|| bad("each label group must be an object"))?;
            let key = obj
                .get("key")
                .and_then(Value::as_str)
                .filter(|k| !k.is_empty())
                .ok_or_else(|| bad("each label group needs a 'key'"))?;
            let kind = parse_kind(obj)?;
            let values = parse_values(obj)?;
            if obj.get("tag").and_then(Value::as_bool).unwrap_or(false) {
                tag_keys.insert(key.to_lowercase());
            }
            if matches!(kind, LabelGroupKind::Value | LabelGroupKind::MultiValues) {
                for v in values.iter().filter(|v| !v.value.is_empty()) {
                    lookup.insert(format!("{key}:{}", v.value).to_lowercase());
                }
            }
            groups.push(LabelGroup {
                key: key.to_string(),
                description: text_field(obj, "description")?,
                kind,
                values,
                fields: parse_fields(obj)?,
            });
        }
        if groups.is_empty() && free_form {
            return Ok(None);
        }
        Ok(Some(Self {
            groups,
            free_form,
            tag_keys,
            lookup,
        }))
    }

    fn is_label_entity(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        if self.lookup.contains(&lower) {
            return true;
        }
        self.groups.iter().any(|g| {
            let prefix = format!("{}:", g.key.to_lowercase());
            match g.kind {
                LabelGroupKind::Text | LabelGroupKind::MultiText => lower.starts_with(&prefix),
                LabelGroupKind::Map => map_path_matches(&lower, &prefix, &g.fields),
                _ => false,
            }
        })
    }

    /// Turn a fact's raw `labels` object into `key:value` entities and tags
    /// (upstream's post-processing plus `_inject_label_tags`).
    pub fn apply(&self, fact: &mut ExtractedFact) {
        let labels = fact.labels.take();
        let mut names = LabelNames::new(fact);
        if let Some(Value::Object(data)) = &labels {
            for group in &self.groups {
                let Some(value) = data.get(&group.key).filter(|v| truthy(v)) else {
                    continue;
                };
                match group.kind {
                    LabelGroupKind::Map => {
                        for item in one_or_many(value) {
                            if let Value::Object(obj) = item {
                                let prefix = format!("{}:", group.key);
                                names.add_map(obj, &group.fields, &prefix);
                            }
                        }
                    }
                    LabelGroupKind::Text | LabelGroupKind::MultiText => {
                        for v in one_or_many(value).filter_map(usable_label) {
                            names.add(format!("{}:{v}", group.key));
                        }
                    }
                    LabelGroupKind::Value | LabelGroupKind::MultiValues => {
                        for v in one_or_many(value).filter_map(usable_label) {
                            let label = format!("{}:{v}", group.key);
                            if self.lookup.contains(&label.to_lowercase()) {
                                names.add(label);
                            } else {
                                eprintln!(
                                    "rustsight: label {label:?} is not an allowed value, skipping"
                                );
                            }
                        }
                    }
                }
            }
        }
        let added = names.added;
        fact.entities
            .extend(added.into_iter().map(|name| ExtractedEntity {
                name,
                etype: EntityType::Other,
            }));
        if !self.free_form {
            fact.entities.retain(|e| self.is_label_entity(&e.name));
        }
        for entity in &fact.entities {
            let Some((key, _)) = entity.name.split_once(':') else {
                continue;
            };
            if self.tag_keys.contains(&key.to_lowercase()) && !fact.tags.contains(&entity.name) {
                fact.tags.push(entity.name.clone());
            }
        }
    }
}

/// New label strings for one fact, case-insensitively deduplicated against
/// its entities.
struct LabelNames {
    seen: HashSet<String>,
    added: Vec<String>,
}

impl LabelNames {
    fn new(fact: &ExtractedFact) -> Self {
        Self {
            seen: fact
                .entities
                .iter()
                .map(|e| e.name.to_lowercase())
                .collect(),
            added: vec![],
        }
    }

    fn add(&mut self, label: String) {
        if self.seen.insert(label.to_lowercase()) {
            self.added.push(label);
        }
    }

    fn add_map(&mut self, obj: &Map<String, Value>, fields: &[(String, MapField)], prefix: &str) {
        for (name, field) in fields {
            let Some(value) = obj.get(name).filter(|v| !v.is_null() && *v != "") else {
                continue;
            };
            let path = format!("{prefix}{name}:");
            if field.kind == MapFieldKind::Map && !field.fields.is_empty() {
                for item in one_or_many(value) {
                    if let Value::Object(child) = item {
                        self.add_map(child, &field.fields, &path);
                    }
                }
            } else {
                for v in one_or_many(value).filter_map(usable_label) {
                    self.add(format!("{path}{v}"));
                }
            }
        }
    }
}

fn map_path_matches(text: &str, prefix: &str, fields: &[(String, MapField)]) -> bool {
    fields.iter().any(|(name, field)| {
        let path = format!("{prefix}{}:", name.to_lowercase());
        if field.kind == MapFieldKind::Map && !field.fields.is_empty() {
            map_path_matches(text, &path, &field.fields)
        } else {
            text.starts_with(&path)
        }
    })
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn one_or_many(value: &Value) -> Box<dyn Iterator<Item = &Value> + '_> {
    match value {
        Value::Array(items) => Box::new(items.iter()),
        other => Box::new(std::iter::once(other)),
    }
}

fn usable_label(value: &Value) -> Option<&str> {
    let text = value.as_str()?.trim();
    let placeholder = ["none", "null", "n/a"]
        .iter()
        .any(|p| text.eq_ignore_ascii_case(p));
    (!text.is_empty() && !placeholder).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryKind;
    use serde_json::json;

    fn fact(entities: &[&str], labels: Value) -> ExtractedFact {
        ExtractedFact {
            text: "t".into(),
            fact_type: MemoryKind::World,
            entities: entities
                .iter()
                .map(|n| ExtractedEntity {
                    name: n.to_string(),
                    etype: EntityType::Other,
                })
                .collect(),
            occurred_start_ms: None,
            occurred_end_ms: None,
            causal: vec![],
            chunk_index: 0,
            labels: Some(labels),
            tags: vec![],
        }
    }

    fn config() -> Value {
        json!([
            {"key": "state", "type": "value", "optional": false, "tag": true,
             "values": [{"value": "current"}, {"value": "historical"}]},
            {"key": "name", "type": "multi-text", "tag": true},
            {"key": "topic", "type": "text"}
        ])
    }

    #[test]
    fn tagged_groups_become_entities_and_tags() {
        let labels = BankLabels::parse(Some(&config()), true).unwrap().unwrap();
        let mut f = fact(
            &["Kubernetes"],
            json!({"state": "current", "name": ["k8s", "kube"], "topic": "infra"}),
        );
        labels.apply(&mut f);
        let names: Vec<&str> = f.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Kubernetes") && names.contains(&"topic:infra"));
        assert_eq!(f.tags, vec!["state:current", "name:k8s", "name:kube"]);
    }

    #[test]
    fn values_outside_a_closed_vocabulary_are_dropped() {
        let labels = BankLabels::parse(Some(&config()), true).unwrap().unwrap();
        let mut f = fact(&[], json!({"state": "invented", "name": ["N/A", " "]}));
        labels.apply(&mut f);
        assert!(f.entities.is_empty() && f.tags.is_empty());
    }

    #[test]
    fn labels_only_mode_drops_free_form_entities() {
        let labels = BankLabels::parse(Some(&config()), false).unwrap().unwrap();
        let mut f = fact(&["Kubernetes"], json!({"name": ["k8s"]}));
        labels.apply(&mut f);
        let names: Vec<&str> = f.entities.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["name:k8s"]);
    }

    #[test]
    fn map_groups_flatten_to_paths() {
        let raw = json!({"attributes": [{"key": "person", "type": "map", "fields": {
            "city": {"type": "text"}, "pets": {"type": "multi-text"}
        }}]});
        let labels = BankLabels::parse(Some(&raw), false).unwrap().unwrap();
        let mut f = fact(
            &["Alice"],
            json!({"person": [{"city": "Oslo", "pets": ["Rex"]}]}),
        );
        labels.apply(&mut f);
        let mut names: Vec<&str> = f.entities.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["person:city:Oslo", "person:pets:Rex"]);
    }

    #[test]
    fn malformed_config_is_rejected_and_empty_means_none() {
        assert!(BankLabels::parse(Some(&json!("x")), true).is_err());
        assert!(BankLabels::parse(Some(&json!([{"type": "value"}])), true).is_err());
        assert!(BankLabels::parse(Some(&json!([])), true).unwrap().is_none());
        assert!(BankLabels::parse(None, false).unwrap().is_some());
    }
}
