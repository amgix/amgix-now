use std::collections::HashSet;

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use serde::de::Error as DeError;
use serde::Deserializer;
use serde_json::Value;

use crate::common::{MAX_COLLECTION_NAME_LENGTH, MAX_METADATA_KEY_LENGTH};
use crate::filter_parser::parse_filter;
use crate::models::MetadataFilter;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[grammar = "join_parser.pest"]
struct JoinParser;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JoinRefKind {
    Id,
    Meta(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JoinSideRef {
    pub kind: JoinRefKind,
}

impl JoinSideRef {
    pub fn id() -> Self {
        Self { kind: JoinRefKind::Id }
    }

    pub fn meta(key: String) -> Self {
        Self { kind: JoinRefKind::Meta(key) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinSpec {
    pub collection_name: String,
    pub parent_ref: JoinSideRef,
    pub child_ref: JoinSideRef,
    pub metadata_filter: Option<MetadataFilter>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JoinField {
    Single(String),
    Multiple(Vec<String>),
}

pub fn deserialize_join_field<'de, D>(deserializer: D) -> Result<Option<JoinField>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Option<Value> = Option::deserialize(deserializer)?;
    match v {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(JoinField::Single(s))),
        Some(Value::Array(arr)) => {
            let mut items = Vec::with_capacity(arr.len());
            for item in arr {
                let s = item
                    .as_str()
                    .ok_or_else(|| DeError::custom("Join list items must be strings"))?
                    .to_string();
                items.push(s);
            }
            Ok(Some(JoinField::Multiple(items)))
        }
        Some(_) => Err(DeError::custom("Join must be a string or list of strings")),
    }
}

impl JoinField {
    pub fn into_specs(self) -> Result<Vec<JoinSpec>, String> {
        let items = match self {
            JoinField::Single(s) => vec![s],
            JoinField::Multiple(v) => v,
        };
        if items.is_empty() {
            return Err("Join list cannot be empty".to_string());
        }
        let mut specs = Vec::with_capacity(items.len());
        let mut seen = HashSet::new();
        for item in items {
            let spec = parse_join(&item)?;
            if !seen.insert(spec.collection_name.clone()) {
                return Err(format!(
                    "Duplicate join collection '{}' in join list",
                    spec.collection_name
                ));
            }
            specs.push(spec);
        }
        Ok(specs)
    }
}

pub fn parse_join(expr: &str) -> Result<JoinSpec, String> {
    let text = expr.trim();
    if text.is_empty() {
        return Err("Join expression cannot be empty".to_string());
    }

    let pairs = JoinParser::parse(Rule::join_expr, text)
        .map_err(|e| format!("Invalid join expression: {e}"))?;

    let pair = pairs
        .into_iter()
        .next()
        .ok_or_else(|| "Empty join expression".to_string())?;

    build_join_expr(pair)
}

fn build_join_expr(pair: Pair<Rule>) -> Result<JoinSpec, String> {
    let mut inner = pair.into_inner();
    let collection = inner
        .next()
        .ok_or_else(|| "Invalid join expression: missing collection name".to_string())?
        .as_str()
        .to_string();
    validate_collection_name(&collection)?;

    let mut parent_ref = JoinSideRef::id();
    let mut child_ref = JoinSideRef::id();
    let mut metadata_filter = None;

    for child in inner {
        match child.as_rule() {
            Rule::join_keys => {
                let (p, c) = build_join_keys(child)?;
                parent_ref = p;
                child_ref = c;
            }
            Rule::filter_part => {
                metadata_filter = Some(build_filter_part(child)?);
            }
            Rule::EOI => {}
            rule => return Err(format!("Unexpected join rule: {rule:?}")),
        }
    }

    Ok(JoinSpec {
        collection_name: collection,
        parent_ref,
        child_ref,
        metadata_filter,
    })
}

fn build_join_keys(pair: Pair<Rule>) -> Result<(JoinSideRef, JoinSideRef), String> {
    let mut inner = pair.into_inner();
    let parent = build_parent_ref(
        inner
            .next()
            .ok_or_else(|| "Invalid join keys: missing parent ref".to_string())?,
    )?;
    let child = build_child_ref(
        inner
            .next()
            .ok_or_else(|| "Invalid join keys: missing child ref".to_string())?,
    )?;
    Ok((parent, child))
}

fn build_parent_ref(pair: Pair<Rule>) -> Result<JoinSideRef, String> {
    let inner = match pair.as_rule() {
        Rule::parent_ref => pair
            .into_inner()
            .next()
            .ok_or_else(|| "Invalid parent join reference".to_string())?,
        Rule::parent_id | Rule::parent_meta => pair,
        rule => return Err(format!("Unexpected parent ref rule: {rule:?}")),
    };
    match inner.as_rule() {
        Rule::parent_id => Ok(JoinSideRef::id()),
        Rule::parent_meta => {
            let key = inner
                .into_inner()
                .next()
                .ok_or_else(|| "Invalid parent metadata ref".to_string())?
                .as_str()
                .to_string();
            validate_meta_key(&key)?;
            Ok(JoinSideRef::meta(key))
        }
        rule => Err(format!("Unexpected parent ref rule: {rule:?}")),
    }
}

fn build_child_ref(pair: Pair<Rule>) -> Result<JoinSideRef, String> {
    let inner = match pair.as_rule() {
        Rule::child_ref => pair
            .into_inner()
            .next()
            .ok_or_else(|| "Invalid child join reference".to_string())?,
        Rule::child_id | Rule::child_meta => pair,
        rule => return Err(format!("Unexpected child ref rule: {rule:?}")),
    };
    match inner.as_rule() {
        Rule::child_id => Ok(JoinSideRef::id()),
        Rule::child_meta => {
            let key = inner
                .into_inner()
                .next()
                .ok_or_else(|| "Invalid child metadata ref".to_string())?
                .as_str()
                .to_string();
            validate_meta_key(&key)?;
            Ok(JoinSideRef::meta(key))
        }
        rule => Err(format!("Unexpected child ref rule: {rule:?}")),
    }
}

fn build_filter_part(pair: Pair<Rule>) -> Result<MetadataFilter, String> {
    let expr = pair
        .into_inner()
        .next()
        .ok_or_else(|| "Invalid join filter: missing expression".to_string())?;
    parse_filter(expr.as_str())
}

fn validate_collection_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_COLLECTION_NAME_LENGTH {
        return Err(format!(
            "Invalid collection name '{name}': must be 1–{MAX_COLLECTION_NAME_LENGTH} characters"
        ));
    }
    Ok(())
}

fn validate_meta_key(key: &str) -> Result<(), String> {
    if key.is_empty() || key.len() > MAX_METADATA_KEY_LENGTH {
        return Err(format!(
            "Metadata key '{key}' exceeds {MAX_METADATA_KEY_LENGTH} character limit"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_collection() {
        let spec = parse_join("child_coll").unwrap();
        assert_eq!(spec.collection_name, "child_coll");
        assert_eq!(spec.parent_ref, JoinSideRef::id());
        assert_eq!(spec.child_ref, JoinSideRef::id());
        assert!(spec.metadata_filter.is_none());
    }

    #[test]
    fn parse_filter_only() {
        let spec = parse_join(r#"child(role = "primary")"#).unwrap();
        assert_eq!(spec.collection_name, "child");
        assert!(spec.metadata_filter.is_some());
    }

    #[test]
    fn parse_keys_and_filter() {
        let spec = parse_join(r#"child[$id=$$.meta.parent_ref](role = "primary")"#).unwrap();
        assert_eq!(spec.parent_ref, JoinSideRef::id());
        assert_eq!(
            spec.child_ref,
            JoinSideRef::meta("parent_ref".to_string())
        );
        assert!(spec.metadata_filter.is_some());
    }

    #[test]
    fn parse_invalid_syntax() {
        let err = parse_join("bad[[syntax").unwrap_err();
        assert!(err.to_lowercase().contains("invalid join"));
    }

    #[test]
    fn parse_metadata_child_ref() {
        let spec = parse_join(r#"child[$id=$$.meta.parent_ref]"#).unwrap();
        assert_eq!(spec.parent_ref, JoinSideRef::id());
        assert_eq!(
            spec.child_ref,
            JoinSideRef::meta("parent_ref".to_string())
        );
    }
}
