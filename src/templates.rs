//! Document/query template parsing and rendering for VectorConfig template mode.
//!
//! Mirrors amgix-server `src/core/common/vector_templates.py`.

use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

use crate::common::DocumentField;
use crate::models::{Document, VectorConfigInternal};

static PLACEHOLDER_RE: OnceLock<Regex> = OnceLock::new();
static META_KEY_RE: OnceLock<Regex> = OnceLock::new();

fn placeholder_re() -> &'static Regex {
    PLACEHOLDER_RE.get_or_init(|| Regex::new(r"\{([^{}]+)\}").expect("placeholder regex"))
}

fn meta_key_re() -> &'static Regex {
    META_KEY_RE.get_or_init(|| Regex::new(r"^[a-zA-Z0-9_-]+$").expect("meta key regex"))
}

const META_PREFIX: &str = ".meta.";

pub fn uses_templates(config: &VectorConfigInternal) -> bool {
    config.doc_template.is_some()
}

/// Physical field slots for a vector config (`index_fields` or synthetic `template`).
pub fn vector_index_fields(config: &VectorConfigInternal) -> Result<Vec<DocumentField>, String> {
    if uses_templates(config) {
        return Ok(vec![DocumentField::Template]);
    }
    match &config.index_fields {
        Some(fields) if !fields.is_empty() => Ok(fields.clone()),
        Some(_) => Err(format!(
            "Vector '{}' has empty index_fields and no templates",
            config.name
        )),
        None => Err(format!(
            "Vector '{}' has no index_fields and no templates",
            config.name
        )),
    }
}

fn parse_placeholders(template: &str) -> Vec<&str> {
    placeholder_re()
        .captures_iter(template)
        .map(|c| c.get(1).map(|m| m.as_str()).unwrap_or(""))
        .collect()
}

pub fn validate_doc_template(template: &str) -> Result<(), String> {
    if template.is_empty() {
        return Err("doc_template cannot be empty".to_string());
    }
    let placeholders = parse_placeholders(template);
    if placeholders.is_empty() {
        return Err(
            "doc_template must contain at least one placeholder \
             ({name}, {description}, {content}, or {.meta.<key>})"
                .to_string(),
        );
    }
    for name in placeholders {
        if matches!(name, "name" | "description" | "content") {
            continue;
        }
        if let Some(key) = name.strip_prefix(META_PREFIX) {
            if key.is_empty() || !meta_key_re().is_match(key) {
                return Err(format!(
                    "Invalid doc_template placeholder '{{{name}}}': \
                     meta key must match [a-zA-Z0-9_-]+"
                ));
            }
            continue;
        }
        return Err(format!(
            "Invalid doc_template placeholder '{{{name}}}': \
             allowed are {{name}}, {{description}}, {{content}}, {{.meta.<key>}}"
        ));
    }
    Ok(())
}

pub fn validate_query_template(template: &str) -> Result<(), String> {
    if template.is_empty() {
        return Err("query_template cannot be empty".to_string());
    }
    let placeholders = parse_placeholders(template);
    if placeholders != ["query"] {
        return Err(
            "query_template must contain exactly one {query} placeholder and no others"
                .to_string(),
        );
    }
    Ok(())
}

fn scalar_to_str(value: &Value) -> Option<String> {
    match value {
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn meta_value_to_template_str(value: &Value) -> Result<String, String> {
    if value.is_null() {
        return Ok(String::new());
    }
    if let Some(s) = scalar_to_str(value) {
        return Ok(s);
    }
    if let Value::Array(items) = value {
        let mut parts = Vec::with_capacity(items.len());
        for item in items {
            match scalar_to_str(item) {
                Some(s) => parts.push(s),
                None => return Ok(String::new()),
            }
        }
        return Ok(parts.join(" "));
    }
    if value.is_object() {
        return Ok(String::new());
    }
    Err(format!(
        "Unsupported metadata value type for templates: {}",
        match value {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    ))
}

fn doc_placeholder_value(document: &Document, name: &str) -> Result<String, String> {
    match name {
        "name" => Ok(document.name.clone().unwrap_or_default()),
        "description" => Ok(document.description.clone().unwrap_or_default()),
        "content" => Ok(document.content.clone().unwrap_or_default()),
        _ if name.starts_with(META_PREFIX) => {
            let key = &name[META_PREFIX.len()..];
            let Some(meta) = document.metadata.as_ref() else {
                return Ok(String::new());
            };
            let Some(value) = meta.get(key) else {
                return Ok(String::new());
            };
            meta_value_to_template_str(value)
        }
        _ => Err(format!("Unexpected doc template placeholder '{{{name}}}'")),
    }
}

pub fn render_doc_template(template: &str, document: &Document) -> Result<String, String> {
    let placeholders = parse_placeholders(template);
    let mut values = Vec::with_capacity(placeholders.len());
    for name in &placeholders {
        values.push(doc_placeholder_value(document, name)?);
    }
    if values.iter().all(|v| v.trim().is_empty()) {
        return Ok(String::new());
    }

    let mut out = String::with_capacity(template.len());
    let mut last = 0;
    for cap in placeholder_re().captures_iter(template) {
        let m = cap.get(0).unwrap();
        let name = cap.get(1).unwrap().as_str();
        out.push_str(&template[last..m.start()]);
        out.push_str(&doc_placeholder_value(document, name)?);
        last = m.end();
    }
    out.push_str(&template[last..]);
    Ok(out)
}

pub fn render_query_template(template: &str, query: &str) -> String {
    template.replace("{query}", query)
}

pub fn resolve_search_field(
    config: &VectorConfigInternal,
    field: Option<DocumentField>,
) -> Result<DocumentField, String> {
    if uses_templates(config) {
        return Ok(DocumentField::Template);
    }
    let field = field.ok_or_else(|| {
        format!(
            "field is required for vector '{}' (not template-based)",
            config.name
        )
    })?;
    let slots = vector_index_fields(config)?;
    if !slots.contains(&field) {
        return Err(format!(
            "Field '{field}' is not configured for vector '{}'. Available fields: {:?}",
            config.name,
            slots.iter().map(|f| f.to_string()).collect::<Vec<_>>()
        ));
    }
    Ok(field)
}
