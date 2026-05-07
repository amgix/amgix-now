//! Incoming-request validation — mirrors Python Pydantic field/model validators
//! in `document.py` and `vector.py`.
//!
//! Call `.validate()` explicitly at the handler boundary only.
//! Internal types are never passed through these functions.

use chrono::DateTime;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::common::{
    DenseDistance, VectorType, MAX_BULK_UPLOAD, MAX_COLLECTION_NAME_LENGTH,
    MAX_DOCUMENT_CONTENT_LENGTH, MAX_DOCUMENT_DESCRIPTION_LENGTH, MAX_DOCUMENT_ID_LENGTH,
    MAX_DOCUMENT_NAME_LENGTH, MAX_DOCUMENT_TAG_LENGTH, MAX_DOCUMENT_TAGS_COUNT,
    MAX_METADATA_KEY_LENGTH, MAX_METADATA_VALUE_LENGTH, MAX_MODEL_NAME_LENGTH,
    MAX_SEARCH_LIMIT, MAX_SEARCH_QUERY_LENGTH, MAX_TOP_K_VALUE, MAX_VECTOR_DIMENSIONS,
    MAX_VECTOR_NAME_LENGTH,
};
use crate::models::{BulkUploadRequest, CollectionConfig, Document, SearchQuery, VectorConfig};

// ---------------------------------------------------------------------------
// Shared regex patterns
// ---------------------------------------------------------------------------

static RE_ALPHANUMERIC: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap());

static RE_LANGUAGE_CODE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z]{2}$").unwrap());

// ---------------------------------------------------------------------------
// Validation error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn err(msg: impl Into<String>) -> ValidationError {
    ValidationError(msg.into())
}

type VResult = Result<(), ValidationError>;

// ---------------------------------------------------------------------------
// collection_name path parameter
// Mirrors: CollectionName = Annotated[str, Path(..., regex=r"^[a-zA-Z0-9_-]+$",
//                                     min_length=1, max_length=MAX_COLLECTION_NAME_LENGTH)]
// ---------------------------------------------------------------------------

pub fn validate_collection_name(name: &str) -> VResult {
    if name.is_empty() {
        return Err(err("ensure this value has at least 1 characters"));
    }
    if name.len() > MAX_COLLECTION_NAME_LENGTH {
        return Err(err(format!(
            "ensure this value has at most {MAX_COLLECTION_NAME_LENGTH} characters"
        )));
    }
    if !RE_ALPHANUMERIC.is_match(name) {
        return Err(err("string does not match regex \"^[a-zA-Z0-9_-]+$\""));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Document
// Mirrors: document.py Document field/model validators
// ---------------------------------------------------------------------------

pub fn validate_document(doc: &Document) -> VResult {
    validate_document_id(&doc.id)?;
    validate_document_timestamp(&doc.timestamp)?;
    if let Some(tags) = &doc.tags {
        validate_tags(tags)?;
    }
    validate_document_name_opt(doc.name.as_deref())?;
    validate_document_description_opt(doc.description.as_deref())?;
    validate_document_content_opt(doc.content.as_deref())?;
    if let Some(metadata) = &doc.metadata {
        validate_metadata(metadata)?;
    }
    validate_at_least_one_field(doc)?;
    Ok(())
}

/// @field_validator('id') — validate_id_format
fn validate_document_id(id: &str) -> VResult {
    let stripped = id.trim();
    if stripped.is_empty() {
        return Err(err("Document ID cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(stripped) {
        return Err(err(
            "Document ID can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if stripped.len() > MAX_DOCUMENT_ID_LENGTH {
        return Err(err(format!(
            "ensure this value has at most {MAX_DOCUMENT_ID_LENGTH} characters"
        )));
    }
    Ok(())
}

/// @field_validator('timestamp') — validate_timestamp_utc
fn validate_document_timestamp(ts: &chrono::DateTime<chrono::Utc>) -> VResult {
    // chrono::DateTime<Utc> is always UTC by type; serde will reject non-datetime JSON.
    // Python also validates tzinfo is present and is UTC — guaranteed by the Rust type.
    let _ = ts;
    Ok(())
}

/// @field_validator('tags') — validate_tags
fn validate_tags(tags: &[String]) -> VResult {
    if tags.len() > MAX_DOCUMENT_TAGS_COUNT {
        return Err(err(format!(
            "ensure this value has at most {MAX_DOCUMENT_TAGS_COUNT} items"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for tag in tags {
        let t = tag.trim();
        if t.is_empty() {
            continue;
        }
        if t.contains('|') {
            return Err(err(format!("Tag '{t}' cannot contain pipe characters (|)")));
        }
        if t.len() > MAX_DOCUMENT_TAG_LENGTH {
            return Err(err(format!(
                "Tag '{t}' exceeds {MAX_DOCUMENT_TAG_LENGTH} character limit"
            )));
        }
        if !seen.insert(t.to_string()) {
            return Err(err("Tags must not contain duplicates"));
        }
    }
    Ok(())
}

fn validate_document_name_opt(name: Option<&str>) -> VResult {
    if let Some(n) = name {
        if n.len() > MAX_DOCUMENT_NAME_LENGTH {
            return Err(err(format!(
                "ensure this value has at most {MAX_DOCUMENT_NAME_LENGTH} characters"
            )));
        }
    }
    Ok(())
}

fn validate_document_description_opt(desc: Option<&str>) -> VResult {
    if let Some(d) = desc {
        if d.len() > MAX_DOCUMENT_DESCRIPTION_LENGTH {
            return Err(err(format!(
                "ensure this value has at most {MAX_DOCUMENT_DESCRIPTION_LENGTH} characters"
            )));
        }
    }
    Ok(())
}

fn validate_document_content_opt(content: Option<&str>) -> VResult {
    if let Some(c) = content {
        if c.len() > MAX_DOCUMENT_CONTENT_LENGTH {
            return Err(err(format!(
                "ensure this value has at most {MAX_DOCUMENT_CONTENT_LENGTH} characters"
            )));
        }
    }
    Ok(())
}

/// @field_validator('metadata', mode='before') — validate_metadata
fn validate_metadata(metadata: &std::collections::HashMap<String, Value>) -> VResult {
    for (key, value) in metadata {
        // key format
        if !RE_ALPHANUMERIC.is_match(key) {
            return Err(err(format!(
                "Metadata key '{key}' can only contain letters, numbers, underscores, and hyphens"
            )));
        }
        if key.len() > MAX_METADATA_KEY_LENGTH {
            return Err(err(format!(
                "Metadata key '{key}' exceeds {MAX_METADATA_KEY_LENGTH} character limit"
            )));
        }

        // value must be a MetaValue dict: {"value": ..., "type": "..."}
        // or a primitive (string / number / bool).
        // Python converts primitives → MetaValue; we validate the raw JSON shape here.
        match value {
            Value::Object(map) => {
                // MetaValue dict form
                let val = map.get("value").ok_or_else(|| {
                    err(format!(
                        "Metadata value for key '{key}' is a dict but missing 'value' or 'type' fields. \
                        For datetime, use {{\"value\": \"...\", \"type\": \"datetime\"}}"
                    ))
                })?;
                let type_str = map
                    .get("type")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| {
                        err(format!(
                            "Metadata value for key '{key}' is a dict but missing 'value' or 'type' fields. \
                            For datetime, use {{\"value\": \"...\", \"type\": \"datetime\"}}"
                        ))
                    })?;
                validate_meta_value_type(key, type_str, val)?;
            }
            Value::String(s) => {
                // raw string → type = "string"
                if s.len() > MAX_METADATA_VALUE_LENGTH {
                    return Err(err(format!(
                        "String metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
                    )));
                }
            }
            Value::Bool(_) => {} // raw bool → type = "boolean"
            Value::Number(n) => {
                // raw int or float — accepted as-is (type inferred)
                let _ = n;
            }
            other => {
                let type_name = json_type_name(other);
                return Err(err(format!(
                    "Metadata value for key '{key}' must be string, int, float, bool, or MetaValue \
                    (required for datetime), got {type_name}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_meta_value_type(key: &str, type_str: &str, val: &Value) -> VResult {
    let allowed = ["string", "integer", "float", "boolean", "datetime"];
    if !allowed.contains(&type_str) {
        return Err(err(format!(
            "Invalid metadata type '{type_str}' for key '{key}'. Allowed types: {allowed:?}"
        )));
    }
    match type_str {
        "string" => {
            match val.as_str() {
                Some(s) if s.len() > MAX_METADATA_VALUE_LENGTH => Err(err(format!(
                    "String metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
                ))),
                Some(_) => Ok(()),
                None => Err(err(format!(
                    "Metadata value for key '{key}' must be string for type='string', got {}",
                    json_type_name(val)
                ))),
            }
        }
        "integer" => {
            if val.is_i64() || val.is_u64() {
                Ok(())
            } else {
                Err(err(format!(
                    "Metadata value for key '{key}' must be integer for type='integer', got {}",
                    json_type_name(val)
                )))
            }
        }
        "float" => {
            if val.is_number() {
                Ok(())
            } else {
                Err(err(format!(
                    "Metadata value for key '{key}' must be number for type='float', got {}",
                    json_type_name(val)
                )))
            }
        }
        "boolean" => {
            if val.is_boolean() {
                Ok(())
            } else {
                Err(err(format!(
                    "Metadata value for key '{key}' must be boolean for type='boolean', got {}",
                    json_type_name(val)
                )))
            }
        }
        "datetime" => match val.as_str() {
            Some(s) => {
                let normalized = s.replace('Z', "+00:00");
                if DateTime::parse_from_rfc3339(&normalized).is_ok() {
                    Ok(())
                } else {
                    Err(err(format!(
                        "Metadata value for key '{key}' must be a valid ISO 8601 datetime string, got '{s}'"
                    )))
                }
            }
            None => Err(err(format!(
                "Metadata value for key '{key}' must be string (ISO 8601) for type='datetime', got {}",
                json_type_name(val)
            ))),
        },
        _ => unreachable!(),
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() { "int" } else { "float" }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// @model_validator — validate_at_least_one_field_has_content
fn validate_at_least_one_field(doc: &Document) -> VResult {
    let has_name = doc.name.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    let has_desc = doc.description.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    let has_content = doc.content.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    if !has_name && !has_desc && !has_content {
        return Err(err(
            "Document must have at least one non-empty field (name, description, or content)",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// BulkUploadRequest
// Mirrors: MAX_BULK_UPLOAD constant check (main.py) + each Document validated
// ---------------------------------------------------------------------------

pub fn validate_bulk_upload(req: &BulkUploadRequest) -> VResult {
    if req.documents.len() > MAX_BULK_UPLOAD {
        return Err(err(format!(
            "ensure this value has at most {MAX_BULK_UPLOAD} items"
        )));
    }
    for doc in &req.documents {
        validate_document(doc)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CollectionConfig / VectorConfig
// Mirrors: vector.py CollectionConfig + VectorConfig field/model validators
// ---------------------------------------------------------------------------

pub fn validate_collection_config(config: &CollectionConfig) -> VResult {
    // @field_validator('vectors') — validate_vectors_not_empty
    if config.vectors.is_empty() {
        return Err(err("Collection must have at least one vector configuration"));
    }
    // @field_validator('vectors') — validate_unique_vector_names
    let mut seen = std::collections::HashSet::new();
    let mut duplicates: Vec<&str> = Vec::new();
    for v in &config.vectors {
        if !seen.insert(v.name.as_str()) {
            duplicates.push(v.name.as_str());
        }
    }
    if !duplicates.is_empty() {
        return Err(err(format!(
            "Duplicate vector names found: {}. Each vector name must be unique within a collection.",
            duplicates.join(", ")
        )));
    }
    for v in &config.vectors {
        validate_vector_config(v)?;
    }
    Ok(())
}

fn validate_vector_config(v: &VectorConfig) -> VResult {
    validate_vector_name(&v.name)?;
    validate_top_k(v.top_k)?;
    validate_wmtr_word_weight(v.wmtr_word_weight)?;
    validate_language_confidence(v.language_confidence)?;
    validate_language_code(Some(v.language_default_code.as_str()))?;
    validate_dense_distance(&v.dense_distance, &v.vector_type)?;
    validate_normalization(v.normalization, &v.vector_type)?;
    validate_model_requirements(v)?;
    validate_custom_vector_config(v)?;
    validate_language_config(v)?;
    if let Some(ref model) = v.model {
        validate_model_name_length("model", model)?;
    }
    if let Some(ref r) = v.revision {
        validate_model_name_length("revision", r)?;
    }
    if let Some(ref qm) = v.query_model {
        validate_model_name_length("query_model", qm)?;
    }
    if let Some(ref qr) = v.query_revision {
        validate_model_name_length("query_revision", qr)?;
    }
    Ok(())
}

fn validate_model_name_length(field: &str, s: &str) -> VResult {
    if s.len() > MAX_MODEL_NAME_LENGTH {
        return Err(err(format!(
            "ensure this value has at most {MAX_MODEL_NAME_LENGTH} characters (field: {field})"
        )));
    }
    Ok(())
}

/// @field_validator('name') on VectorConfig — validate_name_format
fn validate_vector_name(name: &str) -> VResult {
    let stripped = name.trim();
    if stripped.is_empty() {
        return Err(err("Vector configuration name cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(stripped) {
        return Err(err(
            "Vector name can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if stripped.len() > MAX_VECTOR_NAME_LENGTH {
        return Err(err(format!(
            "Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters"
        )));
    }
    Ok(())
}

/// @field_validator('top_k') — validate_top_k_positive
fn validate_top_k(top_k: u32) -> VResult {
    if top_k == 0 {
        return Err(err("top_k must be positive (greater than 0)"));
    }
    if top_k > MAX_TOP_K_VALUE {
        return Err(err(format!("top_k cannot exceed {MAX_TOP_K_VALUE}")));
    }
    Ok(())
}

/// wmtr_word_weight ge=0, le=100
fn validate_wmtr_word_weight(w: u32) -> VResult {
    if w > 100 {
        return Err(err("ensure this value is less than or equal to 100"));
    }
    Ok(())
}

/// @field_validator('language_confidence')
fn validate_language_confidence(c: f64) -> VResult {
    if !(0.0..=1.0).contains(&c) {
        return Err(err("language_confidence must be between 0.0 and 1.0"));
    }
    Ok(())
}

/// @field_validator('language_default_code')
fn validate_language_code(code: Option<&str>) -> VResult {
    if let Some(c) = code {
        if !RE_LANGUAGE_CODE.is_match(c) {
            return Err(err(
                "Language code must be a valid ISO 639-1 code (2 letters)",
            ));
        }
    }
    Ok(())
}

/// @field_validator('dense_distance') + @model_validator dense_distance_for_dense_vectors
fn validate_dense_distance(dist: &DenseDistance, vtype: &VectorType) -> VResult {
    let allowed = ["cosine", "dot", "euclid"];
    let dist_str = dist.to_string();
    if !allowed.contains(&dist_str.as_str()) {
        return Err(err(format!(
            "dense_distance must be one of {allowed:?}"
        )));
    }
    if *dist != DenseDistance::Cosine && !vtype.is_dense() {
        return Err(err(format!(
            "dense_distance can only be specified for dense vectors. Current type: {vtype}"
        )));
    }
    Ok(())
}

/// @model_validator — validate_normalization_for_sparse_vectors
fn validate_normalization(norm: Option<bool>, vtype: &VectorType) -> VResult {
    if norm == Some(true) && !vtype.is_dense() {
        return Err(err(format!(
            "Normalization is not supported for sparse vector type '{vtype}'. \
            Only dense vectors support normalization."
        )));
    }
    Ok(())
}

/// @model_validator — validate_model_requirements
fn validate_model_requirements(v: &VectorConfig) -> VResult {
    if v.vector_type.is_transformer_based() && v.model.is_none() {
        return Err(err(format!(
            "Model is required for {} vector type",
            v.vector_type
        )));
    }
    if v.model.is_some() && v.vector_type.is_custom_tokenization() {
        return Err(err(format!(
            "Model should not be specified for {} vector type",
            v.vector_type
        )));
    }
    Ok(())
}

/// @model_validator — validate_custom_vector_config
fn validate_custom_vector_config(v: &VectorConfig) -> VResult {
    if v.vector_type.is_custom_vectors() {
        if v.model.is_some() {
            return Err(err(format!(
                "Model should not be specified for {} vector type",
                v.vector_type
            )));
        }
        if v.vector_type == VectorType::DenseCustom && v.dimensions.is_none() {
            return Err(err(format!(
                "Dimensions are required for {} vector type",
                v.vector_type
            )));
        }
    }
    if let Some(dims) = v.dimensions {
        if dims == 0 {
            return Err(err("Dimensions must be positive (greater than 0)"));
        }
        if dims > MAX_VECTOR_DIMENSIONS {
            return Err(err(format!("Dimensions cannot exceed {MAX_VECTOR_DIMENSIONS}")));
        }
    }
    Ok(())
}

/// @model_validator — validate_language_config
fn validate_language_config(v: &VectorConfig) -> VResult {
    let needs_lang = matches!(
        v.vector_type,
        VectorType::FullText | VectorType::Whitespace | VectorType::Wmtr
    );
    if needs_lang && v.language_default_code.is_empty() {
        return Err(err(format!(
            "language_default_code is required for {} vector type",
            v.vector_type
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SearchQuery
// Mirrors: vector.py SearchQuery field validators
// ---------------------------------------------------------------------------

pub fn validate_search_query(q: &SearchQuery) -> VResult {
    // @field_validator('query') — validate_query_not_empty
    if q.query.trim().is_empty() {
        return Err(err("Query string cannot be empty or whitespace"));
    }
    if q.query.len() > MAX_SEARCH_QUERY_LENGTH {
        return Err(err(format!(
            "ensure this value has at most {MAX_SEARCH_QUERY_LENGTH} characters"
        )));
    }
    // limit ge=1, le=MAX_SEARCH_LIMIT
    if q.limit < 1 {
        return Err(err("ensure this value is greater than or equal to 1"));
    }
    if q.limit > MAX_SEARCH_LIMIT {
        return Err(err(format!(
            "ensure this value is less than or equal to {MAX_SEARCH_LIMIT}"
        )));
    }
    // @field_validator('score_threshold') — validate_score_threshold_number
    // (always a number in Rust's type system; nothing to check)

    // @field_validator('document_tags') — validate_document_tag_lengths
    if let Some(tags) = &q.document_tags {
        if tags.len() > MAX_DOCUMENT_TAGS_COUNT {
            return Err(err(format!(
                "ensure this value has at most {MAX_DOCUMENT_TAGS_COUNT} items"
            )));
        }
        for tag in tags {
            let t = tag.trim();
            if t.is_empty() {
                return Err(err("Document tags cannot be empty or whitespace"));
            }
            if t.contains('|') {
                return Err(err(format!(
                    "Document tag '{t}' cannot contain pipe characters (|)"
                )));
            }
            if t.len() > MAX_DOCUMENT_TAG_LENGTH {
                return Err(err(format!(
                    "Document tag '{t}' exceeds {MAX_DOCUMENT_TAG_LENGTH} character limit"
                )));
            }
        }
    }
    // vector_weights: each vector_name validated
    for w in &q.vector_weights {
        validate_search_vector_name(&w.vector_name)?;
    }
    Ok(())
}

/// @field_validator('vector_name') on VectorSearchWeight
fn validate_search_vector_name(name: &str) -> VResult {
    let stripped = name.trim();
    if stripped.is_empty() {
        return Err(err("Vector name cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(stripped) {
        return Err(err(
            "Vector name can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if stripped.len() > MAX_VECTOR_NAME_LENGTH {
        return Err(err(format!(
            "Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters"
        )));
    }
    Ok(())
}
