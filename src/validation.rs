//! Incoming-request validation — mirrors Python Pydantic field/model validators
//! in `document.py` and `vector.py`.
//!
//! Call `.validate()` explicitly at the handler boundary only.
//! Internal types are never passed through these functions.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use crate::common::{
    DenseDistance, DocumentField, VectorType, MAX_BULK_UPLOAD, MAX_COLLECTION_NAME_LENGTH,
    MAX_DOCUMENT_CONTENT_LENGTH, MAX_DOCUMENT_DESCRIPTION_LENGTH, MAX_DOCUMENT_ID_LENGTH,
    MAX_DOCUMENT_NAME_LENGTH, MAX_DOCUMENT_TAG_LENGTH, MAX_DOCUMENT_TAGS_COUNT,
    MAX_METADATA_KEY_LENGTH, MAX_METADATA_VALUE_LENGTH,
    MAX_MODEL_NAME_LENGTH,
    MAX_SEARCH_LIMIT, MAX_SEARCH_QUERY_LENGTH, MAX_TOP_K_VALUE, MAX_VECTOR_DIMENSIONS,
    MAX_VECTOR_NAME_LENGTH, MAX_FACET_PREFETCH_MULTIPLIER, MAX_FACET_MAX_VALUES,
};
use crate::models::{
    BulkUploadRequest, CollectionConfig, CollectionConfigInternal, CustomDocumentVector, Document,
    MetadataIndex, SearchQuery, VectorConfig, VectorConfigInternal, VectorData,
};

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
    if name.chars().count() > MAX_COLLECTION_NAME_LENGTH {
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

/// Coercions matching Pydantic field validators that **mutate** the model (`return v.strip()`, etc.).
/// Must run **before** [`validate_document`].
pub fn normalize_document_python(doc: &mut Document) {
    doc.id = doc.id.trim().to_string();
    if let Some(tags) = &mut doc.tags {
        let trimmed: Vec<String> = tags
            .iter()
            .filter_map(|t| {
                let s = t.trim();
                (!s.is_empty()).then(|| s.to_string())
            })
            .collect();
        *tags = trimmed;
    }
    if let Some(ref mut cv) = doc.custom_vectors {
        for c in cv.iter_mut() {
            c.vector_name = c.vector_name.trim().to_string();
        }
    }
}

pub fn validate_document(doc: &Document) -> VResult {
    validate_document_id(&doc.id)?;
    if let Some(tags) = &doc.tags {
        validate_tags(tags)?;
    }
    validate_document_name_opt(doc.name.as_deref())?;
    validate_document_description_opt(doc.description.as_deref())?;
    validate_document_content_opt(doc.content.as_deref())?;
    if let Some(metadata) = &doc.metadata {
        validate_metadata(metadata)?;
    }
    if let Some(cv) = &doc.custom_vectors {
        for c in cv {
            validate_custom_document_vector(c)?;
        }
    }
    Ok(())
}

/// Validate precomputed document vectors when provided on upsert.
/// Mirrors `validate_document_vectors` in amgix-server `database/common.py`.
pub fn validate_document_vectors(
    collection_config: &CollectionConfigInternal,
    document: &Document,
) -> VResult {
    let Some(vectors) = document.vectors.as_ref() else {
        return Ok(());
    };
    if vectors.is_empty() {
        return Err(err(
            "vectors must be omitted or contain the complete non-custom vector set",
        ));
    }

    let mut expected: HashMap<(String, DocumentField), &VectorConfigInternal> = HashMap::new();
    for config in &collection_config.vectors {
        if config.vector_type.is_custom_vectors() {
            continue;
        }
        for field in &config.index_fields {
            expected.insert((config.name.clone(), *field), config);
        }
    }

    let mut provided: HashMap<(String, DocumentField), &VectorData> = HashMap::new();
    for vd in vectors {
        if vd.vector_type.is_custom_vectors() {
            return Err(err(format!(
                "Vector '{}' field '{}' has type '{}'; custom vector types must use custom_vectors",
                vd.vector_name, vd.field, vd.vector_type
            )));
        }
        let key = (vd.vector_name.clone(), vd.field);
        if provided.contains_key(&key) {
            return Err(err(format!(
                "Duplicate vector entry for '{}' field '{}'",
                vd.vector_name, vd.field
            )));
        }
        let Some(config) = expected.get(&key) else {
            return Err(err(format!(
                "Unexpected vector '{}' field '{}' (not a non-custom collection vector slot)",
                vd.vector_name, vd.field
            )));
        };
        if vd.vector_type != config.vector_type {
            return Err(err(format!(
                "Vector '{}' field '{}' has type '{}', expected '{}'",
                vd.vector_name, vd.field, vd.vector_type, config.vector_type
            )));
        }
        validate_provided_vector_shape(vd, config)?;
        provided.insert(key, vd);
    }

    let missing: Vec<_> = expected
        .keys()
        .filter(|k| !provided.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        let mut labels: Vec<String> = missing
            .iter()
            .map(|(name, field)| format!("{name}/{field}"))
            .collect();
        labels.sort();
        return Err(err(format!(
            "Incomplete vectors: missing non-custom slots: {}",
            labels.join(", ")
        )));
    }

    if let Some(custom_vectors) = &document.custom_vectors {
        let custom_keys: HashSet<(String, DocumentField)> = custom_vectors
            .iter()
            .map(|cv| (cv.vector_name.clone(), cv.field))
            .collect();
        let overlap: Vec<_> = provided
            .keys()
            .filter(|k| custom_keys.contains(*k))
            .collect();
        if !overlap.is_empty() {
            let mut labels: Vec<String> = overlap
                .iter()
                .map(|(name, field)| format!("{name}/{field}"))
                .collect();
            labels.sort();
            return Err(err(format!(
                "Duplicate vector slots in vectors and custom_vectors: {}",
                labels.join(", ")
            )));
        }
    }

    Ok(())
}

fn validate_provided_vector_shape(vd: &VectorData, config: &VectorConfigInternal) -> VResult {
    if config.vector_type.is_dense() {
        let Some(dense) = vd.dense_vector.as_ref() else {
            return Err(err(format!(
                "Vector '{}' field '{}' requires dense_vector",
                vd.vector_name, vd.field
            )));
        };
        if dense.is_empty() {
            return Err(err(format!(
                "Vector '{}' field '{}' requires dense_vector",
                vd.vector_name, vd.field
            )));
        }
        if let Some(dim) = config.dimensions {
            if dense.len() != dim as usize {
                return Err(err(format!(
                    "Vector '{}' field '{}' has {} dimensions, expected {}",
                    vd.vector_name,
                    vd.field,
                    dense.len(),
                    dim
                )));
            }
        }
        return Ok(());
    }

    let (Some(indices), Some(values)) = (&vd.sparse_indices, &vd.sparse_values) else {
        return Err(err(format!(
            "Vector '{}' field '{}' requires sparse_indices and sparse_values",
            vd.vector_name, vd.field
        )));
    };
    if indices.len() != values.len() {
        return Err(err(format!(
            "Vector '{}' field '{}': sparse_indices and sparse_values length mismatch",
            vd.vector_name, vd.field
        )));
    }
    if indices.len() > config.top_k as usize {
        return Err(err(format!(
            "Vector '{}' field '{}' has {} entries, max allowed: {}",
            vd.vector_name,
            vd.field,
            indices.len(),
            config.top_k
        )));
    }
    Ok(())
}

/// @field_validator('id') — validate_id_format
fn validate_document_id(id: &str) -> VResult {
    let stripped = id.trim();
    if stripped.is_empty() {
        return Err(err("id: Document ID cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(stripped) {
        return Err(err(
            "id: Document ID can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if stripped.chars().count() > MAX_DOCUMENT_ID_LENGTH {
        return Err(err(format!(
            "id: ensure this value has at most {MAX_DOCUMENT_ID_LENGTH} characters"
        )));
    }
    Ok(())
}

/// @field_validator('tags') — validate_tags
fn validate_tags(tags: &[String]) -> VResult {
    if tags.len() > MAX_DOCUMENT_TAGS_COUNT {
        return Err(err(format!(
            "tags: ensure this value has at most {MAX_DOCUMENT_TAGS_COUNT} items"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for tag in tags {
        let t = tag.trim();
        if t.is_empty() {
            continue;
        }
        if t.contains('|') {
            return Err(err(format!("tags: Tag '{t}' cannot contain pipe characters (|)")));
        }
        if t.chars().count() > MAX_DOCUMENT_TAG_LENGTH {
            return Err(err(format!(
                "tags: Tag '{t}' exceeds {MAX_DOCUMENT_TAG_LENGTH} character limit"
            )));
        }
        if !seen.insert(t.to_string()) {
            return Err(err("tags: Tags must not contain duplicates"));
        }
    }
    Ok(())
}

fn validate_document_name_opt(name: Option<&str>) -> VResult {
    if let Some(n) = name {
        if n.chars().count() > MAX_DOCUMENT_NAME_LENGTH {
            return Err(err(format!(
                "name: ensure this value has at most {MAX_DOCUMENT_NAME_LENGTH} characters"
            )));
        }
    }
    Ok(())
}

fn validate_document_description_opt(desc: Option<&str>) -> VResult {
    if let Some(d) = desc {
        if d.chars().count() > MAX_DOCUMENT_DESCRIPTION_LENGTH {
            return Err(err(format!(
                "description: ensure this value has at most {MAX_DOCUMENT_DESCRIPTION_LENGTH} characters"
            )));
        }
    }
    Ok(())
}

fn validate_document_content_opt(content: Option<&str>) -> VResult {
    if let Some(c) = content {
        if c.chars().count() > MAX_DOCUMENT_CONTENT_LENGTH {
            return Err(err(format!(
                "content: ensure this value has at most {MAX_DOCUMENT_CONTENT_LENGTH} characters"
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
                "metadata.{key}: key can only contain letters, numbers, underscores, and hyphens"
            )));
        }
        if key.chars().count() > MAX_METADATA_KEY_LENGTH {
            return Err(err(format!(
                "metadata.{key}: key exceeds {MAX_METADATA_KEY_LENGTH} character limit"
            )));
        }

        // MetaValue dict form {"value": ..., "type": "..."}, plain object dict, or primitive.
        match value {
            Value::Object(map) => {
                if map.len() == 2 && map.contains_key("value") && map.contains_key("type") {
                    let val = map.get("value").ok_or_else(|| {
                        err(format!(
                            "metadata.{key}: value is a dict but missing 'value' or 'type' fields. \
                            For datetime, use {{\"value\": \"...\", \"type\": \"datetime\"}}"
                        ))
                    })?;
                    let type_str = map
                        .get("type")
                        .and_then(|t| t.as_str())
                        .ok_or_else(|| {
                            err(format!(
                                "metadata.{key}: value is a dict but missing 'value' or 'type' fields. \
                                For datetime, use {{\"value\": \"...\", \"type\": \"datetime\"}}"
                            ))
                        })?;
                    validate_meta_value_type(key, type_str, val)?;
                }
            }
            Value::String(s) => {
                // raw string → type = "string"
                if s.chars().count() > MAX_METADATA_VALUE_LENGTH {
                    return Err(err(format!(
                        "metadata.{key}: string value exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
                    )));
                }
            }
            Value::Bool(_) => {} // raw bool → type = "boolean"
            Value::Number(n) => {
                // raw int or float — accepted as-is (type inferred)
                let _ = n;
            }
            Value::Array(_) => {} // raw array → type = "array"
            Value::Null => {} // raw null → type = "object" with null value
            other => {
                let type_name = json_type_name(other);
                return Err(err(format!(
                    "metadata.{key}: value must be string, int, float, bool, array, null, or MetaValue \
                    (required for datetime and object), got {type_name}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_meta_value_type(key: &str, type_str: &str, val: &Value) -> VResult {
    let allowed = ["string", "integer", "float", "boolean", "datetime", "array", "object"];
    if !allowed.contains(&type_str) {
        return Err(err(format!(
            "metadata.{key}: invalid type '{type_str}'. Allowed types: {allowed:?}"
        )));
    }
    match type_str {
        "string" => {
            match val.as_str() {
                Some(s) if s.chars().count() > MAX_METADATA_VALUE_LENGTH => Err(err(format!(
                    "metadata.{key}: string value exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
                ))),
                Some(_) => Ok(()),
                None => Err(err(format!(
                    "metadata.{key}: value must be string for type='string', got {}",
                    json_type_name(val)
                ))),
            }
        }
        "integer" => {
            if val.is_i64() || val.is_u64() {
                Ok(())
            } else {
                Err(err(format!(
                    "metadata.{key}: value must be integer for type='integer', got {}",
                    json_type_name(val)
                )))
            }
        }
        "float" => {
            if val.is_number() {
                Ok(())
            } else {
                Err(err(format!(
                    "metadata.{key}: value must be number for type='float', got {}",
                    json_type_name(val)
                )))
            }
        }
        "boolean" => {
            if val.is_boolean() {
                Ok(())
            } else {
                Err(err(format!(
                    "metadata.{key}: value must be boolean for type='boolean', got {}",
                    json_type_name(val)
                )))
            }
        }
        "datetime" => match val.as_str() {
            Some(s) => {
                if crate::datetime_parse::is_valid_datetime_string(s) {
                    Ok(())
                } else {
                    Err(err(format!(
                        "metadata.{key}: value must be a valid ISO 8601 datetime string, got '{s}'"
                    )))
                }
            }
            None => Err(err(format!(
                "metadata.{key}: value must be string (ISO 8601) for type='datetime', got {}",
                json_type_name(val)
            ))),
        },
        "array" => {
            if val.is_array() {
                Ok(())
            } else {
                Err(err(format!(
                    "metadata.{key}: value must be array for type='array', got {}",
                    json_type_name(val)
                )))
            }
        }
        "object" => {
            if val.is_object() || val.is_null() {
                Ok(())
            } else {
                Err(err(format!(
                    "metadata.{key}: value must be object or null for type='object', got {}",
                    json_type_name(val)
                )))
            }
        }
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

// ---------------------------------------------------------------------------
// BulkUploadRequest
// Mirrors: MAX_BULK_UPLOAD constant check (main.py) + each Document validated
// ---------------------------------------------------------------------------

pub fn validate_bulk_upload(req: &mut BulkUploadRequest) -> VResult {
    if req.documents.len() > MAX_BULK_UPLOAD {
        return Err(err(format!(
            "documents: ensure this value has at most {MAX_BULK_UPLOAD} items"
        )));
    }
    for (i, doc) in req.documents.iter_mut().enumerate() {
        normalize_document_python(doc);
        validate_document(doc).map_err(|e| {
            ValidationError(format!("documents[{}]: {}", i, e.0))
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CollectionConfig / VectorConfig
// Mirrors: vector.py CollectionConfig + VectorConfig field/model validators
// ---------------------------------------------------------------------------

pub fn validate_collection_config(config: &CollectionConfig) -> VResult {
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
    if let Some(ref indexes) = config.metadata_indexes {
        for mi in indexes {
            validate_metadata_index(mi)?;
        }
    }
    Ok(())
}

fn validate_vector_config(v: &VectorConfig) -> VResult {
    validate_vector_name(&v.name)?;
    validate_top_k(v.top_k)?;
    validate_wmtr_word_ratio(v.wmtr_word_ratio)?;
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
    if s.chars().count() > MAX_MODEL_NAME_LENGTH {
        return Err(err(format!(
            "{field}: ensure this value has at most {MAX_MODEL_NAME_LENGTH} characters"
        )));
    }
    Ok(())
}

/// @field_validator('name') on VectorConfig — validate_name_format
/// Matches `vector.py`: pattern and max length apply to **raw** `name` (`v`), not trimmed.
fn validate_vector_name(name: &str) -> VResult {
    if name.trim().is_empty() {
        return Err(err("name: Vector configuration name cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(name) {
        return Err(err(
            "name: Vector name can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if name.chars().count() > MAX_VECTOR_NAME_LENGTH {
        return Err(err(format!(
            "name: Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters"
        )));
    }
    Ok(())
}

/// Mirrors `vector.py` `MetadataIndex` field validators.
fn validate_metadata_index(mi: &MetadataIndex) -> VResult {
    if !RE_ALPHANUMERIC.is_match(&mi.key) {
        return Err(err(format!(
            "metadata_indexes.{}: key can only contain letters, numbers, underscores, and hyphens",
            mi.key
        )));
    }
    if mi.key.chars().count() > MAX_METADATA_KEY_LENGTH {
        return Err(err(format!(
            "metadata_indexes.{}: key cannot exceed {MAX_METADATA_KEY_LENGTH} characters",
            mi.key
        )));
    }
    let allowed = ["string", "integer", "float", "boolean", "datetime"];
    if !allowed.contains(&mi.value_type.as_str()) {
        return Err(err(format!(
            "metadata_indexes.{}: invalid type '{}'. Allowed types: {allowed:?}",
            mi.key, mi.value_type
        )));
    }
    Ok(())
}

/// Mirrors `CustomVector.vector` in `vector.py`.
fn validate_custom_vector_data_value(v: &Value) -> VResult {
    let arr = v
        .as_array()
        .ok_or_else(|| err("Vector data cannot be empty"))?;
    if arr.is_empty() {
        return Err(err("Vector data cannot be empty"));
    }
    let first = &arr[0];
    if first.is_number() {
        for x in arr {
            if !x.is_number() {
                return Err(err("Dense vector must contain only numbers"));
            }
        }
        return Ok(());
    }
    if first.is_array() {
        for item in arr {
            let tup = item.as_array().ok_or_else(|| {
                err("Sparse vector must contain (index, value) tuples")
            })?;
            if tup.len() != 2 {
                return Err(err(
                    "Sparse vector must contain (index, value) tuples",
                ));
            }
            let idx_ok = tup[0].as_i64().is_some() || tup[0].as_u64().is_some();
            if !idx_ok || !tup[1].is_number() {
                return Err(err(
                    "Sparse vector tuples must be (int, float) pairs",
                ));
            }
        }
        return Ok(());
    }
    Err(err(
        "Vector must be either list of numbers (dense) or list of (index, value) tuples (sparse)",
    ))
}

fn validate_custom_query_vector_name_stripped(raw: &str) -> VResult {
    let t = raw.trim();
    if t.is_empty() {
        return Err(err("vector_name: Vector name cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(t) {
        return Err(err(
            "vector_name: Vector name can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if t.chars().count() > MAX_VECTOR_NAME_LENGTH {
        return Err(err(format!(
            "vector_name: Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters"
        )));
    }
    Ok(())
}

fn validate_custom_document_vector(c: &CustomDocumentVector) -> VResult {
    validate_custom_query_vector_name_stripped(&c.vector_name)?;
    validate_custom_vector_data_value(&c.vector)?;
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

/// wmtr_word_ratio ge=0, le=100
fn validate_wmtr_word_ratio(w: u32) -> VResult {
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

/// Mirrors Pydantic `VectorSearchOption` / `CustomVector.vector_name`: stored names are trimmed
/// **after** validation (see validators returning `strip()`).
pub fn normalize_search_query_python(q: &mut SearchQuery) {
    for w in &mut q.settings.vector_options {
        w.vector_name = w.vector_name.trim().to_string();
    }
    if let Some(ref mut cv) = q.settings.custom_vectors {
        for c in cv.iter_mut() {
            c.vector_name = c.vector_name.trim().to_string();
        }
    }
    // @field_validator('exclude') — dedupe_exclude (preserve first-occurrence order)
    if let Some(ref mut exclude) = q.settings.exclude {
        let mut seen = std::collections::HashSet::new();
        exclude.retain(|f| seen.insert(*f));
    }
}

pub fn validate_search_query(q: &SearchQuery) -> VResult {
    // @field_validator('query') — validate_query_not_empty
    if q.query.trim().is_empty() {
        return Err(err("query: Query string cannot be empty or whitespace"));
    }
    if q.query.chars().count() > MAX_SEARCH_QUERY_LENGTH {
        return Err(err(format!(
            "query: ensure this value has at most {MAX_SEARCH_QUERY_LENGTH} characters"
        )));
    }
    // limit ge=1, le=MAX_SEARCH_LIMIT
    if q.settings.limit < 1 {
        return Err(err("limit: ensure this value is greater than or equal to 1"));
    }
    if q.settings.limit > MAX_SEARCH_LIMIT {
        return Err(err(format!(
            "limit: ensure this value is less than or equal to {MAX_SEARCH_LIMIT}"
        )));
    }
    // @field_validator('score_threshold') — validate_score_threshold_number
    // (always a number in Rust's type system; nothing to check)

    // @field_validator('document_tags') — validate_document_tag_lengths
    if let Some(tags) = &q.settings.document_tags {
        if tags.len() > MAX_DOCUMENT_TAGS_COUNT {
            return Err(err(format!(
                "document_tags: ensure this value has at most {MAX_DOCUMENT_TAGS_COUNT} items"
            )));
        }
        for tag in tags {
            let t = tag.trim();
            if t.is_empty() {
                return Err(err("document_tags: Document tags cannot be empty or whitespace"));
            }
            if t.contains('|') {
                return Err(err(format!(
                    "document_tags: Document tag '{t}' cannot contain pipe characters (|)"
                )));
            }
            if t.chars().count() > MAX_DOCUMENT_TAG_LENGTH {
                return Err(err(format!(
                    "document_tags: Document tag '{t}' exceeds {MAX_DOCUMENT_TAG_LENGTH} character limit"
                )));
            }
        }
    }
    // vector_options: each vector_name validated
    for w in &q.settings.vector_options {
        validate_search_vector_name(&w.vector_name)?;
    }
    if let Some(ref cv) = q.settings.custom_vectors {
        for c in cv {
            validate_custom_query_vector_name_stripped(&c.vector_name)?;
            validate_custom_vector_data_value(&c.vector)?;
        }
    }
    validate_fusion_mode(&q.settings.fusion_mode)?;
    // group_max ge=1, group_max_fetches ge=1
    if q.settings.group_max < 1 {
        return Err(err("group_max: ensure this value is greater than or equal to 1"));
    }
    if q.settings.group_max_fetches < 1 {
        return Err(err("group_max_fetches: ensure this value is greater than or equal to 1"));
    }
    if let Some(ref opts) = q.settings.facet_options {
        if opts.prefetch_multiplier < 1 || opts.prefetch_multiplier > MAX_FACET_PREFETCH_MULTIPLIER {
            return Err(err(format!(
                "facet_options.prefetch_multiplier: ensure this value is between 1 and {MAX_FACET_PREFETCH_MULTIPLIER}"
            )));
        }
        if opts.max_values < 1 || opts.max_values > MAX_FACET_MAX_VALUES {
            return Err(err(format!(
                "facet_options.max_values: ensure this value is between 1 and {MAX_FACET_MAX_VALUES}"
            )));
        }
    }
    Ok(())
}

/// @field_validator('vector_name') on VectorSearchOption — pattern on raw `v`, max len on raw,
/// modeled value trimmed (Rust: [`normalize_search_query_python`] afterward).
fn validate_search_vector_name(name: &str) -> VResult {
    if name.trim().is_empty() {
        return Err(err("vector_options.vector_name: Vector name cannot be empty or whitespace"));
    }
    if !RE_ALPHANUMERIC.is_match(name) {
        return Err(err(
            "vector_options.vector_name: Vector name can only contain letters, numbers, underscores, and hyphens",
        ));
    }
    if name.chars().count() > MAX_VECTOR_NAME_LENGTH {
        return Err(err(format!(
            "vector_options.vector_name: Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters"
        )));
    }
    Ok(())
}

fn validate_fusion_mode(s: &str) -> VResult {
    match s {
        "rrf" | "linear" => Ok(()),
        _ => Err(err(format!(
            "fusion_mode must be one of ['rrf', 'linear'], got {s:?}"
        ))),
    }
}
