//! API-facing request/response types and internal storage models.
//!
//! API types mirror the JSON shapes of `amgix-server` exactly.
//! Internal types (`*Internal`) mirror what `amgix-server` stores in Qdrant,
//! so data written by either service is readable by the other.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Unwrap legacy wrapped metadata entries `{"key": {"value": x, "type": y}}` to flat `{"key": x}`.
/// Only applies when the object contains exactly `value` and `type` keys.
/// Called at the API response boundary for reads of pre-migration storage data.
pub fn flatten_doc_metadata(mut val: Value) -> Value {
    if let Value::Object(ref mut map) = val {
        if let Some(Value::Object(meta)) = map.get_mut("metadata") {
            let flattened: serde_json::Map<String, Value> = meta
                .iter()
                .map(|(k, v)| {
                    let out = if let Value::Object(obj) = v {
                        if obj.len() == 2 && obj.contains_key("value") && obj.contains_key("type") {
                            obj.get("value").cloned().unwrap_or_else(|| v.clone())
                        } else {
                            v.clone()
                        }
                    } else {
                        v.clone()
                    };
                    (k.clone(), out)
                })
                .collect();
            *meta = flattened;
        }
    }
    val
}

/// Null out excluded fields on a search hit's JSON object, recursing into
/// `joined.<collection>[*]` so joined documents are nulled the same way.
/// Fields are set to null rather than removed so the response shape stays stable.
/// Mirrors document.py `apply_search_exclude`.
pub fn apply_search_exclude(val: Value, exclude: &[crate::common::SearchExcludeField]) -> Value {
    match val {
        Value::Object(mut map) => {
            for field in exclude {
                map.insert(field.as_str().to_string(), Value::Null);
            }
            if let Some(Value::Object(joined)) = map.get("joined") {
                let stripped_joined: serde_json::Map<String, Value> = joined
                    .iter()
                    .map(|(coll, docs)| {
                        let stripped_docs = match docs {
                            Value::Array(items) => Value::Array(
                                items
                                    .iter()
                                    .cloned()
                                    .map(|d| apply_search_exclude(d, exclude))
                                    .collect(),
                            ),
                            other => other.clone(),
                        };
                        (coll.clone(), stripped_docs)
                    })
                    .collect();
                map.insert("joined".to_string(), Value::Object(stripped_joined));
            }
            Value::Object(map)
        }
        other => other,
    }
}

fn strip_null_json_fields(val: Value) -> Value {
    match val {
        Value::Object(map) => {
            let stripped: serde_json::Map<String, Value> = map
                .into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_null_json_fields(v)))
                .collect();
            Value::Object(stripped)
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(strip_null_json_fields)
                .collect(),
        ),
        other => other,
    }
}

/// JSON bytes for one document in collection export (API shape, null fields omitted).
pub fn document_to_export_json(doc: &Document) -> Result<Vec<u8>, serde_json::Error> {
    let val = serde_json::to_value(doc)?;
    let val = flatten_doc_metadata(val);
    let val = strip_null_json_fields(val);
    serde_json::to_vec(&val)
}

/// Stable [`Hash`] for [`serde_json::Value`] (object keys sorted so order-independent).
pub(crate) fn hash_json_value<H: Hasher>(v: &Value, state: &mut H) {
    match v {
        Value::Null => {
            0u8.hash(state);
        }
        Value::Bool(b) => {
            1u8.hash(state);
            b.hash(state);
        }
        Value::Number(n) => {
            2u8.hash(state);
            n.to_string().hash(state);
        }
        Value::String(s) => {
            3u8.hash(state);
            s.hash(state);
        }
        Value::Array(arr) => {
            4u8.hash(state);
            arr.len().hash(state);
            for x in arr {
                hash_json_value(x, state);
            }
        }
        Value::Object(map) => {
            5u8.hash(state);
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            keys.len().hash(state);
            for k in keys {
                k.hash(state);
                hash_json_value(&map[k], state);
            }
        }
    }
}

use crate::common::{
    DenseDistance, DocumentField, VectorType, DEFAULT_LANGUAGE_CONFIDENCE, DEFAULT_TOP_K,
    DEFAULT_WMTR_TRIGRAM_WEIGHT, DEFAULT_WMTR_WORD_WEIGHT_PERCENTAGE, DEFAULT_SEARCH_LIMIT,
};

// ---------------------------------------------------------------------------
// Shared response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OkResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<Vec<String>>,
}

impl OkResponse {
    pub fn ok() -> Self {
        OkResponse { ok: true, skipped: None }
    }

    pub fn ok_with_skipped(skipped: Vec<String>) -> Self {
        let skipped = if skipped.is_empty() { None } else { Some(skipped) };
        OkResponse { ok: true, skipped }
    }
}

/// Readiness body — same JSON shape as Python `ReadyResponse` in `amgix-server` (`main.py`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyResponse {
    pub database: bool,
    pub rabbitmq: bool,
    pub index: bool,
    pub query: bool,
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfoResponse {
    pub amgix_version: String,
    pub database_kind: String,
    pub database_version: String,
    #[serde(default)]
    pub database_features: HashMap<String, bool>,
    pub rabbitmq_version: String,
    pub collection_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionExistsResponse {
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueInfo {
    pub queued_upsert: i64,
    pub queued_delete: i64,
    pub requeued_upsert: i64,
    pub requeued_delete: i64,
    pub failed_upsert: i64,
    pub failed_delete: i64,
    pub total: i64,
}

impl QueueInfo {
    pub fn empty() -> Self {
        QueueInfo {
            queued_upsert: 0,
            queued_delete: 0,
            requeued_upsert: 0,
            requeued_delete: 0,
            failed_upsert: 0,
            failed_delete: 0,
            total: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatsResponse {
    pub doc_count: i64,
    pub queue: QueueInfo,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum QueuedDocumentStatus {
    Queued,
    Requeued,
    Indexed,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum QueueOperationType {
    Upsert,
    Delete,
}

/// One row in `DocumentStatusResponse.statuses` — mirrors `document.py` `DocumentStatus`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatus {
    pub status: QueuedDocumentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op_type: Option<QueueOperationType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<String>,
    pub timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub try_count: Option<u32>,
}

/// Mirrors `document.py` `DocumentStatusResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatusResponse {
    pub statuses: Vec<DocumentStatus>,
}

/// Mirrors `document.py` `QueueDocument` — payload stored in `amgix_sys_queue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueDocument {
    pub queue_id: String,
    pub collection_name: String,
    pub collection_id: String,
    pub doc_id: String,
    pub op_type: QueueOperationType,
    pub doc_timestamp: DateTime<Utc>,
    pub status: QueuedDocumentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<Document>,
    pub created_at: DateTime<Utc>,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub try_count: u32,
}

// ---------------------------------------------------------------------------
// Metadata index (used in both API and internal collection configs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataIndex {
    pub key: String,
    #[serde(rename = "type")]
    pub value_type: String,
}

// ---------------------------------------------------------------------------
// VectorConfig — API-facing (mirrors CollectionConfig in vector.py)
// ---------------------------------------------------------------------------

fn default_top_k() -> u32 {
    DEFAULT_TOP_K
}

fn default_wmtr_word_ratio() -> u32 {
    DEFAULT_WMTR_WORD_WEIGHT_PERCENTAGE
}

fn default_language_code() -> String {
    "en".to_string()
}

fn default_language_confidence() -> f64 {
    DEFAULT_LANGUAGE_CONFIDENCE
}

fn default_index_fields() -> Vec<DocumentField> {
    vec![DocumentField::Content]
}

fn default_dense_distance() -> DenseDistance {
    DenseDistance::default()
}

fn default_store_content() -> bool {
    true
}

/// Mirrors `vector.py` `keep_case` Field(default=False).
fn default_keep_case() -> Option<bool> {
    Some(false)
}

/// Mirrors `vector.py` `VectorConfig.set_normalization_default`: dense → true, sparse → false when omitted.
fn apply_normalization_default(norm: Option<bool>, vector_type: &VectorType) -> Option<bool> {
    Some(norm.unwrap_or_else(|| vector_type.is_dense()))
}

fn canonical_vector_type(vt: VectorType) -> VectorType {
    match vt {
        VectorType::Keyword => VectorType::Wmtr,
        vt => vt,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub vector_type: VectorType,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub query_model: Option<String>,
    #[serde(default)]
    pub query_revision: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_wmtr_word_ratio", alias = "wmtr_word_weight")]
    pub wmtr_word_ratio: u32,
    /// Mutually exclusive with `doc_template` / `query_template`. Defaults to `[content]` when both templates are absent.
    #[serde(default)]
    pub index_fields: Option<Vec<DocumentField>>,
    #[serde(default)]
    pub doc_template: Option<String>,
    #[serde(default)]
    pub query_template: Option<String>,
    #[serde(default = "default_language_code")]
    pub language_default_code: String,
    #[serde(default)]
    pub language_detect: bool,
    #[serde(default = "default_language_confidence")]
    pub language_confidence: f64,
    #[serde(default)]
    pub normalization: Option<bool>,
    #[serde(default = "default_dense_distance")]
    pub dense_distance: DenseDistance,
    #[serde(default = "default_keep_case")]
    pub keep_case: Option<bool>,
}

fn default_noop_vectors() -> Vec<VectorConfig> {
    vec![VectorConfig {
        name: "noop".to_string(),
        vector_type: VectorType::Noop,
        model: None,
        revision: None,
        query_model: None,
        query_revision: None,
        dimensions: None,
        top_k: default_top_k(),
        wmtr_word_ratio: default_wmtr_word_ratio(),
        index_fields: Some(vec![DocumentField::Name]),
        doc_template: None,
        query_template: None,
        language_default_code: default_language_code(),
        language_detect: false,
        language_confidence: default_language_confidence(),
        normalization: None,
        dense_distance: default_dense_distance(),
        keep_case: default_keep_case(),
    }]
}

#[derive(Debug, Clone, Serialize)]
pub struct CollectionConfig {
    pub vectors: Vec<VectorConfig>,
    pub store_content: bool,
    pub metadata_indexes: Option<Vec<MetadataIndex>>,
}

impl<'de> Deserialize<'de> for CollectionConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            vectors: Option<Vec<VectorConfig>>,
            #[serde(default = "default_store_content")]
            store_content: bool,
            #[serde(default)]
            metadata_indexes: Option<Vec<MetadataIndex>>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let vectors = match raw.vectors {
            Some(v) if !v.is_empty() => v,
            _ => default_noop_vectors(),
        };
        Ok(CollectionConfig {
            vectors,
            store_content: raw.store_content,
            metadata_indexes: raw.metadata_indexes,
        })
    }
}

// ---------------------------------------------------------------------------
// VectorConfigInternal / CollectionConfigInternal
// Mirrors Python's VectorConfigInternal / CollectionConfigInternal exactly —
// this is the JSON stored as a payload in `amgix_sys_meta`.
// ---------------------------------------------------------------------------

fn default_version() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfigInternal {
    #[serde(default = "default_version")]
    pub version: u32,

    // All fields from VectorConfig
    pub name: String,
    #[serde(rename = "type")]
    pub vector_type: VectorType,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub query_model: Option<String>,
    #[serde(default)]
    pub query_revision: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_wmtr_word_ratio", alias = "wmtr_word_weight")]
    pub wmtr_word_ratio: u32,
    #[serde(default)]
    pub index_fields: Option<Vec<DocumentField>>,
    #[serde(default)]
    pub doc_template: Option<String>,
    #[serde(default)]
    pub query_template: Option<String>,
    #[serde(default = "default_language_code")]
    pub language_default_code: String,
    #[serde(default)]
    pub language_detect: bool,
    #[serde(default = "default_language_confidence")]
    pub language_confidence: f64,
    #[serde(default)]
    pub normalization: Option<bool>,
    #[serde(default = "default_dense_distance")]
    pub dense_distance: DenseDistance,
    #[serde(default = "default_keep_case")]
    pub keep_case: Option<bool>,
}

impl From<VectorConfig> for VectorConfigInternal {
    fn from(v: VectorConfig) -> Self {
        let vector_type = canonical_vector_type(v.vector_type);
        let normalization = apply_normalization_default(v.normalization, &vector_type);
        let (index_fields, doc_template, query_template) =
            if v.doc_template.is_some() || v.query_template.is_some() {
                (None, v.doc_template, v.query_template)
            } else {
                (
                    Some(v.index_fields.unwrap_or_else(default_index_fields)),
                    None,
                    None,
                )
            };
        VectorConfigInternal {
            version: 1_u32,
            name: v.name,
            vector_type,
            model: v.model,
            revision: v.revision,
            query_model: v.query_model,
            query_revision: v.query_revision,
            dimensions: v.dimensions,
            top_k: v.top_k,
            wmtr_word_ratio: v.wmtr_word_ratio,
            index_fields,
            doc_template,
            query_template,
            language_default_code: v.language_default_code,
            language_detect: v.language_detect,
            language_confidence: v.language_confidence,
            normalization,
            dense_distance: v.dense_distance,
            keep_case: v.keep_case,
        }
    }
}

impl From<VectorConfigInternal> for VectorConfig {
    /// Mirrors `vector.py` `internal_to_user_config`: exposed API omits stored
    /// `query_model` / `query_revision` / `keep_case` — they are reset to defaults
    /// (`None`, `None`, `False`). Template mode omits `index_fields`.
    fn from(v: VectorConfigInternal) -> Self {
        let vector_type = canonical_vector_type(v.vector_type);
        let normalization = apply_normalization_default(v.normalization, &vector_type);
        let uses_tmpl = v.doc_template.is_some();
        VectorConfig {
            name: v.name,
            vector_type,
            model: v.model,
            revision: v.revision,
            query_model: None,
            query_revision: None,
            dimensions: v.dimensions,
            top_k: v.top_k,
            wmtr_word_ratio: v.wmtr_word_ratio,
            index_fields: if uses_tmpl {
                None
            } else {
                v.index_fields.or_else(|| Some(default_index_fields()))
            },
            doc_template: v.doc_template,
            query_template: v.query_template,
            language_default_code: v.language_default_code,
            language_detect: v.language_detect,
            language_confidence: v.language_confidence,
            normalization,
            dense_distance: v.dense_distance,
            keep_case: default_keep_case(),
        }
    }
}

impl From<CollectionConfigInternal> for CollectionConfig {
    fn from(c: CollectionConfigInternal) -> Self {
        CollectionConfig {
            vectors: c.vectors.into_iter().map(VectorConfig::from).collect(),
            store_content: c.store_content,
            metadata_indexes: c.metadata_indexes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfigInternal {
    #[serde(default = "default_version")]
    pub version: u32,
    pub collection_id: String,
    pub vectors: Vec<VectorConfigInternal>,
    #[serde(default = "default_store_content")]
    pub store_content: bool,
    #[serde(default)]
    pub metadata_indexes: Option<Vec<MetadataIndex>>,
}

// ---------------------------------------------------------------------------
// Document.timestamp — mirrors `document.py` `@field_validator('timestamp')`
// ---------------------------------------------------------------------------

fn deserialize_document_timestamp<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::String(s) => crate::datetime_parse::parse_utc_datetime(&s).map_err(serde::de::Error::custom),
        _ => Err(serde::de::Error::custom(
            "Timestamp must be a datetime object",
        )),
    }
}

// ---------------------------------------------------------------------------
// Document — API-facing (mirrors document.py)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomDocumentVector {
    pub vector_name: String,
    pub vector: Value,
    pub field: DocumentField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    #[serde(deserialize_with = "deserialize_document_timestamp")]
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    /// Hash of content for revectorization when store_content=false (server-managed).
    #[serde(default)]
    pub content_hash: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub custom_vectors: Option<Vec<CustomDocumentVector>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joined: Option<HashMap<String, Vec<Document>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors: Option<Vec<VectorData>>,
    #[serde(default, skip)]
    pub token_lengths: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// BulkUploadRequest — mirrors BulkUploadRequest in main.py
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkUploadRequest {
    pub documents: Vec<Document>,
}

// ---------------------------------------------------------------------------
// VectorData — pre-computed vector for one (field, vector_name) pair.
// Mirrors vector.py VectorData exactly.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    pub vector_name: String,
    pub field: DocumentField,
    pub vector_type: VectorType,
    #[serde(default)]
    pub dense_vector: Option<Vec<f32>>,
    #[serde(default)]
    pub sparse_indices: Option<Vec<u32>>,
    #[serde(default)]
    pub sparse_values: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Search — API-facing (mirrors vector.py / document.py)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomVector {
    pub vector_name: String,
    pub vector: Value,
}

impl Hash for CustomVector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vector_name.hash(state);
        hash_json_value(&self.vector, state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchOption {
    pub vector_name: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Optional for template-based vectors (ignored if present).
    #[serde(default)]
    pub field: Option<DocumentField>,
    #[serde(default = "default_wmtr_trigram_weight")]
    pub wmtr_trigram_weight: f64,
}

impl PartialEq for VectorSearchOption {
    fn eq(&self, other: &Self) -> bool {
        self.vector_name == other.vector_name
            && self.field == other.field
            && self.weight.to_bits() == other.weight.to_bits()
            && self.wmtr_trigram_weight.to_bits() == other.wmtr_trigram_weight.to_bits()
    }
}

impl Eq for VectorSearchOption {}

impl Hash for VectorSearchOption {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vector_name.hash(state);
        self.weight.to_bits().hash(state);
        self.field.hash(state);
        self.wmtr_trigram_weight.to_bits().hash(state);
    }
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataFilter {
    #[serde(rename = "and", default)]
    pub and_: Option<Vec<MetadataFilter>>,
    #[serde(rename = "or", default)]
    pub or_: Option<Vec<MetadataFilter>>,
    #[serde(rename = "not", default)]
    pub not_: Option<Box<MetadataFilter>>,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub op: Option<String>,
    #[serde(default)]
    pub value: Option<Value>,
}

impl PartialEq for MetadataFilter {
    fn eq(&self, other: &Self) -> bool {
        self.and_ == other.and_
            && self.or_ == other.or_
            && self.not_ == other.not_
            && self.key == other.key
            && self.op == other.op
            && self.value == other.value
    }
}

impl Eq for MetadataFilter {}

impl Hash for MetadataFilter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.and_.hash(state);
        self.or_.hash(state);
        self.not_.hash(state);
        self.key.hash(state);
        self.op.hash(state);
        if let Some(ref v) = self.value {
            hash_json_value(v, state);
        }
    }
}

fn deserialize_metadata_filter<'de, D>(deserializer: D) -> Result<Option<MetadataFilter>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Option<Value> = Option::deserialize(deserializer)?;
    match v {
        None => Ok(None),
        Some(Value::String(s)) => {
            let filter = crate::filter_parser::parse_filter(&s)
                .map_err(serde::de::Error::custom)?;
            Ok(Some(filter))
        }
        Some(other) => {
            let filter: MetadataFilter =
                serde_json::from_value(other).map_err(serde::de::Error::custom)?;
            Ok(Some(filter))
        }
    }
}

fn default_search_limit() -> u32 {
    DEFAULT_SEARCH_LIMIT
}

fn default_wmtr_trigram_weight() -> f64 {
    DEFAULT_WMTR_TRIGRAM_WEIGHT
}

fn default_fusion_mode() -> String {
    "rrf".to_string()
}

fn default_group_max() -> u32 {
    3
}

fn default_group_max_fetches() -> u32 {
    2
}

fn default_facet_prefetch_multiplier() -> u32 {
    crate::common::DEFAULT_FACET_PREFETCH_MULTIPLIER
}

fn default_facet_max_values() -> u32 {
    crate::common::DEFAULT_FACET_MAX_VALUES
}

/// Faceting options for a search query. Mirrors FacetOptions in vector.py.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FacetOptions {
    #[serde(default = "default_facet_prefetch_multiplier")]
    pub prefetch_multiplier: u32,
    #[serde(default = "default_facet_max_values")]
    pub max_values: u32,
}

fn opt_f64_eq(a: &Option<f64>, b: &Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        _ => false,
    }
}

fn hash_opt_f64<H: Hasher>(x: &Option<f64>, state: &mut H) {
    match x {
        None => {
            0u8.hash(state);
        }
        Some(f) => {
            1u8.hash(state);
            f.to_bits().hash(state);
        }
    }
}

/// All [`SearchQuery`] fields except `query`, used for ingress batching keys (`Hash`/`Eq`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuerySettings {
    #[serde(default)]
    pub vector_options: Vec<VectorSearchOption>,
    #[serde(default)]
    pub custom_vectors: Option<Vec<CustomVector>>,
    #[serde(default = "default_search_limit")]
    pub limit: u32,
    #[serde(default)]
    pub score_threshold: Option<f64>,
    #[serde(default)]
    pub document_tags: Option<Vec<String>>,
    #[serde(default)]
    pub document_tags_match_all: bool,
    #[serde(default, deserialize_with = "deserialize_metadata_filter")]
    pub metadata_filter: Option<MetadataFilter>,
    #[serde(default)]
    pub raw_scores: bool,
    #[serde(default = "default_fusion_mode")]
    pub fusion_mode: String,
    #[serde(default, deserialize_with = "crate::join_parser::deserialize_join_field")]
    pub join: Option<crate::join_parser::JoinField>,
    #[serde(default)]
    pub exclude: Option<Vec<crate::common::SearchExcludeField>>,
    #[serde(default)]
    pub group_field: Option<String>,
    #[serde(default = "default_group_max")]
    pub group_max: u32,
    #[serde(default = "default_group_max_fetches")]
    pub group_max_fetches: u32,
    #[serde(default)]
    pub facets: bool,
    #[serde(default)]
    pub facet_options: Option<FacetOptions>,
}

impl PartialEq for SearchQuerySettings {
    fn eq(&self, other: &Self) -> bool {
        self.vector_options == other.vector_options
            && self.custom_vectors == other.custom_vectors
            && self.limit == other.limit
            && opt_f64_eq(&self.score_threshold, &other.score_threshold)
            && self.document_tags == other.document_tags
            && self.document_tags_match_all == other.document_tags_match_all
            && self.metadata_filter == other.metadata_filter
            && self.raw_scores == other.raw_scores
            && self.fusion_mode == other.fusion_mode
            && self.join == other.join
            && self.exclude == other.exclude
            && self.group_field == other.group_field
            && self.group_max == other.group_max
            && self.group_max_fetches == other.group_max_fetches
            && self.facets == other.facets
            && self.facet_options == other.facet_options
    }
}

impl Eq for SearchQuerySettings {}

impl Hash for SearchQuerySettings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vector_options.hash(state);
        self.custom_vectors.hash(state);
        self.limit.hash(state);
        hash_opt_f64(&self.score_threshold, state);
        self.document_tags.hash(state);
        self.document_tags_match_all.hash(state);
        self.metadata_filter.hash(state);
        self.raw_scores.hash(state);
        self.fusion_mode.hash(state);
        self.join.hash(state);
        self.exclude.hash(state);
        self.group_field.hash(state);
        self.group_max.hash(state);
        self.group_max_fetches.hash(state);
        self.facets.hash(state);
        self.facet_options.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchQuery {
    pub query: String,
    #[serde(flatten)]
    pub settings: SearchQuerySettings,
}

// ---------------------------------------------------------------------------
// SearchQueryWithVectors — SearchQuery + pre-computed query vectors.
// Mirrors vector.py SearchQueryWithVectors exactly (flat JSON via [`serde(flatten)`]).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQueryWithVectors {
    pub query: String,
    #[serde(flatten)]
    pub settings: SearchQuerySettings,
    pub vectors: Vec<VectorData>,
}

// ---------------------------------------------------------------------------
// Model validation — mirrors vector.py ModelValidationResult / Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationResult {
    pub valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub results: Option<HashMap<String, ModelValidationResult>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorScore {
    pub field: String,
    pub vector: String,
    pub score: f64,
    pub rank: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    #[serde(flatten)]
    pub document: Document,
    pub score: f64,
    #[serde(default)]
    pub vector_scores: Vec<VectorScore>,
}

/// Internal search result bundling hits with optional facet counts. Mirrors
/// `SearchOutcome` in amgix-server. `facet_counts` is `Some` only when the
/// request enabled faceting.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub facet_counts: Option<std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>>>,
}

impl SearchOutcome {
    pub fn new(results: Vec<SearchResult>) -> Self {
        SearchOutcome { results, facet_counts: None }
    }
}

// ---------------------------------------------------------------------------
// SearchResponse
// Mirrors amgix-server SearchResponse.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query_time_ms: f64,
}

// ---------------------------------------------------------------------------
// DocumentFetchRequest / DocumentFetchResponse
// Mirrors amgix-server DocumentFetchRequest / DocumentFetchResponse.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentFetchRequest {
    #[serde(default = "default_fetch_page_size")]
    pub page_size: u32,
    #[serde(default)]
    pub after: Option<String>,
    #[serde(default, deserialize_with = "deserialize_metadata_filter")]
    pub metadata_filter: Option<MetadataFilter>,
    #[serde(default)]
    pub document_tags: Option<Vec<String>>,
    #[serde(default)]
    pub document_tags_match_all: bool,
    #[serde(default, deserialize_with = "crate::join_parser::deserialize_join_field")]
    pub join: Option<crate::join_parser::JoinField>,
    #[serde(default)]
    pub with_vectors: bool,
}

fn default_fetch_page_size() -> u32 {
    100
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentFetchResponse {
    pub documents: Vec<Document>,
    pub after: Option<String>,
}
