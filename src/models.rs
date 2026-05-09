//! API-facing request/response types and internal storage models.
//!
//! API types mirror the JSON shapes of `amgix-server` exactly.
//! Internal types (`*Internal`) mirror what `amgix-server` stores in Qdrant,
//! so data written by either service is readable by the other.

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QueuedDocumentStatus {
    Queued,
    Requeued,
    Indexed,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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

/// Mirrors `document.py` `DocumentStatusResponse`. In **amgix-now** only `indexed` appears (no queue).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatusResponse {
    pub statuses: Vec<DocumentStatus>,
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

fn default_wmtr_word_weight() -> u32 {
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
    #[serde(default = "default_wmtr_word_weight")]
    pub wmtr_word_weight: u32,
    #[serde(default = "default_index_fields")]
    pub index_fields: Vec<DocumentField>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub vectors: Vec<VectorConfig>,
    #[serde(default = "default_store_content")]
    pub store_content: bool,
    #[serde(default)]
    pub metadata_indexes: Option<Vec<MetadataIndex>>,
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
    #[serde(default = "default_wmtr_word_weight")]
    pub wmtr_word_weight: u32,
    #[serde(default = "default_index_fields")]
    pub index_fields: Vec<DocumentField>,
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
            wmtr_word_weight: v.wmtr_word_weight,
            index_fields: v.index_fields,
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
    /// (`None`, `None`, `False`).
    fn from(v: VectorConfigInternal) -> Self {
        let vector_type = canonical_vector_type(v.vector_type);
        let normalization = apply_normalization_default(v.normalization, &vector_type);
        VectorConfig {
            name: v.name,
            vector_type,
            model: v.model,
            revision: v.revision,
            query_model: None,
            query_revision: None,
            dimensions: v.dimensions,
            top_k: v.top_k,
            wmtr_word_weight: v.wmtr_word_weight,
            index_fields: v.index_fields,
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

/// Parses request JSON strings only. Offset must be UTC (`Z` or `+00:00` / equivalent zero offset).
/// Rejects naive datetimes and non-UTC offsets with the same messages as Python.
pub fn parse_document_timestamp_for_api(s: &str) -> Result<DateTime<Utc>, String> {
    let s = s.trim();
    let z_norm = s.replacen('Z', "+00:00", 1);

    if let Ok(dt) = DateTime::parse_from_rfc3339(&z_norm) {
        if dt.offset().local_minus_utc() != 0 {
            return Err("Timestamp must be in UTC timezone".to_string());
        }
        return Ok(dt.with_timezone(&Utc));
    }

    const NAIVE_FMTS: [&str; 2] = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S"];
    for fmt in NAIVE_FMTS {
        if NaiveDateTime::parse_from_str(s, fmt).is_ok() {
            return Err("Timestamp must include timezone information".to_string());
        }
    }

    Err("Timestamp must be a valid ISO 8601 datetime string".to_string())
}

fn deserialize_document_timestamp<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::String(s) => parse_document_timestamp_for_api(&s).map_err(serde::de::Error::custom),
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
    #[serde(default)]
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub custom_vectors: Option<Vec<CustomDocumentVector>>,
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
// DocumentWithVectors — Document + pre-computed vectors + token lengths.
// Mirrors document.py DocumentWithVectors exactly.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentWithVectors {
    // All Document fields (flattened)
    pub id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub custom_vectors: Option<Vec<CustomDocumentVector>>,
    // Internal fields
    #[serde(default)]
    pub vectors: Vec<VectorData>,
    #[serde(default)]
    pub token_lengths: HashMap<String, usize>,
}

impl From<DocumentWithVectors> for Document {
    fn from(d: DocumentWithVectors) -> Self {
        Document {
            id: d.id,
            timestamp: d.timestamp,
            tags: d.tags,
            name: d.name,
            description: d.description,
            content: d.content,
            metadata: d.metadata,
            custom_vectors: d.custom_vectors,
        }
    }
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
pub struct VectorSearchWeight {
    pub vector_name: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    pub field: DocumentField,
}

impl PartialEq for VectorSearchWeight {
    fn eq(&self, other: &Self) -> bool {
        self.vector_name == other.vector_name
            && self.field == other.field
            && self.weight.to_bits() == other.weight.to_bits()
    }
}

impl Eq for VectorSearchWeight {}

impl Hash for VectorSearchWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vector_name.hash(state);
        self.weight.to_bits().hash(state);
        self.field.hash(state);
    }
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

fn default_search_limit() -> u32 {
    DEFAULT_SEARCH_LIMIT
}

fn default_wmtr_trigram_weight() -> f64 {
    DEFAULT_WMTR_TRIGRAM_WEIGHT
}

fn default_fusion_mode() -> String {
    "rrf".to_string()
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
    pub vector_weights: Vec<VectorSearchWeight>,
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
    #[serde(default)]
    pub metadata_filter: Option<MetadataFilter>,
    #[serde(default)]
    pub raw_scores: bool,
    #[serde(default = "default_wmtr_trigram_weight")]
    pub wmtr_trigram_weight: f64,
    #[serde(default = "default_fusion_mode")]
    pub fusion_mode: String,
}

impl PartialEq for SearchQuerySettings {
    fn eq(&self, other: &Self) -> bool {
        self.vector_weights == other.vector_weights
            && self.custom_vectors == other.custom_vectors
            && self.limit == other.limit
            && opt_f64_eq(&self.score_threshold, &other.score_threshold)
            && self.document_tags == other.document_tags
            && self.document_tags_match_all == other.document_tags_match_all
            && self.metadata_filter == other.metadata_filter
            && self.raw_scores == other.raw_scores
            && self.wmtr_trigram_weight.to_bits() == other.wmtr_trigram_weight.to_bits()
            && self.fusion_mode == other.fusion_mode
    }
}

impl Eq for SearchQuerySettings {}

impl Hash for SearchQuerySettings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vector_weights.hash(state);
        self.custom_vectors.hash(state);
        self.limit.hash(state);
        hash_opt_f64(&self.score_threshold, state);
        self.document_tags.hash(state);
        self.document_tags_match_all.hash(state);
        self.metadata_filter.hash(state);
        self.raw_scores.hash(state);
        self.wmtr_trigram_weight.to_bits().hash(state);
        self.fusion_mode.hash(state);
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
    pub id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, Value>>,
    pub score: f64,
    #[serde(default)]
    pub vector_scores: Vec<VectorScore>,
}
