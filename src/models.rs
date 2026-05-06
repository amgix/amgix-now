//! API-facing request/response types and internal storage models.
//!
//! API types mirror the JSON shapes of `amgix-server` exactly.
//! Internal types (`*Internal`) mirror what `amgix-server` stores in Qdrant,
//! so data written by either service is readable by the other.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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
    #[serde(default)]
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
    #[serde(default)]
    pub keep_case: Option<bool>,
}

impl From<VectorConfig> for VectorConfigInternal {
    fn from(v: VectorConfig) -> Self {
        VectorConfigInternal {
            version: 1_u32,
            name: v.name,
            vector_type: v.vector_type,
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
            normalization: v.normalization,
            dense_distance: v.dense_distance,
            keep_case: v.keep_case,
        }
    }
}

impl From<VectorConfigInternal> for VectorConfig {
    fn from(v: VectorConfigInternal) -> Self {
        VectorConfig {
            name: v.name,
            vector_type: v.vector_type,
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
            normalization: v.normalization,
            dense_distance: v.dense_distance,
            keep_case: v.keep_case,
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

// ---------------------------------------------------------------------------
// SearchQueryWithVectors — SearchQuery + pre-computed query vectors.
// Mirrors vector.py SearchQueryWithVectors exactly.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQueryWithVectors {
    // All SearchQuery fields
    pub query: String,
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
    // Pre-computed query vectors
    pub vectors: Vec<VectorData>,
}

// ---------------------------------------------------------------------------
// Search — API-facing (mirrors vector.py / document.py)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomVector {
    pub vector_name: String,
    pub vector: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchWeight {
    pub vector_name: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    pub field: DocumentField,
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

fn default_search_limit() -> u32 {
    DEFAULT_SEARCH_LIMIT
}

fn default_wmtr_trigram_weight() -> f64 {
    DEFAULT_WMTR_TRIGRAM_WEIGHT
}

fn default_fusion_mode() -> String {
    "rrf".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
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
