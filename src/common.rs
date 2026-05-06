use once_cell::sync::Lazy;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Application identity
// ---------------------------------------------------------------------------

pub const APP_NAME: &str = "Amalgam Index";
pub const APP_PREFIX: &str = "amgix";

/// UUID5 namespace for all document / config IDs.
/// Computed once at startup: uuid5(NAMESPACE_DNS, APP_NAME).
/// Must produce `d953b233-6472-5054-8f32-1999b057711c` — verified against Python.
pub static DOC_NAMESPACE: Lazy<Uuid> =
    Lazy::new(|| Uuid::new_v5(&Uuid::NAMESPACE_DNS, APP_NAME.as_bytes()));

// ---------------------------------------------------------------------------
// Numeric defaults (mirrors constants.py)
// ---------------------------------------------------------------------------

pub const DEFAULT_TOP_K: u32 = 128;
pub const DEFAULT_SEARCH_LIMIT: u32 = 10;
pub const DEFAULT_WMTR_WORD_WEIGHT_PERCENTAGE: u32 = 80;
pub const DEFAULT_WMTR_TRIGRAM_WEIGHT: f64 = 1.0;
pub const DEFAULT_LANGUAGE_CONFIDENCE: f64 = 0.9;
pub const SEARCH_PREFETCH_MULTIPLIER: f64 = 1.5;

// ---------------------------------------------------------------------------
// VectorType — all variants, mirrors enums.py exactly
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorType {
    DenseModel,
    SparseModel,
    FullText,
    Trigrams,
    Whitespace,
    Wmtr,
    Keyword,
    DenseCustom,
    SparseCustom,
}

impl VectorType {
    pub fn is_dense(&self) -> bool {
        matches!(self, VectorType::DenseModel | VectorType::DenseCustom)
    }

    pub fn is_sparse(&self) -> bool {
        !self.is_dense()
    }

    /// Types that use our Rust tokenizers (get IDF modifier in Qdrant).
    pub fn is_custom_tokenization(&self) -> bool {
        matches!(
            self,
            VectorType::FullText
                | VectorType::Trigrams
                | VectorType::Whitespace
                | VectorType::Wmtr
                | VectorType::Keyword
        )
    }

    /// Types backed by transformer models (not implemented yet).
    pub fn is_transformer_based(&self) -> bool {
        matches!(self, VectorType::DenseModel | VectorType::SparseModel)
    }

    /// Types where the caller supplies pre-computed vectors.
    pub fn is_custom_vectors(&self) -> bool {
        matches!(self, VectorType::DenseCustom | VectorType::SparseCustom)
    }

    /// Mirrors `VectorType.sparse_types()` in enums.py — used for BM25/token-length bookkeeping.
    pub fn sparse_types_contains(&self) -> bool {
        matches!(
            self,
            VectorType::SparseModel
                | VectorType::FullText
                | VectorType::Trigrams
                | VectorType::Whitespace
                | VectorType::Wmtr
                | VectorType::Keyword
                | VectorType::SparseCustom
        )
    }
}

impl std::fmt::Display for VectorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            VectorType::DenseModel => "dense_model",
            VectorType::SparseModel => "sparse_model",
            VectorType::FullText => "full_text",
            VectorType::Trigrams => "trigrams",
            VectorType::Whitespace => "whitespace",
            VectorType::Wmtr => "wmtr",
            VectorType::Keyword => "keyword",
            VectorType::DenseCustom => "dense_custom",
            VectorType::SparseCustom => "sparse_custom",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// DenseDistance — mirrors enums.py
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DenseDistance {
    Cosine,
    Dot,
    Euclid,
}

impl Default for DenseDistance {
    fn default() -> Self {
        DenseDistance::Cosine
    }
}

// ---------------------------------------------------------------------------
// DocumentField — mirrors enums.py
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentField {
    Name,
    Description,
    Content,
}

impl std::fmt::Display for DocumentField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DocumentField::Name => "name",
            DocumentField::Description => "description",
            DocumentField::Content => "content",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Collection name helpers — mirrors functions.py
// ---------------------------------------------------------------------------

/// `foo` → `amgix_foo`
pub fn get_real_collection_name(user_name: &str) -> String {
    format!("{APP_PREFIX}_{user_name}")
}

/// `amgix_foo` → `foo`
pub fn get_user_collection_name(real_name: &str) -> &str {
    real_name
        .strip_prefix(&format!("{APP_PREFIX}_"))
        .unwrap_or(real_name)
}

// ---------------------------------------------------------------------------
// UUID helpers — mirrors DatabaseBase._string_to_uuid
// ---------------------------------------------------------------------------

/// Deterministic UUID5 for any string identifier (document IDs, config keys).
/// Matches Python: `uuid.uuid5(DOC_NAMESPACE, string_id)`
pub fn string_to_uuid(s: &str) -> Uuid {
    Uuid::new_v5(&DOC_NAMESPACE, s.as_bytes())
}

/// System collection name, e.g. `amgix_sys_meta`.
pub fn sys_collection_name(suffix: &str) -> String {
    format!("{APP_PREFIX}_sys_{suffix}")
}

/// `AMGIX_DATABASE_URL` uses the `qdrant://host:port` scheme (see `amgix-server` `main.py`).
/// Returns a **gRPC** endpoint URI for [`qdrant_client::Qdrant`] (tonic): `http://…` on port
/// **6334** by default, same as Python `AsyncQdrantClient(..., prefer_grpc=True)`.
pub fn qdrant_client_url(connection_string: &str) -> String {
    let s = connection_string.trim();
    if let Some(rest) = s.strip_prefix("qdrant://") {
        format!("http://{rest}")
    } else if s.contains("://") {
        s.to_string()
    } else {
        format!("http://{s}")
    }
}

/// Short product label from DB URL scheme — mirrors `amgix-server` `_database_kind_label`.
pub const DATABASE_KIND: &str = "Qdrant";
