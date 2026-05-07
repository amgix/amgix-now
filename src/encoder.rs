//! Encoder-layer logic — mirrors `src/encoder/encoder.py` (`update_collection_stats`,
//! `validate_models`, `document_upsert_sync`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, Mutex, OwnedMutexGuard};

use crate::common::{VectorType, DEFAULT_SEARCH_LIMIT, DEFAULT_WMTR_TRIGRAM_WEIGHT};
use crate::functions::normalize_document_metadata_inplace;
use crate::models::{
    CollectionConfigInternal, Document, DocumentWithVectors, ModelValidationResponse,
    ModelValidationResult, SearchQuery, SearchQueryWithVectors, VectorConfigInternal,
    VectorSearchWeight,
};
use crate::qdrant::{CollectionStats, DbError, QdrantDb};
use crate::vectors::vectorizer::Vectorizer;

// ---------------------------------------------------------------------------
// TokenLengthUpdate — per-field bundle mirroring encoder.py's updates dict shape
// ---------------------------------------------------------------------------

pub struct TokenLengthUpdate {
    pub new_doc_count: i64,
    pub new_sum_token_lengths: i64,
    pub update_doc_count: i64,
    pub update_sum_token_lengths: i64,
    pub old_sum_token_lengths: i64,
}

// ---------------------------------------------------------------------------
// NamedLocks — generic async mutex registry keyed by an arbitrary string.
//
// Used for:
//   - stats locks  (key = collection_name)  — mirrors _stats_locks
//   - per-doc locks (key = "{collection}-{doc_id}") — mirrors lock_client per-doc
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct NamedLocks {
    inner: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

impl NamedLocks {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn lock(&self, key: &str) -> OwnedMutexGuard<()> {
        let entry = {
            let mut map = self.inner.lock().await;
            map.entry(key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        Mutex::lock_owned(entry).await
    }
}

// ---------------------------------------------------------------------------
// CollectionConfigCache — TTL cache for collection configs.
//
// Mirrors EncoderBase._collection_info_cache: AMGIXCache(ttl=60, maxsize=1000).
// Keyed by collection_name. Entries expire after TTL_SECS seconds.
// On overflow (> MAX_ENTRIES), the oldest entry is evicted.
// ---------------------------------------------------------------------------

const CACHE_TTL_SECS: u64 = 60;
const CACHE_MAX_ENTRIES: usize = 1000;

struct CacheEntry {
    config: CollectionConfigInternal,
    inserted_at: Instant,
}

#[derive(Clone)]
pub struct CollectionConfigCache {
    inner: Arc<Mutex<HashMap<String, CacheEntry>>>,
}

impl CollectionConfigCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn get(&self, collection_name: &str) -> Option<CollectionConfigInternal> {
        let map = self.inner.lock().await;
        map.get(collection_name).and_then(|e| {
            if e.inserted_at.elapsed() < Duration::from_secs(CACHE_TTL_SECS) {
                Some(e.config.clone())
            } else {
                None
            }
        })
    }

    pub async fn set(&self, collection_name: &str, config: CollectionConfigInternal) {
        let mut map = self.inner.lock().await;
        if map.len() >= CACHE_MAX_ENTRIES && !map.contains_key(collection_name) {
            // Evict the entry with the oldest insertion time.
            if let Some(oldest_key) = map
                .iter()
                .min_by_key(|(_, e)| e.inserted_at)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest_key);
            }
        }
        map.insert(
            collection_name.to_string(),
            CacheEntry { config, inserted_at: Instant::now() },
        );
    }

    pub async fn invalidate(&self, collection_name: &str) {
        self.inner.lock().await.remove(collection_name);
    }
}

/// Fetch collection config, hitting the cache first.
/// Returns `(config, from_cache)` — mirrors `EncoderBase.get_collection_info_cached`.
pub async fn get_collection_info_cached(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    collection_name: &str,
) -> Result<(CollectionConfigInternal, bool), DbError> {
    if let Some(cached) = cache.get(collection_name).await {
        return Ok((cached, true));
    }
    let config = db.get_collection_info_internal(collection_name).await?;
    cache.set(collection_name, config.clone()).await;
    Ok((config, false))
}

// ---------------------------------------------------------------------------
// Collection stats persistence — mirrors encoder.py update_collection_stats
// ---------------------------------------------------------------------------

fn apply_token_length_updates_to_stats(
    stats: &mut CollectionStats,
    updates: &HashMap<String, TokenLengthUpdate>,
) {
    let old_doc_count = stats.doc_count;
    let new_docs_in_batch = updates.values().next().map(|u| u.new_doc_count).unwrap_or(0);
    let new_doc_count = old_doc_count + new_docs_in_batch;

    for (field_vector_name, u) in updates {
        let old_avgdl = stats.avgdls.get(field_vector_name).copied().unwrap_or(0.0);
        let new_avgdl = (old_avgdl * old_doc_count as f64
            - u.old_sum_token_lengths as f64
            + u.new_sum_token_lengths as f64
            + u.update_sum_token_lengths as f64)
            / new_doc_count as f64;
        stats.avgdls.insert(field_vector_name.clone(), new_avgdl);
    }

    stats.doc_count = new_doc_count;
}

async fn persist_stats_maps_for_collection(
    stats_locks: &NamedLocks,
    db: &QdrantDb,
    collection_name: &str,
    maps_in_order: &[&HashMap<String, TokenLengthUpdate>],
) -> Result<(), DbError> {
    let _guard = stats_locks.lock(collection_name).await;
    let mut stats = db.get_collection_stats(collection_name).await?;
    for u in maps_in_order {
        apply_token_length_updates_to_stats(&mut stats, u);
    }
    db.set_collection_stats(collection_name, &stats).await
}

const STATS_BATCH_MAX_JOBS: usize = 10;
const STATS_BATCH_WAIT: Duration = Duration::from_millis(200);
const STATS_BATCH_CHANNEL: usize = 1024;

struct StatsJob {
    collection_name: String,
    updates: HashMap<String, TokenLengthUpdate>,
}

/// Coalesces many stat updates into fewer Qdrant writes (up to [`STATS_BATCH_MAX_JOBS`] jobs or
/// [`STATS_BATCH_WAIT`] after the first job in a window).
#[derive(Clone)]
pub struct StatsUpdateBatcher {
    tx: mpsc::Sender<StatsJob>,
}

/// Drop this **after** the HTTP [`Router`] (and any clones of [`StatsUpdateBatcher`]) are gone,
/// then await [`Self::shutdown_and_wait`], so the worker sees the channel closed, drains the queue,
/// flushes any partial batch, and exits.
pub struct StatsBatcherShutdown {
    keepalive: Option<mpsc::Sender<StatsJob>>,
    join: tokio::task::JoinHandle<()>,
}

impl StatsBatcherShutdown {
    pub async fn shutdown_and_wait(mut self) {
        drop(self.keepalive.take());
        if let Err(e) = self.join.await {
            eprintln!("stats batcher task ended with error: {e}");
        }
    }
}

impl StatsUpdateBatcher {
    pub fn new(db: Arc<QdrantDb>, stats_locks: NamedLocks) -> (Self, StatsBatcherShutdown) {
        let (tx, mut rx) = mpsc::channel::<StatsJob>(STATS_BATCH_CHANNEL);
        let join = tokio::spawn(async move {
            while let Some(batch) = collect_stats_batch(&mut rx).await {
                flush_stats_job_batch(&stats_locks, &db, batch).await;
            }
        });
        let batcher = Self { tx: tx.clone() };
        let shutdown = StatsBatcherShutdown {
            keepalive: Some(tx),
            join,
        };
        (batcher, shutdown)
    }

    /// Queues a stats delta for the background worker. Only waits for channel capacity (bounded
    /// buffer); does **not** wait for Qdrant stats persistence — that is the point of micro-batching.
    pub async fn enqueue(
        &self,
        collection_name: &str,
        updates: HashMap<String, TokenLengthUpdate>,
    ) -> Result<(), DbError> {
        let job = StatsJob {
            collection_name: collection_name.to_string(),
            updates,
        };
        self.tx
            .send(job)
            .await
            .map_err(|_| DbError::Config("stats update batcher shut down".to_string()))
    }
}

async fn collect_stats_batch(rx: &mut mpsc::Receiver<StatsJob>) -> Option<Vec<StatsJob>> {
    let first = rx.recv().await?;
    let mut batch = vec![first];
    if batch.len() >= STATS_BATCH_MAX_JOBS {
        return Some(batch);
    }
    let mut sleep = Box::pin(tokio::time::sleep(STATS_BATCH_WAIT));
    loop {
        tokio::select! {
            _ = sleep.as_mut() => return Some(batch),
            job = rx.recv() => match job {
                Some(j) => {
                    batch.push(j);
                    if batch.len() >= STATS_BATCH_MAX_JOBS {
                        return Some(batch);
                    }
                }
                None => return Some(batch),
            },
        }
    }
}

async fn flush_stats_job_batch(
    stats_locks: &NamedLocks,
    db: &QdrantDb,
    batch: Vec<StatsJob>,
) {
    let mut by_collection: HashMap<String, Vec<StatsJob>> = HashMap::new();
    for job in batch {
        by_collection
            .entry(job.collection_name.clone())
            .or_default()
            .push(job);
    }

    for (collection_name, jobs) in by_collection {
        let maps_in_order: Vec<_> = jobs.iter().map(|j| &j.updates).collect();
        if let Err(e) = persist_stats_maps_for_collection(
            stats_locks,
            db,
            &collection_name,
            &maps_in_order,
        )
        .await
        {
            eprintln!(
                "stats batch persist failed for collection {collection_name}: {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// UpsertSyncError — error type for document_upsert_sync
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum UpsertSyncError {
    NotFound(String),
    Db(DbError),
    Vectorization(String),
}

impl std::fmt::Display for UpsertSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpsertSyncError::NotFound(m) => write!(f, "{m}"),
            UpsertSyncError::Db(e) => write!(f, "{e}"),
            UpsertSyncError::Vectorization(m) => write!(f, "{m}"),
        }
    }
}

impl From<DbError> for UpsertSyncError {
    fn from(e: DbError) -> Self {
        UpsertSyncError::Db(e)
    }
}

// ---------------------------------------------------------------------------
// document_upsert_sync — single-document convenience wrapper around bulk
// ---------------------------------------------------------------------------

/// Returns `Ok(skipped_ids)` — non-empty when the document was stale.
pub async fn document_upsert_sync(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &NamedLocks,
    index_pool: &Arc<rayon::ThreadPool>,
    collection_name: &str,
    document: Document,
) -> Result<Vec<String>, UpsertSyncError> {
    document_upsert_bulk(db, cache, stats_batcher, doc_locks, index_pool, collection_name, vec![document]).await
}

// ---------------------------------------------------------------------------
// document_upsert_bulk — mirrors _document_upsert_bulk_internal (queue-free)
// ---------------------------------------------------------------------------

/// Returns `Ok(skipped_ids)` — IDs of documents that were stale and not indexed.
pub async fn document_upsert_bulk(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &NamedLocks,
    index_pool: &Arc<rayon::ThreadPool>,
    collection_name: &str,
    documents: Vec<Document>,
) -> Result<Vec<String>, UpsertSyncError> {
    if documents.is_empty() {
        return Ok(vec![]);
    }

    let collection_config = match get_collection_info_cached(db, cache, collection_name).await {
        Ok((c, _)) => c,
        Err(DbError::NotFound(_)) => {
            return Err(UpsertSyncError::NotFound(
                "Collection configuration not found".to_string(),
            ))
        }
        Err(e) => return Err(UpsertSyncError::Db(e)),
    };

    // Mirrors Python: validate_metadata_types(collection_config, document) per doc.
    for doc in &documents {
        validate_metadata_types(&collection_config, doc)
            .map_err(|e| UpsertSyncError::Vectorization(e.0))?;
    }

    // Acquire per-doc locks for all documents upfront (in stable order to avoid deadlock).
    let mut doc_ids: Vec<&str> = documents.iter().map(|d| d.id.as_str()).collect();
    doc_ids.sort_unstable();
    doc_ids.dedup();
    let mut guards = Vec::with_capacity(doc_ids.len());
    for id in &doc_ids {
        guards.push(doc_locks.lock(&format!("{collection_name}-{id}")).await);
    }

    // Batch fetch existing documents.
    let all_ids: Vec<&str> = documents.iter().map(|d| d.id.as_str()).collect();
    let existing_results = db.get_documents(collection_name, &all_ids, true).await?;
    let existing_map: HashMap<&str, &DocumentWithVectors> = all_ids
        .iter()
        .zip(existing_results.iter())
        .filter_map(|(id, opt)| opt.as_ref().map(|doc| (*id, doc)))
        .collect();

    // Partition into stale (skip) and to-process.
    let mut skipped: Vec<String> = vec![];
    let mut to_process: Vec<&Document> = vec![];
    let mut is_new_flags: Vec<bool> = vec![];

    for doc in &documents {
        match existing_map.get(doc.id.as_str()) {
            Some(existing) if doc.timestamp <= existing.timestamp => {
                skipped.push(doc.id.clone());
            }
            Some(_) => {
                to_process.push(doc);
                is_new_flags.push(false);
            }
            None => {
                to_process.push(doc);
                is_new_flags.push(true);
            }
        }
    }

    if to_process.is_empty() {
        return Ok(skipped);
    }

    // Build avgdl_dict with defaults for custom-tokenization fields.
    let mut stats = db.get_collection_stats(collection_name).await?;
    for config in &collection_config.vectors {
        if config.vector_type.is_custom_tokenization() {
            for field in &config.index_fields {
                let field_vector_name = format!("{}_{}", field, config.name);
                stats.avgdls.entry(field_vector_name).or_insert(50.0);
            }
        }
    }
    let avgdl_dict: HashMap<String, f64> = stats.avgdls.clone();

    let mut docs_owned: Vec<Document> = to_process.iter().map(|d| (*d).clone()).collect();
    for doc in &mut docs_owned {
        normalize_document_metadata_inplace(doc)
            .map_err(UpsertSyncError::Vectorization)?;
    }
    let vectors_cfg = collection_config.vectors.clone();
    let pool = Arc::clone(index_pool);
    
    let (tx, rx) = tokio::sync::oneshot::channel();
    
    pool.spawn(move || {
        let result = Vectorizer::vectorize_documents(&docs_owned, &vectors_cfg, Some(&avgdl_dict));
        let _ = tx.send(result);
    });

    let docs_with_vectors = rx
        .await
        .map_err(|e| UpsertSyncError::Vectorization(format!("Rayon channel dropped: {e}")))?
        .map_err(UpsertSyncError::Vectorization)?;

    db.add_documents(collection_name, &docs_with_vectors, collection_config.store_content)
        .await?;

    // Build stats updates across the whole batch.
    let new_doc_count_batch = is_new_flags.iter().filter(|&&n| n).count() as i64;
    let update_doc_count_batch = is_new_flags.iter().filter(|&&n| !n).count() as i64;

    let mut updates: HashMap<String, TokenLengthUpdate> = HashMap::new();
    for (doc_idx, doc_with_vectors) in docs_with_vectors.iter().enumerate() {
        let is_new = is_new_flags[doc_idx];
        let existing = existing_map.get(to_process[doc_idx].id.as_str());
        for (field_vector_name, &token_length) in &doc_with_vectors.token_lengths {
            let entry = updates.entry(field_vector_name.clone()).or_insert(TokenLengthUpdate {
                new_doc_count: new_doc_count_batch,
                new_sum_token_lengths: 0,
                update_doc_count: update_doc_count_batch,
                update_sum_token_lengths: 0,
                old_sum_token_lengths: 0,
            });
            if is_new {
                entry.new_sum_token_lengths += token_length as i64;
            } else {
                entry.update_sum_token_lengths += token_length as i64;
                if let Some(existing_doc) = existing {
                    if let Some(&old_len) = existing_doc.token_lengths.get(field_vector_name) {
                        entry.old_sum_token_lengths += old_len as i64;
                    }
                }
            }
        }
    }

    if !updates.is_empty() {
        stats_batcher.enqueue(collection_name, updates).await?;
    }

    Ok(skipped)
}

// ---------------------------------------------------------------------------
// document_delete_sync — mirrors encoder.py RpcService.document_delete_sync
// ---------------------------------------------------------------------------

/// Deletes a document and updates collection stats with negative token lengths.
/// Returns `Ok(())` silently if the document does not exist (already deleted).
pub async fn document_delete_sync(
    db: &QdrantDb,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &NamedLocks,
    collection_name: &str,
    document_id: &str,
) -> Result<(), UpsertSyncError> {
    let doc_lock_key = format!("{collection_name}-{document_id}");
    let _doc_guard = doc_locks.lock(&doc_lock_key).await;

    let existing = db
        .get_documents(collection_name, &[document_id], true)
        .await?
        .into_iter()
        .next()
        .flatten();

    let doc_with_vectors = match existing {
        Some(d) => d,
        None => return Ok(()),
    };

    match db.delete_document(collection_name, document_id).await {
        Ok(()) => {}
        Err(DbError::NotFound(_)) => return Ok(()),
        Err(e) => return Err(UpsertSyncError::Db(e)),
    }

    let mut updates: HashMap<String, TokenLengthUpdate> = HashMap::new();
    for (field_vector_name, &token_length) in &doc_with_vectors.token_lengths {
        updates.insert(field_vector_name.clone(), TokenLengthUpdate {
            new_doc_count: -1,
            new_sum_token_lengths: -(token_length as i64),
            update_doc_count: 0,
            update_sum_token_lengths: 0,
            old_sum_token_lengths: 0,
        });
    }

    if !updates.is_empty() {
        stats_batcher.enqueue(collection_name, updates).await?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// validate_models — mirrors encoder.py EncoderService.validate_models
// ---------------------------------------------------------------------------

/// Validates vector configs by running [`Vectorizer::vectorize_search_query`] with
/// `validation_mode=true` on a dummy query (`"x"`), then summarizing per config.
///
/// Runs vectorization on the blocking thread pool so the Tokio runtime is not stalled.
///
/// On vectorization failure, [`ModelValidationResponse.error`] carries the message for API callers.
pub async fn validate_models(vector_configs: Vec<VectorConfigInternal>) -> ModelValidationResponse {
    match tokio::task::spawn_blocking(move || validate_models_inner(&vector_configs)).await {
        Ok(r) => r,
        Err(e) => ModelValidationResponse {
            results: None,
            error: Some(format!("validate_models task: {e}")),
        },
    }
}

fn validate_models_inner(vector_configs: &[VectorConfigInternal]) -> ModelValidationResponse {
    let vector_weights: Vec<VectorSearchWeight> = vector_configs
        .iter()
        .flat_map(|config| {
            config
                .index_fields
                .iter()
                .copied()
                .map(move |field| VectorSearchWeight {
                    vector_name: config.name.clone(),
                    weight: 1.0,
                    field,
                })
        })
        .collect();

    let dummy_query = SearchQuery {
        query: "x".to_string(),
        vector_weights,
        custom_vectors: None,
        limit: DEFAULT_SEARCH_LIMIT,
        score_threshold: None,
        document_tags: None,
        document_tags_match_all: false,
        metadata_filter: None,
        raw_scores: false,
        wmtr_trigram_weight: DEFAULT_WMTR_TRIGRAM_WEIGHT,
        fusion_mode: "rrf".to_string(),
    };

    match Vectorizer::vectorize_search_query(dummy_query, vector_configs, true) {
        Ok(query_with_vectors) => {
            let mut results: HashMap<String, ModelValidationResult> = HashMap::new();

            for config in vector_configs {
                let result = match config.vector_type {
                    VectorType::DenseModel => {
                        let dense_vector = query_with_vectors
                            .vectors
                            .iter()
                            .find(|v| v.vector_name == config.name && v.dense_vector.is_some())
                            .and_then(|v| v.dense_vector.as_ref());

                        match dense_vector {
                            Some(dv) => ModelValidationResult {
                                valid: true,
                                dimension: u32::try_from(dv.len()).ok(),
                                error: None,
                            },
                            None => ModelValidationResult {
                                valid: false,
                                dimension: None,
                                error: Some("No dense vector generated".to_string()),
                            },
                        }
                    }
                    VectorType::SparseModel => ModelValidationResult {
                        valid: true,
                        dimension: None,
                        error: None,
                    },
                    _ => ModelValidationResult {
                        valid: true,
                        dimension: None,
                        error: None,
                    },
                };

                results.insert(config.name.clone(), result);
            }

            ModelValidationResponse {
                results: Some(results),
                error: None,
            }
        }
        Err(e) => ModelValidationResponse {
            results: None,
            error: Some(e),
        },
    }
}

// ---------------------------------------------------------------------------
// validate_metadata_types — mirrors database/common.py validate_metadata_types
// ---------------------------------------------------------------------------

pub struct MetadataTypeError(pub String);

/// Mirrors `validate_metadata_types` from `database/common.py`.
/// Validates that document metadata value types match the types declared in collection config.
pub fn validate_metadata_types(
    collection_config: &CollectionConfigInternal,
    document: &Document,
) -> Result<(), MetadataTypeError> {
    let indexes = match collection_config.metadata_indexes.as_deref() {
        Some(idx) if !idx.is_empty() => idx,
        _ => return Ok(()),
    };

    let metadata = match document.metadata.as_ref() {
        Some(m) if !m.is_empty() => m,
        _ => return Ok(()),
    };

    for idx in indexes {
        if let Some(value) = metadata.get(&idx.key) {
            let actual_type = infer_metadata_value_type(value);
            if actual_type != idx.value_type {
                return Err(MetadataTypeError(format!(
                    "Metadata key '{}' has type '{}' but collection config expects type '{}'",
                    idx.key, actual_type, idx.value_type
                )));
            }
        }
    }

    Ok(())
}

fn infer_metadata_value_type(value: &serde_json::Value) -> String {
    // Handle both raw primitives and MetaValue dict form {"value": ..., "type": "..."}
    if let Some(map) = value.as_object() {
        if let Some(type_val) = map.get("type").and_then(|t| t.as_str()) {
            return type_val.to_string();
        }
    }
    match value {
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Bool(_) => "boolean".to_string(),
        serde_json::Value::Number(n) => {
            if n.is_i64() || n.is_u64() { "integer".to_string() } else { "float".to_string() }
        }
        _ => "unknown".to_string(),
    }
}

// ---------------------------------------------------------------------------
// validate_metadata_filter — mirrors database/common.py validate_metadata_filter
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MetadataFilterError(pub String);

impl std::fmt::Display for MetadataFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn is_iso_datetime_string(s: &str) -> bool {
    // Mirrors Python: datetime.fromisoformat(value.replace("Z", "+00:00"))
    let normalized = s.replace('Z', "+00:00");
    chrono::DateTime::parse_from_rfc3339(&normalized).is_ok()
        || chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S%.f").is_ok()
        || chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S").is_ok()
        || chrono::NaiveDate::parse_from_str(&normalized, "%Y-%m-%d").is_ok()
}

fn validate_filter_node(
    filter: &crate::models::MetadataFilter,
    indexed_types: &HashMap<&str, &str>,
) -> Result<(), MetadataFilterError> {
    if let Some(ref key) = filter.key {
        let expected_type = indexed_types.get(key.as_str()).ok_or_else(|| {
            MetadataFilterError(format!(
                "Metadata filter key '{key}' is not indexed in collection metadata_indexes"
            ))
        })?;

        let op = filter.op.as_deref().unwrap_or("");
        let value = &filter.value;

        match *expected_type {
            "string" => {
                if op != "eq" {
                    return Err(MetadataFilterError(format!(
                        "Metadata filter operator '{op}' is not supported for string key '{key}'. Use 'eq'."
                    )));
                }
                match value {
                    Some(serde_json::Value::String(_)) => {}
                    _ => return Err(MetadataFilterError(format!(
                        "Metadata filter value for key '{key}' must be a string"
                    ))),
                }
            }
            "integer" => {
                match value {
                    Some(serde_json::Value::Number(n)) if n.is_i64() || n.is_u64() => {}
                    _ => return Err(MetadataFilterError(format!(
                        "Metadata filter value for key '{key}' must be an integer"
                    ))),
                }
            }
            "float" => {
                match value {
                    Some(serde_json::Value::Number(_)) => {}
                    _ => return Err(MetadataFilterError(format!(
                        "Metadata filter value for key '{key}' must be a number"
                    ))),
                }
            }
            "boolean" => {
                if op != "eq" {
                    return Err(MetadataFilterError(format!(
                        "Metadata filter operator '{op}' is not supported for boolean key '{key}'. Use 'eq'."
                    )));
                }
                match value {
                    Some(serde_json::Value::Bool(_)) => {}
                    _ => return Err(MetadataFilterError(format!(
                        "Metadata filter value for key '{key}' must be a boolean"
                    ))),
                }
            }
            "datetime" => {
                match value {
                    Some(serde_json::Value::String(s)) if is_iso_datetime_string(s) => {}
                    _ => return Err(MetadataFilterError(format!(
                        "Metadata filter value for key '{key}' must be an ISO datetime string"
                    ))),
                }
            }
            other => {
                return Err(MetadataFilterError(format!(
                    "Unknown metadata index type '{other}' for key '{key}'"
                )));
            }
        }
    }

    if let Some(ref and_) = filter.and_ {
        for child in and_ {
            validate_filter_node(child, indexed_types)?;
        }
    }
    if let Some(ref or_) = filter.or_ {
        for child in or_ {
            validate_filter_node(child, indexed_types)?;
        }
    }
    if let Some(ref not_) = filter.not_ {
        validate_filter_node(not_, indexed_types)?;
    }

    Ok(())
}

/// Mirrors `validate_metadata_filter` from `database/common.py`.
pub fn validate_metadata_filter(
    collection_config: &CollectionConfigInternal,
    filter: &crate::models::MetadataFilter,
) -> Result<(), MetadataFilterError> {
    let indexes = collection_config.metadata_indexes.as_deref().unwrap_or(&[]);
    if indexes.is_empty() {
        return Err(MetadataFilterError(
            "Collection has no metadata_indexes defined. Cannot filter on metadata.".to_string(),
        ));
    }
    let indexed_types: HashMap<&str, &str> =
        indexes.iter().map(|idx| (idx.key.as_str(), idx.value_type.as_str())).collect();
    validate_filter_node(filter, &indexed_types)
}

// ---------------------------------------------------------------------------
// search — mirrors encoder.py RpcService.search (cache + vectorize + db.search)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SearchError {
    NotFound(String),
    InvalidFilter(String),
    Vectorization(String),
    Db(DbError),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::NotFound(m) => write!(f, "{m}"),
            SearchError::InvalidFilter(m) => write!(f, "{m}"),
            SearchError::Vectorization(m) => write!(f, "{m}"),
            SearchError::Db(e) => write!(f, "{e}"),
        }
    }
}

impl From<DbError> for SearchError {
    fn from(e: DbError) -> Self {
        SearchError::Db(e)
    }
}

async fn vectorize_search_query_blocking(
    collection_config: CollectionConfigInternal,
    query: SearchQuery,
    search_pool: Arc<rayon::ThreadPool>,
) -> Result<SearchQueryWithVectors, SearchError> {
    if let Some(ref filter) = query.metadata_filter {
        validate_metadata_filter(&collection_config, filter)
            .map_err(|e| SearchError::InvalidFilter(e.0))?;
    }

    let (tx, rx) = tokio::sync::oneshot::channel();
    
    search_pool.spawn(move || {
        let result = Vectorizer::vectorize_search_query(query, &collection_config.vectors, false)
            .map_err(SearchError::Vectorization);
        let _ = tx.send(result);
    });

    rx.await
        .map_err(|e| SearchError::Vectorization(format!("Rayon channel dropped: {e}")))?
}

pub async fn search(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    search_pool: &Arc<rayon::ThreadPool>,
    collection_name: &str,
    query: crate::models::SearchQuery,
) -> Result<Vec<crate::models::SearchResult>, SearchError> {
    let (collection_config, from_cache) =
        get_collection_info_cached(db, cache, collection_name).await.map_err(|e| {
            if matches!(e, DbError::NotFound(_)) {
                SearchError::NotFound("Collection configuration not found".to_string())
            } else {
                SearchError::Db(e)
            }
        })?;

    let query_with_vectors = match vectorize_search_query_blocking(
        collection_config.clone(),
        query.clone(),
        Arc::clone(search_pool),
    )
    .await
    {
        Ok(q) => q,
        Err(e) if from_cache => {
            // Retry once with fresh config — mirrors Python cache-invalidate-retry.
            cache.invalidate(collection_name).await;
            let (fresh_config, _) =
                get_collection_info_cached(db, cache, collection_name).await.map_err(|e| {
                    if matches!(e, DbError::NotFound(_)) {
                        SearchError::NotFound("Collection configuration not found".to_string())
                    } else {
                        SearchError::Db(e)
                    }
                })?;
            vectorize_search_query_blocking(fresh_config, query, Arc::clone(search_pool))
                .await
                .map_err(|_| e)?
        }
        Err(e) => return Err(e),
    };

    db.search(collection_name, &query_with_vectors, &collection_config)
        .await
        .map_err(SearchError::Db)
}