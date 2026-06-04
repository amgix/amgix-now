//! Encoder-layer logic — mirrors `src/encoder/encoder.py` (`update_collection_stats`,
//! `validate_models`, `document_upsert_sync`). Ingress uses [`UpsertIngress`]: REST bulk hits the
//! bulk channel (immediate internal); singles use a separate channel drained into micro-batches
//! with bounded concurrency ([`SINGLE_UPSERT_CONCURRENT_MICROBATCH_MAX`] pipelines in flight).
//! Search uses [`SearchIngress`]: a separate bounded queue and the same micro-batch / concurrent
//! pipeline pattern ([`SEARCH_INGRESS_*`] constants).

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, oneshot, Mutex, OwnedMutexGuard};
use tokio::task::JoinSet;

use crate::common::{
    DocumentField, VectorType, DEFAULT_SEARCH_LIMIT, DEFAULT_WMTR_TRIGRAM_WEIGHT,
    MAX_INDEXED_METADATA_VALUE_LENGTH,
};
use crate::functions::normalize_document_metadata_inplace;
use crate::models::{
    CollectionConfigInternal, Document, DocumentWithVectors, ModelValidationResponse,
    ModelValidationResult, SearchQuery, SearchQuerySettings, SearchResult, VectorConfigInternal,
    VectorSearchWeight,
};

fn needs_revectorization(
    incoming: &Document,
    existing: &DocumentWithVectors,
    collection_config: &CollectionConfigInternal,
) -> bool {
    if incoming.custom_vectors.is_some() {
        return true;
    }
    let content_indexed = collection_config
        .vectors
        .iter()
        .any(|v| v.index_fields.contains(&DocumentField::Content));
    if content_indexed && !collection_config.store_content {
        return true;
    }
    for v in &collection_config.vectors {
        for field in &v.index_fields {
            let incoming_val: Option<&str> = match field {
                DocumentField::Name => incoming.name.as_deref(),
                DocumentField::Description => incoming.description.as_deref(),
                DocumentField::Content => incoming.content.as_deref(),
            };
            let existing_val: Option<&str> = match field {
                DocumentField::Name => existing.name.as_deref(),
                DocumentField::Description => existing.description.as_deref(),
                DocumentField::Content => existing.content.as_deref(),
            };
            if incoming_val != existing_val {
                return true;
            }
        }
    }
    false
}
use crate::qdrant::{CollectionStats, DbError, QdrantDb};
use crate::vectors::vectorizer::Vectorizer;

// ---------------------------------------------------------------------------
// Constants — cache, stats batching, upsert ingress, search ingress
// ---------------------------------------------------------------------------

const CACHE_TTL_SECS: u64 = 60;
const CACHE_MAX_ENTRIES: usize = 1000;

const STATS_BATCH_MAX_JOBS: usize = 10;
const STATS_BATCH_WAIT: Duration = Duration::from_millis(200);
const STATS_BATCH_CHANNEL: usize = 1024;
const STATS_SET_STATS_MAX_ATTEMPTS: usize = 3;
const STATS_SET_STATS_RETRY_DELAY: Duration = Duration::from_millis(50);
const ADD_DOCUMENTS_MAX_ATTEMPTS: usize = 3;
const ADD_DOCUMENTS_RETRY_DELAY: Duration = Duration::from_millis(50);
const NAMED_LOCKS_CLEANUP_INTERVAL: Duration = Duration::from_secs(60);
const STATS_LOCKS_CLEANUP_INTERVAL: Duration = Duration::from_secs(600);

/// Max REST **bulk** jobs buffered (`try_send` → HTTP 429 when full).
pub const BULK_UPSERT_QUEUE_CAPACITY: usize = 128;
/// Max bulk jobs processed concurrently.
pub const BULK_UPSERT_CONCURRENT_MAX: usize = 4;
/// Bulk batches are split into chunks of this size before vectorization so each chunk
/// runs on its own rayon thread concurrently with other chunks.
pub const BULK_UPSERT_CHUNK_SIZE: usize = 32;
/// Max single-doc micro-batches pulled off the singles ingress channel **in flight at once**.
/// One task still `recv`s and builds batches; this bounds overlapping `document_upsert_bulk_internal`
/// work (per parallel micro-batch pipeline).
pub const SINGLE_UPSERT_CONCURRENT_MICROBATCH_MAX: usize = 4;
/// Max **single-document** ingress jobs buffered.
pub const SINGLE_UPSERT_QUEUE_CAPACITY: usize = 10240;
/// Max single-doc messages pulled into one micro-batch (including first `recv`).
pub const SINGLE_UPSERT_MICROBATCH_DRAIN_MAX: usize = 32;

/// Max search requests buffered (`try_send` → HTTP 429 when full).
pub const SEARCH_INGRESS_QUEUE_CAPACITY: usize = 10240;
/// Max search micro-batch pipelines in flight at once.
pub const SEARCH_INGRESS_CONCURRENT_MICROBATCH_MAX: usize = 8; // WMTR 32
/// Max search jobs coalesced into one micro-batch (including the first `recv`).
pub const SEARCH_INGRESS_MICROBATCH_DRAIN_MAX: usize = 64; // WMTR 8

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

    /// Spawns a background task that periodically removes idle lock entries (those with no
    /// waiters or holders). The task runs until aborted or the process exits — no graceful
    /// shutdown is needed since there is nothing to drain.
    pub fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        self.start_cleanup_task_with_interval(NAMED_LOCKS_CLEANUP_INTERVAL)
    }

    /// Like [`start_cleanup_task`] but uses the longer interval suitable for stats locks,
    /// which are keyed by collection name and therefore much fewer than doc locks.
    pub fn start_stats_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        self.start_cleanup_task_with_interval(STATS_LOCKS_CLEANUP_INTERVAL)
    }

    fn start_cleanup_task_with_interval(&self, interval: Duration) -> tokio::task::JoinHandle<()> {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let mut map = inner.lock().await;
                map.retain(|_, entry| Arc::strong_count(entry) > 1);
            }
        })
    }
}

// ---------------------------------------------------------------------------
// CollectionConfigCache — TTL cache for collection configs.
//
// Mirrors EncoderBase._collection_info_cache: AMGIXCache(ttl=60, maxsize=1000).
// Keyed by collection_name. Entries expire after [`CACHE_TTL_SECS`] seconds.
// On overflow (> [`CACHE_MAX_ENTRIES`]), the oldest entry is evicted.
// ---------------------------------------------------------------------------

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
    for attempt in 1..=STATS_SET_STATS_MAX_ATTEMPTS {
        match db.set_collection_stats(collection_name, &stats).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < STATS_SET_STATS_MAX_ATTEMPTS => {
                tracing::warn!(
                    attempt,
                    max_attempts = STATS_SET_STATS_MAX_ATTEMPTS,
                    error = %e,
                    "set_collection_stats failed; retrying"
                );
                tokio::time::sleep(STATS_SET_STATS_RETRY_DELAY).await;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

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
            tracing::error!(error = %e, "stats batcher task ended with error");
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
            tracing::error!(
                collection = %collection_name,
                error = %e,
                "stats batch persist failed after retries"
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
    /// Bounded ingress channel has no buffer space (`try_send`) — clients should retry with backoff.
    IngressQueueFull(String),
    /// Reply channel dropped (typically ingress worker exited).
    IngressWorkerExited(String),
}

impl std::fmt::Display for UpsertSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpsertSyncError::NotFound(m) => write!(f, "{m}"),
            UpsertSyncError::Db(e) => write!(f, "{e}"),
            UpsertSyncError::Vectorization(m) => write!(f, "{m}"),
            UpsertSyncError::IngressQueueFull(m) => write!(f, "{m}"),
            UpsertSyncError::IngressWorkerExited(m) => write!(f, "{m}"),
        }
    }
}

impl From<DbError> for UpsertSyncError {
    fn from(e: DbError) -> Self {
        UpsertSyncError::Db(e)
    }
}

struct BulkIngressJob {
    collection_name: String,
    documents: Vec<Document>,
    reply: oneshot::Sender<Result<Vec<String>, UpsertSyncError>>,
}

struct SingleIngressJob {
    collection_name: String,
    document: Document,
    reply: Option<oneshot::Sender<Result<Vec<String>, UpsertSyncError>>>,
}

fn send_single_ingress_reply(
    job: &mut SingleIngressJob,
    msg: Result<Vec<String>, UpsertSyncError>,
) {
    if let Some(tx) = job.reply.take() {
        let _ = tx.send(msg);
    }
}

#[derive(Clone)]
pub struct UpsertIngress {
    bulk_tx: mpsc::Sender<BulkIngressJob>,
    singles_tx: mpsc::Sender<SingleIngressJob>,
}

/// Await [**`shutdown_and_wait`**](Self::shutdown_and_wait) **after** [`axum::serve`] returns so
/// all clones of [`UpsertIngress`] are dropped, ingress [`mpsc`] channels close, workers drain any
/// remaining jobs, then exit. Prefer **before** [`StatsBatcherShutdown::shutdown_and_wait`] so
/// ingestion can enqueue stats updates while drains run.
pub struct UpsertIngressShutdown {
    bulk: tokio::task::JoinHandle<()>,
    singles: tokio::task::JoinHandle<()>,
}

impl UpsertIngressShutdown {
    pub async fn shutdown_and_wait(self) {
        let (bulk, singles) = tokio::join!(self.bulk, self.singles);
        if let Err(e) = bulk {
            tracing::warn!("bulk upsert ingress worker join: {e}");
        }
        if let Err(e) = singles {
            tracing::warn!("single upsert ingress worker join: {e}");
        }
    }
}

fn replicate_upsert_sync_err(src: &UpsertSyncError) -> UpsertSyncError {
    match src {
        UpsertSyncError::NotFound(s) => UpsertSyncError::NotFound(s.clone()),
        UpsertSyncError::Db(e) => UpsertSyncError::Db(DbError::Config(e.to_string())),
        UpsertSyncError::Vectorization(s) => UpsertSyncError::Vectorization(s.clone()),
        UpsertSyncError::IngressQueueFull(s) => UpsertSyncError::IngressQueueFull(s.clone()),
        UpsertSyncError::IngressWorkerExited(s) => UpsertSyncError::IngressWorkerExited(s.clone()),
    }
}

impl UpsertIngress {
    pub fn new(
        db: Arc<QdrantDb>,
        cache: CollectionConfigCache,
        stats_batcher: StatsUpdateBatcher,
        doc_locks: NamedLocks,
        index_pool: Arc<rayon::ThreadPool>,
    ) -> (Self, UpsertIngressShutdown) {
        let (bulk_tx, mut bulk_rx) = mpsc::channel::<BulkIngressJob>(BULK_UPSERT_QUEUE_CAPACITY);
        let bulk = {
            let db = Arc::clone(&db);
            let cache = cache.clone();
            let stats_batcher = stats_batcher.clone();
            let doc_locks = doc_locks.clone();
            let index_pool = Arc::clone(&index_pool);
            tokio::spawn(async move {
                let mut inflight = JoinSet::new();
                loop {
                    while inflight.len() >= BULK_UPSERT_CONCURRENT_MAX {
                        if let Some(res) = inflight.join_next().await {
                            if let Err(e) = res {
                                tracing::warn!("bulk upsert task panicked: {e}");
                            }
                        }
                    }
                    match bulk_rx.recv().await {
                        None => break,
                        Some(BulkIngressJob {
                            collection_name,
                            documents,
                            reply,
                        }) => {
                            let db = Arc::clone(&db);
                            let cache = cache.clone();
                            let stats_batcher = stats_batcher.clone();
                            let doc_locks = doc_locks.clone();
                            let index_pool = Arc::clone(&index_pool);
                            inflight.spawn(async move {
                                // Split into chunks and run each concurrently so vectorization
                                // spreads across rayon threads rather than running serially.
                                let chunks: Vec<Vec<Document>> = documents
                                    .chunks(BULK_UPSERT_CHUNK_SIZE)
                                    .map(|c| c.to_vec())
                                    .collect();
                                let mut chunk_set = JoinSet::new();
                                for chunk in chunks {
                                    let db = Arc::clone(&db);
                                    let cache = cache.clone();
                                    let stats_batcher = stats_batcher.clone();
                                    let doc_locks = doc_locks.clone();
                                    let index_pool = Arc::clone(&index_pool);
                                    let collection_name = collection_name.clone();
                                    chunk_set.spawn(async move {
                                        document_upsert_bulk_internal(
                                            &db,
                                            &cache,
                                            &stats_batcher,
                                            &doc_locks,
                                            &index_pool,
                                            &collection_name,
                                            chunk,
                                        )
                                        .await
                                    });
                                }
                                // Collect all chunk results; fail fast on first error.
                                let mut all_skipped: Vec<String> = Vec::new();
                                let mut first_err: Option<UpsertSyncError> = None;
                                while let Some(res) = chunk_set.join_next().await {
                                    match res {
                                        Ok(Ok(skipped)) => all_skipped.extend(skipped),
                                        Ok(Err(e)) => {
                                            if first_err.is_none() {
                                                first_err = Some(e);
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!("bulk chunk task panicked: {e}");
                                        }
                                    }
                                }
                                let out = match first_err {
                                    Some(e) => Err(e),
                                    None => Ok(all_skipped),
                                };
                                let _ = reply.send(out);
                            });
                        }
                    }
                }
                while let Some(res) = inflight.join_next().await {
                    if let Err(e) = res {
                        tracing::warn!("bulk upsert task panicked: {e}");
                    }
                }
            })
        };

        let (singles_tx, mut singles_rx) =
            mpsc::channel::<SingleIngressJob>(SINGLE_UPSERT_QUEUE_CAPACITY);
        let singles = {
            let db = Arc::clone(&db);
            let cache = cache.clone();
            let stats_batcher = stats_batcher.clone();
            let doc_locks = doc_locks.clone();
            let index_pool = Arc::clone(&index_pool);
            tokio::spawn(async move {
                let mut inflight = JoinSet::new();
                loop {
                    while inflight.len() >= SINGLE_UPSERT_CONCURRENT_MICROBATCH_MAX {
                        if let Some(res) = inflight.join_next().await {
                            if let Err(e) = res {
                                tracing::warn!("single-document micro-batch task panicked: {e}");
                            }
                        }
                    }
                    let first = match singles_rx.recv().await {
                        None => break,
                        Some(f) => f,
                    };
                    let mut batch = vec![first];
                    while batch.len() < SINGLE_UPSERT_MICROBATCH_DRAIN_MAX {
                        match singles_rx.try_recv() {
                            Ok(j) => batch.push(j),
                            Err(_) => break,
                        }
                    }
                    let db = Arc::clone(&db);
                    let cache = cache.clone();
                    let stats_batcher = stats_batcher.clone();
                    let doc_locks = doc_locks.clone();
                    let index_pool = Arc::clone(&index_pool);
                    inflight.spawn(async move {
                        process_single_document_microbatch(
                            batch,
                            &db,
                            &cache,
                            &stats_batcher,
                            &doc_locks,
                            &index_pool,
                        )
                        .await;
                    });
                }
                while let Some(res) = inflight.join_next().await {
                    if let Err(e) = res {
                        tracing::warn!("single-document micro-batch task panicked: {e}");
                    }
                }
            })
        };

        (
            Self {
                bulk_tx,
                singles_tx,
            },
            UpsertIngressShutdown { bulk, singles },
        )
    }
}

// ---------------------------------------------------------------------------
// SearchIngress — bounded queue, micro-batch drain, concurrent pipelines (like singles upsert)
// ---------------------------------------------------------------------------

struct SearchIngressJob {
    collection_name: String,
    query: SearchQuery,
    reply: oneshot::Sender<Result<Vec<SearchResult>, SearchError>>,
}

#[derive(Clone)]
pub struct SearchIngress {
    tx: mpsc::Sender<SearchIngressJob>,
}

pub struct SearchIngressShutdown {
    worker: tokio::task::JoinHandle<()>,
}

impl SearchIngressShutdown {
    pub async fn shutdown_and_wait(self) {
        if let Err(e) = self.worker.await {
            tracing::warn!("search ingress worker join: {e}");
        }
    }
}

impl SearchIngress {
    pub fn new(
        db: Arc<QdrantDb>,
        cache: CollectionConfigCache,
        search_pool: Arc<rayon::ThreadPool>,
    ) -> (Self, SearchIngressShutdown) {
        let (tx, mut rx) = mpsc::channel::<SearchIngressJob>(SEARCH_INGRESS_QUEUE_CAPACITY);
        let worker = tokio::spawn(async move {
            let mut inflight = JoinSet::new();
            loop {
                while inflight.len() >= SEARCH_INGRESS_CONCURRENT_MICROBATCH_MAX {
                    if let Some(res) = inflight.join_next().await {
                        if let Err(e) = res {
                            tracing::warn!("search micro-batch task panicked: {e}");
                        }
                    }
                }
                let first = match rx.recv().await {
                    None => break,
                    Some(j) => j,
                };
                let mut batch = vec![first];
                while batch.len() < SEARCH_INGRESS_MICROBATCH_DRAIN_MAX {
                    match rx.try_recv() {
                        Ok(j) => batch.push(j),
                        Err(_) => break,
                    }
                }
                let db = Arc::clone(&db);
                let cache = cache.clone();
                let search_pool = Arc::clone(&search_pool);
                inflight.spawn(async move {
                    process_search_microbatch(batch, db, cache, search_pool).await;
                });
            }
            while let Some(res) = inflight.join_next().await {
                if let Err(e) = res {
                    tracing::warn!("search micro-batch task panicked: {e}");
                }
            }
        });

        (Self { tx }, SearchIngressShutdown { worker })
    }

    pub async fn search(
        &self,
        collection_name: String,
        query: SearchQuery,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .try_send(SearchIngressJob {
                collection_name,
                query,
                reply: reply_tx,
            })
            .map_err(|_| {
                SearchError::IngressQueueFull(format!(
                    "Search ingress queue full (capacity {SEARCH_INGRESS_QUEUE_CAPACITY}); retry later."
                ))
            })?;
        reply_rx.await.map_err(|_| {
            SearchError::IngressWorkerExited("Search ingress worker stopped.".into())
        })?
    }
}

async fn process_search_microbatch(
    jobs: Vec<SearchIngressJob>,
    db: Arc<QdrantDb>,
    cache: CollectionConfigCache,
    search_pool: Arc<rayon::ThreadPool>,
) {
    let mut by_key: HashMap<(String, SearchQuerySettings), Vec<SearchIngressJob>> = HashMap::new();
    for j in jobs {
        let key = (j.collection_name.clone(), j.query.settings.clone());
        by_key.entry(key).or_default().push(j);
    }

    for ((collection_name, _settings_key), bucket_jobs) in by_key {
        run_search_bucket(
            &collection_name,
            bucket_jobs,
            &db,
            &cache,
            &search_pool,
        )
        .await;
    }
}

async fn run_search_bucket(
    collection_name: &str,
    bucket_jobs: Vec<SearchIngressJob>,
    db: &Arc<QdrantDb>,
    cache: &CollectionConfigCache,
    search_pool: &Arc<rayon::ThreadPool>,
) {
    let (mut collection_config, from_cache) =
        match get_collection_info_cached(db, cache, collection_name).await {
            Ok(x) => x,
            Err(e) => {
                let err = match e {
                    DbError::NotFound(_) => SearchError::NotFound(
                        "Collection configuration not found".to_string(),
                    ),
                    e => SearchError::Db(e),
                };
                for j in bucket_jobs {
                    let _ = j.reply.send(Err(err.clone()));
                }
                return;
            }
        };

    let metadata_filter = bucket_jobs
        .first()
        .and_then(|j| j.query.settings.metadata_filter.as_ref());
    if let Some(filter) = metadata_filter {
        if let Err(e) = validate_metadata_filter(&collection_config, filter) {
            let msg = e.0.clone();
            for j in bucket_jobs {
                let _ = j.reply.send(Err(SearchError::InvalidFilter(msg.clone())));
            }
            return;
        }
    }

    let query_texts: Vec<String> =
        bucket_jobs.iter().map(|j| j.query.query.clone()).collect();
    let settings0 = bucket_jobs[0].query.settings.clone();

    let spawn_vectorize = |cfg: CollectionConfigInternal,
                           qt: Vec<String>,
                           sett: SearchQuerySettings,
                           pool: Arc<rayon::ThreadPool>| {
        let vecs = cfg.vectors.clone();
        let (tx, rx) = oneshot::channel();
        pool.spawn(move || {
            let mut settings = sett;
            let r = Vectorizer::vectorize_search_queries(&qt, &mut settings, &vecs, false);
            let _ = tx.send(r);
        });
        rx
    };

    let pool = Arc::clone(search_pool);
    let rx = spawn_vectorize(
        collection_config.clone(),
        query_texts.clone(),
        settings0.clone(),
        Arc::clone(&pool),
    );
    let mut first = match rx.await {
        Ok(r) => r,
        Err(e) => {
            let err = SearchError::Vectorization(format!("Search pool channel dropped: {e}"));
            for j in bucket_jobs {
                let _ = j.reply.send(Err(err.clone()));
            }
            return;
        }
    };

    if first.is_err() && from_cache {
        cache.invalidate(collection_name).await;
        if let Ok((fresh, _)) = get_collection_info_cached(db, cache, collection_name).await {
            collection_config = fresh;
            let rx = spawn_vectorize(
                collection_config.clone(),
                query_texts,
                settings0,
                pool,
            );
            first = match rx.await {
                Ok(r) => r,
                Err(e) => {
                    let err = SearchError::Vectorization(format!("Search pool channel dropped: {e}"));
                    for j in bucket_jobs {
                        let _ = j.reply.send(Err(err.clone()));
                    }
                    return;
                }
            };
        }
    }

    let vectorized = match first {
        Ok(v) => v,
        Err(msg) => {
            for j in bucket_jobs {
                let _ = j.reply.send(Err(SearchError::Vectorization(msg.clone())));
            }
            return;
        }
    };

    if vectorized.len() != bucket_jobs.len() {
        let msg = "internal: vectorized batch length mismatch".to_string();
        for j in bucket_jobs {
            let _ = j.reply.send(Err(SearchError::Vectorization(msg.clone())));
        }
        return;
    }

    for (j, qv) in bucket_jobs.into_iter().zip(vectorized.into_iter()) {
        let res = db
            .search(collection_name, &qv, &collection_config)
            .await
            .map_err(SearchError::Db);
        let _ = j.reply.send(res);
    }
}

async fn process_single_document_microbatch(
    jobs: Vec<SingleIngressJob>,
    db: &Arc<QdrantDb>,
    cache: &CollectionConfigCache,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &NamedLocks,
    index_pool: &Arc<rayon::ThreadPool>,
) {
    let mut by_collection: HashMap<String, Vec<SingleIngressJob>> = HashMap::new();
    for j in jobs {
        by_collection
            .entry(j.collection_name.clone())
            .or_default()
            .push(j);
    }

    for (collection_name, slots) in by_collection {
        respond_single_microbatch_for_collection(
            &collection_name,
            slots,
            db,
            cache,
            stats_batcher,
            doc_locks,
            index_pool,
        )
        .await;
    }
}

async fn respond_single_microbatch_for_collection(
    collection_name: &str,
    mut slots: Vec<SingleIngressJob>,
    db: &Arc<QdrantDb>,
    cache: &CollectionConfigCache,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &NamedLocks,
    index_pool: &Arc<rayon::ThreadPool>,
) {
    let n = slots.len();
    let mut answered = vec![false; n];
    let mut winner_idx_for_id: HashMap<String, usize> = HashMap::new();

    for idx in 0..n {
        let id = slots[idx].document.id.clone();
        let ts = slots[idx].document.timestamp;
        match winner_idx_for_id.entry(id.clone()) {
            Entry::Vacant(v) => {
                v.insert(idx);
            }
            Entry::Occupied(mut o) => {
                let bi = *o.get();
                let bts = slots[bi].document.timestamp;
                match ts.cmp(&bts) {
                    Ordering::Greater => {
                        let doc_id = slots[bi].document.id.clone();
                        send_single_ingress_reply(&mut slots[bi], Ok(vec![doc_id]));
                        answered[bi] = true;
                        *o.get_mut() = idx;
                    }
                    Ordering::Less => {
                        let doc_id = slots[idx].document.id.clone();
                        send_single_ingress_reply(&mut slots[idx], Ok(vec![doc_id]));
                        answered[idx] = true;
                    }
                    Ordering::Equal => {
                        let doc_id = slots[idx].document.id.clone();
                        send_single_ingress_reply(&mut slots[idx], Ok(vec![doc_id]));
                        answered[idx] = true;
                    }
                }
            }
        }
    }

    let mut uniq_winner_idx: HashSet<usize> = HashSet::with_capacity(winner_idx_for_id.len());
    for &wi in winner_idx_for_id.values() {
        uniq_winner_idx.insert(wi);
    }
    let merged: Vec<Document> = uniq_winner_idx
        .into_iter()
        .map(|i| slots[i].document.clone())
        .collect();

    if merged.is_empty() {
        return;
    }

    match document_upsert_bulk_internal(
        db,
        cache,
        stats_batcher,
        doc_locks,
        index_pool,
        collection_name,
        merged,
    )
    .await
    {
        Ok(skipped) => {
            let skipped_set: HashSet<String> = skipped.into_iter().collect();
            for idx in 0..n {
                if answered[idx] {
                    continue;
                }
                let doc_id = slots[idx].document.id.clone();
                let Some(win) = winner_idx_for_id.get(&doc_id).copied() else {
                    continue;
                };
                if win != idx {
                    continue;
                }
                if skipped_set.contains(&doc_id) {
                    send_single_ingress_reply(&mut slots[idx], Ok(vec![doc_id]));
                } else {
                    send_single_ingress_reply(&mut slots[idx], Ok(vec![]));
                }
            }
        }
        Err(e) => {
            for idx in 0..n {
                if answered[idx] {
                    continue;
                }
                send_single_ingress_reply(
                    &mut slots[idx],
                    Err(replicate_upsert_sync_err(&e)),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// document_upsert_sync — single-document path (micro-batch singles queue).
// ---------------------------------------------------------------------------

/// Returns `Ok(skipped_ids)` — non-empty when the document was stale.
pub async fn document_upsert_sync(
    ingress: &UpsertIngress,
    collection_name: &str,
    document: Document,
) -> Result<Vec<String>, UpsertSyncError> {
    let (reply_tx, reply_rx) = oneshot::channel();
    ingress
        .singles_tx
        .try_send(SingleIngressJob {
            collection_name: collection_name.to_string(),
            document,
            reply: Some(reply_tx),
        })
        .map_err(|_| {
            UpsertSyncError::IngressQueueFull(format!(
                "Single-document ingress queue full (capacity {SINGLE_UPSERT_QUEUE_CAPACITY}); retry later.",
            ))
        })?;
    reply_rx
        .await
        .map_err(|_| UpsertSyncError::IngressWorkerExited("Single-document ingress worker stopped.".into()))?
}

// ---------------------------------------------------------------------------
// document_upsert_bulk — REST bulk path (immediate per message).
// ---------------------------------------------------------------------------

/// Returns `Ok(skipped_ids)` — IDs of documents that were stale and not indexed.
pub async fn document_upsert_bulk(
    ingress: &UpsertIngress,
    collection_name: &str,
    documents: Vec<Document>,
) -> Result<Vec<String>, UpsertSyncError> {
    if documents.is_empty() {
        return Ok(vec![]);
    }
    let (reply_tx, reply_rx) = oneshot::channel();
    ingress
        .bulk_tx
        .try_send(BulkIngressJob {
            collection_name: collection_name.to_string(),
            documents,
            reply: reply_tx,
        })
        .map_err(|_| {
            UpsertSyncError::IngressQueueFull(format!(
                "Bulk ingest queue full (capacity {BULK_UPSERT_QUEUE_CAPACITY}); retry later.",
            ))
        })?;
    reply_rx
        .await
        .map_err(|_| UpsertSyncError::IngressWorkerExited("Bulk ingest worker stopped.".into()))?
}

// ---------------------------------------------------------------------------
// document_upsert_bulk_internal
// ---------------------------------------------------------------------------

/// Returns `Ok(skipped_ids)` — IDs of documents that were stale and not indexed.
pub(crate) async fn document_upsert_bulk_internal(
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

    // Partition into stale (skip), patch-only, and to-vectorize.
    let mut skipped: Vec<String> = vec![];
    let mut to_vectorize: Vec<&Document> = vec![];
    let mut to_patch: Vec<&Document> = vec![];
    let mut is_new_flags: Vec<bool> = vec![];

    for doc in &documents {
        match existing_map.get(doc.id.as_str()) {
            Some(existing) if doc.timestamp <= existing.timestamp => {
                skipped.push(doc.id.clone());
            }
            Some(existing) => {
                if needs_revectorization(doc, existing, &collection_config) {
                    to_vectorize.push(doc);
                    is_new_flags.push(false);
                } else {
                    to_patch.push(doc);
                }
            }
            None => {
                to_vectorize.push(doc);
                is_new_flags.push(true);
            }
        }
    }

    if !to_patch.is_empty() {
        let patch_docs: Vec<Document> = to_patch.iter().map(|d| (*d).clone()).collect();
        db.patch_documents(collection_name, &patch_docs, collection_config.store_content)
            .await
            .map_err(UpsertSyncError::Db)?;
    }

    if to_vectorize.is_empty() {
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

    let mut docs_owned: Vec<Document> = to_vectorize.iter().map(|d| (*d).clone()).collect();
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

    for attempt in 1..=ADD_DOCUMENTS_MAX_ATTEMPTS {
        match db
            .add_documents(collection_name, &docs_with_vectors, collection_config.store_content)
            .await
        {
            Ok(()) => break,
            Err(e) if attempt < ADD_DOCUMENTS_MAX_ATTEMPTS => {
                tracing::warn!(
                    attempt,
                    max_attempts = ADD_DOCUMENTS_MAX_ATTEMPTS,
                    error = %e,
                    "add_documents failed; retrying"
                );
                tokio::time::sleep(ADD_DOCUMENTS_RETRY_DELAY).await;
            }
            Err(e) => return Err(UpsertSyncError::Db(e)),
        }
    }

    // Build stats updates for vectorized docs only.
    let new_doc_count_batch = is_new_flags.iter().filter(|&&n| n).count() as i64;
    let update_doc_count_batch = is_new_flags.iter().filter(|&&n| !n).count() as i64;

    let mut updates: HashMap<String, TokenLengthUpdate> = HashMap::new();
    for (doc_idx, doc_with_vectors) in docs_with_vectors.iter().enumerate() {
        let is_new = is_new_flags[doc_idx];
        let existing = existing_map.get(to_vectorize[doc_idx].id.as_str());
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
/// Returns `Ok(skipped_ids)` — non-empty when the delete was stale and not applied.
/// Missing documents return `Ok(vec![])` (idempotent success).
pub async fn document_delete_sync(
    db: &QdrantDb,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &NamedLocks,
    collection_name: &str,
    document_id: &str,
    request_timestamp: DateTime<Utc>,
) -> Result<Vec<String>, UpsertSyncError> {
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
        None => return Ok(vec![]),
    };

    if request_timestamp <= doc_with_vectors.timestamp {
        return Ok(vec![document_id.to_string()]);
    }

    match db.delete_document(collection_name, document_id).await {
        Ok(()) => {}
        Err(DbError::NotFound(_)) => return Ok(vec![]),
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

    Ok(vec![])
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
        settings: SearchQuerySettings {
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
        },
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
            if idx.value_type == "string" {
                let len = metadata_string_value_len(value);
                if len > MAX_INDEXED_METADATA_VALUE_LENGTH {
                    return Err(MetadataTypeError(format!(
                        "String metadata value for key '{}' exceeds {} character limit",
                        idx.key, MAX_INDEXED_METADATA_VALUE_LENGTH
                    )));
                }
            }
        }
    }

    Ok(())
}

fn metadata_string_value_len(value: &serde_json::Value) -> usize {
    if let Some(map) = value.as_object() {
        if let Some(v) = map.get("value").and_then(|v| v.as_str()) {
            return v.chars().count();
        }
    }
    value.as_str().map(|s| s.chars().count()).unwrap_or(0)
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
// SearchError — returned by [`SearchIngress::search`]
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SearchError {
    NotFound(String),
    InvalidFilter(String),
    Vectorization(String),
    Db(DbError),
    IngressQueueFull(String),
    IngressWorkerExited(String),
}

impl Clone for SearchError {
    fn clone(&self) -> Self {
        match self {
            SearchError::NotFound(m) => SearchError::NotFound(m.clone()),
            SearchError::InvalidFilter(m) => SearchError::InvalidFilter(m.clone()),
            SearchError::Vectorization(m) => SearchError::Vectorization(m.clone()),
            SearchError::Db(e) => SearchError::Db(DbError::Config(e.to_string())),
            SearchError::IngressQueueFull(m) => SearchError::IngressQueueFull(m.clone()),
            SearchError::IngressWorkerExited(m) => SearchError::IngressWorkerExited(m.clone()),
        }
    }
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::NotFound(m) => write!(f, "{m}"),
            SearchError::InvalidFilter(m) => write!(f, "{m}"),
            SearchError::Vectorization(m) => write!(f, "{m}"),
            SearchError::Db(e) => write!(f, "{e}"),
            SearchError::IngressQueueFull(m) => write!(f, "{m}"),
            SearchError::IngressWorkerExited(m) => write!(f, "{m}"),
        }
    }
}

impl From<DbError> for SearchError {
    fn from(e: DbError) -> Self {
        SearchError::Db(e)
    }
}