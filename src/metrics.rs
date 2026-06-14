//! Metrics collection — mirrors `amgix-server/src/core/common/metrics_service.py`.
//!
//! `MetricsCollector` accumulates metric events via a bounded sync channel (capacity 100 000,
//! same as Python's `deque(maxlen=100_000)`). A dedicated `std::thread` runs its own
//! tokio runtime (multi-thread, 1 worker) and loops every [`REPORT_INTERVAL_S`] seconds,
//! matching `metrics_service.py` report thread logic.
//!
//! Only started when `AMGIX_AMQP_URL` is set (cluster mode). No-op in standalone.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::bunny_talk::BunnyTalk;
use crate::qdrant::QdrantDb;

// ---------------------------------------------------------------------------
// Constants — mirrors metrics_service.py
// ---------------------------------------------------------------------------

const REPORT_INTERVAL_S: f64 = 10.0;
const LIVE_BUCKET_SECONDS: i64 = 5;
const _1M_BUCKET_SECONDS: i64 = 60;
const _5M_BUCKET_SECONDS: i64 = 300;
const PENDING_BUCKET_MAX_AGE_S: f64 = 86_400.0;
const EVENT_BUFFER_MAXLEN: usize = 100_000;
const WINDOWS: &[i64] = &[30, 60];
const MAX_WINDOW: i64 = 60;

// ---------------------------------------------------------------------------
// Metric keys — mirrors metrics_definitions.py MetricKey
// ---------------------------------------------------------------------------

pub mod keys {
    pub const API_REQUESTS: &str = "api_requests";
    pub const API_REQUEST_MS: &str = "api_request_ms";
    pub const API_ASYNC_UPLOAD: &str = "api_async_upload";
    pub const API_ASYNC_UPLOAD_MS: &str = "api_async_upload_ms";
    pub const API_SYNC_UPLOAD: &str = "api_sync_upload";
    pub const API_SYNC_UPLOAD_MS: &str = "api_sync_upload_ms";
    pub const API_BULK_UPLOAD: &str = "api_bulk_upload";
    pub const API_BULK_UPLOAD_MS: &str = "api_bulk_upload_ms";
    pub const API_SEARCH: &str = "api_search";
    pub const API_SEARCH_MS: &str = "api_search_ms";
    pub const API_ASYNC_DELETE: &str = "api_async_delete";
    pub const API_ASYNC_DELETE_MS: &str = "api_async_delete_ms";
    pub const API_SYNC_DELETE: &str = "api_sync_delete";
    pub const API_SYNC_DELETE_MS: &str = "api_sync_delete_ms";
    pub const API_ERROR_4XX: &str = "api_error_4xx";
    pub const API_ERROR_5XX: &str = "api_error_5xx";
    pub const INDEX_QUEUE_DOCS_SKIPPED_STALE: &str = "index_queue_docs_skipped_stale";
    pub const INDEX_QUEUE_DOCS_NEW: &str = "index_queue_docs_new";
    pub const INDEX_QUEUE_DOCS_UPDATED: &str = "index_queue_docs_updated";
    pub const INDEX_QUEUE_DOCS_DELETED: &str = "index_queue_docs_deleted";
    pub const INDEX_QUEUE_DELETE_JOB_MS: &str = "index_queue_delete_job_ms";
    pub const INDEX_QUEUE_FAILED: &str = "index_queue_failed";
    pub const INDEX_QUEUE_REQUEUED: &str = "index_queue_requeued";
    pub const INDEX_QUEUE_JOB_MS: &str = "index_queue_job_ms";
    pub const INDEX_BULK_BATCHES: &str = "index_bulk_batches";
    pub const INDEX_BULK_BATCH_SIZE: &str = "index_bulk_batch_size";
    pub const INDEX_BULK_FAILED: &str = "index_bulk_failed";
    pub const INDEX_BULK_REQUEUED: &str = "index_bulk_requeued";
    pub const INDEX_BULK_JOB_MS: &str = "index_bulk_job_ms";
    pub const EMBED_BATCHES_ORIGIN: &str = "embed_batches_origin";
    pub const EMBED_PASSAGES_ORIGIN: &str = "embed_passages_origin";
    pub const EMBED_INFERENCE_ORIGIN_MS: &str = "embed_inference_origin_ms";
    pub const EMBED_INFERENCE_ORIGIN_ERRORS: &str = "embed_inference_origin_errors";
    pub const EMBED_BATCHES: &str = "embed_batches";
    pub const EMBED_PASSAGES: &str = "embed_passages";
    pub const EMBED_INFERENCE_MS: &str = "embed_inference_ms";
    pub const EMBED_HOPS: &str = "embed_hops";
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Agg {
    Sum,
    Avg,
}

struct MetricEvent {
    /// `[key, dim0, dim1, ...]` — key is always index 0.
    key: Vec<String>,
    value: f64,
    agg: Agg,
    /// Denominator for AVG metrics. None for SUM.
    n: Option<i64>,
    wall_ts: f64,
}

/// One 5-second live bucket — kept mutable until the bucket slot rolls over.
#[derive(Clone)]
struct LiveBucket {
    key: String,
    dims: Vec<String>,
    bucket_start: i64,
    value: f64,
    /// Some(n) for AVG, None for SUM.
    n: Option<i64>,
}

/// A completed, immutable bucket ready to be flushed to Qdrant.
#[derive(Clone)]
pub struct MetricsBucket {
    pub key: String,
    pub dims: Vec<String>,
    pub bucket_start: i64,
    pub bucket_seconds: i64,
    pub value: f64,
    pub n: Option<i64>,
}

// ---------------------------------------------------------------------------
// MetricsCollector — the public handle
// ---------------------------------------------------------------------------

/// `(type, model, revision) → last used Instant`.
pub type ModelLastUsed = Arc<Mutex<HashMap<(String, String, String), Instant>>>;

/// Cloneable handle. `record()` is non-blocking; events are dropped (with a one-time warning)
/// when the channel is full.
#[derive(Clone)]
pub struct MetricsCollector {
    tx: SyncSender<MetricEvent>,
    /// Set to true when a buffer-full warning was already emitted; cleared on next successful send.
    dropping: Arc<AtomicBool>,
    /// Set to true after the first disconnected error is logged; never cleared.
    thread_dead_warned: Arc<AtomicBool>,
    /// Updated on each successful embed; serialized into node meta every tick.
    pub model_last_used: ModelLastUsed,
    /// Total RAM in GB — read once at startup.
    total_ram_gb: f64,
    /// Whether GPU inference is available — detected once at startup.
    gpu_available: bool,
    /// Total VRAM in GB for GPU 0 — `None` if no NVIDIA GPU present.
    total_vram_gb: Option<f64>,
}

pub struct MetricsCollectorShutdown {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MetricsCollector {
    /// Record a metric event. Non-blocking. Silently drops (with a one-time warning) if the
    /// channel is full. `dims` is optional extra dimensions appended to the key tuple.
    /// `n` is the denominator for AVG aggregation; pass `None` for SUM metrics.
    pub fn record(&self, key: &str, dims: &[&str], value: f64, n: Option<i64>) {
        let agg = if n.is_some() { Agg::Avg } else { Agg::Sum };
        let mut nk = Vec::with_capacity(1 + dims.len());
        nk.push(key.to_string());
        for d in dims {
            nk.push(d.to_string());
        }
        let event = MetricEvent {
            key: nk,
            value,
            agg,
            n,
            wall_ts: unix_now(),
        };
        match self.tx.try_send(event) {
            Ok(()) => {
                // If we were previously dropping, clear the flag so the next drop re-warns.
                self.dropping.store(false, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) => {
                if !self.dropping.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        capacity = EVENT_BUFFER_MAXLEN,
                        "MetricsCollector: event buffer full; dropping metric events until drained"
                    );
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                if !self.thread_dead_warned.swap(true, Ordering::Relaxed) {
                    tracing::error!("MetricsCollector: report thread has exited; all subsequent metric events will be lost");
                }
            }
        }
    }

    /// Start the report thread. Returns the shutdown handle.
    /// `amqp_url` is used to build a fresh `BunnyTalk` inside the thread (own runtime).
    pub fn new(
        amqp_url: String,
        db: Arc<QdrantDb>,
        hostname: String,
    ) -> (Self, MetricsCollectorShutdown) {
        let (tx, rx) = mpsc::sync_channel::<MetricEvent>(EVENT_BUFFER_MAXLEN);
        let stop = Arc::new(AtomicBool::new(false));
        let dropping = Arc::new(AtomicBool::new(false));
        let model_last_used: ModelLastUsed = Arc::new(Mutex::new(HashMap::new()));

        // Read once at startup — these don't change at runtime.
        let total_ram_gb = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            sys.total_memory() as f64 / (1024.0_f64.powi(3))
        };
        let gpu_available = crate::vectors::model_cache::is_gpu_inference();

        // Detect total VRAM once — re-initialize NVML inside the report thread for per-tick queries.
        let total_vram_gb: Option<f64> = nvml_total_vram_gb();

        let stop_thread = Arc::clone(&stop);
        let model_last_used_thread = Arc::clone(&model_last_used);
        let thread = std::thread::Builder::new()
            .name("metrics-report".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .thread_name("metrics-report-rt")
                    .build()
                    .expect("metrics report thread: failed to build tokio runtime");
                rt.block_on(report_thread_main(
                    rx,
                    stop_thread,
                    amqp_url,
                    db,
                    hostname,
                    total_ram_gb,
                    gpu_available,
                    total_vram_gb,
                    model_last_used_thread,
                ));
            })
            .expect("failed to spawn metrics report thread");

        let collector = MetricsCollector {
            tx,
            dropping,
            thread_dead_warned: Arc::new(AtomicBool::new(false)),
            model_last_used,
            total_ram_gb,
            gpu_available,
            total_vram_gb,
        };
        let shutdown = MetricsCollectorShutdown {
            stop,
            thread: Some(thread),
        };
        (collector, shutdown)
    }
}

impl MetricsCollectorShutdown {
    pub fn stop_and_join(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            if let Err(e) = t.join() {
                tracing::error!("MetricsCollector: report thread panicked: {e:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Report thread main loop
// ---------------------------------------------------------------------------

async fn report_thread_main(
    rx: mpsc::Receiver<MetricEvent>,
    stop: Arc<AtomicBool>,
    amqp_url: String,
    db: Arc<QdrantDb>,
    hostname: String,
    total_ram_gb: f64,
    gpu_available: bool,
    total_vram_gb: Option<f64>,
    model_last_used: ModelLastUsed,
) {
    // NVML is not Send, so we init it here in the report thread for per-tick VRAM queries.
    let nvml = nvml_wrapper::Nvml::init().ok();
    let bunny = match BunnyTalk::create(&amqp_url).await {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("MetricsCollector: failed to connect to RabbitMQ: {e}; metrics reporting disabled");
            return;
        }
    };

    let mut state = ReportState::new();
    let mut pending_flush: Option<tokio::task::JoinHandle<()>> = None;

    while !stop.load(Ordering::Relaxed) {
        let (snap, completed_1m) = state.drain_and_process(&rx, true);
        let series = snap_to_series(&snap, &state.last_seen);
        let payload = build_payload(&hostname, &series, total_ram_gb, gpu_available, total_vram_gb, nvml.as_ref(), &model_last_used);
        if let Err(e) = bunny
            .talk("metrics-leader", json!({ "payload": payload }), true, None)
            .await
        {
            tracing::warn!(error = %e, "MetricsCollector: failed to publish to metrics-leader");
        }

        let now = unix_now();
        let completed_5m = state.collect_completed_5m_from_1m(&completed_1m, now);
        let had_completed_1m = !completed_1m.is_empty();
        let had_completed_5m = !completed_5m.is_empty();

        let cutoff = now - PENDING_BUCKET_MAX_AGE_S;
        {
            let mut pending = state.pending_1m.lock().expect("pending_1m mutex poisoned");
            pending.retain(|b| b.bucket_start as f64 >= cutoff);
            if had_completed_1m {
                pending.extend(completed_1m);
            }
            if had_completed_5m {
                pending.extend(completed_5m);
            }
        }

        let has_pending = !state.pending_1m.lock().expect("pending_1m mutex poisoned").is_empty();
        if had_completed_1m || had_completed_5m {
            if pending_flush.as_ref().is_none_or(|h| h.is_finished()) {
                pending_flush = Some(schedule_flush(
                    Arc::clone(&db),
                    hostname.clone(),
                    Arc::clone(&state.pending_1m),
                ));
            }
        } else if has_pending && pending_flush.as_ref().is_none_or(|h| h.is_finished()) {
            pending_flush = Some(schedule_flush(
                Arc::clone(&db),
                hostname.clone(),
                Arc::clone(&state.pending_1m),
            ));
        }

        if wait_for_stop(&stop, REPORT_INTERVAL_S).await {
            break;
        }
    }

    // Drain on shutdown: wait for any in-flight flush, then do a final one.
    if let Some(handle) = pending_flush {
        if let Err(e) = handle.await {
            tracing::error!(error = %e, "MetricsCollector: in-flight flush task failed on shutdown");
        }
    }
    if !state.pending_1m.lock().expect("pending_1m mutex poisoned").is_empty() {
        flush_pending_1m(&db, &hostname, &state.pending_1m).await;
    }

    bunny.close().await;
}

/// Mirrors `_wait_for_stop` — returns true if stop was set (caller should break).
async fn wait_for_stop(stop: &AtomicBool, timeout_s: f64) -> bool {
    let deadline = unix_now() + timeout_s;
    let chunk = 0.25;
    while !stop.load(Ordering::Relaxed) {
        let remaining = deadline - unix_now();
        if remaining <= 0.0 {
            break;
        }
        tokio::time::sleep(Duration::from_secs_f64(remaining.min(chunk))).await;
    }
    stop.load(Ordering::Relaxed)
}

fn schedule_flush(
    db: Arc<QdrantDb>,
    hostname: String,
    pending: Arc<Mutex<Vec<MetricsBucket>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        flush_pending_1m(&db, &hostname, &pending).await;
    })
}

async fn flush_pending_1m(
    db: &QdrantDb,
    hostname: &str,
    pending: &Arc<Mutex<Vec<MetricsBucket>>>,
) {
    let batch = {
        let guard = pending.lock().expect("pending_1m mutex poisoned");
        guard.clone()
    };
    if batch.is_empty() {
        return;
    }
    match db.append_metric_buckets(hostname, "amgix-now", &batch).await {
        Ok(()) => {
            let flushed: HashSet<BucketId> = batch.iter().map(bucket_id).collect();
            pending
                .lock()
                .expect("pending_1m mutex poisoned")
                .retain(|b| !flushed.contains(&bucket_id(b)));
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "MetricsCollector: failed to flush metric buckets to Qdrant"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ReportState — all mutable state owned by the report thread
// ---------------------------------------------------------------------------

type NormalizedKey = Vec<String>;

struct ReportState {
    agg: HashMap<NormalizedKey, Agg>,
    /// Keyed by NormalizedKey, values are a VecDeque of 5s live buckets in chronological order.
    data: HashMap<NormalizedKey, VecDeque<LiveBucket>>,
    last_seen: HashMap<NormalizedKey, f64>,
    last_flushed_1m_start: i64,
    last_flushed_5m_start: i64,
    pending_1m: Arc<Mutex<Vec<MetricsBucket>>>,
    /// 1m buckets accumulating toward the next 5m rollup.
    pending_1m_for_5m: Vec<MetricsBucket>,
}

impl ReportState {
    fn new() -> Self {
        ReportState {
            agg: HashMap::new(),
            data: HashMap::new(),
            last_seen: HashMap::new(),
            last_flushed_1m_start: 0,
            last_flushed_5m_start: 0,
            pending_1m: Arc::new(Mutex::new(Vec::new())),
            pending_1m_for_5m: Vec::new(),
        }
    }

    /// Mirrors `_drain_and_process`.
    fn drain_and_process(
        &mut self,
        rx: &mpsc::Receiver<MetricEvent>,
        collect_1m: bool,
    ) -> (HashMap<NormalizedKey, HashMap<i64, (f64, Option<i64>)>>, Vec<MetricsBucket>) {
        self.drain_events(rx);
        let now = unix_now();
        let snap = self.snapshot(now);
        let completed = if collect_1m {
            self.collect_completed_1m(now)
        } else {
            vec![]
        };
        self.trim_buckets(now);
        (snap, completed)
    }

    /// Drain all pending events from the channel into live buckets.
    fn drain_events(&mut self, rx: &mpsc::Receiver<MetricEvent>) {
        loop {
            match rx.try_recv() {
                Ok(event) => self.apply_event(event),
                Err(_) => break,
            }
        }
    }

    fn apply_event(&mut self, event: MetricEvent) {
        let k = &event.key;
        match self.agg.get(k) {
            None => {
                self.agg.insert(k.clone(), event.agg);
                self.data.insert(k.clone(), VecDeque::new());
            }
            Some(&existing) if existing != event.agg => {
                tracing::error!(
                    key = ?k,
                    "MetricsCollector: key recorded with conflicting aggregation; event dropped"
                );
                return;
            }
            _ => {}
        }

        self.last_seen.insert(k.clone(), event.wall_ts);
        let bucket_start = bucket_start_for(event.wall_ts);
        let buckets = self.data.get_mut(k).unwrap();

        if buckets.back().map(|b| b.bucket_start) == Some(bucket_start) {
            let b = buckets.back_mut().unwrap();
            b.value += event.value;
            if event.agg == Agg::Avg {
                *b.n.get_or_insert(0) += event.n.unwrap_or(0);
            }
        } else {
            buckets.push_back(LiveBucket {
                key: k[0].clone(),
                dims: k[1..].to_vec(),
                bucket_start,
                value: event.value,
                n: if event.agg == Agg::Avg { Some(event.n.unwrap_or(0)) } else { None },
            });
        }
    }

    /// Mirrors `_snapshot_locked`.
    fn snapshot(&self, now: f64) -> HashMap<NormalizedKey, HashMap<i64, (f64, Option<i64>)>> {
        let current_bucket_start = bucket_start_for(now);
        let mut result = HashMap::new();

        for (k, agg) in &self.agg {
            let buckets = &self.data[k];
            let mut windows = HashMap::new();
            for &win in WINDOWS {
                let bucket_cut = current_bucket_start - win + LIVE_BUCKET_SECONDS;
                let mut total = 0.0f64;
                let mut count = 0i64;
                for bucket in buckets.iter().rev() {
                    if bucket.bucket_start < bucket_cut {
                        break;
                    }
                    total += bucket.value;
                    if *agg == Agg::Avg {
                        count += bucket.n.unwrap_or(0);
                    }
                }
                let n_val = if *agg == Agg::Avg { Some(count) } else { None };
                windows.insert(win, (total, n_val));
            }
            result.insert(k.clone(), windows);
        }
        result
    }

    /// Mirrors `_collect_completed_1m_buckets_locked`.
    fn collect_completed_1m(&mut self, now: f64) -> Vec<MetricsBucket> {
        let current_1m_start = floor_to(now as i64, _1M_BUCKET_SECONDS);
        if current_1m_start <= self.last_flushed_1m_start {
            return vec![];
        }

        let mut result: Vec<MetricsBucket> = Vec::new();

        for (k, agg) in &self.agg {
            let mut merged: HashMap<i64, MetricsBucket> = HashMap::new();
            let buckets = &self.data[k];
            for bucket in buckets {
                let slot = floor_to(bucket.bucket_start, _1M_BUCKET_SECONDS);
                if slot < self.last_flushed_1m_start || slot >= current_1m_start {
                    continue;
                }
                let entry = merged.entry(slot).or_insert_with(|| MetricsBucket {
                    key: k[0].clone(),
                    dims: k[1..].to_vec(),
                    bucket_start: slot,
                    bucket_seconds: _1M_BUCKET_SECONDS,
                    value: 0.0,
                    n: if *agg == Agg::Avg { Some(0) } else { None },
                });
                entry.value += bucket.value;
                if *agg == Agg::Avg {
                    *entry.n.get_or_insert(0) += bucket.n.unwrap_or(0);
                }
            }
            result.extend(merged.into_values());
        }

        // Only advance last_flushed_1m_start if there's no raw data remaining in the completed range.
        let has_raw_in_completed = self.data.values().any(|buckets| {
            buckets.iter().any(|b| {
                let slot = floor_to(b.bucket_start, _1M_BUCKET_SECONDS);
                slot >= self.last_flushed_1m_start && slot < current_1m_start
            })
        });
        if !result.is_empty() || !has_raw_in_completed {
            self.last_flushed_1m_start = current_1m_start;
        }

        result
    }

    /// Mirrors `_collect_completed_5m_buckets_from_1m`.
    fn collect_completed_5m_from_1m(
        &mut self,
        completed_1m: &[MetricsBucket],
        now: f64,
    ) -> Vec<MetricsBucket> {
        let current_5m_start = floor_to(now as i64, _5M_BUCKET_SECONDS);
        if current_5m_start <= self.last_flushed_5m_start {
            return vec![];
        }

        let cutoff = now - PENDING_BUCKET_MAX_AGE_S;
        self.pending_1m_for_5m.retain(|b| b.bucket_start as f64 >= cutoff);
        self.pending_1m_for_5m.extend_from_slice(completed_1m);

        type BucketKey = (String, Vec<String>, i64);
        let mut merged: HashMap<BucketKey, MetricsBucket> = HashMap::new();
        let mut is_avg: HashMap<BucketKey, bool> = HashMap::new();
        let low_5m = self.last_flushed_5m_start;

        for bucket in &self.pending_1m_for_5m {
            let slot = floor_to(bucket.bucket_start, _5M_BUCKET_SECONDS);
            if slot < low_5m || slot >= current_5m_start {
                continue;
            }
            let bkey: BucketKey = (bucket.key.clone(), bucket.dims.clone(), slot);
            let avg = bucket.n.is_some();
            let entry = merged.entry(bkey.clone()).or_insert_with(|| MetricsBucket {
                key: bucket.key.clone(),
                dims: bucket.dims.clone(),
                bucket_start: slot,
                bucket_seconds: _5M_BUCKET_SECONDS,
                value: 0.0,
                n: if avg { Some(0) } else { None },
            });
            is_avg.entry(bkey).or_insert(avg);
            entry.value += bucket.value;
            if avg {
                *entry.n.get_or_insert(0) += bucket.n.unwrap_or(0);
            }
        }

        let has_1m_in_completed_5m = self.pending_1m_for_5m.iter().any(|b| {
            let slot = floor_to(b.bucket_start, _5M_BUCKET_SECONDS);
            slot >= low_5m && slot < current_5m_start
        });

        if !merged.is_empty() || !has_1m_in_completed_5m {
            self.pending_1m_for_5m.retain(|b| {
                floor_to(b.bucket_start, _5M_BUCKET_SECONDS) >= current_5m_start
            });
            self.last_flushed_5m_start = current_5m_start;
        }

        merged.into_values().collect()
    }

    /// Mirrors `_trim_buckets_locked`.
    fn trim_buckets(&mut self, now: f64) {
        let current_bucket_start = bucket_start_for(now);
        let window_cutoff = current_bucket_start - MAX_WINDOW + LIVE_BUCKET_SECONDS;
        let trim_cutoff = window_cutoff.min(self.last_flushed_1m_start);

        let mut empty_keys: Vec<NormalizedKey> = Vec::new();
        for (k, buckets) in &mut self.data {
            while buckets.front().map(|b| b.bucket_start).unwrap_or(i64::MAX) < trim_cutoff {
                buckets.pop_front();
            }
            if buckets.is_empty() {
                empty_keys.push(k.clone());
            }
        }
        for k in empty_keys {
            self.data.remove(&k);
            self.agg.remove(&k);
            self.last_seen.remove(&k);
        }
    }
}

// ---------------------------------------------------------------------------
// Payload construction
// ---------------------------------------------------------------------------

fn snap_to_series(
    snap: &HashMap<NormalizedKey, HashMap<i64, (f64, Option<i64>)>>,
    last_seen: &HashMap<NormalizedKey, f64>,
) -> Vec<Value> {
    snap.iter()
        .map(|(k, windows)| {
            let key = &k[0];
            let dims: Vec<&str> = k[1..].iter().map(|s| s.as_str()).collect();
            let ls = last_seen.get(k).copied();
            let windows_json: serde_json::Map<String, Value> = windows
                .iter()
                .map(|(win, (value, n))| {
                    (
                        win.to_string(),
                        json!({ "value": value, "n": n }),
                    )
                })
                .collect();
            json!({
                "key": key,
                "dims": dims,
                "windows": windows_json,
                "last_seen": ls,
            })
        })
        .collect()
}

fn build_payload(
    hostname: &str,
    series: &[Value],
    total_ram_gb: f64,
    gpu_available: bool,
    total_vram_gb: Option<f64>,
    nvml: Option<&nvml_wrapper::Nvml>,
    model_last_used: &ModelLastUsed,
) -> Value {
    use crate::vectors::dense_model::dense_model_cache_snapshot;
    use crate::vectors::sparse_model::sparse_model_cache_snapshot;

    // Read free RAM every tick — it changes.
    let free_ram_gb = {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        sys.available_memory() as f64 / (1024.0_f64.powi(3))
    };

    // Loaded models from both caches.
    let mut all_models: Vec<(String, String, Option<String>, Instant)> = dense_model_cache_snapshot();
    all_models.extend(sparse_model_cache_snapshot());
    all_models.sort_by(|a, b| {
        a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0))
    });
    let now_instant = Instant::now();
    let loaded_models: Vec<Value> = all_models
        .iter()
        .map(|(type_str, model, revision, loaded_at)| {
            let model_key = json!([type_str, model, revision.as_deref().unwrap_or("")]);
            let loaded_at_unix = unix_now() - now_instant.duration_since(*loaded_at).as_secs_f64();
            let label = if revision.as_deref().unwrap_or("").is_empty() {
                model.clone()
            } else {
                format!("{}:{}", model, revision.as_deref().unwrap_or(""))
            };
            json!({ "model_key": model_key, "loaded_at": loaded_at_unix, "label": label })
        })
        .collect();

    // Last-used timestamps per model key — trim entries older than MODEL_IDLE_GRACE_SECONDS.
    const MODEL_IDLE_GRACE_SECONDS: f64 = 300.0;
    let model_last_used_json: Vec<Value> = {
        let mut guard = model_last_used.lock().expect("model_last_used poisoned");
        guard.retain(|_, ts| now_instant.duration_since(*ts).as_secs_f64() < MODEL_IDLE_GRACE_SECONDS);
        guard
            .iter()
            .map(|(k, ts)| {
                let model_key = json!([k.0, k.1, k.2]);
                let last_used_at = unix_now() - now_instant.duration_since(*ts).as_secs_f64();
                json!({ "model_key": model_key, "last_used_at": last_used_at })
            })
            .collect()
    };

    let meta = json!({
        "load_models": true,
        "at_capacity": false,
        "total_ram_gb": (total_ram_gb * 100.0).round() / 100.0,
        "free_ram_gb": (free_ram_gb * 100.0).round() / 100.0,
        "total_vram_gb": total_vram_gb.map(|v| (v * 100.0).round() / 100.0),
        "free_vram_gb": nvml_free_vram_gb(nvml).map(|v| (v * 100.0).round() / 100.0),
        "gpu_support": gpu_available,
        "gpu_available": gpu_available,
        "loaded_models": loaded_models,
        "model_last_used": model_last_used_json,
    });

    json!({
        "probe": false,
        "hostname": hostname,
        "source": "amgix-now",
        "role": "now",
        "metrics": series,
        "meta": meta,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64()
}

fn bucket_start_for(ts: f64) -> i64 {
    (ts as i64 / LIVE_BUCKET_SECONDS) * LIVE_BUCKET_SECONDS
}

fn floor_to(ts: i64, bucket: i64) -> i64 {
    (ts / bucket) * bucket
}

type BucketId = (String, Vec<String>, i64, i64);

fn bucket_id(b: &MetricsBucket) -> BucketId {
    (b.key.clone(), b.dims.clone(), b.bucket_start, b.bucket_seconds)
}

/// Returns total VRAM in GB for GPU 0, or `None` if no NVIDIA GPU is present.
/// Initializes NVML temporarily — only called once at startup.
fn nvml_total_vram_gb() -> Option<f64> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;
    let info = device.memory_info().ok()?;
    Some(info.total as f64 / (1024.0_f64.powi(3)))
}

/// Returns free VRAM in GB for GPU 0 using an already-initialized NVML instance.
fn nvml_free_vram_gb(nvml: Option<&nvml_wrapper::Nvml>) -> Option<f64> {
    let nvml = nvml?;
    let device = nvml.device_by_index(0).ok()?;
    let info = device.memory_info().ok()?;
    Some(info.free as f64 / (1024.0_f64.powi(3)))
}
