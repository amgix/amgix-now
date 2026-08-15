use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::Instant;

use embed_anything::embeddings::embed::TextEmbedder;
use embed_anything::embeddings::local::bert::{BertEmbedder, SparseBertEmbedder};
use embed_anything::embeddings::local::modernbert::ModernBertEmbedder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde_json::Value;

use crate::common::cache_base_dir;
use crate::vectors::st_pooling::StPoolingConfig;

// ---------------------------------------------------------------------------
// GPU inference serialization
// ---------------------------------------------------------------------------

static GPU_INFERENCE: OnceLock<bool> = OnceLock::new();

/// Max overlapping GPU forwards (dense + sparse BERT paths) inside this process.
/// `1` matches the legacy single global mutex; increase only when you accept higher GPU
/// parallelism (device memory and driver limits). Unused on CPU-only inference.
pub const GPU_MODEL_INFERENCE_CONCURRENCY_MAX: usize = 1;

const _: () = assert!(GPU_MODEL_INFERENCE_CONCURRENCY_MAX >= 1);

/// Returns true if GPU (CUDA or Metal) is available for model inference.
/// Detected once on first call; subsequent calls are a single bool read.
pub fn is_gpu_inference() -> bool {
    *GPU_INFERENCE.get_or_init(|| {
        let cuda = candle_core::Device::cuda_if_available(0)
            .map(|d| d.is_cuda())
            .unwrap_or(false);
        let metal = candle_core::Device::new_metal(0)
            .map(|d| d.is_metal())
            .unwrap_or(false);
        cuda || metal
    })
}

/// Counting limiter serializing overlapping GPU forwards across rayon pools.
/// `[`GpuInferenceLimiter::release`]` must run once per successful `acquire` (use [`GpuInferencePermit`]).
struct GpuInferenceLimiter {
    max: usize,
    available: Mutex<usize>,
    cvar: Condvar,
}

impl GpuInferenceLimiter {
    fn new(max: usize) -> Self {
        Self {
            max,
            available: Mutex::new(max),
            cvar: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let mut n = self.available.lock().unwrap();
        while *n == 0 {
            n = self.cvar.wait(n).unwrap();
        }
        *n -= 1;
    }

    fn release(&self) {
        let mut n = self.available.lock().unwrap();
        *n = (*n + 1).min(self.max);
        self.cvar.notify_one();
    }
}

/// RAII slot from [`maybe_gpu_inference_permit`].
pub(crate) struct GpuInferencePermit {
    limiter: Arc<GpuInferenceLimiter>,
}

impl Drop for GpuInferencePermit {
    fn drop(&mut self) {
        self.limiter.release();
    }
}

static GPU_INFERENCE_LIMITER: OnceLock<Arc<GpuInferenceLimiter>> = OnceLock::new();

fn gpu_inference_limiter() -> &'static Arc<GpuInferenceLimiter> {
    GPU_INFERENCE_LIMITER.get_or_init(|| {
        Arc::new(GpuInferenceLimiter::new(GPU_MODEL_INFERENCE_CONCURRENCY_MAX))
    })
}

/// When [`is_gpu_inference`] is false, returns `None` (no limiting). Otherwise blocks until a GPU
/// forward slot is available and returns a permit whose drop releases the slot.
pub(crate) fn maybe_gpu_inference_permit() -> Option<GpuInferencePermit> {
    if !is_gpu_inference() {
        return None;
    }
    let limiter = Arc::clone(gpu_inference_limiter());
    limiter.acquire();
    Some(GpuInferencePermit { limiter })
}

fn model_cache_size() -> usize {
    std::env::var("AMGIX_MODEL_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

/// Returns the HuggingFace local cache directory, mirrors HF_CACHE_DIR in Python.
pub fn hf_cache_dir() -> String {
    cache_base_dir()
        .join("huggingface")
        .to_string_lossy()
        .into_owned()
}

/// Check if a model is from a trusted organization.
/// Mirrors VectorBase.is_trusted_model: splits on '/' and checks the left part.
pub fn is_trusted_model(model_name: &str, trusted_organizations: &HashSet<String>) -> bool {
    if !model_name.contains('/') {
        return false;
    }
    let org = model_name.split('/').next().unwrap_or("").to_lowercase();
    trusted_organizations.contains(&org)
}

/// Load trusted organizations at startup, mirrors embed_router_service._load_trusted_organizations.
/// Enabled via AMGIX_TRUSTED_ORGS=true; file path from AMGIX_TRUSTED_ORGS_FILE.
/// Returns None when the feature is disabled (no organization filtering).
pub fn load_trusted_organizations() -> Option<HashSet<String>> {
    let enabled = std::env::var("AMGIX_TRUSTED_ORGS")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(false);
    if !enabled {
        return None;
    }

    let file_path = std::env::var("AMGIX_TRUSTED_ORGS_FILE")
        .unwrap_or_else(|_| "trusted_orgs.txt".to_string());

    match std::fs::read_to_string(&file_path) {
        Ok(contents) => {
            let orgs: HashSet<String> = contents
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| l.to_string())
                .collect();
            tracing::info!(
                count = orgs.len(),
                path = %file_path,
                "loaded trusted organizations"
            );
            Some(orgs)
        }
        Err(e) => {
            tracing::warn!(
                path = %file_path,
                error = %e,
                "failed to load trusted organizations; using empty set"
            );
            Some(HashSet::new())
        }
    }
}

static TRUSTED_ORGANIZATIONS: OnceLock<Option<HashSet<String>>> = OnceLock::new();

/// Returns the globally-initialized trusted organizations (None = disabled, Some = enforced).
pub fn trusted_organizations() -> Option<&'static HashSet<String>> {
    TRUSTED_ORGANIZATIONS
        .get_or_init(load_trusted_organizations)
        .as_ref()
}

/// Sets HF_HOME so that `ApiBuilder::from_env()` inside embed_anything uses our cache dir.
pub fn set_hf_home() {
    let base = cache_base_dir();
    if std::env::var("HF_HOME").is_err() {
        unsafe { std::env::set_var("HF_HOME", base.join("huggingface")); }
    }
    if std::env::var("CUDA_CACHE_PATH").is_err() {
        unsafe { std::env::set_var("CUDA_CACHE_PATH", base.join("cuda")); }
    }
}

/// Hugging Face Hub token from `HF_TOKEN` (private / gated models).
fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
}

/// HF Hub API: prefer `HF_TOKEN` when set; otherwise keep hf-hub's token-file default.
fn hf_api() -> Result<hf_hub::api::sync::Api, String> {
    let mut builder = ApiBuilder::new().with_progress(false);
    if let Some(token) = hf_token() {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|e| format!("Failed to build HF API: {e}"))
}

// ---------------------------------------------------------------------------
// Dense model cache
// ---------------------------------------------------------------------------

const ST_POOLING_CONFIG_PATH: &str = "1_Pooling/config.json";

fn read_hf_config_json(model_id: &str, revision: Option<&str>) -> Result<Value, String> {
    let api = hf_api()?;
    let repo = match revision {
        Some(rev) => Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(model_id.to_string(), RepoType::Model),
    };
    let config_path = api
        .repo(repo)
        .get("config.json")
        .map_err(|e| format!("Failed to fetch config.json for '{model_id}': {e}"))?;
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config.json for '{model_id}': {e}"))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse config.json for '{model_id}': {e}"))
}

fn is_modern_bert_architecture(architecture: &str) -> bool {
    matches!(architecture, "ModernBertModel" | "ModernBertForMaskedLM")
}

fn load_dense_model(model_id: &str, revision: Option<&str>) -> Result<DenseModelHandle, String> {
    let config = read_hf_config_json(model_id, revision)?;
    let architecture = config["architectures"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let token = hf_token();
    let token = token.as_deref();

    if architecture == "BertModel" {
        let pooling = load_st_pooling_config(model_id, revision);
        let embedder = BertEmbedder::new(
            model_id.to_string(),
            revision.map(|s| s.to_string()),
            token,
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;
        Ok(DenseModelHandle::Bert { embedder, pooling })
    } else if is_modern_bert_architecture(architecture) {
        let pooling = dense_pooling_config(&config, model_id, revision);
        let embedder = ModernBertEmbedder::new(
            model_id.to_string(),
            revision.map(|s| s.to_string()),
            token,
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;
        Ok(DenseModelHandle::ModernBert { embedder, pooling })
    } else {
        let embedder = TextEmbedder::from_pretrained_hf(
            architecture,
            model_id,
            revision,
            token,
            None,
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;
        Ok(DenseModelHandle::Generic(embedder))
    }
}

/// Dense model loaded once at model load time.
///
/// BERT and ModernBERT models use the optimized on-device candle pipeline (tokenize → tensor →
/// forward → pool → normalize). All other architectures use `embed_anything`'s `TextEmbedder`.
pub enum DenseModelHandle {
    Bert {
        embedder: BertEmbedder,
        pooling: StPoolingConfig,
    },
    ModernBert {
        embedder: ModernBertEmbedder,
        pooling: StPoolingConfig,
    },
    Generic(TextEmbedder),
}

fn dense_pooling_config(hf_config: &Value, model_id: &str, revision: Option<&str>) -> StPoolingConfig {
    if let Some(cfg) = try_load_st_pooling_config(model_id, revision) {
        return cfg;
    }
    match hf_config.get("classifier_pooling").and_then(|v| v.as_str()) {
        Some("cls") => StPoolingConfig::cls_only(),
        Some("mean") => StPoolingConfig::mean_only(),
        _ => StPoolingConfig::mean_only(),
    }
}

fn try_load_st_pooling_config(model_id: &str, revision: Option<&str>) -> Option<StPoolingConfig> {
    set_hf_home();
    let api = hf_api().ok()?;
    let repo = match revision {
        Some(rev) => Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(model_id.to_string(), RepoType::Model),
    };
    let path = api.repo(repo).get(ST_POOLING_CONFIG_PATH).ok()?;
    Some(load_st_pooling_config_from_path(&path))
}

fn load_st_pooling_config(model_id: &str, revision: Option<&str>) -> StPoolingConfig {
    set_hf_home();
    let Ok(api) = hf_api() else {
        return StPoolingConfig::mean_only();
    };
    let repo = match revision {
        Some(rev) => Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(model_id.to_string(), RepoType::Model),
    };
    let Ok(path) = api.repo(repo).get(ST_POOLING_CONFIG_PATH) else {
        return StPoolingConfig::mean_only();
    };
    load_st_pooling_config_from_path(&path)
}

fn load_st_pooling_config_from_path(path: &Path) -> StPoolingConfig {
    let Ok(text) = std::fs::read_to_string(path) else {
        return StPoolingConfig::mean_only();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return StPoolingConfig::mean_only();
    };
    StPoolingConfig::from_json_value(&value)
}

type ModelKey = (String, Option<String>);
type DenseModelEntry = (Arc<DenseModelHandle>, Instant);

enum InFlightState<V> {
    Loading,
    Ready(Result<Arc<V>, String>),
}

struct InFlightEntry<V> {
    state: Mutex<InFlightState<V>>,
    cvar: Condvar,
}

type InFlight<V> = Mutex<HashMap<ModelKey, Arc<InFlightEntry<V>>>>;

fn bump_cache_hit<K, V>(cache: &mut HashMap<K, (Arc<V>, Instant)>, key: &K) -> Option<Arc<V>>
where
    K: Eq + Hash,
{
    cache.get_mut(key).map(|(model, last_used)| {
        *last_used = Instant::now();
        Arc::clone(model)
    })
}

fn evict_lru_if_full<K, V>(cache: &mut HashMap<K, (Arc<V>, Instant)>, max_size: usize)
where
    K: Clone + Eq + Hash,
{
    while cache.len() >= max_size {
        let lru_key = cache
            .iter()
            .min_by_key(|(_, (_, last_used))| *last_used)
            .map(|(k, _)| k.clone());
        match lru_key {
            Some(k) => {
                cache.remove(&k);
            }
            None => break,
        }
    }
}

fn cache_hit<V>(
    inner: &RwLock<HashMap<ModelKey, (Arc<V>, Instant)>>,
    key: &ModelKey,
) -> Option<Arc<V>> {
    if !inner.read().unwrap().contains_key(key) {
        return None;
    }
    let mut cache = inner.write().unwrap();
    bump_cache_hit(&mut cache, key)
}

fn insert_loaded<V>(
    inner: &RwLock<HashMap<ModelKey, (Arc<V>, Instant)>>,
    inflight: &InFlight<V>,
    max_size: usize,
    key: ModelKey,
    loaded: Arc<V>,
) -> Arc<V>
where
    V: Send + Sync + 'static,
{
    let mut cache = inner.write().unwrap();
    if let Some(existing) = bump_cache_hit(&mut cache, &key) {
        inflight.lock().unwrap().remove(&key);
        return existing;
    }
    evict_lru_if_full(&mut cache, max_size);
    cache.insert(key.clone(), (Arc::clone(&loaded), Instant::now()));
    inflight.lock().unwrap().remove(&key);
    loaded
}

fn wait_for_inflight_load<V, F>(
    entry: &InFlightEntry<V>,
    is_owner: bool,
    load: F,
) -> Result<Arc<V>, String>
where
    V: Send + Sync + 'static,
    F: FnOnce() -> Result<Arc<V>, String>,
{
    let mut guard = entry.state.lock().unwrap();
    loop {
        match &*guard {
            InFlightState::Ready(Ok(handle)) => return Ok(Arc::clone(handle)),
            InFlightState::Ready(Err(error)) => return Err(error.clone()),
            InFlightState::Loading => {
                if is_owner {
                    let result = load();
                    *guard = InFlightState::Ready(result.clone());
                    entry.cvar.notify_all();
                    return result;
                }
                guard = entry.cvar.wait(guard).unwrap();
            }
        }
    }
}

fn load_with_inflight<V, F>(
    inner: &RwLock<HashMap<ModelKey, (Arc<V>, Instant)>>,
    inflight: &InFlight<V>,
    max_size: usize,
    key: ModelKey,
    load: F,
) -> Result<Arc<V>, String>
where
    V: Send + Sync + 'static,
    F: FnOnce() -> Result<Arc<V>, String>,
{
    if let Some(handle) = cache_hit(inner, &key) {
        return Ok(handle);
    }

    let (entry, is_owner) = {
        let mut loads = inflight.lock().unwrap();
        if let Some(existing) = loads.get(&key) {
            (Arc::clone(existing), false)
        } else {
            let entry = Arc::new(InFlightEntry {
                state: Mutex::new(InFlightState::Loading),
                cvar: Condvar::new(),
            });
            loads.insert(key.clone(), Arc::clone(&entry));
            (entry, true)
        }
    };

    if let Some(handle) = cache_hit(inner, &key) {
        inflight.lock().unwrap().remove(&key);
        return Ok(handle);
    }

    let loaded = match wait_for_inflight_load(&entry, is_owner, load) {
        Ok(handle) => handle,
        Err(error) => {
            inflight.lock().unwrap().remove(&key);
            return Err(error);
        }
    };

    Ok(insert_loaded(inner, inflight, max_size, key, loaded))
}

pub struct DenseModelCache {
    inner: RwLock<HashMap<ModelKey, DenseModelEntry>>,
    inflight: InFlight<DenseModelHandle>,
    max_size: usize,
}

impl DenseModelCache {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
            max_size: model_cache_size(),
        }
    }

    pub fn get_or_load(
        &self,
        model_id: &str,
        revision: Option<&str>,
        trusted_organizations: Option<&HashSet<String>>,
    ) -> Result<Arc<DenseModelHandle>, String> {
        let key = (model_id.to_string(), revision.map(|s| s.to_string()));

        if let Some(handle) = cache_hit(&self.inner, &key) {
            return Ok(handle);
        }

        if let Some(orgs) = trusted_organizations {
            if !is_trusted_model(model_id, orgs) {
                return Err(format!(
                    "Model '{}' is not from a trusted organization. Trusted organizations: {}",
                    model_id,
                    {
                        let mut sorted: Vec<_> = orgs.iter().cloned().collect();
                        sorted.sort();
                        sorted.join(", ")
                    }
                ));
            }
        }

        let model_id = model_id.to_string();
        let revision = revision.map(|s| s.to_string());
        load_with_inflight(&self.inner, &self.inflight, self.max_size, key, || {
            set_hf_home();
            let handle = Arc::new(load_dense_model(&model_id, revision.as_deref())?);
            tracing::info!(
                model = %model_id,
                revision = revision.as_deref().unwrap_or("(default)"),
                kind = "dense",
                "loaded model"
            );
            Ok(handle)
        })
    }

    /// Returns cache entries as `(type_str, model, revision, last_used_at)`.
    pub fn snapshot(&self) -> Vec<(String, String, Option<String>, Instant)> {
        let cache = self.inner.read().unwrap();
        cache
            .iter()
            .map(|((model, revision), (_, last_used))| {
                (
                    "dense_model".to_string(),
                    model.clone(),
                    revision.clone(),
                    *last_used,
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sparse model cache
// ---------------------------------------------------------------------------

type SparseModelEntry = (Arc<SparseBertEmbedder>, Instant);

pub struct SparseModelCache {
    inner: RwLock<HashMap<ModelKey, SparseModelEntry>>,
    inflight: InFlight<SparseBertEmbedder>,
    max_size: usize,
}

impl SparseModelCache {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
            max_size: model_cache_size(),
        }
    }

    pub fn get_or_load(
        &self,
        model_id: &str,
        revision: Option<&str>,
        trusted_organizations: Option<&HashSet<String>>,
    ) -> Result<Arc<SparseBertEmbedder>, String> {
        let key = (model_id.to_string(), revision.map(|s| s.to_string()));

        if let Some(model) = cache_hit(&self.inner, &key) {
            return Ok(model);
        }

        if let Some(orgs) = trusted_organizations {
            if !is_trusted_model(model_id, orgs) {
                return Err(format!(
                    "Model '{}' is not from a trusted organization. Trusted organizations: {}",
                    model_id,
                    {
                        let mut sorted: Vec<_> = orgs.iter().cloned().collect();
                        sorted.sort();
                        sorted.join(", ")
                    }
                ));
            }
        }

        let model_id = model_id.to_string();
        let revision = revision.map(|s| s.to_string());
        load_with_inflight(&self.inner, &self.inflight, self.max_size, key, || {
            set_hf_home();
            let token = hf_token();
            let embedder = Arc::new(
                SparseBertEmbedder::new(
                    model_id.clone(),
                    revision.clone(),
                    token.as_deref(),
                )
                .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?,
            );
            tracing::info!(
                model = %model_id,
                revision = revision.as_deref().unwrap_or("(default)"),
                kind = "sparse",
                "loaded model"
            );
            Ok(embedder)
        })
    }

    /// Returns cache entries as `(type_str, model, revision, last_used_at)`.
    pub fn snapshot(&self) -> Vec<(String, String, Option<String>, Instant)> {
        let cache = self.inner.read().unwrap();
        cache
            .iter()
            .map(|((model, revision), (_, last_used))| {
                (
                    "sparse_model".to_string(),
                    model.clone(),
                    revision.clone(),
                    *last_used,
                )
            })
            .collect()
    }
}
