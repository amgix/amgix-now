use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::Instant;

use embed_anything::embeddings::local::bert::{BertEmbedder, SparseBertEmbedder};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde_json::Value;

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
/// AMGIX_CACHE_DIR defaults to /data/amgix/cache; HF cache lives under it at /huggingface.
pub fn hf_cache_dir() -> String {
    let base = std::env::var("AMGIX_CACHE_DIR").unwrap_or_else(|_| "/data/amgix/cache".to_string());
    format!("{base}/huggingface")
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
    let base = std::env::var("AMGIX_CACHE_DIR").unwrap_or_else(|_| "/data/amgix/cache".to_string());
    if std::env::var("HF_HOME").is_err() {
        unsafe { std::env::set_var("HF_HOME", format!("{base}/huggingface")); }
    }
    if std::env::var("CUDA_CACHE_PATH").is_err() {
        unsafe { std::env::set_var("CUDA_CACHE_PATH", format!("{base}/cuda")); }
    }
}

// ---------------------------------------------------------------------------
// Dense model cache
// ---------------------------------------------------------------------------

const ST_POOLING_CONFIG_PATH: &str = "1_Pooling/config.json";

/// Dense model plus Sentence-Transformers pooling config loaded once at model load time.
pub struct DenseModelHandle {
    pub embedder: BertEmbedder,
    pub pooling: StPoolingConfig,
}

fn load_st_pooling_config(model_id: &str, revision: Option<&str>) -> StPoolingConfig {
    set_hf_home();
    let Ok(api) = ApiBuilder::new().with_progress(false).build() else {
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

type DenseModelEntry = (Arc<DenseModelHandle>, Instant);

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

pub struct DenseModelCache {
    inner: RwLock<HashMap<(String, Option<String>), DenseModelEntry>>,
    max_size: usize,
}

impl DenseModelCache {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
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

        if self.inner.read().unwrap().contains_key(&key) {
            let mut cache = self.inner.write().unwrap();
            if let Some(handle) = bump_cache_hit(&mut cache, &key) {
                return Ok(handle);
            }
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

        {
            let mut cache = self.inner.write().unwrap();
            if let Some(handle) = bump_cache_hit(&mut cache, &key) {
                return Ok(handle);
            }
            evict_lru_if_full(&mut cache, self.max_size);
        }

        set_hf_home();
        let pooling = load_st_pooling_config(model_id, revision);
        let embedder = BertEmbedder::new(
            model_id.to_string(),
            revision.map(|s| s.to_string()),
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;

        let handle = Arc::new(DenseModelHandle { embedder, pooling });
        let mut cache = self.inner.write().unwrap();
        if let Some(existing) = bump_cache_hit(&mut cache, &key) {
            return Ok(existing);
        }
        cache.insert(key, (Arc::clone(&handle), Instant::now()));
        Ok(handle)
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
    inner: RwLock<HashMap<(String, Option<String>), SparseModelEntry>>,
    max_size: usize,
}

impl SparseModelCache {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
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

        if self.inner.read().unwrap().contains_key(&key) {
            let mut cache = self.inner.write().unwrap();
            if let Some(model) = bump_cache_hit(&mut cache, &key) {
                return Ok(model);
            }
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

        {
            let mut cache = self.inner.write().unwrap();
            if let Some(model) = bump_cache_hit(&mut cache, &key) {
                return Ok(model);
            }
            evict_lru_if_full(&mut cache, self.max_size);
        }

        set_hf_home();
        let embedder = SparseBertEmbedder::new(
            model_id.to_string(),
            revision.map(|s| s.to_string()),
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;

        let arc = Arc::new(embedder);
        let mut cache = self.inner.write().unwrap();
        if let Some(existing) = bump_cache_hit(&mut cache, &key) {
            return Ok(existing);
        }
        cache.insert(key, (Arc::clone(&arc), Instant::now()));
        Ok(arc)
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
