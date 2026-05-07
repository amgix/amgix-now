use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use embed_anything::embeddings::local::bert::{BertEmbedder, SparseBertEmbedder};

// ---------------------------------------------------------------------------
// GPU inference serialization
// ---------------------------------------------------------------------------

static GPU_INFERENCE: OnceLock<bool> = OnceLock::new();

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

/// Global mutex serializing GPU model inference across all rayon threads and pools.
/// Only acquired when [`is_gpu_inference`] returns true.
static MODEL_INFERENCE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn model_inference_lock() -> &'static Mutex<()> {
    MODEL_INFERENCE_LOCK.get_or_init(|| Mutex::new(()))
}

const MODEL_CACHE_TTL: Duration = Duration::from_secs(60 * 60); // 1 hour, mirrors MODEL_CACHE_TTL

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
            eprintln!("Loaded {} trusted organizations from {file_path}", orgs.len());
            Some(orgs)
        }
        Err(e) => {
            eprintln!("Failed to load trusted organizations from {file_path}: {e}. Using empty set.");
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
    let dir = hf_cache_dir();
    // Only set if not already set by the user
    if std::env::var("HF_HOME").is_err() {
        unsafe { std::env::set_var("HF_HOME", &dir); }
    }
}

// ---------------------------------------------------------------------------
// Dense model cache
// ---------------------------------------------------------------------------

type DenseModelEntry = (Arc<BertEmbedder>, Instant);

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
    ) -> Result<Arc<BertEmbedder>, String> {
        let key = (model_id.to_string(), revision.map(|s| s.to_string()));

        // Fast path: read lock — check cache
        {
            let cache = self.inner.read().unwrap();
            if let Some((model, loaded_at)) = cache.get(&key) {
                if loaded_at.elapsed() < MODEL_CACHE_TTL {
                    return Ok(Arc::clone(model));
                }
            }
        }

        // Slow path: write lock — check again then load
        let mut cache = self.inner.write().unwrap();
        if let Some((model, loaded_at)) = cache.get(&key) {
            if loaded_at.elapsed() < MODEL_CACHE_TTL {
                return Ok(Arc::clone(model));
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

        set_hf_home();
        let embedder = BertEmbedder::new(
            model_id.to_string(),
            revision.map(|s| s.to_string()),
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;

        // Evict expired and oldest entries if at capacity
        Self::evict_if_needed(&mut cache, self.max_size);

        let arc = Arc::new(embedder);
        cache.insert(key, (Arc::clone(&arc), Instant::now()));
        Ok(arc)
    }

    fn evict_if_needed(
        cache: &mut HashMap<(String, Option<String>), DenseModelEntry>,
        max_size: usize,
    ) {
        // Remove expired entries first
        cache.retain(|_, (_, loaded_at)| loaded_at.elapsed() < MODEL_CACHE_TTL);

        // If still at capacity, remove the oldest entry
        while cache.len() >= max_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                cache.remove(&k);
            } else {
                break;
            }
        }
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

        {
            let cache = self.inner.read().unwrap();
            if let Some((model, loaded_at)) = cache.get(&key) {
                if loaded_at.elapsed() < MODEL_CACHE_TTL {
                    return Ok(Arc::clone(model));
                }
            }
        }

        let mut cache = self.inner.write().unwrap();
        if let Some((model, loaded_at)) = cache.get(&key) {
            if loaded_at.elapsed() < MODEL_CACHE_TTL {
                return Ok(Arc::clone(model));
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

        set_hf_home();
        let embedder = SparseBertEmbedder::new(
            model_id.to_string(),
            revision.map(|s| s.to_string()),
            None,
        )
        .map_err(|e| format!("Failed to load model '{model_id}': {e}"))?;

        Self::evict_if_needed(&mut cache, self.max_size);

        let arc = Arc::new(embedder);
        cache.insert(key, (Arc::clone(&arc), Instant::now()));
        Ok(arc)
    }

    fn evict_if_needed(
        cache: &mut HashMap<(String, Option<String>), SparseModelEntry>,
        max_size: usize,
    ) {
        cache.retain(|_, (_, loaded_at)| loaded_at.elapsed() < MODEL_CACHE_TTL);
        while cache.len() >= max_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                cache.remove(&k);
            } else {
                break;
            }
        }
    }
}
