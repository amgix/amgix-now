use std::sync::OnceLock;
use std::time::Instant;

use candle_core::{DType, Tensor};

use crate::models::VectorConfigInternal;
use crate::vectors::model_cache::{maybe_gpu_inference_permit, trusted_organizations, DenseModelCache, DenseModelHandle};
use crate::vectors::st_pooling::pool_sentence_embeddings;
use crate::vectors::vector_base::VectorBase;

const DENSE_MODEL_BATCH_SIZE: usize = 8;

static DENSE_CACHE: OnceLock<DenseModelCache> = OnceLock::new();

fn cache() -> &'static DenseModelCache {
    DENSE_CACHE.get_or_init(DenseModelCache::new)
}

/// Returns currently loaded (non-expired) dense models: `(type, model, revision, loaded_at)`.
pub fn dense_model_cache_snapshot() -> Vec<(String, String, Option<String>, Instant)> {
    DENSE_CACHE
        .get()
        .map(|c| c.snapshot())
        .unwrap_or_default()
}

pub struct DenseModelVector;

impl VectorBase for DenseModelVector {
    fn get_sparse_vector_single(
        &self,
        _config: &VectorConfigInternal,
        _text: &str,
        _avgdl: f64,
        _trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        Err("DenseModelVector does not produce sparse vectors".to_string())
    }

    fn get_dense_vector(
        &self,
        config: &VectorConfigInternal,
        docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        let model_id = config
            .model
            .as_deref()
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| {
                "DenseModelVector requires 'model' to be specified in VectorConfig".to_string()
            })?;

        let handle =
            cache().get_or_load(model_id, config.revision.as_deref(), trusted_organizations())?;

        let normalize = config.normalization.unwrap_or(true);
        embed_dense_batch(&handle, docs, normalize, model_id)
    }
}

/// Runs dense embedding directly on the model's candle tensors, keeping all intermediate
/// results on device (GPU or CPU) and only copying to host once per mini-batch at the end.
/// This avoids the per-batch GPU→CPU copy that `BertEmbed::embed()` does internally.
fn embed_dense_batch(
    handle: &DenseModelHandle,
    docs: &[String],
    normalize: bool,
    model_id: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let embedder = &handle.embedder;
    let pooling = &handle.pooling;
    let device = &embedder.model.device;
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(docs.len());

    for chunk in docs.chunks(DENSE_MODEL_BATCH_SIZE) {
        let chunk_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let batch = chunk_refs.len();

        // Tokenize — encode_batch with add_special_tokens=true pads all sequences to the
        // same length, so every encoding has identical len.
        let tokens = embedder
            .tokenizer
            .encode_batch(chunk_refs, true)
            .map_err(|e| format!("Tokenization failed for '{model_id}': {e}"))?;

        let seq_len = tokens[0].get_ids().len();

        // Build flat [batch * seq_len] buffers — one H2D transfer each instead of
        // batch-many small transfers followed by Tensor::stack.
        let mut flat_ids: Vec<u32> = Vec::with_capacity(batch * seq_len);
        let mut flat_mask: Vec<u32> = Vec::with_capacity(batch * seq_len);
        for t in &tokens {
            flat_ids.extend_from_slice(t.get_ids());
            flat_mask.extend_from_slice(t.get_attention_mask());
        }

        // Serialize GPU model inference across all threads when running on GPU.
        let _gpu_guard = maybe_gpu_inference_permit();

        let token_ids = Tensor::from_slice(&flat_ids, (batch, seq_len), device)
            .map_err(|e| format!("token_ids tensor failed: {e}"))?;
        let attention_mask = Tensor::from_slice(&flat_mask, (batch, seq_len), device)
            .map_err(|e| format!("attention_mask tensor failed: {e}"))?;
        let token_type_ids = token_ids
            .zeros_like()
            .map_err(|e| format!("zeros_like failed: {e}"))?;

        // Forward pass — stays on device.
        // Output shape: [batch, seq_len, hidden_size]
        let hidden = embedder
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("Model forward failed for '{model_id}': {e}"))?;

        // ST pooling on device — reads 1_Pooling/config.json at model load time.
        let pooled = pool_sentence_embeddings(&hidden, &attention_mask, pooling)
            .map_err(|e| format!("Pooling failed for '{model_id}': {e}"))?;

        // Optional L2 normalization on device.
        let pooled = if normalize {
            l2_normalize_tensor(&pooled)
                .map_err(|e| format!("L2 normalize failed for '{model_id}': {e}"))?
        } else {
            pooled
        };

        // Copy to host; cast to F32 only if the output isn't already F32.
        let pooled_f32 = if pooled.dtype() == DType::F32 {
            pooled
        } else {
            pooled
                .to_dtype(DType::F32)
                .map_err(|e| format!("to_dtype(f32) failed: {e}"))?
        };

        drop(_gpu_guard);

        let batch_vecs = pooled_f32
            .to_vec2::<f32>()
            .map_err(|e| format!("to_vec2 failed: {e}"))?;

        results.extend(batch_vecs);
    }

    Ok(results)
}

/// L2-normalize a [batch, dim] tensor on-device.
fn l2_normalize_tensor(t: &Tensor) -> candle_core::Result<Tensor> {
    t.broadcast_div(&t.sqr()?.sum_keepdim(1)?.sqrt()?)
}
