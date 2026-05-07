use std::sync::OnceLock;

use candle_core::{DType, Tensor};
use embed_anything::embeddings::local::bert::BertEmbedder;

use crate::models::VectorConfigInternal;
use crate::vectors::model_cache::{is_gpu_inference, model_inference_lock, trusted_organizations, DenseModelCache};
use crate::vectors::vector_base::VectorBase;

const DENSE_MODEL_BATCH_SIZE: usize = 32;

static DENSE_CACHE: OnceLock<DenseModelCache> = OnceLock::new();

fn cache() -> &'static DenseModelCache {
    DENSE_CACHE.get_or_init(DenseModelCache::new)
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

        let embedder =
            cache().get_or_load(model_id, config.revision.as_deref(), trusted_organizations())?;

        let normalize = config.normalization.unwrap_or(true);
        embed_dense_batch(&embedder, docs, normalize, model_id)
    }
}

/// Runs dense embedding directly on the model's candle tensors, keeping all intermediate
/// results on device (GPU or CPU) and only copying to host once per mini-batch at the end.
/// This avoids the per-batch GPU→CPU copy that `BertEmbed::embed()` does internally.
fn embed_dense_batch(
    embedder: &BertEmbedder,
    docs: &[String],
    normalize: bool,
    model_id: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let device = &embedder.model.device;
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(docs.len());

    for chunk in docs.chunks(DENSE_MODEL_BATCH_SIZE) {
        let chunk_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();

        // Tokenize — produces padded tensors already on device.
        let tokens = embedder
            .tokenizer
            .encode_batch(chunk_refs, true)
            .map_err(|e| format!("Tokenization failed for '{model_id}': {e}"))?;

        // Serialize GPU model inference across all threads when running on GPU.
        let _gpu_guard = if is_gpu_inference() {
            Some(model_inference_lock().lock().unwrap())
        } else {
            None
        };

        let token_id_vecs: Vec<Tensor> = tokens
            .iter()
            .map(|t| Tensor::new(t.get_ids(), device))
            .collect::<candle_core::Result<_>>()
            .map_err(|e| format!("Tensor creation failed: {e}"))?;
        let mask_vecs: Vec<Tensor> = tokens
            .iter()
            .map(|t| Tensor::new(t.get_attention_mask(), device))
            .collect::<candle_core::Result<_>>()
            .map_err(|e| format!("Tensor creation failed: {e}"))?;

        let token_ids = Tensor::stack(&token_id_vecs, 0)
            .map_err(|e| format!("stack token_ids failed: {e}"))?;
        let attention_mask = Tensor::stack(&mask_vecs, 0)
            .map_err(|e| format!("stack attention_mask failed: {e}"))?;
        let token_type_ids = token_ids
            .zeros_like()
            .map_err(|e| format!("zeros_like failed: {e}"))?;

        // Forward pass — stays on device.
        // Output shape: [batch, seq_len, hidden_size]
        let hidden = embedder
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("Model forward failed for '{model_id}': {e}"))?;

        // Mean pooling on device — mirrors embed_anything's Pooling::Mean.
        // expanded_mask: [batch, seq_len, hidden_size], same dtype as hidden
        let expanded_mask = attention_mask
            .unsqueeze(2)
            .map_err(|e| format!("unsqueeze failed: {e}"))?
            .expand(hidden.shape())
            .map_err(|e| format!("expand failed: {e}"))?
            .to_dtype(hidden.dtype())
            .map_err(|e| format!("to_dtype failed: {e}"))?;

        let mask_sum = expanded_mask
            .sum_all()
            .map_err(|e| format!("sum_all failed: {e}"))?
            .clamp(1e-10_f64, f64::MAX)
            .map_err(|e| format!("clamp failed: {e}"))?;

        // pooled: [batch, hidden_size]
        let pooled = hidden
            .mul(&expanded_mask)
            .map_err(|e| format!("mul failed: {e}"))?
            .sum(1)
            .map_err(|e| format!("sum(1) failed: {e}"))?
            .broadcast_div(&mask_sum)
            .map_err(|e| format!("broadcast_div failed: {e}"))?;

        // Optional L2 normalization on device.
        let pooled = if normalize {
            l2_normalize_tensor(&pooled)
                .map_err(|e| format!("L2 normalize failed for '{model_id}': {e}"))?
        } else {
            pooled
        };

        // Single host copy for the whole mini-batch.
        let batch_vecs = pooled
            .to_dtype(DType::F32)
            .map_err(|e| format!("to_dtype(f32) failed: {e}"))?
            .to_vec2::<f32>()
            .map_err(|e| format!("to_vec2 failed: {e}"))?;

        drop(_gpu_guard);

        results.extend(batch_vecs);
    }

    Ok(results)
}

/// L2-normalize a [batch, dim] tensor on-device.
fn l2_normalize_tensor(t: &Tensor) -> candle_core::Result<Tensor> {
    t.broadcast_div(&t.sqr()?.sum_keepdim(1)?.sqrt()?)
}
