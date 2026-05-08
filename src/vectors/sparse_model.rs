use std::collections::HashSet;
use std::sync::OnceLock;

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::models::VectorConfigInternal;
use crate::vectors::model_cache::{maybe_gpu_inference_permit, trusted_organizations, SparseModelCache};
use crate::vectors::vector_base::{preprocess_text, preprocess_text_keep_case, VectorBase};

const SPARSE_MODEL_BATCH_SIZE: usize = 8;

static SPARSE_CACHE: OnceLock<SparseModelCache> = OnceLock::new();

fn cache() -> &'static SparseModelCache {
    SPARSE_CACHE.get_or_init(SparseModelCache::new)
}

pub struct SparseModelVector;

impl SparseModelVector {
    /// Mirrors Python `SparseModelVector.get_sparse_vector` (without avgdl).
    pub fn get_sparse_vector_batch(
        config: &VectorConfigInternal,
        docs: &[String],
        _trigram_weight: f64,
    ) -> Result<Vec<(Vec<u32>, Vec<f32>)>, String> {
        let model_id = config
            .model
            .as_deref()
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| {
                "SparseModelVector requires 'model' to be specified in VectorConfig".to_string()
            })?;

        let processed_docs: Vec<String> = docs
            .iter()
            .map(|doc| {
                if config.keep_case.unwrap_or(false) {
                    preprocess_text_keep_case(doc)
                } else {
                    preprocess_text(doc)
                }
            })
            .collect();

        let embedder =
            cache().get_or_load(model_id, config.revision.as_deref(), trusted_organizations())?;

        let tokenizer = &embedder.tokenizer;
        let model = &embedder.model;
        let device = &embedder.device;
        let top_k = config.top_k as usize;

        let special_ids = collect_special_ids(tokenizer);

        let mut results: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(docs.len());

        for chunk in processed_docs.chunks(SPARSE_MODEL_BATCH_SIZE) {
            let chunk_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();

            let tokens = tokenizer
                .encode_batch(chunk_refs, true)
                .map_err(|e| format!("Tokenization failed for '{model_id}': {e}"))?;

            // Serialize GPU model inference across all threads when running on GPU.
            let _gpu_guard = maybe_gpu_inference_permit();

            let token_id_tensors: Vec<Tensor> = tokens
                .iter()
                .map(|t| Tensor::new(t.get_ids(), device))
                .collect::<candle_core::Result<_>>()
                .map_err(|e| format!("Tensor creation failed: {e}"))?;
            // Attention mask as u8 — where_cond requires the condition to be a u8/bool tensor
            let attention_mask_tensors: Vec<Tensor> = tokens
                .iter()
                .map(|t| {
                    let mask_u8: Vec<u8> = t.get_attention_mask().iter().map(|&v| v as u8).collect();
                    Tensor::new(mask_u8.as_slice(), device)
                })
                .collect::<candle_core::Result<_>>()
                .map_err(|e| format!("Tensor creation failed: {e}"))?;

            let token_ids_batch = Tensor::stack(&token_id_tensors, 0)
                .map_err(|e| format!("Tensor stack failed: {e}"))?;
            let attention_mask_batch = Tensor::stack(&attention_mask_tensors, 0)
                .map_err(|e| format!("Tensor stack failed: {e}"))?;

            let token_type_ids = token_ids_batch
                .zeros_like()
                .map_err(|e| format!("zeros_like failed: {e}"))?;

            // logits: [batch, seq_len, vocab_size]
            let logits = model
                .forward(&token_ids_batch, &token_type_ids, Some(&attention_mask_batch))
                .map_err(|e| format!("Model forward failed for '{model_id}': {e}"))?;

            let batch_size = chunk.len();
            for i in 0..batch_size {
                // doc_logits: [1, seq_len, vocab_size]
                let doc_logits = logits
                    .narrow(0, i, 1)
                    .map_err(|e| format!("narrow(batch) failed: {e}"))?;

                // doc_mask: [1, seq_len] → [1, seq_len, 1] for broadcast
                let doc_mask = attention_mask_batch
                    .narrow(0, i, 1)
                    .map_err(|e| format!("narrow(mask) failed: {e}"))?
                    .unsqueeze(2)
                    .map_err(|e| format!("unsqueeze failed: {e}"))?;

                // masked_fill: fill where mask == 0 with -inf, mirrors Python:
                //   masked_logits = doc_logits.masked_fill(doc_mask == 0, float("-inf"))
                let neg_inf = Tensor::full(f32::NEG_INFINITY, doc_logits.shape(), device)
                    .map_err(|e| format!("full tensor failed: {e}"))?;
                let mask_f32 = doc_mask
                    .broadcast_as(doc_logits.shape())
                    .map_err(|e| format!("broadcast_as failed: {e}"))?;
                let masked_logits = mask_f32
                    .where_cond(&doc_logits, &neg_inf)
                    .map_err(|e| format!("where_cond failed: {e}"))?;

                // max over seq dim → [vocab_size], mirrors .max(dim=1).values.squeeze(0)
                let per_token_scores = masked_logits
                    .max(1)
                    .map_err(|e| format!("max(1) failed: {e}"))?
                    .squeeze(0)
                    .map_err(|e| format!("squeeze failed: {e}"))?;

                // relu → log1p, mirrors torch.relu / torch.log1p
                let per_token_scores = per_token_scores
                    .relu()
                    .map_err(|e| format!("relu failed: {e}"))?;
                let one = Tensor::ones_like(&per_token_scores)
                    .map_err(|e| format!("ones_like failed: {e}"))?;
                let per_token_scores = one
                    .add(&per_token_scores)
                    .map_err(|e| format!("add(1) failed: {e}"))?
                    .log()
                    .map_err(|e| format!("log failed: {e}"))?;

                let mut scores_vec = per_token_scores
                    .to_vec1::<f32>()
                    .map_err(|e| format!("to_vec1 failed: {e}"))?;

                // Zero out special token positions, mirrors index_fill_(0, idx_tensor, 0.0)
                for &special_id in &special_ids {
                    if (special_id as usize) < scores_vec.len() {
                        scores_vec[special_id as usize] = 0.0;
                    }
                }

                // Keep positive entries only, mirrors positive_mask
                let mut positive: Vec<(u32, f32)> = scores_vec
                    .iter()
                    .enumerate()
                    .filter(|&(_, v)| *v > 0.0)
                    .map(|(idx, &v)| (idx as u32, v))
                    .collect();

                if positive.is_empty() {
                    results.push((vec![], vec![]));
                    continue;
                }

                // Top-K by score descending, mirrors torch.topk(largest=True, sorted=True)
                let k = top_k.min(positive.len());
                positive.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                positive.truncate(k);

                let idxs: Vec<u32> = positive.iter().map(|(i, _)| *i).collect();
                let vals: Vec<f32> = positive.iter().map(|(_, v)| *v).collect();
                results.push((idxs, vals));
            }
            drop(_gpu_guard);
        }

        Ok(results)
    }
}

/// Collect all special token IDs from the tokenizer.
/// Mirrors Python: `set(getattr(tokenizer, "all_special_ids", []) or [])` + pad_token_id.
fn collect_special_ids(tokenizer: &Tokenizer) -> HashSet<u32> {
    tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, tok)| if tok.special { Some(id) } else { None })
        .collect()
}

impl VectorBase for SparseModelVector {
    fn get_sparse_vector_single(
        &self,
        config: &VectorConfigInternal,
        text: &str,
        _avgdl: f64,
        trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        Self::get_sparse_vector_batch(config, &[text.to_string()], trigram_weight)?
            .into_iter()
            .next()
            .ok_or_else(|| "Empty sparse model result".to_string())
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("SparseModelVector does not produce dense vectors".to_string())
    }
}
