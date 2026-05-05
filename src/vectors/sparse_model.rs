use crate::models::VectorConfigInternal;
use crate::vectors::vector_base::{preprocess_text, preprocess_text_keep_case, VectorBase};

pub struct SparseModelVector;

impl SparseModelVector {
    /// Batch sparse embedding mirroring Python `SparseModelVector.get_sparse_vector` (signature
    /// without `avgdls`) — preprocess with `keep_case` from config, then model inference (stub).
    pub fn get_sparse_vector_batch(
        config: &VectorConfigInternal,
        docs: &[String],
        trigram_weight: f64,
    ) -> Result<Vec<(Vec<u32>, Vec<f32>)>, String> {
        let _trigram_weight = trigram_weight;
        let _processed_docs: Vec<String> = docs
            .iter()
            .map(|doc| {
                if config.keep_case.unwrap_or(false) {
                    preprocess_text_keep_case(doc)
                } else {
                    preprocess_text(doc)
                }
            })
            .collect();
        SparseModelVector::get_sparse_vector_batch_impl(config, &_processed_docs)
    }

    fn get_sparse_vector_batch_impl(
        _config: &VectorConfigInternal,
        _processed_docs: &[String],
    ) -> Result<Vec<(Vec<u32>, Vec<f32>)>, String> {
        Err("SparseModelVector: not implemented".to_string())
    }
}

impl VectorBase for SparseModelVector {
    fn get_sparse_vector_single(
        &self,
        _config: &VectorConfigInternal,
        _text: &str,
        _avgdl: f64,
        _trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        Err("SparseModelVector: not implemented".to_string())
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("SparseModelVector does not produce dense vectors".to_string())
    }
}
