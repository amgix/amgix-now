use crate::models::VectorConfigInternal;
use crate::vectors::vector_base::VectorBase;

pub struct NoopVector;

impl VectorBase for NoopVector {
    fn get_sparse_vector_single(
        &self,
        _config: &VectorConfigInternal,
        _text: &str,
        _avgdl: f64,
        _trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        Ok((vec![], vec![]))
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("NoopVector only supports sparse vectors".to_string())
    }
}
