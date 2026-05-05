use crate::models::VectorConfigInternal;
use crate::vectors::vector_base::VectorBase;

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
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("DenseModelVector: not implemented".to_string())
    }
}
