use crate::amgix::tokenize_fulltext as aa_tokenize_fulltext;
use crate::models::VectorConfigInternal;
use crate::vectors::vector_base::{get_language_code, validate_text, VectorBase};

pub struct FullTextVector;

impl VectorBase for FullTextVector {
    fn get_sparse_vector_single(
        &self,
        config: &VectorConfigInternal,
        text: &str,
        avgdl: f64,
        _trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        if !validate_text(text) {
            return Ok((vec![], vec![]));
        }
        let lang_code = get_language_code(config, text)?;
        let (indices, values) = aa_tokenize_fulltext(
            text.to_string(),
            lang_code,
            config.top_k as usize,
            true,
            avgdl as f32,
        );
        Ok((indices, values))
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("FullTextVector does not support dense vectors".to_string())
    }
}
