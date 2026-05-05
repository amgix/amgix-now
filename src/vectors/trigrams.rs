use crate::amgix::tokenize_trigrams as aa_tokenize_trigrams;
use crate::models::VectorConfigInternal;
use crate::vectors::vector_base::{validate_text, VectorBase};

pub struct TrigramsVector;

impl VectorBase for TrigramsVector {
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
        // Trigrams is language-agnostic — no lang_code needed.
        let (indices, values) = aa_tokenize_trigrams(
            text.to_string(),
            config.top_k as usize,
            avgdl as f32,
        );
        Ok((indices, values))
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("TrigramsVector only supports sparse vectors".to_string())
    }
}
