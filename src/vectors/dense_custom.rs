use std::collections::HashMap;

use crate::models::{Document, VectorConfigInternal};
use crate::vectors::vector_base::VectorBase;

pub struct CustomDenseVector;

impl VectorBase for CustomDenseVector {
    fn get_sparse_vector_single(
        &self,
        _config: &VectorConfigInternal,
        _text: &str,
        _avgdl: f64,
        _trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        Err("CustomDenseVector does not produce sparse vectors".to_string())
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("CustomDenseVector does not generate from text".to_string())
    }
}

impl CustomDenseVector {
    /// Mirrors CustomDenseVector.extract_for_documents.
    /// Returns {doc_idx: {field: dense_vector}}.
    pub fn extract_for_documents(
        config: &VectorConfigInternal,
        documents: &[Document],
    ) -> Result<HashMap<usize, HashMap<String, Vec<f32>>>, String> {
        let mut per_doc: HashMap<usize, HashMap<String, Vec<f32>>> = HashMap::new();
        for (idx, doc) in documents.iter().enumerate() {
            let mut per_field: HashMap<String, Vec<f32>> = HashMap::new();
            let custom_vectors = doc.custom_vectors.as_deref().ok_or_else(|| {
                format!(
                    "Custom dense vector '{}' requires custom vectors but document has none",
                    config.name
                )
            })?;
            for field in &config.index_fields {
                let field_str = field.to_string();
                let cv = custom_vectors
                    .iter()
                    .find(|cv| cv.vector_name == config.name && cv.field == *field)
                    .ok_or_else(|| {
                        format!(
                            "Custom dense vector '{}' for field '{}' not provided",
                            config.name, field_str
                        )
                    })?;
                let vec: Vec<f32> = serde_json::from_value(cv.vector.clone())
                    .map_err(|e| format!("Failed to parse dense vector: {e}"))?;
                if let Some(dims) = config.dimensions {
                    if vec.len() != dims as usize {
                        return Err(format!(
                            "Custom dense vector '{}' has {} dimensions, expected {}",
                            config.name,
                            vec.len(),
                            dims
                        ));
                    }
                }
                per_field.insert(field_str, vec);
            }
            per_doc.insert(idx, per_field);
        }
        Ok(per_doc)
    }

    /// Mirrors CustomDenseVector.extract_for_query.
    pub fn extract_for_query(
        config: &VectorConfigInternal,
        custom_vectors: &[crate::models::CustomVector],
    ) -> Result<Vec<f32>, String> {
        let cv = custom_vectors
            .iter()
            .find(|cv| cv.vector_name == config.name)
            .ok_or_else(|| {
                format!("Custom dense vector '{}' not provided in query", config.name)
            })?;
        let vec: Vec<f32> = serde_json::from_value(cv.vector.clone())
            .map_err(|e| format!("Failed to parse dense vector: {e}"))?;
        if let Some(dims) = config.dimensions {
            if vec.len() != dims as usize {
                return Err(format!(
                    "Custom dense vector '{}' has {} dimensions, expected {}",
                    config.name,
                    vec.len(),
                    dims
                ));
            }
        }
        Ok(vec)
    }
}
