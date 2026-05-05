use std::collections::HashMap;

use crate::models::{Document, VectorConfigInternal};
use crate::vectors::vector_base::VectorBase;

pub struct CustomSparseVector;

impl VectorBase for CustomSparseVector {
    fn get_sparse_vector_single(
        &self,
        _config: &VectorConfigInternal,
        _text: &str,
        _avgdl: f64,
        _trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        Err("CustomSparseVector does not generate from text".to_string())
    }

    fn get_dense_vector(
        &self,
        _config: &VectorConfigInternal,
        _docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("CustomSparseVector does not produce dense vectors".to_string())
    }
}

impl CustomSparseVector {
    /// Mirrors CustomSparseVector.extract_for_documents.
    /// Returns {doc_idx: {field: (indices, values)}}.
    pub fn extract_for_documents(
        config: &VectorConfigInternal,
        documents: &[Document],
    ) -> Result<HashMap<usize, HashMap<String, (Vec<u32>, Vec<f32>)>>, String> {
        let mut per_doc: HashMap<usize, HashMap<String, (Vec<u32>, Vec<f32>)>> = HashMap::new();
        for (idx, doc) in documents.iter().enumerate() {
            let mut per_field: HashMap<String, (Vec<u32>, Vec<f32>)> = HashMap::new();
            let custom_vectors = doc.custom_vectors.as_deref().ok_or_else(|| {
                format!(
                    "Custom sparse vector '{}' requires custom vectors but document has none",
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
                            "Custom sparse vector '{}' for field '{}' not provided",
                            config.name, field_str
                        )
                    })?;
                // cv.vector is a list of [index, value] pairs, matching Python's cv.vector
                let pairs: Vec<(u32, f32)> = serde_json::from_value(cv.vector.clone())
                    .map_err(|e| format!("Failed to parse sparse vector: {e}"))?;
                if pairs.len() > config.top_k as usize {
                    return Err(format!(
                        "Custom sparse vector '{}' has {} entries, max allowed: {}",
                        config.name,
                        pairs.len(),
                        config.top_k
                    ));
                }
                let indices: Vec<u32> = pairs.iter().map(|(i, _)| *i).collect();
                let values: Vec<f32> = pairs.iter().map(|(_, v)| *v).collect();
                per_field.insert(field_str, (indices, values));
            }
            per_doc.insert(idx, per_field);
        }
        Ok(per_doc)
    }

    /// Mirrors CustomSparseVector.extract_for_query.
    pub fn extract_for_query(
        config: &VectorConfigInternal,
        custom_vectors: &[crate::models::CustomVector],
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let cv = custom_vectors
            .iter()
            .find(|cv| cv.vector_name == config.name)
            .ok_or_else(|| {
                format!("Custom sparse vector '{}' not provided in query", config.name)
            })?;
        let pairs: Vec<(u32, f32)> = serde_json::from_value(cv.vector.clone())
            .map_err(|e| format!("Failed to parse sparse vector: {e}"))?;
        if pairs.len() > config.top_k as usize {
            return Err(format!(
                "Custom sparse vector '{}' has {} entries, max allowed: {}",
                config.name,
                pairs.len(),
                config.top_k
            ));
        }
        let indices: Vec<u32> = pairs.iter().map(|(i, _)| *i).collect();
        let values: Vec<f32> = pairs.iter().map(|(_, v)| *v).collect();
        Ok((indices, values))
    }
}
