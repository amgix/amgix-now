//! Mirrors `amgix-server/src/core/vector/vectorizer.py`.
//!
//! Embedding is routed through [`route_embed_dispatch`] instead of a Python `EmbedRouter` closure.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::common::{
    DocumentField, VectorType, DEFAULT_WMTR_TRIGRAM_WEIGHT,
};
use crate::models::{
    Document, DocumentWithVectors, SearchQuery, SearchQueryWithVectors, VectorConfigInternal,
    VectorData, VectorSearchWeight,
};
use crate::vectors::dense_custom::CustomDenseVector;
use crate::vectors::dense_model::DenseModelVector;
use crate::vectors::full_text::FullTextVector;
use crate::vectors::sparse_custom::CustomSparseVector;
use crate::vectors::sparse_model::SparseModelVector;
use crate::vectors::trigrams::TrigramsVector;
use crate::vectors::vector_base::VectorBase;
use crate::vectors::whitespace::WhiteSpaceVector;
use crate::vectors::wmtr::WMTRVector;

#[derive(Debug)]
pub enum RoutedEmbed {
    Dense(Vec<Vec<f32>>),
    Sparse(Vec<(Vec<u32>, Vec<f32>)>),
}

/// Mirrors `EmbedRouterService.embed` routing (dense vs sparse vs custom-token batch path).
/// Do **not** call this for `dense_custom` / `sparse_custom` — those are handled like Python.
pub fn route_embed_dispatch(
    config: &VectorConfigInternal,
    docs: &[String],
    avgdls: Option<&[f64]>,
    trigram_weight: f64,
) -> Result<RoutedEmbed, String> {
    match config.vector_type {
        VectorType::DenseModel => Ok(RoutedEmbed::Dense(
            DenseModelVector.get_dense_vector(config, docs)?,
        )),
        VectorType::SparseModel => Ok(RoutedEmbed::Sparse(
            SparseModelVector::get_sparse_vector_batch(config, docs, trigram_weight)?,
        )),
        VectorType::FullText => {
            let av = avgdls.ok_or_else(|| {
                format!(
                    "avgdl entries required for custom tokenization vector '{}' ({:?})",
                    config.name,
                    VectorType::FullText,
                )
            })?;
            Ok(RoutedEmbed::Sparse(
                FullTextVector.get_sparse_vector(config, docs, av, trigram_weight)?,
            ))
        }
        VectorType::Trigrams => {
            let av = avgdls.ok_or_else(|| {
                format!(
                    "avgdl entries required for custom tokenization vector '{}' ({:?})",
                    config.name,
                    VectorType::Trigrams,
                )
            })?;
            Ok(RoutedEmbed::Sparse(
                TrigramsVector.get_sparse_vector(config, docs, av, trigram_weight)?,
            ))
        }
        VectorType::Whitespace => {
            let av = avgdls.ok_or_else(|| {
                format!(
                    "avgdl entries required for custom tokenization vector '{}' ({:?})",
                    config.name,
                    VectorType::Whitespace,
                )
            })?;
            Ok(RoutedEmbed::Sparse(
                WhiteSpaceVector.get_sparse_vector(config, docs, av, trigram_weight)?,
            ))
        }
        VectorType::Wmtr | VectorType::Keyword => {
            let av = avgdls.ok_or_else(|| {
                format!(
                    "avgdl entries required for custom tokenization vector '{}' ({:?})",
                    config.name,
                    config.vector_type,
                )
            })?;
            Ok(RoutedEmbed::Sparse(
                WMTRVector.get_sparse_vector(config, docs, av, trigram_weight)?,
            ))
        }
        VectorType::DenseCustom | VectorType::SparseCustom => Err(
            "route_embed_dispatch must not be called for dense_custom/sparse_custom".to_string(),
        ),
    }
}

fn dense_config_with_query_embedding_model(mut c: VectorConfigInternal) -> VectorConfigInternal {
    if c.query_model.is_some() {
        c.model = c.query_model.clone();
        c.revision = c.query_revision.clone();
    }
    c
}

fn sparse_model_config_query_embedding(mut c: VectorConfigInternal) -> VectorConfigInternal {
    if c.vector_type == VectorType::SparseModel && c.query_model.is_some() {
        c.model = c.query_model.clone();
        c.revision = c.query_revision.clone();
    }
    c
}

fn get_field_text(document: &Document, field: DocumentField) -> String {
    match field {
        DocumentField::Name => document.name.clone().unwrap_or_default(),
        DocumentField::Description => document.description.clone().unwrap_or_default(),
        DocumentField::Content => document.content.clone().unwrap_or_default(),
    }
}

pub struct Vectorizer;

impl Vectorizer {
    /// Mirrors `Vectorizer.vectorize_documents` (sans async — routing is synchronous here).
    pub fn vectorize_documents(
        documents: &[Document],
        vector_configs: &[VectorConfigInternal],
        avgdl_dict: Option<&HashMap<String, f64>>,
    ) -> Result<Vec<DocumentWithVectors>, String> {
        let mut vectors_per_doc: Vec<Vec<VectorData>> = vec![Vec::new(); documents.len()];
        let mut token_lengths_per_doc: Vec<HashMap<String, usize>> =
            vec![HashMap::new(); documents.len()];

        for config in vector_configs {
            let cfg_result: Result<(), String> = (|| {
                match config.vector_type {
                    VectorType::DenseModel => {
                        let mut texts: Vec<String> = Vec::new();
                        for doc in documents {
                            for field in &config.index_fields {
                                texts.push(get_field_text(doc, *field));
                            }
                        }
                        let RoutedEmbed::Dense(dense_vectors) = route_embed_dispatch(
                            config,
                            &texts,
                            None,
                            DEFAULT_WMTR_TRIGRAM_WEIGHT,
                        )?
                        else {
                            return Err(format!(
                                "dense_model vector '{}' routed to non-dense embedding",
                                config.name
                            ));
                        };

                        let mut idx = 0_usize;
                        for doc_idx in 0..documents.len() {
                            for field in &config.index_fields {
                                let dense_vector = dense_vectors[idx].clone();
                                if let Some(dim) = config.dimensions {
                                    if dense_vector.len() != dim as usize {
                                        return Err(format!(
                                            "Specified dimensions {} don't match generated dimensions {} for vector '{}' field '{}'",
                                            dim,
                                            dense_vector.len(),
                                            config.name,
                                            field
                                        ));
                                    }
                                }
                                vectors_per_doc[doc_idx].push(VectorData {
                                    vector_name: config.name.clone(),
                                    field: *field,
                                    vector_type: config.vector_type.clone(),
                                    dense_vector: Some(dense_vector),
                                    sparse_indices: None,
                                    sparse_values: None,
                                });
                                idx += 1;
                            }
                        }
                    }
                    VectorType::DenseCustom => {
                        let per_doc = CustomDenseVector::extract_for_documents(config, documents)?;
                        for (doc_idx, field_map) in per_doc {
                            for field in &config.index_fields {
                                let field_key = field.to_string();
                                let vec = field_map.get(&field_key).ok_or_else(|| {
                                    format!(
                                        "Custom dense vector '{}' for field '{}' not provided",
                                        config.name, field_key
                                    )
                                })?;
                                vectors_per_doc[doc_idx].push(VectorData {
                                    vector_name: config.name.clone(),
                                    field: *field,
                                    vector_type: config.vector_type.clone(),
                                    dense_vector: Some(vec.clone()),
                                    sparse_indices: None,
                                    sparse_values: None,
                                });
                            }
                        }
                    }
                    VectorType::SparseCustom => {
                        let per_doc = CustomSparseVector::extract_for_documents(config, documents)?;
                        for (doc_idx, field_map) in per_doc {
                            for field in &config.index_fields {
                                let field_key = field.to_string();
                                let pair = field_map.get(&field_key).ok_or_else(|| {
                                    format!(
                                        "Custom sparse vector '{}' for field '{}' not provided",
                                        config.name, field_key
                                    )
                                })?;
                                let field_vector_name = format!("{}_{}", field_key, config.name);
                                let (indices, values) = pair;
                                let token_length = indices.len();
                                vectors_per_doc[doc_idx].push(VectorData {
                                    vector_name: config.name.clone(),
                                    field: *field,
                                    vector_type: config.vector_type.clone(),
                                    dense_vector: None,
                                    sparse_indices: Some(indices.clone()),
                                    sparse_values: Some(values.clone()),
                                });
                                token_lengths_per_doc[doc_idx]
                                    .insert(field_vector_name, token_length);
                            }
                        }
                    }
                    VectorType::SparseModel
                    | VectorType::FullText
                    | VectorType::Trigrams
                    | VectorType::Whitespace
                    | VectorType::Wmtr
                    | VectorType::Keyword => {
                        let mut texts: Vec<String> = Vec::new();
                        let mut avgdls: Vec<f64> = Vec::new();
                        let is_custom = config.vector_type.is_custom_tokenization();
                        for doc in documents {
                            for field in &config.index_fields {
                                texts.push(get_field_text(doc, *field));
                                if is_custom {
                                    let field_vector_name = format!("{}_{}", field, config.name);
                                    let table = avgdl_dict.ok_or_else(|| {
                                        format!(
                                            "avgdl_dict required for '{}' ({:?})",
                                            config.name, config.vector_type
                                        )
                                    })?;
                                    let avgdl_val = table.get(&field_vector_name).ok_or_else(|| {
                                        format!(
                                            "Missing avgdl_dict entry '{field_vector_name}' for vector '{}'",
                                            config.name
                                        )
                                    })?;
                                    avgdls.push(*avgdl_val);
                                }
                            }
                        }

                        let sparse_vectors = match route_embed_dispatch(
                            config,
                            &texts,
                            if is_custom { Some(avgdls.as_slice()) } else { None },
                            DEFAULT_WMTR_TRIGRAM_WEIGHT,
                        )? {
                            RoutedEmbed::Sparse(v) => v,
                            RoutedEmbed::Dense(_) => {
                                return Err(format!(
                                    "sparse vector '{}' routed to dense embedding",
                                    config.name
                                ));
                            }
                        };

                        let mut idx = 0_usize;
                        for doc_idx in 0..documents.len() {
                            for field in &config.index_fields {
                                let (indices, values) = sparse_vectors[idx].clone();
                                let field_vector_name = format!("{}_{}", field, config.name);
                                let token_length = indices.len();
                                vectors_per_doc[doc_idx].push(VectorData {
                                    vector_name: config.name.clone(),
                                    field: *field,
                                    vector_type: config.vector_type.clone(),
                                    dense_vector: None,
                                    sparse_indices: Some(indices),
                                    sparse_values: Some(values),
                                });
                                if config.vector_type.sparse_types_contains() {
                                    token_lengths_per_doc[doc_idx]
                                        .insert(field_vector_name.clone(), token_length);
                                }
                                idx += 1;
                            }
                        }
                    }
                }
                Ok(())
            })();

            if let Err(e) = cfg_result {
                return if matches!(
                    config.vector_type,
                    VectorType::DenseCustom | VectorType::SparseCustom
                ) {
                    Err(e)
                } else {
                    Err(format!(
                        "Failed to generate vector '{}' for fields {:?}: {e}",
                        config.name,
                        config
                            .index_fields
                            .iter()
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    ))
                };
            }
        }

        let mut out = Vec::with_capacity(documents.len());
        for (i, doc) in documents.iter().enumerate() {
            out.push(DocumentWithVectors {
                id: doc.id.clone(),
                timestamp: doc.timestamp,
                tags: doc.tags.clone(),
                name: doc.name.clone(),
                description: doc.description.clone(),
                content: doc.content.clone(),
                metadata: doc.metadata.clone(),
                custom_vectors: doc.custom_vectors.clone(),
                vectors: vectors_per_doc[i].clone(),
                token_lengths: token_lengths_per_doc[i].clone(),
            });
        }
        Ok(out)
    }

    /// Mirrors `Vectorizer.vectorize_search_query` (sequential encoding per vector group —
    /// same outcome as asyncio gather for deterministic CPU-heavy embedders).
    pub fn vectorize_search_query(
        mut query: SearchQuery,
        vector_configs: &[VectorConfigInternal],
        validation_mode: bool,
    ) -> Result<SearchQueryWithVectors, String> {
        if query.query.trim().is_empty() {
            return Err("Search query cannot be empty".to_string());
        }

        if validation_mode {
            for config in vector_configs {
                if config.vector_type == VectorType::DenseCustom && config.dimensions.is_none() {
                    return Err(format!(
                        "Dense custom vector '{}' requires dimensions to be specified",
                        config.name
                    ));
                }
                // Python also checks `sparse_custom` + `top_k is None`; Rust `VectorConfigInternal.top_k`
                // is always present after deserialization (default), so that branch is not represented.
            }
        }

        let mut vectors_to_generate: HashSet<(String, DocumentField)> = HashSet::new();

        if !query.vector_weights.is_empty() {
            for weight in &query.vector_weights {
                if weight.weight != 0.0 {
                    vectors_to_generate
                        .insert((weight.vector_name.clone(), weight.field));
                }
            }
        } else {
            for config in vector_configs {
                for field in &config.index_fields {
                    vectors_to_generate.insert((config.name.clone(), *field));
                }
            }
            if !vectors_to_generate.is_empty() {
                let equal_weight =
                    1.0_f64 / (vectors_to_generate.len() as f64);
                query.vector_weights = vectors_to_generate
                    .iter()
                    .map(|(name, field)| VectorSearchWeight {
                        vector_name: name.clone(),
                        field: *field,
                        weight: equal_weight,
                    })
                    .collect();
            }
        }

        let config_map: HashMap<String, &VectorConfigInternal> = vector_configs
            .iter()
            .map(|c| (c.name.clone(), c))
            .collect();

        let mut vector_groups: HashMap<String, Vec<DocumentField>> = HashMap::new();
        for (vector_name, field) in vectors_to_generate {
            vector_groups.entry(vector_name).or_default().push(field);
        }

        let groups: Vec<(String, Vec<DocumentField>)> = vector_groups.into_iter().collect();

        // Validate all groups before doing any embedding work.
        for (vector_name, fields) in &groups {
            let config = config_map.get(vector_name).ok_or_else(|| {
                format!(
                    "Vector configuration '{vector_name}' not found. Available vectors: {}",
                    config_map.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            })?;
            for field in fields {
                if !config.index_fields.contains(field) {
                    return Err(format!(
                        "Field '{field}' is not configured for vector '{vector_name}'. Available fields: {:?}",
                        config.index_fields.iter().map(|f| f.to_string()).collect::<Vec<_>>()
                    ));
                }
            }
        }

        let vectors: Vec<VectorData> = if groups.len() > 1 {
            groups
                .par_iter()
                .map(|(vector_name, fields)| {
                    let config = *config_map.get(vector_name).expect("validated above");
                    Self::generate_vector_for_query(config, &query, fields, validation_mode)
                        .map_err(|e| format!("Failed to generate vector '{vector_name}': {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect()
        } else {
            let mut out = Vec::new();
            for (vector_name, fields) in &groups {
                let config = *config_map.get(vector_name).expect("validated above");
                let batch =
                    Self::generate_vector_for_query(config, &query, fields, validation_mode)
                        .map_err(|e| format!("Failed to generate vector '{vector_name}': {e}"))?;
                out.extend(batch);
            }
            out
        };

        Ok(SearchQueryWithVectors {
            query: query.query,
            vector_weights: query.vector_weights,
            custom_vectors: query.custom_vectors,
            limit: query.limit,
            score_threshold: query.score_threshold,
            document_tags: query.document_tags,
            document_tags_match_all: query.document_tags_match_all,
            metadata_filter: query.metadata_filter,
            raw_scores: query.raw_scores,
            wmtr_trigram_weight: query.wmtr_trigram_weight,
            fusion_mode: query.fusion_mode,
            vectors,
        })
    }

    fn generate_vector_for_query(
        config: &VectorConfigInternal,
        query: &SearchQuery,
        fields: &[DocumentField],
        validation_mode: bool,
    ) -> Result<Vec<VectorData>, String> {
        let mut vector_data_list = Vec::new();

        match config.vector_type {
            VectorType::DenseModel => {
                let effective_config = dense_config_with_query_embedding_model(config.clone());

                let RoutedEmbed::Dense(dense_batch) = route_embed_dispatch(
                    &effective_config,
                    &[query.query.clone()],
                    None,
                    DEFAULT_WMTR_TRIGRAM_WEIGHT,
                )?
                else {
                    return Err("dense routing returned sparse".to_string());
                };
                let dense_vector = dense_batch
                    .into_iter()
                    .next()
                    .ok_or_else(|| "empty dense batch for query".to_string())?;

                if let Some(dim) = config.dimensions {
                    if dense_vector.len() != dim as usize {
                        return Err(format!(
                            "Specified dimensions {} don't match generated dimensions {} for vector '{}'",
                            dim,
                            dense_vector.len(),
                            config.name
                        ));
                    }
                }

                for field in fields {
                    vector_data_list.push(VectorData {
                        vector_name: config.name.clone(),
                        field: *field,
                        vector_type: config.vector_type.clone(),
                        dense_vector: Some(dense_vector.clone()),
                        sparse_indices: None,
                        sparse_values: None,
                    });
                }
            }
            VectorType::DenseCustom => {
                if validation_mode {
                    return Ok(vector_data_list);
                }
                let custom = query.custom_vectors.as_deref().ok_or_else(|| {
                    format!(
                        "Custom dense vector '{}' requires custom vectors but query has none",
                        config.name
                    )
                })?;
                let vec = CustomDenseVector::extract_for_query(config, custom)?;
                for field in fields {
                    vector_data_list.push(VectorData {
                        vector_name: config.name.clone(),
                        field: *field,
                        vector_type: config.vector_type.clone(),
                        dense_vector: Some(vec.clone()),
                        sparse_indices: None,
                        sparse_values: None,
                    });
                }
            }
            VectorType::SparseCustom => {
                if validation_mode {
                    return Ok(vector_data_list);
                }
                let custom = query.custom_vectors.as_deref().ok_or_else(|| {
                    format!(
                        "Custom sparse vector '{}' requires custom vectors but query has none",
                        config.name
                    )
                })?;
                let (indices, values) = CustomSparseVector::extract_for_query(config, custom)?;
                for field in fields {
                    vector_data_list.push(VectorData {
                        vector_name: config.name.clone(),
                        field: *field,
                        vector_type: config.vector_type.clone(),
                        dense_vector: None,
                        sparse_indices: Some(indices.clone()),
                        sparse_values: Some(values.clone()),
                    });
                }
            }
            VectorType::SparseModel
            | VectorType::FullText
            | VectorType::Trigrams
            | VectorType::Whitespace
            | VectorType::Wmtr
            | VectorType::Keyword => {
                let effective_config =
                    sparse_model_config_query_embedding(config.clone());

                let routed = if config.vector_type.is_custom_tokenization() {
                    route_embed_dispatch(
                        &effective_config,
                        &[query.query.clone()],
                        Some(&[5.0_f64]),
                        query.wmtr_trigram_weight,
                    )
                } else {
                    route_embed_dispatch(
                        &effective_config,
                        &[query.query.clone()],
                        None,
                        query.wmtr_trigram_weight,
                    )
                }?;

                let RoutedEmbed::Sparse(sparse_batch) = routed else {
                    return Err("sparse routing returned dense".to_string());
                };

                let (indices, values) = sparse_batch
                    .into_iter()
                    .next()
                    .ok_or_else(|| "empty sparse batch for query".to_string())?;

                for field in fields {
                    vector_data_list.push(VectorData {
                        vector_name: config.name.clone(),
                        field: *field,
                        vector_type: config.vector_type.clone(),
                        dense_vector: None,
                        sparse_indices: Some(indices.clone()),
                        sparse_values: Some(values.clone()),
                    });
                }
            }
        }

        Ok(vector_data_list)
    }
}
