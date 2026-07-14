//! Mirrors `amgix-server/src/core/vector/vectorizer.py`.
//!
//! Embedding is routed through [`route_embed_dispatch`] instead of a Python `EmbedRouter` closure.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

static INDEX_THREADS: OnceLock<usize> = OnceLock::new();

/// Set at startup from `AMGIX_NOW_INDEX_THREADS`. Controls both `RAYON_NUM_THREADS` (matmul
/// parallelism inside candle/gemm) and the number of native threads spawned when vectorizing
/// multiple configs concurrently.
pub fn set_index_threads(n: usize) {
    let _ = INDEX_THREADS.set(n);
}

fn index_threads() -> usize {
    *INDEX_THREADS.get().unwrap_or(&1)
}

use rayon::prelude::*;

use crate::common::{
    DocumentField, VectorType, DEFAULT_WMTR_TRIGRAM_WEIGHT,
};
use crate::metrics::MetricsCollector;
use crate::models::{
    Document, DocumentWithVectors, SearchQuery, SearchQuerySettings, SearchQueryWithVectors,
    VectorConfigInternal, VectorData, VectorSearchOption,
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
use crate::vectors::noop::NoopVector;

#[derive(Debug)]
pub enum RoutedEmbed {
    Dense(Vec<Vec<f32>>),
    Sparse(Vec<(Vec<u32>, Vec<f32>)>),
}

/// Mirrors `EmbedRouterService.embed` routing (dense vs sparse vs custom-token batch path).
/// Do **not** call this for `dense_custom` / `sparse_custom` — those are handled like Python.
///
/// Records all embed metric keys when `metrics` is `Some`. amgix-now always has `hops=0`
/// (no routing), so origin metrics equal local metrics — same as Python when `hops==0`.
pub fn route_embed_dispatch(
    config: &VectorConfigInternal,
    docs: &[String],
    avgdls: Option<&[f64]>,
    trigram_weight: f64,
    metrics: Option<&MetricsCollector>,
) -> Result<RoutedEmbed, String> {
    let t0 = Instant::now();
    let n_passages = docs.len();
    let type_str = config.vector_type.to_string();
    let model_str = config.model.as_deref().unwrap_or("");
    let revision_str = config.revision.as_deref().unwrap_or("");
    let dim = &[type_str.as_str(), model_str, revision_str];

    let result = match config.vector_type {
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
        VectorType::Noop => {
            let av = avgdls.ok_or_else(|| {
                format!(
                    "avgdl entries required for custom tokenization vector '{}' ({:?})",
                    config.name,
                    VectorType::Noop,
                )
            })?;
            Ok(RoutedEmbed::Sparse(
                NoopVector.get_sparse_vector(config, docs, av, trigram_weight)?,
            ))
        }
        VectorType::DenseCustom | VectorType::SparseCustom => Err(
            "route_embed_dispatch must not be called for dense_custom/sparse_custom".to_string(),
        ),
    };

    if let Some(m) = metrics {
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => {
                let n = n_passages as f64;
                m.record(crate::metrics::keys::EMBED_BATCHES, dim, 1.0, None);
                m.record(crate::metrics::keys::EMBED_BATCHES_ORIGIN, dim, 1.0, None);
                m.record(crate::metrics::keys::EMBED_PASSAGES, dim, n, None);
                m.record(crate::metrics::keys::EMBED_PASSAGES_ORIGIN, dim, n, None);
                m.record(crate::metrics::keys::EMBED_INFERENCE_MS, dim, elapsed_ms, Some(1));
                m.record(crate::metrics::keys::EMBED_INFERENCE_ORIGIN_MS, dim, elapsed_ms, Some(1));
                m.record(crate::metrics::keys::EMBED_HOPS, dim, 0.0, Some(1));
                // Track last-used timestamp for this model key (mirrors mark_last_used).
                let key = (type_str.clone(), model_str.to_string(), revision_str.to_string());
                if let Ok(mut guard) = m.model_last_used.lock() {
                    guard.insert(key, std::time::Instant::now());
                }
            }
            Err(_) => {
                m.record(crate::metrics::keys::EMBED_INFERENCE_ORIGIN_ERRORS, dim, 1.0, None);
            }
        }
    }

    result
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

/// Vectorize all documents for a single vector config. Used by both the sequential path
/// (single-config collections) and the native-thread path (multi-config collections).
fn vectorize_single_config(
    config: &VectorConfigInternal,
    documents: &[Document],
    avgdl_dict: Option<&HashMap<String, f64>>,
    metrics: Option<&MetricsCollector>,
) -> Result<(Vec<Vec<VectorData>>, Vec<HashMap<String, usize>>), String> {
    let n = documents.len();
    let mut vectors: Vec<Vec<VectorData>> = vec![Vec::new(); n];
    let mut token_lengths: Vec<HashMap<String, usize>> = vec![HashMap::new(); n];

    let result: Result<(), String> = (|| {
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
                    metrics,
                )?
                else {
                    return Err(format!(
                        "dense_model vector '{}' routed to non-dense embedding",
                        config.name
                    ));
                };

                let mut idx = 0_usize;
                for doc_idx in 0..n {
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
                        vectors[doc_idx].push(VectorData {
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
                        vectors[doc_idx].push(VectorData {
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
                        vectors[doc_idx].push(VectorData {
                            vector_name: config.name.clone(),
                            field: *field,
                            vector_type: config.vector_type.clone(),
                            dense_vector: None,
                            sparse_indices: Some(indices.clone()),
                            sparse_values: Some(values.clone()),
                        });
                        token_lengths[doc_idx].insert(field_vector_name, token_length);
                    }
                }
            }
            VectorType::SparseModel
            | VectorType::FullText
            | VectorType::Trigrams
            | VectorType::Whitespace
            | VectorType::Wmtr
            | VectorType::Keyword
            | VectorType::Noop => {
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
                    metrics,
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
                for doc_idx in 0..n {
                    for field in &config.index_fields {
                        let (indices, values) = sparse_vectors[idx].clone();
                        let field_vector_name = format!("{}_{}", field, config.name);
                        let token_length = indices.len();
                        vectors[doc_idx].push(VectorData {
                            vector_name: config.name.clone(),
                            field: *field,
                            vector_type: config.vector_type.clone(),
                            dense_vector: None,
                            sparse_indices: Some(indices),
                            sparse_values: Some(values),
                        });
                        if config.vector_type.sparse_types_contains() {
                            token_lengths[doc_idx].insert(field_vector_name.clone(), token_length);
                        }
                        idx += 1;
                    }
                }
            }
        }
        Ok(())
    })();

    match result {
        Ok(()) => Ok((vectors, token_lengths)),
        Err(e) => Err(if matches!(
            config.vector_type,
            VectorType::DenseCustom | VectorType::SparseCustom
        ) {
            e
        } else {
            format!(
                "Failed to generate vector '{}' for fields {:?}: {e}",
                config.name,
                config
                    .index_fields
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        }),
    }
}

impl Vectorizer {
    /// Mirrors `Vectorizer.vectorize_documents` (sans async — routing is synchronous here).
    pub fn vectorize_documents(
        documents: &[Document],
        vector_configs: &[VectorConfigInternal],
        avgdl_dict: Option<&HashMap<String, f64>>,
        metrics: Option<Arc<MetricsCollector>>,
    ) -> Result<Vec<DocumentWithVectors>, String> {
        let n = documents.len();

        // For multiple configs: spawn native threads (not Rayon workers) so candle/gemm can
        // freely use the global Rayon pool for intra-matmul parallelism without nesting.
        // Number of threads is capped to index_threads() (set from AMGIX_NOW_INDEX_THREADS).
        // Single-config collections skip threading overhead entirely.
        let per_config_results: Vec<Result<(Vec<Vec<VectorData>>, Vec<HashMap<String, usize>>), String>> =
            if vector_configs.len() > 1 {
                let max_threads = index_threads().min(vector_configs.len());
                let chunk_size = (vector_configs.len() + max_threads - 1) / max_threads;
                let handles: Vec<_> = vector_configs
                    .chunks(chunk_size)
                    .map(|chunk| {
                        let chunk: Vec<VectorConfigInternal> = chunk.to_vec();
                        let documents: Vec<Document> = documents.to_vec();
                        let avgdl_dict: Option<HashMap<String, f64>> = avgdl_dict.map(|m| m.clone());
                        let metrics = metrics.clone();
                        std::thread::spawn(move || {
                            chunk
                                .iter()
                                .map(|config| {
                                    vectorize_single_config(
                                        config,
                                        &documents,
                                        avgdl_dict.as_ref(),
                                        metrics.as_deref(),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap_or_else(|_| {
                        vec![Err("vectorize thread panicked".to_string())]
                    }))
                    .collect()
            } else {
                vector_configs
                    .iter()
                    .map(|config| vectorize_single_config(config, documents, avgdl_dict, metrics.as_deref()))
                    .collect()
            };

        // Merge per-config results into final per-doc vectors.
        let mut vectors_per_doc: Vec<Vec<VectorData>> = vec![Vec::new(); n];
        let mut token_lengths_per_doc: Vec<HashMap<String, usize>> = vec![HashMap::new(); n];
        for result in per_config_results {
            let (cfg_vectors, cfg_token_lengths) = result?;
            for doc_idx in 0..n {
                vectors_per_doc[doc_idx].extend(cfg_vectors[doc_idx].iter().cloned());
                token_lengths_per_doc[doc_idx].extend(cfg_token_lengths[doc_idx].iter().map(|(k, v)| (k.clone(), *v)));
            }
        }

        let mut out = Vec::with_capacity(n);
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

    /// Single-query wrapper around [`Self::vectorize_search_queries`].
    pub fn vectorize_search_query(
        mut query: SearchQuery,
        vector_configs: &[VectorConfigInternal],
        validation_mode: bool,
        metrics: Option<Arc<MetricsCollector>>,
    ) -> Result<SearchQueryWithVectors, String> {
        let q = query.query.clone();
        let mut batch =
            Self::vectorize_search_queries(std::slice::from_ref(&q), &mut query.settings, vector_configs, validation_mode, metrics)?;
        batch.pop().ok_or_else(|| "internal: empty search batch".to_string())
    }

    /// Shared settings and multiple query strings: one embedding batch per vector group, then one
    /// [`SearchQueryWithVectors`] per query string (same layout as repeated single calls).
    pub fn vectorize_search_queries(
        query_texts: &[String],
        settings: &mut SearchQuerySettings,
        vector_configs: &[VectorConfigInternal],
        validation_mode: bool,
        metrics: Option<Arc<MetricsCollector>>,
    ) -> Result<Vec<SearchQueryWithVectors>, String> {
        if query_texts.is_empty() {
            return Err("Search query batch cannot be empty".to_string());
        }
        for t in query_texts {
            if t.trim().is_empty() {
                return Err("Search query cannot be empty".to_string());
            }
        }

        if validation_mode {
            for config in vector_configs {
                if config.vector_type == VectorType::DenseCustom && config.dimensions.is_none() {
                    return Err(format!(
                        "Dense custom vector '{}' requires dimensions to be specified",
                        config.name
                    ));
                }
            }
        }

        let mut vectors_to_generate: HashSet<(String, DocumentField)> = HashSet::new();

        if !settings.vector_options.is_empty() {
            for option in &settings.vector_options {
                if option.weight != 0.0 {
                    vectors_to_generate
                        .insert((option.vector_name.clone(), option.field));
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
                settings.vector_options = vectors_to_generate
                    .iter()
                    .map(|(name, field)| VectorSearchOption {
                        vector_name: name.clone(),
                        field: *field,
                        weight: equal_weight,
                        wmtr_trigram_weight: DEFAULT_WMTR_TRIGRAM_WEIGHT,
                    })
                    .collect();
            }
        }

        let option_map: HashMap<(String, DocumentField), &VectorSearchOption> = settings
            .vector_options
            .iter()
            .map(|o| ((o.vector_name.clone(), o.field), o))
            .collect();

        let config_map: HashMap<String, &VectorConfigInternal> = vector_configs
            .iter()
            .map(|c| (c.name.clone(), c))
            .collect();

        // Group non-WMTR vectors by name; WMTR gets one task per field (trigram weight is per option).
        let mut normal_groups: HashMap<String, Vec<DocumentField>> = HashMap::new();
        let mut wmtr_work: Vec<(&VectorConfigInternal, DocumentField, f64)> = Vec::new();
        for (vector_name, field) in vectors_to_generate {
            let config = config_map.get(&vector_name).ok_or_else(|| {
                format!(
                    "Vector configuration '{vector_name}' not found. Available vectors: {}",
                    config_map.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            })?;
            if !config.index_fields.contains(&field) {
                return Err(format!(
                    "Field '{field}' is not configured for vector '{vector_name}'. Available fields: {:?}",
                    config
                        .index_fields
                        .iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<_>>()
                ));
            }
            if config.vector_type == VectorType::Wmtr {
                let option = option_map.get(&(vector_name.clone(), field));
                let trigram_weight = option
                    .map(|o| o.wmtr_trigram_weight)
                    .unwrap_or(DEFAULT_WMTR_TRIGRAM_WEIGHT);
                wmtr_work.push((config, field, trigram_weight));
            } else {
                normal_groups.entry(vector_name).or_default().push(field);
            }
        }

        let mut work_items: Vec<(String, Vec<DocumentField>, f64)> = normal_groups
            .into_iter()
            .map(|(vector_name, fields)| (vector_name, fields, DEFAULT_WMTR_TRIGRAM_WEIGHT))
            .collect();
        for (config, field, trigram_weight) in wmtr_work {
            work_items.push((config.name.clone(), vec![field], trigram_weight));
        }

        for (vector_name, fields, _) in &work_items {
            let config = config_map.get(vector_name).expect("validated above");
            for field in fields {
                if !config.index_fields.contains(field) {
                    return Err(format!(
                        "Field '{field}' is not configured for vector '{vector_name}'. Available fields: {:?}",
                        config.index_fields.iter().map(|f| f.to_string()).collect::<Vec<_>>()
                    ));
                }
            }
        }

        let n = query_texts.len();
        let group_results: Vec<Vec<Vec<VectorData>>> = if work_items.len() > 1 {
            work_items
                .par_iter()
                .map(|(vector_name, fields, trigram_weight)| {
                    let config = *config_map.get(vector_name).expect("validated above");
                    Self::generate_vectors_for_query_batch(
                        config,
                        query_texts,
                        fields,
                        settings,
                        validation_mode,
                        *trigram_weight,
                        metrics.as_deref(),
                    )
                    .map_err(|e| format!("Failed to generate vector '{vector_name}': {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut out = Vec::new();
            for (vector_name, fields, trigram_weight) in &work_items {
                let config = *config_map.get(vector_name).expect("validated above");
                let batch = Self::generate_vectors_for_query_batch(
                    config,
                    query_texts,
                    fields,
                    settings,
                    validation_mode,
                    *trigram_weight,
                    metrics.as_deref(),
                )
                .map_err(|e| format!("Failed to generate vector '{vector_name}': {e}"))?;
                out.push(batch);
            }
            out
        };

        let settings_out = settings.clone();
        let mut results = Vec::with_capacity(n);
        for qi in 0..n {
            let mut vectors = Vec::new();
            for gr in &group_results {
                vectors.extend(gr[qi].iter().cloned());
            }
            results.push(SearchQueryWithVectors {
                query: query_texts[qi].clone(),
                settings: settings_out.clone(),
                vectors,
            });
        }
        Ok(results)
    }

    fn generate_vectors_for_query_batch(
        config: &VectorConfigInternal,
        query_texts: &[String],
        fields: &[DocumentField],
        settings: &SearchQuerySettings,
        validation_mode: bool,
        trigram_weight: f64,
        metrics: Option<&MetricsCollector>,
    ) -> Result<Vec<Vec<VectorData>>, String> {
        let n = query_texts.len();
        let mut per_query: Vec<Vec<VectorData>> = vec![Vec::new(); n];

        match config.vector_type {
            VectorType::DenseModel => {
                let effective_config = dense_config_with_query_embedding_model(config.clone());

                let RoutedEmbed::Dense(dense_batch) = route_embed_dispatch(
                    &effective_config,
                    query_texts,
                    None,
                    DEFAULT_WMTR_TRIGRAM_WEIGHT,
                    metrics,
                )?
                else {
                    return Err("dense routing returned sparse".to_string());
                };
                if dense_batch.len() != n {
                    return Err(format!(
                        "dense batch length {} != query count {}",
                        dense_batch.len(),
                        n
                    ));
                }

                for (qi, dense_vector) in dense_batch.into_iter().enumerate() {
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
                        per_query[qi].push(VectorData {
                            vector_name: config.name.clone(),
                            field: *field,
                            vector_type: config.vector_type.clone(),
                            dense_vector: Some(dense_vector.clone()),
                            sparse_indices: None,
                            sparse_values: None,
                        });
                    }
                }
            }
            VectorType::DenseCustom => {
                if validation_mode {
                    return Ok(per_query);
                }
                let custom = settings.custom_vectors.as_deref().ok_or_else(|| {
                    format!(
                        "Custom dense vector '{}' requires custom vectors but query has none",
                        config.name
                    )
                })?;
                let vec = CustomDenseVector::extract_for_query(config, custom)?;
                for qi in 0..n {
                    for field in fields {
                        per_query[qi].push(VectorData {
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
                if validation_mode {
                    return Ok(per_query);
                }
                let custom = settings.custom_vectors.as_deref().ok_or_else(|| {
                    format!(
                        "Custom sparse vector '{}' requires custom vectors but query has none",
                        config.name
                    )
                })?;
                let (indices, values) = CustomSparseVector::extract_for_query(config, custom)?;
                for qi in 0..n {
                    for field in fields {
                        per_query[qi].push(VectorData {
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
            VectorType::SparseModel
            | VectorType::FullText
            | VectorType::Trigrams
            | VectorType::Whitespace
            | VectorType::Wmtr
            | VectorType::Keyword
            | VectorType::Noop => {
                let effective_config =
                    sparse_model_config_query_embedding(config.clone());

                let avgdls_5 = vec![5.0_f64; n];
                let routed = if config.vector_type.is_custom_tokenization() {
                    route_embed_dispatch(
                        &effective_config,
                        query_texts,
                        Some(avgdls_5.as_slice()),
                        trigram_weight,
                        metrics,
                    )
                } else {
                    route_embed_dispatch(
                        &effective_config,
                        query_texts,
                        None,
                        trigram_weight,
                        metrics,
                    )
                }?;

                let RoutedEmbed::Sparse(sparse_batch) = routed else {
                    return Err("sparse routing returned dense".to_string());
                };
                if sparse_batch.len() != n {
                    return Err(format!(
                        "sparse batch length {} != query count {}",
                        sparse_batch.len(),
                        n
                    ));
                }

                for (qi, (indices, values)) in sparse_batch.into_iter().enumerate() {
                    for field in fields {
                        per_query[qi].push(VectorData {
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
        }

        Ok(per_query)
    }
}
