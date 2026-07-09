//! Qdrant database layer — mirrors `amgix-server/src/core/database/qdrant.py`.
//!
//! **Transport:** `qdrant-client` uses **gRPC** (tonic), not Qdrant REST. URLs look like
//! `http://host:6334`; that is the gRPC endpoint (HTTP/2), same port as Python `prefer_grpc=True`.
//!
//! Pure storage layer: configure, create_collection, delete_collection,
//! get_collection_info_internal, get/set_collection_stats, add_documents, search.
//! No business logic — no stats accumulation, no fusion algorithms.

use std::collections::HashMap;
use std::time::Duration;

use qdrant_client::qdrant::{
    points_selector::PointsSelectorOneOf,
    points_update_operation::{Operation, OverwritePayload},
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DatetimeRange,
    DeletePointsBuilder, Distance, FieldCondition, FieldType, Filter, GetPointsBuilder,
    Match, PointId, PointStruct, PointsIdsList, PointsSelector, PointsUpdateOperation,
    QueryBatchPointsBuilder, QueryPointsBuilder, Range,
    RepeatedStrings, SparseIndexConfig, SparseVector, SparseVectorConfig,
    SparseVectorParams, ScrollPointsBuilder, Timestamp, UpdateBatchPointsBuilder, UpsertPointsBuilder,
    VectorInput, VectorParams, VectorParamsMap, VectorsConfig, Modifier, Query,
    PayloadIncludeSelector,
};
use qdrant_client::Qdrant;

use crate::common::{
    string_to_uuid, sys_collection_name, DenseDistance,
    MAX_DATABASE_WAIT_SECONDS, search_prefetch_limit,
};
use tokio::time::sleep;
use crate::functions::{
    doc_payload_only, doc_to_payload, linear_weighted_score_fuse, qdrant_val_to_json,
    rrf_fuse, scored_point_id, search_result_from_point, split_first_underscore,
};
use crate::models::{
    CollectionConfigInternal, Document, DocumentFetchRequest, DocumentFetchResponse,
    DocumentWithVectors, MetadataFilter, MetadataIndex,
    SearchQueryWithVectors, SearchResult, VectorScore,
};

const QDRANT_GRPC_CHANNEL_POOL_SIZE: usize = 20;
/// Tonic request deadline for all Qdrant gRPC calls on this client (replaces `qdrant-client`'s 5s default).
const QDRANT_GRPC_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum DbError {
    NotFound(String),
    Qdrant(qdrant_client::QdrantError),
    Config(String),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::NotFound(msg) => write!(f, "Not found: {msg}"),
            DbError::Qdrant(e) => write!(f, "Qdrant error: {e}"),
            DbError::Config(msg) => write!(f, "Config error: {msg}"),
        }
    }
}

impl From<qdrant_client::QdrantError> for DbError {
    fn from(e: qdrant_client::QdrantError) -> Self {
        DbError::Qdrant(e)
    }
}

// ---------------------------------------------------------------------------
// Collection stats (stored under "stats" payload key in meta)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollectionStats {
    pub doc_count: i64,
    pub avgdls: HashMap<String, f64>,
}

impl Default for CollectionStats {
    fn default() -> Self {
        CollectionStats { doc_count: 0, avgdls: HashMap::new() }
    }
}

// ---------------------------------------------------------------------------
// QdrantDb
// ---------------------------------------------------------------------------

pub struct QdrantDb {
    pub client: Qdrant,
    pub meta_collection: String,
    pub metrics_collection: String,
    /// When true, upserts/deletes pass Qdrant `wait=true` so the API returns after data is visible.
    sync_db_writes: bool,
}

impl QdrantDb {
    /// `url` is passed to [`Qdrant::from_url`](qdrant_client::Qdrant::from_url): gRPC URI, e.g.
    /// `http://localhost:6334` (not Qdrant REST on 6333).
    /// `sync_db_writes`: maps to Qdrant [`UpsertPoints`](qdrant_client::Qdrant::upsert_points) /
    /// [`delete_points`](qdrant_client::Qdrant::delete_points) `wait` (`AMGIX_NOW_SYNC_DB_WRITES`).
    pub fn new(url: &str, sync_db_writes: bool) -> Result<Self, DbError> {
        // `qdrant-client`'s default `check_compatibility` runs `health_check` inside `build()` before
        // we can `wait_connected`, printing to stdout when Qdrant is still starting; we defer checks.
        let mut builder = Qdrant::from_url(url)
            .skip_compatibility_check()
            .timeout(QDRANT_GRPC_TIMEOUT);
        builder.set_pool_size(QDRANT_GRPC_CHANNEL_POOL_SIZE);
        let client = builder.build()?;
        Ok(QdrantDb {
            client,
            meta_collection: sys_collection_name("meta"),
            metrics_collection: sys_collection_name("metrics"),
            sync_db_writes,
        })
    }

    /// Mirrors Python `QdrantDatabase.probe()`: fetches server version via `health_check`.
    pub async fn probe(&self) -> Result<String, DbError> {
        let reply = self.client.health_check().await?;
        Ok(reply.version)
    }

    // -----------------------------------------------------------------------
    // configure — ensure amgix_sys_meta exists
    // -----------------------------------------------------------------------

    pub async fn configure(&self) -> Result<(), DbError> {
        self.configure_meta_collection().await?;
        self.configure_metrics_collection().await?;
        Ok(())
    }

    async fn configure_meta_collection(&self) -> Result<(), DbError> {
        if !self.client.collection_exists(&self.meta_collection).await? {
            tracing::info!("Creating system meta collection");
            let vectors_config = VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::ParamsMap(
                    VectorParamsMap {
                        map: HashMap::from([(
                            "dummy".to_string(),
                            VectorParams {
                                size: 1,
                                distance: Distance::Dot as i32,
                                ..Default::default()
                            },
                        )]),
                    },
                )),
            };
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.meta_collection)
                        .vectors_config(vectors_config),
                )
                .await?;
        }
        Ok(())
    }

    async fn configure_metrics_collection(&self) -> Result<(), DbError> {
        if !self.client.collection_exists(&self.metrics_collection).await? {
            tracing::info!("Creating system metrics collection");
            let vectors_config = VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::ParamsMap(
                    VectorParamsMap {
                        map: HashMap::from([(
                            "dummy".to_string(),
                            VectorParams {
                                size: 1,
                                distance: Distance::Dot as i32,
                                ..Default::default()
                            },
                        )]),
                    },
                )),
            };
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.metrics_collection)
                        .vectors_config(vectors_config),
                )
                .await?;

            for (field, field_type) in [
                ("key", FieldType::Keyword),
                ("bucket_seconds", FieldType::Integer),
                ("bucket_start", FieldType::Integer),
            ] {
                self.client
                    .create_field_index(
                        CreateFieldIndexCollectionBuilder::new(
                            &self.metrics_collection,
                            field,
                            field_type,
                        )
                        .wait(true),
                    )
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn is_connected(&self) -> bool {
        self.client.health_check().await.is_ok()
    }

    /// Mirrors Python `DatabaseBase.wait_connected()` (`base.py`): retry until reachable,
    /// sleep starting at 2s, +2 each attempt, capped at [`MAX_DATABASE_WAIT_SECONDS`].
    pub async fn wait_connected(&self) {
        let mut wait_secs: u64 = 2;
        loop {
            if self.is_connected().await {
                return;
            }
            tracing::warn!(
                "Database is not available, will retry in {} seconds",
                wait_secs
            );
            sleep(Duration::from_secs(wait_secs)).await;
            wait_secs = (wait_secs + 2).min(MAX_DATABASE_WAIT_SECONDS);
        }
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, DbError> {
        let resp = self.client.list_collections().await?;
        let user_prefix = format!("{}_", crate::common::APP_PREFIX);
        let sys_prefix = format!("{}_sys_", crate::common::APP_PREFIX);
        Ok(resp
            .collections
            .into_iter()
            .map(|c| c.name)
            .filter(|n| n.starts_with(&user_prefix) && !n.starts_with(&sys_prefix))
            .collect())
    }

    // -----------------------------------------------------------------------
    // create_collection
    // -----------------------------------------------------------------------

    pub async fn create_collection(
        &self,
        collection_name: &str,
        config: &CollectionConfigInternal,
    ) -> Result<bool, DbError> {
        let mut dense_params: HashMap<String, VectorParams> = HashMap::new();
        let mut sparse_params: HashMap<String, SparseVectorParams> = HashMap::new();

        for vc in &config.vectors {
            let vname = &vc.name;

            if vc.vector_type.is_dense() {
                let dims = vc.dimensions.ok_or_else(|| {
                    DbError::Config(format!(
                        "Dimensions required for dense vector '{vname}'"
                    ))
                })?;
                let distance = match &vc.dense_distance {
                    DenseDistance::Cosine => Distance::Cosine,
                    DenseDistance::Dot => Distance::Dot,
                    DenseDistance::Euclid => Distance::Euclid,
                };
                for field in &vc.index_fields {
                    dense_params.insert(
                        format!("{field}_{vname}"),
                        VectorParams {
                            size: dims as u64,
                            distance: distance as i32,
                            ..Default::default()
                        },
                    );
                }
            } else if vc.vector_type.is_sparse() {
                let modifier = if vc.vector_type.is_custom_tokenization() {
                    Some(Modifier::Idf as i32)
                } else {
                    None
                };
                for field in &vc.index_fields {
                    sparse_params.insert(
                        format!("{field}_{vname}"),
                        SparseVectorParams {
                            index: Some(SparseIndexConfig {
                                on_disk: Some(true),
                                ..Default::default()
                            }),
                            modifier,
                            ..Default::default()
                        },
                    );
                }
            }
        }

        let vectors_config = VectorsConfig {
            config: Some(qdrant_client::qdrant::vectors_config::Config::ParamsMap(
                VectorParamsMap { map: dense_params },
            )),
        };

        let mut builder =
            CreateCollectionBuilder::new(collection_name).vectors_config(vectors_config);

        if !sparse_params.is_empty() {
            builder = builder.sparse_vectors_config(SparseVectorConfig { map: sparse_params });
        }

        self.client.create_collection(builder).await?;

        // Store config as a point in meta
        self.upsert_meta_point(
            &format!("{collection_name}_config"),
            "config",
            serde_json::to_value(config)
                .map_err(|e| DbError::Config(format!("Serialization error: {e}")))?,
        )
        .await?;

        // Index on tags
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    collection_name,
                    "tags",
                    FieldType::Keyword,
                )
                .wait(true),
            )
            .await?;

        // Indexes for declared metadata fields
        if let Some(indexes) = &config.metadata_indexes {
            for mi in indexes {
                self.create_metadata_index(collection_name, mi).await?;
            }
        }

        Ok(true)
    }

    async fn create_metadata_index(
        &self,
        collection_name: &str,
        mi: &MetadataIndex,
    ) -> Result<(), DbError> {
        let field_path = format!("metadata.{}", mi.key);
        let field_type = match mi.value_type.as_str() {
            "string" => FieldType::Keyword,
            "integer" => FieldType::Integer,
            "float" => FieldType::Float,
            "boolean" => FieldType::Bool,
            "datetime" => FieldType::Datetime,
            _ => return Ok(()), // unknown types skipped, matching Python `continue`
        };
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    collection_name,
                    &field_path,
                    field_type,
                )
                .wait(true),
            )
            .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // delete_collection
    // -----------------------------------------------------------------------

    pub async fn delete_collection(&self, collection_name: &str) -> Result<bool, DbError> {
        self.client.delete_collection(collection_name).await?;

        // Delete config and stats points from meta
        let ids: Vec<String> = vec![
            string_to_uuid(&format!("{collection_name}_config")).to_string(),
            string_to_uuid(&format!("{collection_name}_stats")).to_string(),
        ];
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.meta_collection)
                    .points(ids)
                    .wait(true),
            )
            .await?;

        Ok(true)
    }

    pub async fn empty_collection(&self, collection_name: &str) -> Result<bool, DbError> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(collection_name)
                    .points(Filter::default())
                    .wait(true),
            )
            .await?;

        let stats_id = string_to_uuid(&format!("{collection_name}_stats")).to_string();
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.meta_collection)
                    .points(vec![stats_id])
                    .wait(true),
            )
            .await?;

        Ok(true)
    }

    // -----------------------------------------------------------------------
    // get_collection_info_internal
    // -----------------------------------------------------------------------

    pub async fn get_collection_info_internal(
        &self,
        collection_name: &str,
    ) -> Result<CollectionConfigInternal, DbError> {
        let id: PointId = string_to_uuid(&format!("{collection_name}_config")).to_string().into();
        let result = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.meta_collection, vec![id])
                    .with_payload(true),
            )
            .await?;

        let point = result.result.into_iter().next().ok_or_else(|| {
            DbError::NotFound(format!("Configuration not found for '{collection_name}'"))
        })?;

        let config_val = point
            .payload
            .get("config")
            .ok_or_else(|| DbError::Config("Missing 'config' key in meta payload".into()))?;

        let config: CollectionConfigInternal =
            serde_json::from_value(qdrant_val_to_json(config_val))
                .map_err(|e| DbError::Config(format!("Deserialization error: {e}")))?;

        Ok(config)
    }

    // -----------------------------------------------------------------------
    // get_document_count
    // -----------------------------------------------------------------------

    pub async fn get_document_count(&self, collection_name: &str) -> Result<u64, DbError> {
        let info = self.client.collection_info(collection_name).await?;
        info.result
            .and_then(|r| r.points_count)
            .ok_or_else(|| DbError::Config("Collection points_count is unavailable".into()))
    }

    // get_collection_stats / set_collection_stats
    // -----------------------------------------------------------------------

    pub async fn get_collection_stats(
        &self,
        collection_name: &str,
    ) -> Result<CollectionStats, DbError> {
        let id: PointId = string_to_uuid(&format!("{collection_name}_stats")).to_string().into();
        let result = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.meta_collection, vec![id])
                    .with_payload(true),
            )
            .await?;

        if let Some(point) = result.result.into_iter().next() {
            if let Some(val) = point.payload.get("stats") {
                let stats: CollectionStats = serde_json::from_value(qdrant_val_to_json(val))
                    .unwrap_or_default();
                return Ok(stats);
            }
        }
        Ok(CollectionStats::default())
    }

    pub async fn set_collection_stats(
        &self,
        collection_name: &str,
        stats: &CollectionStats,
    ) -> Result<(), DbError> {
        self.upsert_meta_point(
            &format!("{collection_name}_stats"),
            "stats",
            serde_json::to_value(stats)
                .map_err(|e| DbError::Config(format!("Serialization error: {e}")))?,
        )
        .await
    }

    // -----------------------------------------------------------------------
    // append_metric_buckets — mirrors qdrant.py append_metric_buckets
    // -----------------------------------------------------------------------

    pub async fn append_metric_buckets(
        &self,
        hostname: &str,
        source: &str,
        buckets: &[crate::metrics::MetricsBucket],
    ) -> Result<(), DbError> {
        if buckets.is_empty() {
            return Ok(());
        }

        let mut points = Vec::with_capacity(buckets.len());
        for bucket in buckets {
            let identity = format!(
                "{}:{}:{}:{}:{}:{}",
                hostname,
                source,
                bucket.key,
                bucket.dims.join(":"),
                bucket.bucket_start,
                bucket.bucket_seconds,
            );
            let point_id = string_to_uuid(&identity).to_string();
            let mut payload = serde_json::Map::new();
            payload.insert("hostname".into(), serde_json::json!(hostname));
            payload.insert("source".into(), serde_json::json!(source));
            payload.insert("key".into(), serde_json::json!(bucket.key));
            payload.insert("dims".into(), serde_json::json!(bucket.dims));
            payload.insert("bucket_start".into(), serde_json::json!(bucket.bucket_start));
            payload.insert("bucket_seconds".into(), serde_json::json!(bucket.bucket_seconds));
            payload.insert("value".into(), serde_json::json!(bucket.value));
            payload.insert("n".into(), serde_json::json!(bucket.n));

            let dummy_vectors: HashMap<String, Vec<f32>> =
                HashMap::from([("dummy".to_string(), vec![0.0_f32])]);
            points.push(PointStruct::new(point_id, dummy_vectors, payload));
        }

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.metrics_collection, points).wait(true),
            )
            .await?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // add_documents
    //
    // Mirrors QdrantDatabase.add_documents exactly — accepts pre-computed
    // DocumentWithVectors (vectorization already done upstream).
    // Converts dense/sparse VectorData → Qdrant named vectors and upserts.
    // Stats accumulation is the encoder layer's responsibility.
    // -----------------------------------------------------------------------

    pub async fn add_documents(
        &self,
        collection_name: &str,
        documents_with_vectors: &[DocumentWithVectors],
        store_content: bool,
    ) -> Result<(), DbError> {
        let mut points: Vec<PointStruct> = Vec::with_capacity(documents_with_vectors.len());

        for doc_with_vectors in documents_with_vectors {
            // Convert VectorData → Qdrant named vector map — mirrors Python lines 426-437.
            let mut qdrant_vectors: HashMap<String, qdrant_client::qdrant::Vector> =
                HashMap::new();
            for vector_data in &doc_with_vectors.vectors {
                let field_vector_name =
                    format!("{}_{}", vector_data.field, vector_data.vector_name);

                if vector_data.vector_type.is_dense() {
                    if let Some(ref dv) = vector_data.dense_vector {
                        qdrant_vectors.insert(field_vector_name, dv.clone().into());
                    }
                } else {
                    let indices = vector_data.sparse_indices.clone().unwrap_or_default();
                    let values = vector_data.sparse_values.clone().unwrap_or_default();
                    qdrant_vectors.insert(
                        field_vector_name,
                        SparseVector { indices, values }.into(),
                    );
                }
            }

            let payload = doc_to_payload(doc_with_vectors, store_content)?;
            let doc_uuid = string_to_uuid(&doc_with_vectors.id).to_string();
            points.push(PointStruct::new(doc_uuid, qdrant_vectors, payload));
        }

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(collection_name, points).wait(self.sync_db_writes),
            )
            .await?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // patch_documents — update payload only, vectors unchanged.
    // Mirrors QdrantDatabase.patch_documents (Python).
    // -----------------------------------------------------------------------

    pub async fn patch_documents(
        &self,
        collection_name: &str,
        documents: &[Document],
        store_content: bool,
    ) -> Result<(), DbError> {
        if documents.is_empty() {
            return Ok(());
        }
        let mut operations: Vec<PointsUpdateOperation> = Vec::with_capacity(documents.len());
        for doc in documents {
            let payload_map = doc_payload_only(doc, store_content)?;
            let payload: HashMap<String, qdrant_client::qdrant::Value> =
                qdrant_client::Payload::from(payload_map).into();
            let uuid = string_to_uuid(&doc.id).to_string();
            operations.push(PointsUpdateOperation {
                operation: Some(Operation::OverwritePayload(OverwritePayload {
                    points_selector: Some(PointsSelector {
                        points_selector_one_of: Some(PointsSelectorOneOf::Points(
                            PointsIdsList { ids: vec![uuid.into()] },
                        )),
                    }),
                    payload,
                    ..Default::default()
                })),
            });
        }
        self.client
            .update_points_batch(
                UpdateBatchPointsBuilder::new(collection_name, operations)
                    .wait(self.sync_db_writes),
            )
            .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // get_documents
    //
    // Mirrors QdrantDatabase.get_documents exactly.
    // Returns one Option<DocumentWithVectors> per input id, in the same order.
    // None for missing documents when suppress_not_found=true; error otherwise.
    // vectors is always empty (payload-only retrieval, matching Python).
    // -----------------------------------------------------------------------

    pub async fn get_documents(
        &self,
        collection_name: &str,
        document_ids: &[&str],
        suppress_not_found: bool,
    ) -> Result<Vec<Option<DocumentWithVectors>>, DbError> {
        let point_ids: Vec<PointId> = document_ids
            .iter()
            .map(|id| string_to_uuid(id).to_string().into())
            .collect();

        let result = self
            .client
            .get_points(
                GetPointsBuilder::new(collection_name, point_ids).with_payload(true),
            )
            .await?;

        // Build uuid-string → DocumentWithVectors map.
        let mut doc_map: HashMap<String, DocumentWithVectors> = HashMap::new();
        for point in result.result {
            let uuid_str = match &point.id {
                Some(pid) => match &pid.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u.clone(),
                    _ => continue,
                },
                None => continue,
            };

            let payload_json = serde_json::Value::Object(
                point.payload.iter().map(|(k, v)| (k.clone(), qdrant_val_to_json(v))).collect(),
            );
            let mut doc: DocumentWithVectors = serde_json::from_value(payload_json)
                .map_err(|e| DbError::Config(format!("Document deserialization error: {e}")))?;
            doc.vectors = vec![];
            doc_map.insert(uuid_str, doc);
        }

        // Re-order by input order; error on missing if not suppressed.
        let mut out: Vec<Option<DocumentWithVectors>> = Vec::with_capacity(document_ids.len());
        let mut missing: Vec<&str> = vec![];
        for id in document_ids {
            let uuid_str = string_to_uuid(id).to_string();
            match doc_map.remove(&uuid_str) {
                Some(doc) => out.push(Some(doc)),
                None => {
                    missing.push(id);
                    out.push(None);
                }
            }
        }

        if !suppress_not_found && !missing.is_empty() {
            return Err(DbError::NotFound(format!(
                "Documents not found for document_ids: {}",
                missing.join(", ")
            )));
        }

        Ok(out)
    }

    // -----------------------------------------------------------------------
    // delete_document
    //
    // Mirrors QdrantDatabase.delete_document — deletes a single point by
    // document ID (UUID5). Returns NotFound if the document does not exist.
    // -----------------------------------------------------------------------

    pub async fn delete_document(
        &self,
        collection_name: &str,
        document_id: &str,
    ) -> Result<(), DbError> {
        let uuid = string_to_uuid(document_id);
        let point_id: PointId = uuid.to_string().into();

        let result = self
            .client
            .get_points(
                GetPointsBuilder::new(collection_name, vec![point_id.clone()]).with_payload(false),
            )
            .await?;

        if result.result.is_empty() {
            return Err(DbError::NotFound(format!(
                "Document '{document_id}' not found in collection '{collection_name}'"
            )));
        }

        self.client
            .delete_points(
                DeletePointsBuilder::new(collection_name)
                    .points(vec![point_id])
                    .wait(self.sync_db_writes),
            )
            .await?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // search
    //
    // Mirrors QdrantDatabase.search exactly — accepts pre-computed
    // SearchQueryWithVectors (vectorization already done upstream).
    // Iterates query.vectors, skips weight-0 entries, builds batch requests,
    // then fuses with RRF or linear fusion.
    // -----------------------------------------------------------------------

    pub async fn search(
        &self,
        collection_name: &str,
        query: &SearchQueryWithVectors,
        collection_config: &CollectionConfigInternal,
    ) -> Result<Vec<SearchResult>, DbError> {
        let final_filter = build_search_filter(query, collection_config)?;

        // weight_lookup: (vector_name, field) → weight — mirrors Python lines 585-586.
        let weight_lookup: HashMap<(String, String), f64> = query
            .settings
            .vector_weights
            .iter()
            .map(|w| ((w.vector_name.clone(), w.field.to_string()), w.weight))
            .collect();

        let prefetch_limit = search_prefetch_limit(query.settings.limit);

        let mut batch_requests: Vec<qdrant_client::qdrant::QueryPoints> = Vec::new();
        let mut batch_vector_names: Vec<String> = Vec::new();
        let mut weight_map: HashMap<String, f64> = HashMap::new();

        // Iterate query.vectors exactly as Python lines 601-642.
        for vector_data in &query.vectors {
            let field_vector_name =
                format!("{}_{}", vector_data.field, vector_data.vector_name);

            let weight = weight_lookup
                .get(&(vector_data.vector_name.clone(), vector_data.field.to_string()))
                .copied()
                .unwrap_or(1.0);

            // Skip weight-0 vectors — they contribute nothing (Python line 610).
            if weight == 0.0 {
                continue;
            }
            weight_map.insert(field_vector_name.clone(), weight);

            let query_variant = if vector_data.vector_type.is_dense() {
                let dv = vector_data.dense_vector.clone().unwrap_or_default();
                qdrant_client::qdrant::query::Variant::Nearest(VectorInput {
                    variant: Some(qdrant_client::qdrant::vector_input::Variant::Dense(
                        qdrant_client::qdrant::DenseVector { data: dv },
                    )),
                })
            } else {
                let sparse_vec = SparseVector {
                    indices: vector_data.sparse_indices.clone().unwrap_or_default(),
                    values: vector_data.sparse_values.clone().unwrap_or_default(),
                };
                qdrant_client::qdrant::query::Variant::Nearest(VectorInput {
                    variant: Some(
                        qdrant_client::qdrant::vector_input::Variant::Sparse(sparse_vec),
                    ),
                })
            };

            // Mirrors Python line 616: only fetch payload fields needed for SearchResult.
            let payload_selector = PayloadIncludeSelector {
                fields: vec![
                    "id".to_string(),
                    "timestamp".to_string(),
                    "name".to_string(),
                    "description".to_string(),
                    "metadata".to_string(),
                    "tags".to_string(),
                ],
            };

            let mut qb = QueryPointsBuilder::new(collection_name)
                .using(field_vector_name.clone())
                .query(Query { variant: Some(query_variant) })
                .limit(prefetch_limit)
                .with_payload(payload_selector)
                .with_vectors(false);

            if let Some(ref f) = final_filter {
                qb = qb.filter(f.clone());
            }

            batch_requests.push(qb.build());
            batch_vector_names.push(field_vector_name);
        }

        if batch_requests.is_empty() {
            return Ok(vec![]);
        }

        let batch_response = self
            .client
            .query_batch(QueryBatchPointsBuilder::new(collection_name, batch_requests))
            .await?;

        // Build arm_weights in the same order as batch_vector_names.
        let arm_weights: Vec<f64> = batch_vector_names
            .iter()
            .map(|n| *weight_map.get(n).unwrap_or(&1.0))
            .collect();

        // Collect ranked ids, scored lists, point lookup, and raw scores —
        // mirrors Python lines 661-689.
        let mut id_lists: Vec<Vec<String>> = Vec::new();
        let mut scored_lists: Vec<Vec<(String, f64)>> = Vec::new();
        let mut point_lookup: HashMap<String, qdrant_client::qdrant::ScoredPoint> =
            HashMap::new();
        let mut raw_scores_map: HashMap<String, Vec<VectorScore>> = HashMap::new();

        for (idx, response) in batch_response.result.iter().enumerate() {
            let field_vector_name = &batch_vector_names[idx];
            let mut ids: Vec<String> = Vec::new();
            let mut scored_arm: Vec<(String, f64)> = Vec::new();

            // enumerate with 1-based rank — mirrors Python `enumerate(response.points, 1)`
            for (rank, point) in response.result.iter().enumerate().map(|(i, p)| (i + 1, p)) {
                let point_id = scored_point_id(point);
                ids.push(point_id.clone());
                scored_arm.push((point_id.clone(), point.score as f64));
                point_lookup.entry(point_id.clone()).or_insert_with(|| point.clone());

                if query.settings.raw_scores {
                    let (field_part, vector_part) = split_first_underscore(field_vector_name);
                    raw_scores_map
                        .entry(point_id.clone())
                        .or_default()
                        .push(VectorScore {
                            field: field_part.to_string(),
                            vector: vector_part.to_string(),
                            score: point.score as f64,
                            rank: rank as u32,
                        });
                }
            }

            id_lists.push(ids);
            scored_lists.push(scored_arm);
        }

        // Fuse — mirrors Python lines 691-705.
        let fused = if query.settings.fusion_mode == "linear" {
            linear_weighted_score_fuse(
                &scored_lists,
                &arm_weights,
                query.settings.limit as usize,
                query.settings.score_threshold,
            )
        } else {
            rrf_fuse(
                &id_lists,
                &arm_weights,
                query.settings.limit as usize,
                query.settings.score_threshold,
                2,
            )
        };

        // Convert fused results to SearchResult — mirrors Python lines 711-719.
        let mut results: Vec<SearchResult> = Vec::with_capacity(fused.len());
        for (item_id, fused_score) in fused {
            if let Some(point) = point_lookup.get(&item_id) {
                let vector_scores = if query.settings.raw_scores {
                    raw_scores_map.get(&item_id).cloned().unwrap_or_default()
                } else {
                    vec![]
                };
                results.push(search_result_from_point(point, fused_score, vector_scores)?);
            }
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Internal helper — upsert a single point in the meta collection
    // -----------------------------------------------------------------------

    async fn upsert_meta_point(
        &self,
        key: &str,
        payload_field: &str,
        value: serde_json::Value,
    ) -> Result<(), DbError> {
        let mut payload_map = serde_json::Map::new();
        payload_map.insert(payload_field.to_string(), value);

        // dummy vector so meta collection accepts the point
        let dummy_vectors: HashMap<String, Vec<f32>> =
            HashMap::from([("dummy".to_string(), vec![0.0_f32])]);

        let point = PointStruct::new(
            string_to_uuid(key).to_string(),
            dummy_vectors,
            payload_map,
        );

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.meta_collection, vec![point]).wait(true),
            )
            .await?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// fetch_documents — cursor-based pagination via Qdrant scroll
// ---------------------------------------------------------------------------

impl QdrantDb {
    pub async fn fetch_documents(
        &self,
        collection_name: &str,
        request: &DocumentFetchRequest,
        collection_config: &CollectionConfigInternal,
    ) -> Result<DocumentFetchResponse, DbError> {
        let scroll_filter = build_fetch_filter(request, collection_config)?;

        let offset: Option<PointId> = request.after.as_deref().map(|s| s.to_string().into());

        let mut builder = ScrollPointsBuilder::new(collection_name)
            .limit(request.page_size)
            .with_payload(true)
            .with_vectors(false);
        if let Some(f) = scroll_filter {
            builder = builder.filter(f);
        }
        if let Some(off) = offset {
            builder = builder.offset(off);
        }

        let response = self.client.scroll(builder).await?;

        let documents: Vec<Document> = response
            .result
            .iter()
            .map(|point| {
                let payload_json = serde_json::Value::Object(
                    point.payload.iter().map(|(k, v)| (k.clone(), qdrant_val_to_json(v))).collect(),
                );
                serde_json::from_value(payload_json)
                    .map_err(|e| DbError::Config(format!("Document deserialization error: {e}")))
            })
            .collect::<Result<_, _>>()?;

        let after = response.next_page_offset.map(|pid| match pid.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u,
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n.to_string(),
            None => String::new(),
        });

        Ok(DocumentFetchResponse { documents, after })
    }

    pub async fn fetch_documents_by_metadata_values(
        &self,
        collection_name: &str,
        metadata_key: &str,
        values: &[serde_json::Value],
        metadata_filter: Option<&MetadataFilter>,
        collection_config: &CollectionConfigInternal,
        max_documents: usize,
    ) -> Result<Vec<Document>, DbError> {
        if values.is_empty() || max_documents == 0 {
            return Ok(vec![]);
        }

        let field_path = format!("metadata.{metadata_key}");
        let in_condition = join_values_match_condition(&field_path, values)?;

        let extra_filter = metadata_filter
            .map(|mf| convert_metadata_filter(mf, collection_config))
            .transpose()?;

        let scroll_filter = match extra_filter {
            Some(extra) => Filter {
                must: vec![in_condition, extra.into()],
                ..Default::default()
            },
            None => Filter {
                must: vec![in_condition],
                ..Default::default()
            },
        };

        let response = self
            .client
            .scroll(
                ScrollPointsBuilder::new(collection_name)
                    .limit(max_documents as u32)
                    .with_payload(true)
                    .with_vectors(false)
                    .filter(scroll_filter),
            )
            .await?;

        response
            .result
            .iter()
            .map(|point| {
                let payload_json = serde_json::Value::Object(
                    point
                        .payload
                        .iter()
                        .map(|(k, v)| (k.clone(), qdrant_val_to_json(v)))
                        .collect(),
                );
                serde_json::from_value(payload_json)
                    .map_err(|e| DbError::Config(format!("Document deserialization error: {e}")))
            })
            .collect()
    }
}

fn join_values_match_condition(
    field_path: &str,
    values: &[serde_json::Value],
) -> Result<Condition, DbError> {
    if values.len() == 1 {
        let match_value = json_to_match_value(&values[0])?;
        return Ok(FieldCondition {
            key: field_path.to_string(),
            r#match: Some(Match {
                match_value: Some(match_value),
            }),
            ..Default::default()
        }
        .into());
    }

    let should: Result<Vec<Condition>, DbError> = values
        .iter()
        .map(|v| {
            let match_value = json_to_match_value(v)?;
            Ok(FieldCondition {
                key: field_path.to_string(),
                r#match: Some(Match {
                    match_value: Some(match_value),
                }),
                ..Default::default()
            }
            .into())
        })
        .collect();

    Ok(Filter {
        should: should?,
        ..Default::default()
    }
    .into())
}

fn json_to_match_value(
    v: &serde_json::Value,
) -> Result<qdrant_client::qdrant::r#match::MatchValue, DbError> {
    use qdrant_client::qdrant::r#match::MatchValue;
    match v {
        serde_json::Value::String(s) => Ok(MatchValue::Keyword(s.clone())),
        serde_json::Value::Bool(b) => Ok(MatchValue::Boolean(*b)),
        serde_json::Value::Number(n) => Ok(MatchValue::Integer(n.as_i64().unwrap_or(0))),
        _ => Err(DbError::Config(
            "Join metadata value must be a string, boolean, or number".to_string(),
        )),
    }
}

fn build_fetch_filter(
    request: &DocumentFetchRequest,
    collection_config: &CollectionConfigInternal,
) -> Result<Option<Filter>, DbError> {
    let mut conditions: Vec<Condition> = Vec::new();

    if let Some(tags) = &request.document_tags {
        if !tags.is_empty() {
            if request.document_tags_match_all {
                for tag in tags {
                    conditions.push(
                        FieldCondition {
                            key: "tags".to_string(),
                            r#match: Some(Match {
                                match_value: Some(
                                    qdrant_client::qdrant::r#match::MatchValue::Keyword(tag.clone()),
                                ),
                            }),
                            ..Default::default()
                        }
                        .into(),
                    );
                }
            } else {
                conditions.push(
                    FieldCondition {
                        key: "tags".to_string(),
                        r#match: Some(Match {
                            match_value: Some(
                                qdrant_client::qdrant::r#match::MatchValue::Keywords(
                                    RepeatedStrings { strings: tags.clone() },
                                ),
                            ),
                        }),
                        ..Default::default()
                    }
                    .into(),
                );
            }
        }
    }

    let tags_filter = if conditions.is_empty() {
        None
    } else {
        Some(Filter { must: conditions, ..Default::default() })
    };

    let metadata_filter = request
        .metadata_filter
        .as_ref()
        .map(|mf| convert_metadata_filter(mf, collection_config))
        .transpose()?;

    Ok(match (tags_filter, metadata_filter) {
        (Some(tf), Some(mf)) => Some(Filter {
            must: vec![tf.into(), mf.into()],
            ..Default::default()
        }),
        (Some(tf), None) => Some(tf),
        (None, Some(mf)) => Some(mf),
        (None, None) => None,
    })
}

// ---------------------------------------------------------------------------
// Search filter — mirrors _convert_metadata_filter_to_qdrant
// ---------------------------------------------------------------------------

fn parse_datetime_for_filter(s: &str) -> Result<Timestamp, DbError> {
    let normalized = s.replace('Z', "+00:00");
    // RFC3339 / ISO 8601 with timezone
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&normalized) {
        return Ok(Timestamp { seconds: dt.timestamp(), nanos: dt.timestamp_subsec_nanos() as i32 });
    }
    // Naive datetime with fractional seconds
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S%.f") {
        return Ok(Timestamp { seconds: dt.and_utc().timestamp(), nanos: dt.and_utc().timestamp_subsec_nanos() as i32 });
    }
    // Naive datetime without fractional seconds
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S") {
        return Ok(Timestamp { seconds: dt.and_utc().timestamp(), nanos: 0 });
    }
    // Date only — treat as start of day UTC
    if let Ok(d) = chrono::NaiveDate::parse_from_str(&normalized, "%Y-%m-%d") {
        let dt = d.and_hms_opt(0, 0, 0).unwrap().and_utc();
        return Ok(Timestamp { seconds: dt.timestamp(), nanos: 0 });
    }
    Err(DbError::Config(format!(
        "Invalid datetime value '{s}': expected ISO 8601 (e.g. '2021-05-01', '2021-05-01T00:00:00Z')"
    )))
}

fn build_search_filter(
    query: &SearchQueryWithVectors,
    collection_config: &CollectionConfigInternal,
) -> Result<Option<Filter>, DbError> {
    let mut search_conditions: Vec<Condition> = Vec::new();

    if let Some(tags) = &query.settings.document_tags {
        if !tags.is_empty() {
            if query.settings.document_tags_match_all {
                for tag in tags {
                    search_conditions.push(
                        FieldCondition {
                            key: "tags".to_string(),
                            r#match: Some(Match {
                                match_value: Some(
                                    qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                        tag.clone(),
                                    ),
                                ),
                            }),
                            ..Default::default()
                        }
                        .into(),
                    );
                }
            } else {
                search_conditions.push(
                    FieldCondition {
                        key: "tags".to_string(),
                        r#match: Some(Match {
                            match_value: Some(
                                qdrant_client::qdrant::r#match::MatchValue::Keywords(
                                    RepeatedStrings { strings: tags.clone() },
                                ),
                            ),
                        }),
                        ..Default::default()
                    }
                    .into(),
                );
            }
        }
    }

    let tags_filter = if search_conditions.is_empty() {
        None
    } else {
        Some(Filter { must: search_conditions, ..Default::default() })
    };

    let metadata_filter = query
        .settings
        .metadata_filter
        .as_ref()
        .map(|mf| convert_metadata_filter(mf, collection_config))
        .transpose()?;

    Ok(match (tags_filter, metadata_filter) {
        (Some(tf), Some(mf)) => Some(Filter {
            must: vec![tf.into(), mf.into()],
            ..Default::default()
        }),
        (Some(tf), None) => Some(tf),
        (None, Some(mf)) => Some(mf),
        (None, None) => None,
    })
}

fn convert_metadata_filter(
    node: &MetadataFilter,
    collection_config: &CollectionConfigInternal,
) -> Result<Filter, DbError> {
    let condition = convert_metadata_node(node, collection_config)?;
    Ok(match condition {
        Condition {
            condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(f)),
        } => f,
        other => Filter { must: vec![other], ..Default::default() },
    })
}

fn convert_metadata_node(
    node: &MetadataFilter,
    collection_config: &CollectionConfigInternal,
) -> Result<Condition, DbError> {
    if let Some(key) = &node.key {
        let field_path = format!("metadata.{key}");

        let is_datetime = collection_config
            .metadata_indexes
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .any(|mi| mi.key == *key && mi.value_type == "datetime");

        let op = node.op.as_deref().unwrap_or("");
        let raw_value = node.value.as_ref();

        let condition: Condition = if op == "eq" || op == "neq" {
            let match_value = match raw_value {
                Some(serde_json::Value::String(s)) => {
                    qdrant_client::qdrant::r#match::MatchValue::Keyword(s.clone())
                }
                Some(serde_json::Value::Bool(b)) => {
                    qdrant_client::qdrant::r#match::MatchValue::Boolean(*b)
                }
                Some(serde_json::Value::Number(n)) => {
                    qdrant_client::qdrant::r#match::MatchValue::Integer(n.as_i64().unwrap_or(0))
                }
                _ => return Err(DbError::Config(format!(
                    "Metadata filter for key '{key}': missing or unsupported value type for operator '{op}'"
                ))),
            };
            let field_condition: Condition = FieldCondition {
                key: field_path,
                r#match: Some(Match { match_value: Some(match_value) }),
                ..Default::default()
            }
            .into();
            if op == "neq" {
                Filter { must_not: vec![field_condition], ..Default::default() }.into()
            } else {
                field_condition
            }
        } else {
            let val_str = raw_value.and_then(|v| v.as_str()).map(str::to_string);

            if is_datetime {
                let s = val_str.as_deref().ok_or_else(|| DbError::Config(format!(
                    "Metadata filter for key '{key}': datetime value must be a string"
                )))?;
                let ts = parse_datetime_for_filter(s)?;
                let mut dr = DatetimeRange::default();
                match op {
                    "gt" => dr.gt = Some(ts),
                    "gte" => dr.gte = Some(ts),
                    "lt" => dr.lt = Some(ts),
                    "lte" => dr.lte = Some(ts),
                    _ => return Err(DbError::Config(format!(
                        "Unsupported operator '{op}' for datetime key '{key}'"
                    ))),
                }
                FieldCondition { key: field_path, datetime_range: Some(dr), ..Default::default() }
                    .into()
            } else {
                let val_f64 = raw_value
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| DbError::Config(format!(
                        "Metadata filter for key '{key}': numeric value required for operator '{op}'"
                    )))?;
                let mut r = Range::default();
                match op {
                    "gt" => r.gt = Some(val_f64),
                    "gte" => r.gte = Some(val_f64),
                    "lt" => r.lt = Some(val_f64),
                    "lte" => r.lte = Some(val_f64),
                    _ => return Err(DbError::Config(format!(
                        "Unsupported operator '{op}' for key '{key}'"
                    ))),
                }
                FieldCondition { key: field_path, range: Some(r), ..Default::default() }.into()
            }
        };

        return Ok(condition);
    }

    let must: Vec<Condition> = node
        .and_
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|c| convert_metadata_node(c, collection_config))
        .collect::<Result<_, _>>()?;
    let should: Vec<Condition> = node
        .or_
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|c| convert_metadata_node(c, collection_config))
        .collect::<Result<_, _>>()?;
    let must_not: Vec<Condition> = node
        .not_
        .as_deref()
        .map(|n| convert_metadata_node(n, collection_config))
        .transpose()?
        .into_iter()
        .collect();

    Ok(Condition {
        condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(
            Filter { must, should, must_not, ..Default::default() },
        )),
    })
}

