mod amgix;
mod bunny_talk;
mod lock_client;
mod metrics;
mod common;
mod platform;
mod encoder;
mod filter_parser;
mod functions;
mod join_parser;
mod models;
mod qdrant;
mod search_join;
mod validation;
mod vectors;

use std::convert::Infallible;
use std::io::Write;
use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    extract::{
        rejection::JsonRejection,
        Path, Query, Request, State,
    },
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use bytes::Bytes;
use chrono::Utc;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use common::{
    get_real_collection_name, get_user_collection_name, qdrant_client_url,
    AMGIX_VARIANT, AMGIX_VERSION, VectorType, DATABASE_KIND, MAX_DOCUMENT_FETCH_PAGE_SIZE,
};
use encoder::{
    document_delete_sync, document_upsert_bulk, document_upsert_sync, validate_metadata_filter,
    validate_models, CollectionConfigCache, LockBackend, NamedLocks, SearchError, SearchIngress,
    StatsUpdateBatcher, UpsertIngress, UpsertSyncError,
};
use models::{
    parse_document_timestamp_for_api, BulkUploadRequest, CollectionConfig, CollectionConfigInternal,
    CollectionExistsResponse, CollectionStatsResponse, Document, DocumentFetchRequest,
    DocumentStatus, DocumentStatusResponse, OkResponse, QueueInfo,
    QueuedDocumentStatus, ReadyResponse, SearchQuery, SearchResponse,
    SystemInfoResponse, VectorConfigInternal, VersionResponse,
};
use qdrant::{DbError, QdrantDb};
use validation::{
    normalize_document_python, normalize_search_query_python,
    validate_bulk_upload, validate_collection_config, validate_collection_name, validate_document,
    validate_document_vectors, validate_search_query,
};

#[derive(Clone)]
struct AppState {
    db: Arc<QdrantDb>,
    qdrant_version: String,
    #[allow(dead_code)]
    amgix_version: String,
    #[allow(dead_code)]
    amgix_variant: String,
    amgix_version_display: String,
    collection_cache: CollectionConfigCache,
    stats_batcher: StatsUpdateBatcher,
    upsert_ingress: UpsertIngress,
    search_ingress: SearchIngress,
    doc_locks: LockBackend,
    bunny: Option<Arc<bunny_talk::BunnyTalk>>,
    metrics: Option<Arc<metrics::MetricsCollector>>,
}

fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({ "detail": msg.into() })),
    )
}

/// FastAPI / Pydantic shape: `{ "detail": [ { "type", "loc", "msg", "input" }, ... ] }`.
///
/// Validation error messages use `"field: message"` format; this function splits on the first `: `
/// to build a proper `loc` array matching Pydantic's output.
fn validation_error_detail_list(msg: impl Into<String>) -> Value {
    let msg = msg.into();
    let (loc, clean_msg) = if let Some(colon_pos) = msg.find(": ") {
        let field_path = &msg[..colon_pos];
        let rest = &msg[colon_pos + 2..];
        // Build loc from path segments split by '.'
        // e.g. "documents[0].description" -> ["body", "documents[0]", "description"]
        let mut loc_parts: Vec<Value> = vec![json!("body")];
        for part in field_path.split('.') {
            loc_parts.push(json!(part));
        }
        (json!(loc_parts), rest.to_string())
    } else {
        (json!(["body"]), msg)
    };
    json!([
        {
            "type": "validation_error",
            "loc": loc,
            "msg": clean_msg,
            "input": Value::Null
        }
    ])
}

fn infer_json_error_loc(msg: &str) -> Value {
    const NEEDLE: &str = "missing field `";
    if let Some(start) = msg.find(NEEDLE) {
        let after = &msg[start + NEEDLE.len()..];
        if let Some(end) = after.find('`') {
            let field = &after[..end];
            return json!(["body", field]);
        }
    }
    json!(["body"])
}

fn json_rejection_response(rejection: JsonRejection) -> (StatusCode, Json<Value>) {
    let (status, msg) = match &rejection {
        JsonRejection::JsonDataError(err) => (StatusCode::UNPROCESSABLE_ENTITY, err.to_string()),
        JsonRejection::JsonSyntaxError(err) => (StatusCode::BAD_REQUEST, err.to_string()),
        JsonRejection::MissingJsonContentType(_) => {
            (StatusCode::UNSUPPORTED_MEDIA_TYPE, rejection.to_string())
        }
        JsonRejection::BytesRejection(err) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::BAD_REQUEST, rejection.to_string()),
    };
    let loc = infer_json_error_loc(&msg);
    (
        status,
        Json(json!({
            "detail": [
                {
                    "type": "validation_error",
                    "loc": loc,
                    "msg": msg,
                    "input": Value::Null
                }
            ]
        })),
    )
}

fn validation_error(e: validation::ValidationError) -> (StatusCode, Json<Value>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({ "detail": validation_error_detail_list(e.to_string()) })),
    )
}

fn query_validation_error(field: &str, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({
            "detail": [
                {
                    "type": "validation_error",
                    "loc": ["query", field],
                    "msg": msg.into(),
                    "input": Value::Null
                }
            ]
        })),
    )
}

#[derive(Debug, Deserialize)]
struct DeleteDocumentQuery {
    request_timestamp: String,
}

#[derive(Debug, Deserialize)]
struct GetDocumentQuery {
    #[serde(default)]
    with_vectors: bool,
}

#[derive(Debug, Deserialize)]
struct ExportDocumentsQuery {
    #[serde(default)]
    with_vectors: bool,
}

struct GzipStreamWriter {
    tx: tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
}

impl Write for GzipStreamWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        self.tx
            .blocking_send(Ok(Bytes::copy_from_slice(buf)))
            .map_err(|_| std::io::Error::other("export stream consumer dropped"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Full Amgix API compatibility surface that **amgix-now** intentionally omits — same paths as FastAPI (`api/main.py`).
const AMGIX_NOW_NOT_IMPLEMENTED_MSG: &str = "Not implemented in Amgix Now";

/// `501` stubs for **`GET /v1/metrics/*`** routes present in Python.
async fn not_implemented_amgix_now_metrics() -> impl IntoResponse {
    api_error(StatusCode::NOT_IMPLEMENTED, AMGIX_NOW_NOT_IMPLEMENTED_MSG)
}

/// `501` stubs for **`.../queue/...`** collection routes present in Python; validates `{collection_name}` like other handlers.
async fn not_implemented_amgix_now_collection_queue(Path(collection_name): Path<String>) -> impl IntoResponse {
    match validate_collection_name(&collection_name) {
        Err(e) => validation_error(e).into_response(),
        Ok(()) => api_error(StatusCode::NOT_IMPLEMENTED, AMGIX_NOW_NOT_IMPLEMENTED_MSG).into_response(),
    }
}

/// Axum middleware — mirrors `ApiMetricsMiddleware` + `record_api_http_request` from `api_metrics.py`.
///
/// Records `api_requests`, `api_request_ms`, per-operation classified metrics, and 4xx/5xx errors.
/// No-op when `AppState.metrics` is `None` (standalone mode).
async fn api_metrics_middleware(
    State(app): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();

    let t0 = std::time::Instant::now();
    let response = next.run(request).await;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if let Some(ref m) = app.metrics {
        m.record(metrics::keys::API_REQUESTS, &[], 1.0, None);
        m.record(metrics::keys::API_REQUEST_MS, &[], elapsed_ms, Some(1));

        let status = response.status().as_u16();
        if status >= 500 {
            m.record(metrics::keys::API_ERROR_5XX, &[], 1.0, None);
        } else if status >= 400 {
            m.record(metrics::keys::API_ERROR_4XX, &[], 1.0, None);
        }

        // Classify by method + path pattern — mirrors _CLASSIFIED_OPERATIONS.
        let path = uri.path();
        match (method.as_str(), classify_api_route(path)) {
            ("POST", Some(RouteKind::AsyncUpload)) => {
                m.record(metrics::keys::API_ASYNC_UPLOAD, &[], 1.0, None);
                m.record(metrics::keys::API_ASYNC_UPLOAD_MS, &[], elapsed_ms, Some(1));
            }
            ("POST", Some(RouteKind::SyncUpload)) => {
                m.record(metrics::keys::API_SYNC_UPLOAD, &[], 1.0, None);
                m.record(metrics::keys::API_SYNC_UPLOAD_MS, &[], elapsed_ms, Some(1));
            }
            ("POST", Some(RouteKind::BulkUpload)) => {
                m.record(metrics::keys::API_BULK_UPLOAD, &[], 1.0, None);
                m.record(metrics::keys::API_BULK_UPLOAD_MS, &[], elapsed_ms, Some(1));
            }
            ("POST", Some(RouteKind::Search)) => {
                m.record(metrics::keys::API_SEARCH, &[], 1.0, None);
                m.record(metrics::keys::API_SEARCH_MS, &[], elapsed_ms, Some(1));
            }
            ("DELETE", Some(RouteKind::AsyncDelete)) => {
                m.record(metrics::keys::API_ASYNC_DELETE, &[], 1.0, None);
                m.record(metrics::keys::API_ASYNC_DELETE_MS, &[], elapsed_ms, Some(1));
            }
            ("DELETE", Some(RouteKind::SyncDelete)) => {
                m.record(metrics::keys::API_SYNC_DELETE, &[], 1.0, None);
                m.record(metrics::keys::API_SYNC_DELETE_MS, &[], elapsed_ms, Some(1));
            }
            _ => {}
        }
    }

    response
}

enum RouteKind {
    AsyncUpload,
    SyncUpload,
    BulkUpload,
    Search,
    AsyncDelete,
    SyncDelete,
}

fn classify_api_route(path: &str) -> Option<RouteKind> {
    // Check specific suffixes before the generic segment match.
    if path.ends_with("/documents/bulk") {
        return Some(RouteKind::BulkUpload);
    }
    if path.ends_with("/documents/sync") {
        return Some(RouteKind::SyncUpload);
    }
    if path.ends_with("/search") {
        return Some(RouteKind::Search);
    }
    if let Some(after) = path.strip_prefix("/v1/collections/") {
        // splitn(5) gives us up to: {c}, "documents", {id}, "sync", remainder
        let segments: Vec<&str> = after.splitn(5, '/').collect();
        match segments.as_slice() {
            // POST /v1/collections/{c}/documents  → upsert_document (async)
            [_, "documents"] => return Some(RouteKind::AsyncUpload),
            // DELETE /v1/collections/{c}/documents/{id}/sync  → delete_document_sync
            [_, "documents", _id, "sync"] => return Some(RouteKind::SyncDelete),
            // DELETE /v1/collections/{c}/documents/{id}  → delete_document (async)
            [_, "documents", _id] => return Some(RouteKind::AsyncDelete),
            _ => {}
        }
    }
    None
}

/// `GET /v1/health/check` — process is up and serving HTTP (mirrors Python `health_check`).
async fn health_check() -> Json<OkResponse> {
    Json(OkResponse::ok())
}

/// `GET /v1/health/ready` — same status rules and JSON body as Python `readiness_check`.
/// In **amgix-now** there are no separate index/query workers; those probes stay `true`.
/// In cluster mode (`AMGIX_AMQP_URL` set), `rabbitmq` mirrors amgix-server: local connection + channel open.
async fn health_ready(State(app): State<AppState>) -> (StatusCode, Json<ReadyResponse>) {
    const PARTIAL_READY: u16 = 218;

    let database = app.db.is_connected().await;
    let rabbitmq = app
        .bunny
        .as_ref()
        .map(|b| b.is_connected())
        .unwrap_or(true);
    let index = true;
    let query = true;
    let ready = database && rabbitmq && index && query;

    let body = ReadyResponse {
        database,
        rabbitmq,
        index,
        query,
        ready,
    };

    let status = if !database || !rabbitmq || (!index && !query) {
        StatusCode::SERVICE_UNAVAILABLE
    } else if ready {
        StatusCode::OK
    } else {
        StatusCode::from_u16(PARTIAL_READY).expect("218 is a valid status code")
    };

    (status, Json(body))
}

/// `GET /v1/version` — mirrors Python `version`.
async fn version_endpoint(State(app): State<AppState>) -> Json<VersionResponse> {
    Json(VersionResponse {
        version: app.amgix_version_display.clone(),
    })
}

/// `GET /v1/system/info` — mirrors Python `system_info` (no connection URLs).
async fn system_info(
    State(app): State<AppState>,
) -> Result<Json<SystemInfoResponse>, (StatusCode, Json<Value>)> {
    let database_version = app.qdrant_version.clone();
    let collection_names = app.db.list_collections().await.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to list collections: {e}"),
        )
    })?;

    Ok(Json(SystemInfoResponse {
        amgix_version: app.amgix_version_display.clone(),
        database_kind: DATABASE_KIND.to_string(),
        database_version,
        database_features: Default::default(),
        rabbitmq_version: "unknown".to_string(),
        collection_count: collection_names.len() as u64,
    }))
}

/// `GET /v1/collections` — mirrors Python `list_collections`.
async fn list_collections(
    State(app): State<AppState>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<Value>)> {
    let names = app.db.list_collections().await.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to list collections: {e}"),
        )
    })?;
    Ok(Json(names.iter().map(|n| get_user_collection_name(n).to_string()).collect()))
}

/// `GET /v1/collections/{collection_name}` — mirrors Python `get_collection_config`.
async fn get_collection_config(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CollectionConfig>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let internal = app.db.get_collection_info_internal(&real_collection_name).await.map_err(|e| match e {
        DbError::NotFound(_) => api_error(
            StatusCode::NOT_FOUND,
            format!("Collection '{collection_name}' not found"),
        ),
        e => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to get collection config: {e}"),
        ),
    })?;
    Ok(Json(CollectionConfig::from(internal)))
}

async fn create_collection(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    payload: Result<Json<CollectionConfig>, JsonRejection>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    let Json(config) = payload.map_err(|e| json_rejection_response(e))?;
    validate_collection_name(&collection_name).map_err(validation_error)?;
    validate_collection_config(&config).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);

    match app.db.get_collection_info_internal(&real_collection_name).await {
        Ok(_) => {
            return Err(api_error(
                StatusCode::CONFLICT,
                format!("Collection '{collection_name}' already exists"),
            ));
        }
        Err(DbError::NotFound(_)) => {}
        Err(e) => {
            return Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to check collection: {e}"),
            ));
        }
    }

    let internal_for_validation: Vec<VectorConfigInternal> =
        config.vectors.iter().cloned().map(VectorConfigInternal::from).collect();

    let validation = validate_models(internal_for_validation).await;
    if let Some(err) = validation.error {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("Model validation failed: {err}"),
        ));
    }

    let results = validation.results.ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "Model validation returned no results map",
        )
    })?;

    let mut full_vectors: Vec<VectorConfigInternal> =
        Vec::with_capacity(config.vectors.len());
    for vc in &config.vectors {
        if vc.vector_type == VectorType::DenseModel {
            let r = results.get(&vc.name).ok_or_else(|| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    format!("Model validation failed for {}", vc.name),
                )
            })?;
            if !r.valid {
                let msg = r
                    .error
                    .clone()
                    .unwrap_or_else(|| "Model validation failed".to_string());
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!("Model {} is not valid: {msg}", vc.name),
                ));
            }
            let dim = r.dimension.ok_or_else(|| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Model {} is valid but has no reported dimension after validation",
                        vc.name
                    ),
                )
            })?;
            let mut full = VectorConfigInternal::from(vc.clone());
            full.dimensions = Some(dim);
            full_vectors.push(full);
        } else {
            full_vectors.push(VectorConfigInternal::from(vc.clone()));
        }
    }

    let full_collection_config = CollectionConfigInternal {
        version: 1,
        collection_id: Uuid::new_v4().to_string(),
        vectors: full_vectors,
        store_content: config.store_content,
        metadata_indexes: config.metadata_indexes.clone(),
    };

    match app
        .db
        .create_collection(&real_collection_name, &full_collection_config)
        .await
    {
        Ok(true) => {
            app.collection_cache.invalidate(&real_collection_name).await;
            Ok(Json(OkResponse::ok()))
        }
        Ok(false) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create collection '{collection_name}'"),
        )),
        Err(e) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create collection: {e}"),
        )),
    }
}

async fn delete_collection(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);

    match app.db.delete_collection(&real_collection_name).await {
        Ok(true) => {
            app.collection_cache.invalidate(&real_collection_name).await;
            Ok(Json(OkResponse::ok()))
        }
        Ok(false) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete collection '{collection_name}'"),
        )),
        Err(e) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete collection: {e}"),
        )),
    }
}

/// Mirrors `main.py` `collection_exists` — always **200** with `exists` true/false, except on
/// unexpected database errors (**500**).
async fn collection_exists(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CollectionExistsResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);

    match app.db.get_collection_info_internal(&real_collection_name).await {
        Ok(_) => Ok(Json(CollectionExistsResponse { exists: true })),
        Err(DbError::NotFound(_)) => Ok(Json(CollectionExistsResponse { exists: false })),
        Err(e) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to check collection: {e}"),
        )),
    }
}

/// `GET /v1/collections/{collection_name}/stats` — mirrors Python `get_collection_stats`.
/// `amgix-now` has no queue; `queue` is always all zeros.
async fn get_collection_stats(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CollectionStatsResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);

    match app.db.get_collection_info_internal(&real_collection_name).await {
        Ok(_) => {}
        Err(DbError::NotFound(_)) => {
            return Err(api_error(
                StatusCode::NOT_FOUND,
                format!("Collection '{collection_name}' not found"),
            ));
        }
        Err(e) => {
            return Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to check collection: {e}"),
            ));
        }
    }

    let doc_count = app.db.get_document_count(&real_collection_name).await.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to get document count: {e}"),
        )
    })?;

    Ok(Json(CollectionStatsResponse {
        doc_count: doc_count as i64,
        queue: QueueInfo::empty(),
    }))
}

/// `POST /v1/collections/{collection_name}/empty` — mirrors Python `empty_collection`.
async fn empty_collection(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let ok = app.db.empty_collection(&real_collection_name).await.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to empty collection: {e}"),
        )
    })?;
    Ok(Json(OkResponse { ok, skipped: None }))
}

async fn upsert_document(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    payload: Result<Json<Document>, JsonRejection>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    let Json(mut document) = payload.map_err(|e| json_rejection_response(e))?;
    validate_collection_name(&collection_name).map_err(validation_error)?;
    normalize_document_python(&mut document);
    validate_document(&document).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let collection_config = app
        .db
        .get_collection_info_internal(&real_collection_name)
        .await
        .map_err(|e| match e {
            DbError::NotFound(_) => api_error(
                StatusCode::NOT_FOUND,
                format!("Collection '{collection_name}' not found"),
            ),
            e => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get collection config: {e}"),
            ),
        })?;
    validate_document_vectors(&collection_config, &document)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, e.0))?;
    match document_upsert_sync(&app.upsert_ingress, &real_collection_name, document).await {
        Ok(skipped) => Ok(Json(OkResponse::ok_with_skipped(skipped))),
        Err(UpsertSyncError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(UpsertSyncError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(UpsertSyncError::IngressQueueFull(m)) => {
            Err(api_error(StatusCode::TOO_MANY_REQUESTS, m))
        }
        Err(UpsertSyncError::IngressWorkerExited(m)) => {
            Err(api_error(StatusCode::SERVICE_UNAVAILABLE, m))
        }
        Err(UpsertSyncError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
        }
    }
}

async fn upsert_documents_bulk(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    payload: Result<Json<BulkUploadRequest>, JsonRejection>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    let Json(mut request) = payload.map_err(|e| json_rejection_response(e))?;
    validate_collection_name(&collection_name).map_err(validation_error)?;
    validate_bulk_upload(&mut request).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let collection_config = app
        .db
        .get_collection_info_internal(&real_collection_name)
        .await
        .map_err(|e| match e {
            DbError::NotFound(_) => api_error(
                StatusCode::NOT_FOUND,
                format!("Collection '{collection_name}' not found"),
            ),
            e => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get collection config: {e}"),
            ),
        })?;
    for (i, doc) in request.documents.iter().enumerate() {
        validate_document_vectors(&collection_config, doc).map_err(|e| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("documents[{i}]: {}", e.0),
            )
        })?;
    }
    match document_upsert_bulk(
        &app.upsert_ingress,
        &real_collection_name,
        request.documents,
    )
    .await
    {
        Ok(skipped) => Ok(Json(OkResponse::ok_with_skipped(skipped))),
        Err(UpsertSyncError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(UpsertSyncError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(UpsertSyncError::IngressQueueFull(m)) => {
            Err(api_error(StatusCode::TOO_MANY_REQUESTS, m))
        }
        Err(UpsertSyncError::IngressWorkerExited(m)) => {
            Err(api_error(StatusCode::SERVICE_UNAVAILABLE, m))
        }
        Err(UpsertSyncError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
        }
    }
}

/// `GET /v1/collections/{collection_name}/documents/export` — mirrors Python `export_documents`.
async fn export_documents(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    Query(query): Query<ExportDocumentsQuery>,
) -> Result<Response<Body>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let collection_config = app
        .db
        .get_collection_info_internal(&real_collection_name)
        .await
        .map_err(|e| match e {
            DbError::NotFound(m) => api_error(StatusCode::NOT_FOUND, m),
            e => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get collection config: {e}"),
            ),
        })?;

    let user_name = get_user_collection_name(&real_collection_name);
    let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    let filename = format!("{user_name}-{timestamp}.json.gz");

    let (body_tx, body_rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(32);
    let (plain_tx, plain_rx) = std::sync::mpsc::channel::<Vec<u8>>();

    let with_vectors = query.with_vectors;

    std::thread::spawn(move || {
        let mut gz = GzEncoder::new(
            GzipStreamWriter {
                tx: body_tx.clone(),
            },
            Compression::default(),
        );
        for chunk in plain_rx {
            if chunk.is_empty() {
                break;
            }
            if gz.write_all(&chunk).is_err() {
                return;
            }
        }
        let _ = gz.finish();
        drop(body_tx);
    });

    let db = app.db.clone();
    tokio::spawn(async move {
        let send_plain = |bytes: Vec<u8>| {
            let _ = plain_tx.send(bytes);
        };

        send_plain(b"[".to_vec());
        let mut first = true;
        let mut after: Option<String> = None;

        loop {
            let request = DocumentFetchRequest {
                page_size: MAX_DOCUMENT_FETCH_PAGE_SIZE,
                after: after.clone(),
                metadata_filter: None,
                document_tags: None,
                document_tags_match_all: false,
                join: None,
                with_vectors,
            };

            let page = match db
                .fetch_documents(&real_collection_name, &request, &collection_config)
                .await
            {
                Ok(page) => page,
                Err(_) => break,
            };

            for doc in page.documents {
                match models::document_to_export_json(&doc) {
                    Ok(json_bytes) => {
                        let chunk = if first {
                            first = false;
                            json_bytes
                        } else {
                            let mut v = Vec::with_capacity(1 + json_bytes.len());
                            v.push(b',');
                            v.extend(json_bytes);
                            v
                        };
                        send_plain(chunk);
                    }
                    Err(_) => break,
                }
            }

            if page.after.is_none() {
                break;
            }
            after = page.after;
        }

        send_plain(b"]".to_vec());
        send_plain(Vec::new());
    });

    let body = Body::from_stream(ReceiverStream::new(body_rx));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, HeaderValue::from_static("application/gzip"))
        .header(
            header::CONTENT_DISPOSITION,
            HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
                .map_err(|e| {
                    api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Invalid Content-Disposition: {e}"),
                    )
                })?,
        )
        .body(body)
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build export response: {e}"),
            )
        })
}

/// `GET /v1/collections/{collection_name}/documents/{document_id}` — mirrors Python `get_document`.
async fn get_document(
    State(app): State<AppState>,
    Path((collection_name, document_id)): Path<(String, String)>,
    Query(query): Query<GetDocumentQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let collection_config = if query.with_vectors {
        Some(
            app.db
                .get_collection_info_internal(&real_collection_name)
                .await
                .map_err(|e| match e {
                    DbError::NotFound(m) => api_error(StatusCode::NOT_FOUND, m),
                    e => api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Database error: {e}"),
                    ),
                })?,
        )
    } else {
        None
    };
    let rows = app
        .db
        .get_documents(
            &real_collection_name,
            &[document_id.as_str()],
            false,
            query.with_vectors,
            collection_config.as_ref(),
        )
        .await
        .map_err(|e| match e {
            DbError::NotFound(m) => api_error(StatusCode::NOT_FOUND, m),
            DbError::Config(m) => api_error(StatusCode::BAD_REQUEST, m),
            e => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {e}"),
            ),
        })?;

    let doc = rows
        .into_iter()
        .next()
        .flatten()
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                format!("Document '{document_id}' not found"),
            )
        })?;

    let val = serde_json::to_value(doc)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization error: {e}")))?;
    Ok(Json(models::flatten_doc_metadata(val)))
}

/// `GET /v1/collections/{collection_name}/documents/{document_id}/status` — mirrors Python
/// `get_document_status` / `get_queue_statuses`. No queue in **amgix-now**; only `indexed` when present.
async fn get_document_status(
    State(app): State<AppState>,
    Path((collection_name, document_id)): Path<(String, String)>,
) -> Result<Json<DocumentStatusResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let rows = app
        .db
        .get_documents(&real_collection_name, &[document_id.as_str()], true, false, None)
        .await
        .map_err(|e| match e {
            DbError::Config(m) => api_error(StatusCode::BAD_REQUEST, m),
            e => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {e}"),
            ),
        })?;

    let doc_with = rows.into_iter().next().flatten();
    let statuses = if let Some(doc) = doc_with {
        vec![DocumentStatus {
            status: QueuedDocumentStatus::Indexed,
            op_type: None,
            info: None,
            timestamp: doc.timestamp,
            queue_id: None,
            try_count: None,
        }]
    } else {
        vec![]
    };

    if statuses.is_empty() {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            format!("Document {document_id} not found in collection {collection_name}"),
        ));
    }

    Ok(Json(DocumentStatusResponse { statuses }))
}

async fn delete_document(
    State(app): State<AppState>,
    Path((collection_name, document_id)): Path<(String, String)>,
    Query(query): Query<DeleteDocumentQuery>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let request_timestamp = parse_document_timestamp_for_api(&query.request_timestamp).map_err(|_| {
        query_validation_error(
            "request_timestamp",
            format!(
                "'request_timestamp' must include a UTC timezone (e.g. 2024-01-01T00:00:00Z). Got {}",
                query.request_timestamp
            ),
        )
    })?;
    let real_collection_name = get_real_collection_name(&collection_name);
    match document_delete_sync(
        &app.db,
        &app.stats_batcher,
        &app.doc_locks,
        app.metrics.as_deref(),
        &real_collection_name,
        &document_id,
        request_timestamp,
    )
    .await
    {
        Ok(skipped) => Ok(Json(OkResponse::ok_with_skipped(skipped))),
        Err(UpsertSyncError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(UpsertSyncError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(UpsertSyncError::IngressQueueFull(m)) => {
            Err(api_error(StatusCode::TOO_MANY_REQUESTS, m))
        }
        Err(UpsertSyncError::IngressWorkerExited(m)) => {
            Err(api_error(StatusCode::SERVICE_UNAVAILABLE, m))
        }
        Err(UpsertSyncError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
        }
    }
}

async fn fetch_documents(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    payload: Result<Json<DocumentFetchRequest>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(request) = payload.map_err(|e| json_rejection_response(e))?;
    validate_collection_name(&collection_name).map_err(validation_error)?;

    if request.page_size == 0 || request.page_size > 1000 {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "page_size must be between 1 and 1000".to_string(),
        ));
    }

    let real_collection_name = get_real_collection_name(&collection_name);
    let collection_config = app
        .db
        .get_collection_info_internal(&real_collection_name)
        .await
        .map_err(|e| match e {
            DbError::NotFound(m) => api_error(StatusCode::NOT_FOUND, m),
            e => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")),
        })?;

    if let Some(mf) = &request.metadata_filter {
        validate_metadata_filter(&collection_config, mf)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, e.to_string()))?;
    }

    let mut response = app
        .db
        .fetch_documents(&real_collection_name, &request, &collection_config)
        .await
        .map_err(|e| match e {
            DbError::Config(m) => api_error(StatusCode::BAD_REQUEST, m),
            e => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")),
        })?;

    if let Some(ref join) = request.join {
        crate::search_join::enrich_documents_with_joins(
            &app.db,
            &app.collection_cache,
            &mut response.documents,
            join,
            request.page_size,
        )
        .await
        .map_err(|e| match e {
            SearchError::InvalidFilter(m) => api_error(StatusCode::BAD_REQUEST, m),
            SearchError::Db(e) => {
                api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}"))
            }
            SearchError::NotFound(m) => api_error(StatusCode::NOT_FOUND, m),
            SearchError::Vectorization(m) => api_error(StatusCode::BAD_REQUEST, m),
            SearchError::IngressQueueFull(m) => api_error(StatusCode::TOO_MANY_REQUESTS, m),
            SearchError::IngressWorkerExited(m) => api_error(StatusCode::SERVICE_UNAVAILABLE, m),
        })?;
    }

    let documents: Vec<Value> = response
        .documents
        .into_iter()
        .map(|d| {
            serde_json::to_value(d)
                .map(models::flatten_doc_metadata)
                .unwrap_or(Value::Null)
        })
        .collect();

    Ok(Json(serde_json::json!({
        "documents": documents,
        "after": response.after,
    })))
}

async fn search(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    payload: Result<Json<SearchQuery>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(mut query) = payload.map_err(|e| json_rejection_response(e))?;
    validate_collection_name(&collection_name).map_err(validation_error)?;
    validate_search_query(&query).map_err(validation_error)?;
    normalize_search_query_python(&mut query);
    let real_collection_name = get_real_collection_name(&collection_name);
    let t0 = std::time::Instant::now();
    match app
        .search_ingress
        .search(real_collection_name.to_string(), query)
        .await
    {
        Ok(results) => {
            let response = SearchResponse {
                results,
                query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            };
            let vals: Vec<Value> = response
                .results
                .into_iter()
                .map(|r| {
                    serde_json::to_value(r)
                        .map(models::flatten_doc_metadata)
                        .unwrap_or(Value::Null)
                })
                .collect();
            Ok(Json(serde_json::json!({
                "results": vals,
                "query_time_ms": response.query_time_ms,
            })))
        }
        Err(SearchError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(SearchError::InvalidFilter(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(SearchError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(SearchError::IngressQueueFull(m)) => {
            Err(api_error(StatusCode::TOO_MANY_REQUESTS, m))
        }
        Err(SearchError::IngressWorkerExited(m)) => {
            Err(api_error(StatusCode::SERVICE_UNAVAILABLE, m))
        }
        Err(SearchError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
        }
    }
}

/// Read directives for [`tracing_subscriber::EnvFilter`]: `AMGIX_LOG_LEVEL`,
/// or **`info`** when unset. Single-token values are normalized (`WARNING`→`warn`, …);
/// strings with `=` are passed through for crate-specific overrides.
fn log_env_filter_directives() -> String {
    let raw = std::env::var("AMGIX_LOG_LEVEL").unwrap_or_else(|_| "info".into());
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "info".into();
    }
    if trimmed.contains('=') {
        return trimmed.to_string();
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "warning" => "warn".into(),
        "critical" => "error".into(),
        other => other.to_string(),
    }
}

fn init_tracing_from_env() {
    let directives = log_env_filter_directives();
    let env_filter =
        tracing_subscriber::EnvFilter::try_new(&directives).unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("info")
        });
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();
}

/// Central place: log any endpoint that finishes with an HTTP failure status (not 2xx,
/// excluding 218 used by `/v1/health/ready` partial-ready).
async fn log_failed_http_responses(request: Request, next: Next) -> Response {
    const MAX_BODY_CAPTURE: usize = 256 * 1024;

    fn truncate_body_for_log(raw: &str) -> std::borrow::Cow<'_, str> {
        const MAX_CHARS: usize = 8192;
        let count = raw.chars().count();
        if count <= MAX_CHARS {
            return std::borrow::Cow::Borrowed(raw);
        }
        let prefix: String = raw.chars().take(MAX_CHARS).collect();
        std::borrow::Cow::Owned(format!("{prefix}… (truncated, {count} chars)"))
    }

    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    let status = response.status();

    if status.is_success() || status.as_u16() == 218 {
        return response;
    }

    let server_err = status.is_server_error();

    let (parts, body) = response.into_parts();

    match to_bytes(body, MAX_BODY_CAPTURE).await {
        Ok(bytes) => {
            let decoded = String::from_utf8_lossy(&bytes);
            let body_for_log = truncate_body_for_log(&decoded);
            if server_err {
                tracing::error!(%method, path, %status, body = %body_for_log, "HTTP server error response");
            } else {
                tracing::warn!(%method, path, %status, body = %body_for_log, "HTTP client error response");
            }
            Response::from_parts(parts, Body::from(bytes))
        }
        Err(e) => {
            if server_err {
                tracing::error!(%method, path, %status, capture_error = %e, "HTTP server error response (body not captured)");
            } else {
                tracing::warn!(%method, path, %status, capture_error = %e, "HTTP client error response (body not captured)");
            }
            Response::from_parts(parts, Body::empty())
        }
    }
}

async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let sigterm = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sig =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        let _ = sig.recv().await;
    };

    #[cfg(not(unix))]
    let sigterm = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = sigterm => {}
    }
}

/// HTTP listen address (`host:port`). Default: `127.0.0.1:8235`.
fn amgix_now_listen_addr_from_env() -> String {
    std::env::var("AMGIX_NOW_LISTEN_ADDR").unwrap_or_else(|_| "127.0.0.1:8235".into())
}

/// `AMGIX_NOW_SYNC_DB_WRITES`: `true` / `1` / `yes` (case-insensitive) → Qdrant `wait=true` on document upserts and deletes.
fn amgix_now_sync_db_writes_from_env() -> bool {
    std::env::var("AMGIX_NOW_SYNC_DB_WRITES").map_or(false, |s| {
        matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    })
}

#[tokio::main]
async fn main() {
    platform::init();
    init_tracing_from_env();

    let db_url = std::env::var("AMGIX_DATABASE_URL")
        .unwrap_or_else(|_| "qdrant://localhost:6334".to_string());

    let amqp_url = std::env::var("AMGIX_AMQP_URL").ok();

    let bunny = if let Some(ref url) = amqp_url {
        tracing::info!("AMGIX_AMQP_URL set — connecting to RabbitMQ");
        match bunny_talk::BunnyTalk::create(url).await {
            Ok(b) => {
                tracing::info!("RabbitMQ connected");
                Some(b)
            }
            Err(e) => {
                tracing::error!("RabbitMQ connect failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // In cluster mode distributed locks require writes to be visible immediately.
    let sync_db_writes = bunny.is_some() || amgix_now_sync_db_writes_from_env();

    let lock_client = bunny
        .as_ref()
        .map(|b| Arc::new(lock_client::LockClient::new(Arc::clone(b))));
    if let Some(lc) = lock_client.as_deref() {
        lc.start_cleanup_task();
    }

    let db = match QdrantDb::new(&qdrant_client_url(&db_url), sync_db_writes) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            tracing::error!("Qdrant client (AMGIX_DATABASE_URL): {e}");
            std::process::exit(1);
        }
    };

    db.wait_connected().await;

    if let Some(lc) = lock_client.as_deref() {
        tracing::info!("Acquiring database-configure lock...");
        let _guard = match lc.acquire(&["database-configure"], std::time::Duration::from_secs(30)).await {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("Failed to acquire database-configure lock: {e}");
                std::process::exit(1);
            }
        };
        if let Err(e) = db.configure().await {
            tracing::error!("Qdrant configure: {e}");
            std::process::exit(1);
        }
    } else if let Err(e) = db.configure().await {
        tracing::error!("Qdrant configure: {e}");
        std::process::exit(1);
    }

    let qdrant_version = match db.probe().await {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Qdrant probe failed: {e}");
            std::process::exit(1);
        }
    };

    let amgix_version = AMGIX_VERSION.to_string();
    let amgix_variant = AMGIX_VARIANT.to_string();
    let amgix_version_display = format!("{} ({})", amgix_version, AMGIX_VARIANT);

    println!(r#"                              _         _   _                 
     /\                      (_)       | \ | |                
    /  \    _ __ ___    __ _  _ __  __ |  \| |  ___ __      __
   / /\ \  | '_ ` _ \  / _` || |\ \/ / | . ` | / _ \\ \ /\ / /
  / ____ \ | | | | | || (_| || | >  <  | |\  || (_) |\ V  V / 
 /_/    \_\|_| |_| |_| \__, ||_|/_/\_\ |_| \_| \___/  \_/\_/  
                        __/ |                                 
                       |___/                                  
    "#);

    let cluster_mode = if bunny.is_some() { "cluster" } else { "standalone" };
    tracing::info!("Amgix version: {amgix_version_display}");
    tracing::info!("Qdrant version: {qdrant_version}");
    tracing::info!("Cluster mode: {cluster_mode}");
    tracing::info!("Synchronous Database Writes: {sync_db_writes}");

    let stats_locks = NamedLocks::new();
    stats_locks.start_stats_cleanup_task();
    let (stats_batcher, stats_shutdown) =
        StatsUpdateBatcher::new(Arc::clone(&db), stats_locks, bunny.clone());

    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    let index_threads = std::env::var("AMGIX_NOW_INDEX_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (num_cpus / 2).max(1));
    let search_threads = std::env::var("AMGIX_NOW_SEARCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (num_cpus).max(2));

    // Tell candle/gemm how many threads to use for CPU matrix multiplication.
    // Must be set before any embedding call; candle reads it once on first matmul.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", index_threads.to_string()); }

    // Register index_threads so vectorize_documents can cap native thread spawning.
    vectors::vectorizer::set_index_threads(index_threads);

    let search_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(search_threads)
            .thread_name(|i| format!("search-{i}"))
            .build()
            .expect("failed to build search thread pool"),
    );
    let web_threads = std::env::var("TOKIO_WORKER_THREADS").unwrap_or_default();
    tracing::info!("Web pool: {web_threads} threads, Index pool: {index_threads} threads, Search pool: {search_threads} threads");

    let (metrics_collector, metrics_shutdown) = if let Some(ref url) = amqp_url {
        let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
        let (c, s) = metrics::MetricsCollector::new(url.clone(), Arc::clone(&db), hostname);
        (Some(Arc::new(c)), Some(s))
    } else {
        (None, None)
    };

    let collection_cache = CollectionConfigCache::new(bunny.is_some());
    let doc_locks = match lock_client {
        Some(lc) => LockBackend::distributed(lc),
        None => {
            let locks = NamedLocks::new();
            locks.start_cleanup_task();
            LockBackend::local(locks)
        }
    };
    let (upsert_ingress, upsert_shutdown) = UpsertIngress::new(
        Arc::clone(&db),
        collection_cache.clone(),
        stats_batcher.clone(),
        doc_locks.clone(),
        metrics_collector.clone(),
    );
    let (search_ingress, search_shutdown) =
        SearchIngress::new(Arc::clone(&db), collection_cache.clone(), Arc::clone(&search_pool), metrics_collector.clone());

    let state = AppState {
        db,
        qdrant_version,
        amgix_version,
        amgix_variant,
        amgix_version_display,
        collection_cache,
        stats_batcher,
        upsert_ingress,
        search_ingress,
        doc_locks,
        bunny: bunny.clone(),
        metrics: metrics_collector,
    };

    let app = Router::new()
        .route("/v1/version", get(version_endpoint))
        .route("/v1/system/info", get(system_info))
        .route("/v1/health/check", get(health_check))
        .route("/v1/health/ready", get(health_ready))
        .route("/v1/metrics/current", get(not_implemented_amgix_now_metrics))
        .route("/v1/metrics/prometheus", get(not_implemented_amgix_now_metrics))
        .route("/v1/metrics/trends", get(not_implemented_amgix_now_metrics))
        .route("/v1/metrics/definitions", get(not_implemented_amgix_now_metrics))
        .route("/v1/collections", get(list_collections))
        .route(
            "/v1/collections/{collection_name}",
            get(get_collection_config).post(create_collection).delete(delete_collection),
        )
        .route(
            "/v1/collections/{collection_name}/exists",
            get(collection_exists),
        )
        .route(
            "/v1/collections/{collection_name}/stats",
            get(get_collection_stats),
        )
        .route(
            "/v1/collections/{collection_name}/queue/info",
            get(not_implemented_amgix_now_collection_queue),
        )
        .route(
            "/v1/collections/{collection_name}/queue",
            delete(not_implemented_amgix_now_collection_queue),
        )
        .route(
            "/v1/collections/{collection_name}/empty",
            post(empty_collection),
        )
        .route(
            "/v1/collections/{collection_name}/documents",
            post(upsert_document),
        )
        .route(
            "/v1/collections/{collection_name}/documents/sync",
            post(upsert_document),
        )
        .route(
            "/v1/collections/{collection_name}/documents/bulk",
            post(upsert_documents_bulk),
        )
        .route(
            "/v1/collections/{collection_name}/documents/export",
            get(export_documents),
        )
        .route(
            "/v1/collections/{collection_name}/documents/{document_id}/sync",
            delete(delete_document),
        )
        .route(
            "/v1/collections/{collection_name}/documents/{document_id}/status",
            get(get_document_status),
        )
        .route(
            "/v1/collections/{collection_name}/documents/{document_id}",
            get(get_document).delete(delete_document),
        )
        .route(
            "/v1/collections/{collection_name}/documents/fetch",
            post(fetch_documents),
        )
        .route(
            "/v1/collections/{collection_name}/search",
            post(search),
        )
        .layer(middleware::from_fn_with_state(state.clone(), api_metrics_middleware))
        .layer(middleware::from_fn(log_failed_http_responses))
        .with_state(state);

    let listen_addr = amgix_now_listen_addr_from_env();
    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    tracing::info!("Listening http://{listen_addr}");

    let shutdown = async {
        wait_for_shutdown_signal().await;
        tracing::info!("shutdown signal received, finishing in-flight requests");
    };

    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;

    upsert_shutdown.shutdown_and_wait().await;

    search_shutdown.shutdown_and_wait().await;

    stats_shutdown.shutdown_and_wait().await;

    if let Some(s) = metrics_shutdown {
        s.stop_and_join();
    }

    if let Some(ref b) = bunny {
        b.close().await;
    }

    if let Err(e) = serve_result {
        tracing::error!("server error: {e}");
        std::process::exit(1);
    }
}
