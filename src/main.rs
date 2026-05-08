mod amgix;
mod common;
mod encoder;
mod functions;
mod models;
mod qdrant;
mod validation;
mod vectors;

use std::sync::Arc;

use rayon::ThreadPool;

use axum::{
    body::{to_bytes, Body},
    extract::{
        rejection::JsonRejection,
        Path, Request, State,
    },
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::{json, Value};
use uuid::Uuid;

use common::{
    get_real_collection_name, get_user_collection_name, qdrant_client_url, VectorType, DATABASE_KIND,
};
use encoder::{
    document_delete_sync, document_upsert_bulk, document_upsert_sync, validate_models,
    CollectionConfigCache, NamedLocks, SearchError, StatsUpdateBatcher, UpsertSyncError,
};
use encoder::search as encoder_search;
use models::{
    BulkUploadRequest, CollectionConfig, CollectionConfigInternal, CollectionExistsResponse,
    CollectionStatsResponse, Document, DocumentStatus, DocumentStatusResponse, OkResponse,
    QueueInfo, QueuedDocumentStatus, ReadyResponse, SearchQuery, SearchResult,
    SystemInfoResponse, VectorConfigInternal, VersionResponse,
};
use qdrant::{DbError, QdrantDb};
use validation::{
    normalize_document_python, normalize_search_query_python,
    validate_bulk_upload, validate_collection_config, validate_collection_name, validate_document,
    validate_search_query,
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
    doc_locks: NamedLocks,
    index_pool: Arc<ThreadPool>,
    search_pool: Arc<ThreadPool>,
}

fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({ "detail": msg.into() })),
    )
}

/// FastAPI / Pydantic shape: `{ "detail": [ { "type", "loc", "msg", "input" }, ... ] }`.
fn validation_error_detail_list(msg: impl Into<String>) -> Value {
    json!([
        {
            "type": "validation_error",
            "loc": ["body"],
            "msg": msg.into(),
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

/// `GET /v1/health/check` — process is up and serving HTTP (mirrors Python `health_check`).
async fn health_check() -> Json<OkResponse> {
    Json(OkResponse::ok())
}

/// `GET /v1/health/ready` — same status rules and JSON body as Python `readiness_check`.
/// In **amgix-now** there is no RabbitMQ or separate index/query workers; those probes are
/// reported `true` when not applicable so `ready` tracks Qdrant connectivity.
async fn health_ready(State(app): State<AppState>) -> (StatusCode, Json<ReadyResponse>) {
    const PARTIAL_READY: u16 = 218;

    let database = app.db.is_connected().await;
    let rabbitmq = true;
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

    let stats = app.db.get_collection_stats(&real_collection_name).await.map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to get collection stats: {e}"),
        )
    })?;

    Ok(Json(CollectionStatsResponse {
        doc_count: stats.doc_count,
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
    match document_upsert_sync(
        &app.db,
        &app.collection_cache,
        &app.stats_batcher,
        &app.doc_locks,
        &app.index_pool,
        &real_collection_name,
        document,
    )
    .await
    {
        Ok(skipped) => Ok(Json(OkResponse::ok_with_skipped(skipped))),
        Err(UpsertSyncError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(UpsertSyncError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
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
    match document_upsert_bulk(
        &app.db,
        &app.collection_cache,
        &app.stats_batcher,
        &app.doc_locks,
        &app.index_pool,
        &real_collection_name,
        request.documents,
    )
    .await
    {
        Ok(skipped) => Ok(Json(OkResponse::ok_with_skipped(skipped))),
        Err(UpsertSyncError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(UpsertSyncError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(UpsertSyncError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
        }
    }
}

/// `GET /v1/collections/{collection_name}/documents/{document_id}` — mirrors Python `get_document`.
async fn get_document(
    State(app): State<AppState>,
    Path((collection_name, document_id)): Path<(String, String)>,
) -> Result<Json<Document>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    let rows = app
        .db
        .get_documents(&real_collection_name, &[document_id.as_str()], false)
        .await
        .map_err(|e| match e {
            DbError::NotFound(m) => api_error(StatusCode::NOT_FOUND, m),
            DbError::Config(m) => api_error(StatusCode::BAD_REQUEST, m),
            e => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {e}"),
            ),
        })?;

    let doc_with = rows
        .into_iter()
        .next()
        .flatten()
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                format!("Document '{document_id}' not found"),
            )
        })?;

    Ok(Json(Document::from(doc_with)))
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
        .get_documents(&real_collection_name, &[document_id.as_str()], true)
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
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    validate_collection_name(&collection_name).map_err(validation_error)?;
    let real_collection_name = get_real_collection_name(&collection_name);
    match document_delete_sync(
        &app.db,
        &app.stats_batcher,
        &app.doc_locks,
        &real_collection_name,
        &document_id,
    )
    .await
    {
        Ok(()) => Ok(Json(OkResponse::ok())),
        Err(UpsertSyncError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(UpsertSyncError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(UpsertSyncError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
        }
    }
}

async fn search(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    payload: Result<Json<SearchQuery>, JsonRejection>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, Json<Value>)> {
    let Json(mut query) = payload.map_err(|e| json_rejection_response(e))?;
    validate_collection_name(&collection_name).map_err(validation_error)?;
    validate_search_query(&query).map_err(validation_error)?;
    normalize_search_query_python(&mut query);
    let real_collection_name = get_real_collection_name(&collection_name);
    match encoder_search(&app.db, &app.collection_cache, &app.search_pool, &real_collection_name, query).await {
        Ok(results) => Ok(Json(results)),
        Err(SearchError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(SearchError::InvalidFilter(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(SearchError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
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

#[tokio::main]
async fn main() {
    init_tracing_from_env();

    let db_url = std::env::var("AMGIX_DATABASE_URL")
        .unwrap_or_else(|_| "qdrant://localhost:6334".to_string());

    let db = match QdrantDb::new(&qdrant_client_url(&db_url)) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            tracing::error!("Qdrant client (AMGIX_DATABASE_URL): {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = db.configure().await {
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

    let amgix_version = std::env::var("AMGIX_VERSION").unwrap_or_default();
    let amgix_version = amgix_version.trim().to_string();
    let amgix_variant = std::env::var("AMGIX_VARIANT").unwrap_or_default();
    let amgix_variant = amgix_variant.trim().to_string();
    let amgix_version_display = if amgix_variant.is_empty() {
        amgix_version.clone()
    } else {
        format!("{} ({})", amgix_version, amgix_variant)
    };

    tracing::info!("Amgix version: {amgix_version_display}");
    tracing::info!("Qdrant version: {qdrant_version}");

    let (stats_batcher, stats_shutdown) = StatsUpdateBatcher::new(Arc::clone(&db), NamedLocks::new());

    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    let index_threads = std::env::var("AMGIX_NOW_INDEX_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (num_cpus / 2).max(1));
    let search_threads = std::env::var("AMGIX_NOW_SEARCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (num_cpus / 2).max(1));

    let index_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(index_threads)
            .thread_name(|i| format!("ingest-{i}"))
            .build()
            .expect("failed to build ingest thread pool"),
    );
    let search_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(search_threads)
            .thread_name(|i| format!("search-{i}"))
            .build()
            .expect("failed to build search thread pool"),
    );
    let web_threads = std::env::var("TOKIO_WORKER_THREADS").unwrap_or_default();
    tracing::info!("Web pool: {web_threads} threads, Index pool: {index_threads} threads, Search pool: {search_threads} threads");

    let state = AppState {
        db,
        qdrant_version,
        amgix_version,
        amgix_variant,
        amgix_version_display,
        collection_cache: CollectionConfigCache::new(),
        stats_batcher,
        doc_locks: NamedLocks::new(),
        index_pool,
        search_pool,
    };

    let app = Router::new()
        .route("/v1/version", get(version_endpoint))
        .route("/v1/system/info", get(system_info))
        .route("/v1/health/check", get(health_check))
        .route("/v1/health/ready", get(health_ready))
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
            "/v1/collections/{collection_name}/search",
            post(search),
        )
        .layer(middleware::from_fn(log_failed_http_responses))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8235").await.unwrap();
    tracing::info!("Listening  http://0.0.0.0:8235");

    let shutdown = async {
        wait_for_shutdown_signal().await;
        tracing::info!("shutdown signal received, finishing in-flight requests");
    };

    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;

    stats_shutdown.shutdown_and_wait().await;

    if let Err(e) = serve_result {
        tracing::error!("server error: {e}");
        std::process::exit(1);
    }
}
