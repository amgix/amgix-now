mod amgix;
mod common;
mod encoder;
mod functions;
mod models;
mod qdrant;
mod vectors;

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::{json, Value};
use uuid::Uuid;

use common::{
    get_real_collection_name, qdrant_client_url, VectorType, DATABASE_KIND,
};
use encoder::{
    document_delete_sync, document_upsert_bulk, document_upsert_sync, validate_models,
    CollectionConfigCache, NamedLocks, SearchError, StatsUpdateBatcher, UpsertSyncError,
};
use encoder::search as encoder_search;
use models::{
    BulkUploadRequest, CollectionConfig, CollectionConfigInternal, CollectionExistsResponse,
    Document, OkResponse, ReadyResponse, SearchQuery, SearchResult, SystemInfoResponse,
    VectorConfigInternal, VersionResponse,
};
use qdrant::{DbError, QdrantDb};

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
}

fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({ "detail": msg.into() })),
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

async fn create_collection(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    Json(config): Json<CollectionConfig>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
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

async fn upsert_document(
    State(app): State<AppState>,
    Path(collection_name): Path<String>,
    Json(document): Json<Document>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    let real_collection_name = get_real_collection_name(&collection_name);
    match document_upsert_sync(
        &app.db,
        &app.collection_cache,
        &app.stats_batcher,
        &app.doc_locks,
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
    Json(request): Json<BulkUploadRequest>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
    let real_collection_name = get_real_collection_name(&collection_name);
    match document_upsert_bulk(
        &app.db,
        &app.collection_cache,
        &app.stats_batcher,
        &app.doc_locks,
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

async fn delete_document(
    State(app): State<AppState>,
    Path((collection_name, document_id)): Path<(String, String)>,
) -> Result<Json<OkResponse>, (StatusCode, Json<Value>)> {
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
    Json(query): Json<SearchQuery>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, Json<Value>)> {
    let real_collection_name = get_real_collection_name(&collection_name);
    match encoder_search(&app.db, &app.collection_cache, &real_collection_name, query).await {
        Ok(results) => Ok(Json(results)),
        Err(SearchError::NotFound(m)) => Err(api_error(StatusCode::NOT_FOUND, m)),
        Err(SearchError::InvalidFilter(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(SearchError::Vectorization(m)) => Err(api_error(StatusCode::BAD_REQUEST, m)),
        Err(SearchError::Db(e)) => {
            Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}")))
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
    let db_url = std::env::var("AMGIX_DATABASE_URL")
        .unwrap_or_else(|_| "qdrant://localhost:6334".to_string());

    let db = match QdrantDb::new(&qdrant_client_url(&db_url)) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Qdrant client (AMGIX_DATABASE_URL): {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = db.configure().await {
        eprintln!("Qdrant configure: {e}");
        std::process::exit(1);
    }

    let qdrant_version = match db.probe().await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Qdrant probe failed: {e}");
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

    println!("Amgix version: {amgix_version_display}");
    println!("Qdrant version: {qdrant_version}");

    let (stats_batcher, stats_shutdown) = StatsUpdateBatcher::new(Arc::clone(&db), NamedLocks::new());

    let state = AppState {
        db,
        qdrant_version,
        amgix_version,
        amgix_variant,
        amgix_version_display,
        collection_cache: CollectionConfigCache::new(),
        stats_batcher,
        doc_locks: NamedLocks::new(),
    };

    let app = Router::new()
        .route("/v1/version", get(version_endpoint))
        .route("/v1/system/info", get(system_info))
        .route("/v1/health/check", get(health_check))
        .route("/v1/health/ready", get(health_ready))
        .route(
            "/v1/collections/{collection_name}",
            post(create_collection).delete(delete_collection),
        )
        .route(
            "/v1/collections/{collection_name}/exists",
            get(collection_exists),
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
            "/v1/collections/{collection_name}/documents/{document_id}",
            delete(delete_document),
        )
        .route(
            "/v1/collections/{collection_name}/search",
            post(search),
        )
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8235").await.unwrap();
    println!("Listening  http://0.0.0.0:8235");

    let shutdown = async {
        wait_for_shutdown_signal().await;
        println!("shutdown signal received, finishing in-flight requests");
    };

    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;

    stats_shutdown.shutdown_and_wait().await;

    if let Err(e) = serve_result {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
