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

use common::{get_real_collection_name, qdrant_client_url, VectorType};
use encoder::{
    document_delete_sync, document_upsert_bulk, document_upsert_sync, validate_models,
    CollectionConfigCache, NamedLocks, SearchError, UpsertSyncError,
};
use encoder::search as encoder_search;
use models::{
    BulkUploadRequest, CollectionConfig, CollectionConfigInternal, CollectionExistsResponse,
    Document, OkResponse, SearchQuery, SearchResult, VectorConfigInternal,
};
use qdrant::{DbError, QdrantDb};

#[derive(Clone)]
struct AppState {
    db: Arc<QdrantDb>,
    collection_cache: CollectionConfigCache,
    stats_locks: NamedLocks,
    doc_locks: NamedLocks,
}

fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (
        status,
        Json(json!({ "detail": msg.into() })),
    )
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

    let validation = validate_models(&internal_for_validation);
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
        &app.stats_locks,
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
        &app.stats_locks,
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
        &app.stats_locks,
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

    let state = AppState {
        db,
        collection_cache: CollectionConfigCache::new(),
        stats_locks: NamedLocks::new(),
        doc_locks: NamedLocks::new(),
    };

    let app = Router::new()
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
    println!("amgix-now listening on http://0.0.0.0:8235");
    axum::serve(listener, app).await.unwrap();
}
