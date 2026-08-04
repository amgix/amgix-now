//! Standalone async ingestion poller — drains `amgix_sys_queue` when RabbitMQ is unset.
//!
//! Upserts are grouped by collection and sent through the bulk ingress (waits when full) with
//! per-document queue tickets so re-verify under lock still applies. Multiple standalone
//! instances against one Qdrant are unsupported.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use crate::common::{
    MAX_DB_RETRIES, MAX_QUEUE_DELIVERY_ATTEMPTS, MAX_RETRY_SLEEP_SECONDS, QUEUE_POLLER_BATCH_SIZE,
    QUEUE_POLLER_IDLE_MS,
};
use crate::encoder::{
    document_delete_sync, document_upsert_bulk_blocking, LockBackend, QueueVerify,
    StatsUpdateBatcher, UpsertIngress, UpsertSyncError,
};
use crate::metrics::MetricsCollector;
use crate::models::{Document, QueueDocument, QueueOperationType, QueuedDocumentStatus};
use crate::qdrant::QdrantDb;

fn retry_delay_secs(try_count: u32) -> f64 {
    let base = 2.0 * f64::from(try_count);
    let jitter = 0.1 + f64::from(try_count.wrapping_mul(2654435761) % 1400) / 1000.0;
    (base + jitter).min(MAX_RETRY_SLEEP_SECONDS)
}

fn backoff_elapsed(entry: &QueueDocument) -> bool {
    if entry.try_count == 0 {
        return true;
    }
    let needed = retry_delay_secs(entry.try_count);
    let elapsed = (Utc::now() - entry.timestamp).num_milliseconds() as f64 / 1000.0;
    elapsed >= needed
}

async fn mark_outcome(
    db: &QdrantDb,
    entry: &QueueDocument,
    err: &UpsertSyncError,
) -> Result<(), crate::qdrant::DbError> {
    let new_try = entry.try_count.saturating_add(1);
    let info = err.to_string();
    let status = match err {
        UpsertSyncError::QueueEntryGone => {
            // Should not be passed here; poller no-ops this variant.
            return Ok(());
        }
        UpsertSyncError::Validation(_)
        | UpsertSyncError::QueueCollectionMismatch(_)
        | UpsertSyncError::NotFound(_) => QueuedDocumentStatus::Failed,
        UpsertSyncError::Vectorization(_) => {
            if new_try < MAX_QUEUE_DELIVERY_ATTEMPTS {
                QueuedDocumentStatus::Requeued
            } else {
                QueuedDocumentStatus::Failed
            }
        }
        _ => {
            if new_try < MAX_DB_RETRIES {
                QueuedDocumentStatus::Requeued
            } else {
                QueuedDocumentStatus::Failed
            }
        }
    };
    db.update_queue_status(&[(entry.queue_id.clone(), new_try)], status, &info)
        .await
}

/// Drop / await after HTTP serve returns and **before** upsert ingress shutdown so in-flight
/// queue work can still complete through the bulk MPSC.
pub struct QueuePollerShutdown {
    stop_tx: watch::Sender<bool>,
    join: tokio::task::JoinHandle<()>,
}

impl QueuePollerShutdown {
    pub async fn shutdown_and_wait(self) {
        let _ = self.stop_tx.send(true);
        match self.join.await {
            Ok(()) => info!("Standalone queue poller exited"),
            Err(e) => error!("Standalone queue poller join error: {e}"),
        }
    }
}

/// Returns `true` when the poller should exit.
async fn idle_or_stop(stop_rx: &mut watch::Receiver<bool>) -> bool {
    if *stop_rx.borrow() {
        return true;
    }
    tokio::select! {
        result = stop_rx.changed() => {
            result.is_err() || *stop_rx.borrow()
        }
        _ = sleep(Duration::from_millis(QUEUE_POLLER_IDLE_MS)) => {
            *stop_rx.borrow()
        }
    }
}

pub fn spawn_queue_poller(
    db: Arc<QdrantDb>,
    upsert_ingress: UpsertIngress,
    stats_batcher: StatsUpdateBatcher,
    doc_locks: LockBackend,
    metrics: Option<Arc<MetricsCollector>>,
) -> QueuePollerShutdown {
    let (stop_tx, mut stop_rx) = watch::channel(false);
    let join = tokio::spawn(async move {
        info!("Standalone queue poller started");
        loop {
            if *stop_rx.borrow() {
                break;
            }

            let pending = match db.claim_pending_queue(QUEUE_POLLER_BATCH_SIZE).await {
                Ok(p) => p,
                Err(e) => {
                    error!("Queue poller claim failed: {e}");
                    if idle_or_stop(&mut stop_rx).await {
                        break;
                    }
                    continue;
                }
            };

            // Stop requested during claim: leave rows queued; do not start a new batch.
            if *stop_rx.borrow() {
                break;
            }

            let eligible: Vec<_> = pending.into_iter().filter(backoff_elapsed).collect();
            if eligible.is_empty() {
                if idle_or_stop(&mut stop_rx).await {
                    break;
                }
                continue;
            }

            // Finish the whole batch before re-checking stop so ack/delete stay consistent.
            if let Err(e) = process_batch(
                &db,
                &upsert_ingress,
                &stats_batcher,
                &doc_locks,
                metrics.as_deref(),
                eligible,
            )
            .await
            {
                error!("Queue poller batch failed: {e}");
                if idle_or_stop(&mut stop_rx).await {
                    break;
                }
            }
        }
        info!("Standalone queue poller stopping");
    });

    QueuePollerShutdown { stop_tx, join }
}

/// One claimed upsert ready to send on the bulk path.
struct UpsertWork {
    entry: QueueDocument,
    document: Document,
}

async fn process_batch(
    db: &QdrantDb,
    upsert_ingress: &UpsertIngress,
    stats_batcher: &StatsUpdateBatcher,
    doc_locks: &LockBackend,
    metrics: Option<&MetricsCollector>,
    eligible: Vec<QueueDocument>,
) -> Result<(), crate::qdrant::DbError> {
    let mut upserts = Vec::new();
    let mut deletes = Vec::new();
    for entry in eligible {
        match entry.op_type {
            QueueOperationType::Upsert => upserts.push(entry),
            QueueOperationType::Delete => deletes.push(entry),
        }
    }

    // Group upserts by collection for the bulk ingress.
    let mut by_collection: HashMap<String, Vec<UpsertWork>> = HashMap::new();
    for entry in upserts {
        let Some(document) = entry.document.clone() else {
            warn!(
                "Queue entry {} missing document payload; marking failed",
                entry.queue_id
            );
            db.update_queue_status(
                &[(entry.queue_id.clone(), entry.try_count.saturating_add(1))],
                QueuedDocumentStatus::Failed,
                "missing document payload",
            )
            .await?;
            continue;
        };
        by_collection
            .entry(entry.collection_name.clone())
            .or_default()
            .push(UpsertWork { entry, document });
    }

    for (collection_name, mut works) in by_collection {
        // Same doc_id twice in one claim: keep newest timestamp, ack-delete the rest as stale.
        works.sort_by(|a, b| {
            a.document
                .id
                .cmp(&b.document.id)
                .then(a.document.timestamp.cmp(&b.document.timestamp))
        });
        let mut winners: Vec<UpsertWork> = Vec::with_capacity(works.len());
        let mut superseded: Vec<QueueDocument> = Vec::new();
        for work in works {
            if let Some(prev) = winners.last_mut() {
                if prev.document.id == work.document.id {
                    // `works` sorted ascending by timestamp — newer replaces older.
                    superseded.push(std::mem::replace(prev, work).entry);
                    continue;
                }
            }
            winners.push(work);
        }
        for entry in superseded {
            if let Err(e) = db.delete_from_queue(&[entry.queue_id.clone()]).await {
                error!(
                    "Failed to delete superseded queue entry {}: {e}",
                    entry.queue_id
                );
            }
        }

        if winners.is_empty() {
            continue;
        }

        let documents: Vec<Document> = winners.iter().map(|w| w.document.clone()).collect();
        let queue_verifies: Vec<Option<QueueVerify>> = winners
            .iter()
            .map(|w| {
                Some(QueueVerify {
                    queue_id: w.entry.queue_id.clone(),
                    collection_id: w.entry.collection_id.clone(),
                })
            })
            .collect();

        match document_upsert_bulk_blocking(
            upsert_ingress,
            &collection_name,
            documents,
            queue_verifies,
        )
        .await
        {
            Ok(outcome) => {
                let drained: std::collections::HashSet<&str> =
                    outcome.drained.iter().map(|s| s.as_str()).collect();
                let mut to_delete = Vec::new();
                for work in &winners {
                    if drained.contains(work.document.id.as_str()) {
                        debug!(
                            "Queue upsert {} entry gone under lock; leaving queue row alone",
                            work.entry.queue_id
                        );
                    } else {
                        to_delete.push(work.entry.queue_id.clone());
                    }
                }
                if !to_delete.is_empty() {
                    if let Err(e) = db.delete_from_queue(&to_delete).await {
                        error!("Failed to delete queue entries after bulk upsert: {e}");
                    }
                }
            }
            Err(UpsertSyncError::QueueEntryGone) => {
                // Whole-batch variant should not appear from bulk_internal; treat as no-op.
                debug!(
                    "Bulk queue upsert for {collection_name} reported QueueEntryGone; leaving rows"
                );
            }
            Err(e) => {
                debug!("Bulk queue upsert for {collection_name} failed: {e}");
                for work in &winners {
                    if let Err(ue) = mark_outcome(db, &work.entry, &e).await {
                        error!(
                            "Failed to update queue status for {}: {ue}",
                            work.entry.queue_id
                        );
                    }
                }
            }
        }
    }

    for entry in deletes {
        let request_timestamp = entry.doc_timestamp;
        let queue = QueueVerify {
            queue_id: entry.queue_id.clone(),
            collection_id: entry.collection_id.clone(),
        };
        match document_delete_sync(
            db,
            stats_batcher,
            doc_locks,
            metrics,
            &entry.collection_name,
            &entry.doc_id,
            request_timestamp,
            Some(&queue),
        )
        .await
        {
            Ok(_) => {
                if let Err(e) = db.delete_from_queue(&[entry.queue_id.clone()]).await {
                    error!("Failed to delete queue entry {}: {e}", entry.queue_id);
                }
            }
            Err(UpsertSyncError::QueueEntryGone) => {
                debug!(
                    "Queue delete {} entry gone under lock; leaving queue row alone",
                    entry.queue_id
                );
            }
            Err(e) => {
                debug!("Queue delete {} failed: {e}", entry.queue_id);
                if let Err(ue) = mark_outcome(db, &entry, &e).await {
                    error!("Failed to update queue status for {}: {ue}", entry.queue_id);
                }
            }
        }
    }

    Ok(())
}
