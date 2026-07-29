//! Standalone async ingestion poller — drains `amgix_sys_queue` when RabbitMQ is unset.
//!
//! Upserts go through [`document_upsert_blocking`] (singles MPSC, waits when full) so
//! downstream micro-batching still applies and ingress backpressure does not fail queue rows.
//! Multiple standalone instances against one Qdrant are unsupported.

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
    document_delete_sync, document_upsert_blocking, LockBackend, QueueVerify, StatsUpdateBatcher,
    UpsertIngress, UpsertSyncError,
};
use crate::metrics::MetricsCollector;
use crate::models::{QueueDocument, QueueOperationType, QueuedDocumentStatus};
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
/// queue work can still complete through the singles MPSC.
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

    // Pipeline upserts through singles MPSC concurrently so micro-batching can form.
    let mut handles = Vec::with_capacity(upserts.len());
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
        let collection_name = entry.collection_name.clone();
        let ingress = upsert_ingress.clone();
        let queue = Some(QueueVerify {
            queue_id: entry.queue_id.clone(),
            collection_id: entry.collection_id.clone(),
        });
        handles.push(tokio::spawn(async move {
            let result =
                document_upsert_blocking(&ingress, &collection_name, document, queue).await;
            (entry, result)
        }));
    }

    for handle in handles {
        match handle.await {
            Ok((entry, Ok(_))) => {
                if let Err(e) = db.delete_from_queue(&[entry.queue_id.clone()]).await {
                    error!("Failed to delete queue entry {}: {e}", entry.queue_id);
                }
            }
            Ok((entry, Err(UpsertSyncError::QueueEntryGone))) => {
                // Row already drained (or re-verify miss). Do not ack-delete or mark_outcome —
                // true drain means the point is gone; a false miss must remain for retry.
                debug!(
                    "Queue upsert {} entry gone under lock; leaving queue row alone",
                    entry.queue_id
                );
            }
            Ok((entry, Err(e))) => {
                debug!("Queue upsert {} failed: {e}", entry.queue_id);
                if let Err(ue) = mark_outcome(db, &entry, &e).await {
                    error!("Failed to update queue status for {}: {ue}", entry.queue_id);
                }
            }
            Err(e) => error!("Queue upsert task join error: {e}"),
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
