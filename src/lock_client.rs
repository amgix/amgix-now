use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use tracing::debug;
use uuid::Uuid;

use crate::bunny_talk::BunnyTalk;

const LOCK_SERVICE: &str = "lock-service";
// Per-attempt RPC timeout inside the retry loop
const RPC_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(2);
// Default release retry attempts on drop
const RELEASE_MAX_ATTEMPTS: u32 = 3;

#[derive(Debug)]
pub enum LockError {
    AcquisitionTimeout { locks: Vec<String>, elapsed: Duration },
    Rpc(String),
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::AcquisitionTimeout { locks, elapsed } => write!(
                f,
                "Failed to acquire locks within {:.1}s: {:?}",
                elapsed.as_secs_f64(),
                locks
            ),
            LockError::Rpc(msg) => write!(f, "Lock RPC error: {}", msg),
        }
    }
}

impl std::error::Error for LockError {}

pub struct LockClient {
    bunny: Arc<BunnyTalk>,
    owner_id: String,
}

impl LockClient {
    pub fn new(bunny: Arc<BunnyTalk>) -> Self {
        let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
        let pid = std::process::id();
        let owner_id = format!("{}-{}-{}", hostname, pid, Uuid::new_v4());
        Self { bunny, owner_id }
    }

    /// Acquire one or more locks, retrying until `timeout` elapses.
    /// Returns a `LockGuard` that releases on drop.
    pub async fn acquire(
        &self,
        lock_names: &[&str],
        timeout: Duration,
    ) -> Result<LockGuard, LockError> {
        let names: Vec<String> = lock_names.iter().map(|s| s.to_string()).collect();
        let deadline = Instant::now() + timeout;
        let mut attempt: u32 = 0;
        let mut last_err: Option<String> = None;

        while Instant::now() < deadline {
            match self.rpc_acquire(&names).await {
                Ok(true) => {
                    return Ok(LockGuard {
                        bunny: Arc::clone(&self.bunny),
                        owner_id: self.owner_id.clone(),
                        lock_names: names,
                        released: false,
                    });
                }
                Ok(false) => {
                    // Lock held by someone else — clear error and retry
                    last_err = None;
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }

            attempt += 1;
            let backoff = Duration::from_millis((attempt as u64 * 100).min(1000));
            tokio::time::sleep(backoff).await;
        }

        if let Some(e) = last_err {
            Err(LockError::Rpc(e))
        } else {
            Err(LockError::AcquisitionTimeout {
                locks: names,
                elapsed: timeout,
            })
        }
    }

    async fn rpc_acquire(&self, lock_names: &[String]) -> Result<bool, String> {
        let resp = self
            .bunny
            .rpc(
                LOCK_SERVICE,
                json!({
                    "action": "acquire",
                    "lock_names": lock_names,
                    "owner_id": self.owner_id,
                }),
                Some(RPC_ATTEMPT_TIMEOUT),
            )
            .await?;

        if !resp.success {
            return Err(resp.error.unwrap_or_else(|| "lock-service returned failure".to_string()));
        }

        Ok(resp.result.and_then(|v| v.as_bool()).unwrap_or(false))
    }

    pub async fn release(&self, lock_names: &[String]) -> Result<(), String> {
        let resp = self
            .bunny
            .rpc(
                LOCK_SERVICE,
                json!({
                    "action": "release",
                    "lock_names": lock_names,
                    "owner_id": self.owner_id,
                }),
                Some(RPC_ATTEMPT_TIMEOUT),
            )
            .await?;

        if !resp.success {
            return Err(resp.error.unwrap_or_else(|| "lock-service release failed".to_string()));
        }

        Ok(())
    }

    pub async fn refresh(&self, lock_names: &[String]) -> Result<(), String> {
        let resp = self
            .bunny
            .rpc(
                LOCK_SERVICE,
                json!({
                    "action": "refresh",
                    "lock_names": lock_names,
                    "owner_id": self.owner_id,
                }),
                Some(RPC_ATTEMPT_TIMEOUT),
            )
            .await?;

        if !resp.success {
            return Err(resp.error.unwrap_or_else(|| "lock-service refresh failed".to_string()));
        }

        Ok(())
    }
}

/// RAII guard. Releases all locks on drop.
pub struct LockGuard {
    bunny: Arc<BunnyTalk>,
    owner_id: String,
    lock_names: Vec<String>,
    released: bool,
}

impl LockGuard {
    /// Explicitly release before drop. Errors are logged; drop will skip retry.
    pub async fn release(mut self) -> Result<(), String> {
        self.released = true;
        release_with_retry(&self.bunny, &self.lock_names, &self.owner_id).await
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // Synchronous context — spawn best-effort release. Locks auto-expire server-side
        // via FAILSAFE_TTL (RPC_TIMEOUT_SECONDS + 5s) if this task is dropped abruptly.
        let bunny = Arc::clone(&self.bunny);
        let names = self.lock_names.clone();
        let owner = self.owner_id.clone();
        tokio::spawn(async move {
            if let Err(e) = release_with_retry(&bunny, &names, &owner).await {
                debug!("LockGuard drop: release failed (locks will auto-expire): {}", e);
            }
        });
    }
}

async fn release_with_retry(
    bunny: &Arc<BunnyTalk>,
    lock_names: &[String],
    owner_id: &str,
) -> Result<(), String> {
    let mut last_err = String::new();
    for attempt in 0..RELEASE_MAX_ATTEMPTS {
        let resp = bunny
            .rpc(
                LOCK_SERVICE,
                json!({
                    "action": "release",
                    "lock_names": lock_names,
                    "owner_id": owner_id,
                }),
                Some(RPC_ATTEMPT_TIMEOUT),
            )
            .await;

        match resp {
            Ok(r) if r.success => return Ok(()),
            Ok(r) => {
                // Server-side ownership error — no point retrying
                let msg = r.error.unwrap_or_else(|| "release failed".to_string());
                return Err(msg);
            }
            Err(e) => {
                last_err = e;
                if attempt + 1 < RELEASE_MAX_ATTEMPTS {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }
    Err(last_err)
}
