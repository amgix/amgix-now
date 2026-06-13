use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lapin::options::{
    BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, ExchangeDeclareOptions,
    QueueBindOptions, QueueDeclareOptions,
};
use lapin::types::{AMQPValue, FieldTable, LongString, ShortString};
use lapin::{BasicProperties, Channel, Connection, ConnectionProperties, ExchangeKind};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{oneshot, Mutex};
use tracing::{debug, error, warn};
use uuid::Uuid;

const APP_PREFIX: &str = "amgix";
const RPC_TIMEOUT_SECONDS: u64 = 60;
// RPC_TIMEOUT_SECONDS / 2 + 5
const HEARTBEAT_SECONDS: u16 = 35;
const MAX_RETRIES: u32 = 10;
const RETRY_DELAY_SECONDS: u64 = 5;

// CLASSIC_QUEUE_ARGUMENTS values from constants.py
const MAX_QUEUE_MESSAGES: i64 = 500_000;
const MAX_QUEUE_SIZE_BYTES: i64 = 1 * 1024 * 1024 * 1024; // 1 GB
// max((RPC_TIMEOUT_SECONDS + 5) * 1000, 10000)
const REPLY_QUEUE_EXPIRES_MS: i64 = ((RPC_TIMEOUT_SECONDS + 5) * 1000) as i64;

fn hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())
}

fn classic_queue_args() -> FieldTable {
    let mut args = FieldTable::default();
    args.insert(
        "x-queue-type".into(),
        AMQPValue::LongString("classic".into()),
    );
    args.insert(
        "x-max-length".into(),
        AMQPValue::LongLongInt(MAX_QUEUE_MESSAGES),
    );
    args.insert(
        "x-max-length-bytes".into(),
        AMQPValue::LongLongInt(MAX_QUEUE_SIZE_BYTES),
    );
    args.insert(
        "x-overflow".into(),
        AMQPValue::LongString("reject-publish".into()),
    );
    let expires = REPLY_QUEUE_EXPIRES_MS.max(10_000);
    args.insert("x-expires".into(), AMQPValue::LongLongInt(expires));
    args
}

#[derive(Debug, Deserialize)]
pub struct RpcResponse {
    pub success: bool,
    pub result: Option<Value>,
    pub error: Option<String>,
    pub error_type: Option<String>,
    pub result_type: Option<String>,
}

struct PendingRpc {
    tx: oneshot::Sender<Result<RpcResponse, String>>,
}

pub struct BunnyTalk {
    connection: Connection,
    channel: Channel,
    exchange_name: String,
    reply_queue_name: String,
    pending: Arc<Mutex<HashMap<String, PendingRpc>>>,
}

impl BunnyTalk {
    pub async fn create(amqp_url: &str) -> Result<Arc<Self>, String> {
        let host = hostname();
        let url_with_heartbeat = if amqp_url.contains('?') {
            format!("{}&heartbeat={}", amqp_url, HEARTBEAT_SECONDS)
        } else {
            format!("{}?heartbeat={}", amqp_url, HEARTBEAT_SECONDS)
        };

        let conn_name: LongString = format!("{}-now-{}", APP_PREFIX, host).into();
        let props = ConnectionProperties::default()
            .with_connection_name(conn_name)
            .enable_auto_recover();

        let mut last_err = String::new();
        for attempt in 1..=MAX_RETRIES {
            match Connection::connect(&url_with_heartbeat, props.clone()).await {
                Ok(connection) => {
                    return Self::setup(connection, &host).await;
                }
                Err(e) => {
                    last_err = e.to_string();
                    if attempt < MAX_RETRIES {
                        error!(
                            "Failed to connect to RabbitMQ (attempt {}/{}): {}",
                            attempt, MAX_RETRIES, e
                        );
                        error!("Retrying in {} seconds...", RETRY_DELAY_SECONDS);
                        tokio::time::sleep(Duration::from_secs(RETRY_DELAY_SECONDS)).await;
                    }
                }
            }
        }
        Err(format!(
            "Failed to connect to RabbitMQ after {} attempts: {}",
            MAX_RETRIES, last_err
        ))
    }

    async fn setup(connection: Connection, host: &str) -> Result<Arc<Self>, String> {
        let channel = connection
            .create_channel()
            .await
            .map_err(|e| format!("Failed to create channel: {}", e))?;

        let exchange_name = format!("{}.topic", APP_PREFIX);
        channel
            .exchange_declare(
                exchange_name.as_str().into(),
                ExchangeKind::Topic,
                ExchangeDeclareOptions {
                    durable: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|e| format!("Failed to declare exchange: {}", e))?;

        let reply_queue_name = format!(
            "{}-rpc-{}-{}",
            APP_PREFIX,
            host,
            &Uuid::new_v4().to_string()[..8]
        );

        channel
            .queue_declare(
                reply_queue_name.as_str().into(),
                QueueDeclareOptions {
                    durable: true,
                    exclusive: true,
                    auto_delete: true,
                    ..Default::default()
                },
                classic_queue_args(),
            )
            .await
            .map_err(|e| format!("Failed to declare reply queue: {}", e))?;

        channel
            .queue_bind(
                reply_queue_name.as_str().into(),
                exchange_name.as_str().into(),
                reply_queue_name.as_str().into(),
                QueueBindOptions::default(),
                FieldTable::default(),
            )
            .await
            .map_err(|e| format!("Failed to bind reply queue: {}", e))?;

        let pending: Arc<Mutex<HashMap<String, PendingRpc>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let instance = Arc::new(Self {
            connection,
            channel: channel.clone(),
            exchange_name: exchange_name.clone(),
            reply_queue_name: reply_queue_name.clone(),
            pending: pending.clone(),
        });

        // Start reply consumer loop
        let consumer = channel
            .basic_consume(
                reply_queue_name.as_str().into(),
                format!("{}-reply-consumer", APP_PREFIX).as_str().into(),
                BasicConsumeOptions {
                    no_ack: false,
                    exclusive: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|e| format!("Failed to start reply consumer: {}", e))?;

        let pending_for_delegate = pending.clone();
        consumer.set_delegate(move |delivery: lapin::message::DeliveryResult| {
            let pending = pending_for_delegate.clone();
            async move {
                handle_rpc_reply(delivery, pending).await;
            }
        });

        Ok(instance)
    }

    pub async fn talk(
        &self,
        routing_key: &str,
        kwargs: Value,
        start_trace: bool,
        trace_meta: Option<Value>,
    ) -> Result<(), lapin::Error> {
        let (trace_id, trace_chain) = build_trace(start_trace, routing_key, "talk");
        let effective_meta = trace_meta.unwrap_or(json!({}));

        let body = serde_json::to_vec(&json!({
            "args": [],
            "kwargs": kwargs,
        }))
        .expect("kwargs must be serializable");

        let headers = trace_headers(&trace_id, &trace_chain, &effective_meta);

        self.channel
            .basic_publish(
                self.exchange_name.as_str().into(),
                routing_key.into(),
                BasicPublishOptions::default(),
                &body,
                BasicProperties::default()
                    .with_delivery_mode(2) // persistent
                    .with_headers(headers),
            )
            .await?
            .await?;

        Ok(())
    }

    pub async fn rpc(
        &self,
        routing_key: &str,
        kwargs: Value,
        timeout: Option<Duration>,
    ) -> Result<RpcResponse, String> {
        let timeout = timeout.unwrap_or(Duration::from_secs(RPC_TIMEOUT_SECONDS));
        let correlation_id = Uuid::new_v4().to_string();

        let (trace_id, trace_chain) = build_trace(false, routing_key, "rpc");
        let effective_meta = json!({});

        let body = serde_json::to_vec(&json!({
            "args": [],
            "kwargs": kwargs,
        }))
        .expect("kwargs must be serializable");

        let expiration_ms = format!("{}", (timeout.as_secs() + 5) * 1000);
        let headers = trace_headers(&trace_id, &trace_chain, &effective_meta);

        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .await
            .insert(correlation_id.clone(), PendingRpc { tx });

        let publish_result = self
            .channel
            .basic_publish(
                self.exchange_name.as_str().into(),
                routing_key.into(),
                BasicPublishOptions::default(),
                &body,
                BasicProperties::default()
                    .with_reply_to(self.reply_queue_name.as_str().into())
                    .with_correlation_id(correlation_id.as_str().into())
                    .with_expiration(expiration_ms.as_str().into())
                    .with_headers(headers),
            )
            .await;

        if let Err(e) = publish_result {
            self.pending.lock().await.remove(&correlation_id);
            return Err(format!("RPC publish failed: {}", e));
        }
        if let Err(e) = publish_result.unwrap().await {
            self.pending.lock().await.remove(&correlation_id);
            return Err(format!("RPC publish confirm failed: {}", e));
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.pending.lock().await.remove(&correlation_id);
                Err(format!(
                    "RPC channel closed for routing_key={}",
                    routing_key
                ))
            }
            Err(_) => {
                self.pending.lock().await.remove(&correlation_id);
                Err(format!(
                    "RPC call to '{}' timed out after {} seconds",
                    routing_key,
                    timeout.as_secs()
                ))
            }
        }
    }

    pub async fn close(&self) {
        if !self.channel.status().connected() {
            return;
        }
        if let Err(e) = self.channel.close(200, "normal shutdown".into()).await {
            error!("Error closing BunnyTalk channel: {}", e);
        }
        if let Err(e) = self.connection.close(200, "normal shutdown".into()).await {
            error!("Error closing BunnyTalk connection: {}", e);
        }

        // Fail any pending RPC futures that are still waiting
        let mut pending = self.pending.lock().await;
        for (id, rpc) in pending.drain() {
            debug!("Dropping pending RPC future on close: correlation_id={}", id);
            let _ = rpc.tx.send(Err("BunnyTalk closed".to_string()));
        }
    }
}

async fn handle_rpc_reply(
    delivery_result: lapin::message::DeliveryResult,
    pending: Arc<Mutex<HashMap<String, PendingRpc>>>,
) {
    let delivery = match delivery_result {
        Err(e) => {
            warn!("RPC reply consumer error: {}", e);
            return;
        }
        Ok(None) => {
            // Consumer cancelled by server
            return;
        }
        Ok(Some(d)) => d,
    };

    let correlation_id = match delivery.properties.correlation_id() {
        Some(id) => id.to_string(),
        None => {
            warn!("RPC reply with no correlation_id; dropping");
            if let Err(e) = delivery.ack(BasicAckOptions::default()).await {
                error!("Failed to ack reply with no correlation_id: {}", e);
            }
            return;
        }
    };

    let rpc = pending.lock().await.remove(&correlation_id);
    if rpc.is_none() {
        debug!(
            "RPC reply received with unknown correlation_id={}; dropping",
            correlation_id
        );
        if let Err(e) = delivery.ack(BasicAckOptions::default()).await {
            error!("Failed to ack unknown-correlation reply: {}", e);
        }
        return;
    }
    let rpc = rpc.unwrap();

    let send_result = parse_rpc_reply(&delivery.data);
    if let Err(e) = delivery.ack(BasicAckOptions::default()).await {
        error!("Failed to ack RPC reply: {}", e);
    }
    let _ = rpc.tx.send(send_result);
}

fn parse_rpc_reply(data: &[u8]) -> Result<RpcResponse, String> {
    let payload: Value =
        serde_json::from_slice(data).map_err(|e| format!("Invalid RPC reply JSON: {}", e))?;

    let response_val = payload
        .get("kwargs")
        .and_then(|k| k.get("response"))
        .ok_or_else(|| "RPC reply missing kwargs.response".to_string())?;

    serde_json::from_value(response_val.clone())
        .map_err(|e| format!("Failed to parse RpcResponse: {}", e))
}

fn build_trace(start_trace: bool, routing_key: &str, method: &str) -> (String, Value) {
    let trace_id = Uuid::new_v4().to_string();
    let _ = start_trace; // no app-wide context — always start fresh (see plan: minimal wire headers)
    let call_info = json!({
        "hostname": hostname(),
        "function": format!("{}->{}->{}", APP_PREFIX, method, routing_key),
        "timestamp": Utc::now().to_rfc3339(),
        "call_id": Uuid::new_v4().to_string(),
    });
    let chain = json!([call_info]);
    (trace_id, chain)
}

fn trace_headers(trace_id: &str, trace_chain: &Value, trace_meta: &Value) -> FieldTable {
    let mut headers = FieldTable::default();
    let key_id: ShortString = format!("{}_trace_id", APP_PREFIX).into();
    let key_chain: ShortString = format!("{}_trace_chain", APP_PREFIX).into();
    let key_meta: ShortString = format!("{}_trace_meta", APP_PREFIX).into();
    headers.insert(
        key_id,
        AMQPValue::LongString(trace_id.to_string().into()),
    );
    headers.insert(
        key_chain,
        AMQPValue::LongString(
            serde_json::to_string(trace_chain)
                .unwrap_or_else(|_| "[]".to_string())
                .into(),
        ),
    );
    headers.insert(
        key_meta,
        AMQPValue::LongString(
            serde_json::to_string(trace_meta)
                .unwrap_or_else(|_| "{}".to_string())
                .into(),
        ),
    );
    headers
}
