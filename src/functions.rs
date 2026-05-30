use std::collections::HashMap;

use crate::common::MAX_METADATA_VALUE_LENGTH;
use crate::models::{Document, DocumentWithVectors, SearchResult, VectorScore};
use crate::qdrant::DbError;

// ---------------------------------------------------------------------------
// Multi-retriever fusion — mirrors `amgix-server` `DatabaseBase`
// (`rrf_fuse`, `linear_weighted_score_fuse`).
// ---------------------------------------------------------------------------

/// Reciprocal rank fusion: `weight / (k + rank)` summed per id across arms.
/// `rank` here is **1-based** (`rank_idx + 1` matches Python `enumerate(..., start=1)`).
pub fn rrf_fuse(
    id_lists: &[Vec<String>],
    weights: &[f64],
    limit: usize,
    score_threshold: Option<f64>,
    k: usize,
) -> Vec<(String, f64)> {
    let mut fused: HashMap<String, f64> = HashMap::new();
    for (list_idx, ids) in id_lists.iter().enumerate() {
        let weight = weights[list_idx];
        for (rank_idx, item_id) in ids.iter().enumerate() {
            *fused.entry(item_id.clone()).or_insert(0.0) += weight / (k + rank_idx + 1) as f64;
        }
    }
    let mut items: Vec<(String, f64)> = fused
        .into_iter()
        .filter(|(_, s)| score_threshold.map_or(true, |t| *s >= t))
        .collect();
    items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    items.truncate(limit);
    items
}

/// Min-max normalizes each arm's scores, then sums `weight * normalized_score`.
/// If min == max for an arm, normalized score is **1.0** for each id.
pub fn linear_weighted_score_fuse(
    scored_lists: &[Vec<(String, f64)>],
    weights: &[f64],
    limit: usize,
    score_threshold: Option<f64>,
) -> Vec<(String, f64)> {
    let mut candidates: HashMap<String, f64> = HashMap::new();
    for (list_idx, arm) in scored_lists.iter().enumerate() {
        if arm.is_empty() {
            continue;
        }
        let weight = weights[list_idx];
        let mn = arm.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        let mx = arm.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);

        let norm_map: HashMap<&str, f64> = if mx == mn {
            arm.iter().map(|(id, _)| (id.as_str(), 1.0)).collect()
        } else {
            let scale = mx - mn;
            arm.iter().map(|(id, s)| (id.as_str(), (s - mn) / scale)).collect()
        };

        for (item_id, nscore) in norm_map {
            *candidates.entry(item_id.to_string()).or_insert(0.0) += weight * nscore;
        }
    }
    let mut items: Vec<(String, f64)> = candidates
        .into_iter()
        .filter(|(_, s)| score_threshold.map_or(true, |t| *s >= t))
        .collect();
    items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    items.truncate(limit);
    items
}

// ---------------------------------------------------------------------------
// Metadata normalization — mirrors `document.py` `Document.validate_metadata`
// converting primitives + `{value,type}` dicts → canonical MetaValue maps for payload.
// Qdrant / API JSON must match Python `Document.model_dump()` shape.
// ---------------------------------------------------------------------------

/// Mirrors Python `validate_metadata`: every entry becomes `{ "value": …, "type": "…" }`.
///
/// Caller must ensure API validation (`validation::validate_document`) already passed.
pub fn normalize_document_metadata_inplace(doc: &mut Document) -> Result<(), String> {
    let Some(ref mut md) = doc.metadata else {
        return Ok(());
    };

    let mut out: HashMap<String, serde_json::Value> = HashMap::with_capacity(md.len());
    for (key, value) in md.drain() {
        let normalized = normalize_one_metadata_value(&key, value)?;
        out.insert(key, normalized);
    }
    *md = out;
    Ok(())
}

fn normalize_one_metadata_value(key: &str, value: serde_json::Value) -> Result<serde_json::Value, String> {
    match value {
        serde_json::Value::Object(map) => {
            let val = map.get("value").ok_or_else(|| {
                format!(
                    "Metadata value for key '{key}' is a dict but missing 'value' or 'type' fields. For datetime, use {{\"value\": \"...\", \"type\": \"datetime\"}}"
                )
            })?;
            let type_str = map
                .get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| {
                    format!(
                        "Metadata value for key '{key}' is a dict but missing 'value' or 'type' fields. For datetime, use {{\"value\": \"...\", \"type\": \"datetime\"}}"
                    )
                })?;
            let allowed = ["string", "integer", "float", "boolean", "datetime"];
            if !allowed.contains(&type_str) {
                return Err(format!(
                    "Invalid metadata type '{type_str}' for key '{key}'. Allowed types: {allowed:?}"
                ));
            }
            let inner = validate_and_clone_meta_inner(key, type_str, val)?;
            Ok(serde_json::json!({
                "value": inner,
                "type": type_str
            }))
        }
        serde_json::Value::String(s) => {
            if s.len() > MAX_METADATA_VALUE_LENGTH {
                return Err(format!(
                    "String metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
                ));
            }
            Ok(serde_json::json!({
                "value": s,
                "type": "string"
            }))
        }
        serde_json::Value::Bool(b) => Ok(serde_json::json!({
            "value": b,
            "type": "boolean"
        })),
        serde_json::Value::Number(n) => {
            if n.as_i64().is_some() || n.as_u64().is_some() {
                Ok(serde_json::json!({
                    "value": n,
                    "type": "integer"
                }))
            } else {
                Ok(serde_json::json!({
                    "value": n,
                    "type": "float"
                }))
            }
        }
        other => Err(format!(
            "Metadata value for key '{key}' must be string, int, float, bool, or MetaValue (required for datetime), got {}",
            json_type_name(&other)
        )),
    }
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "list",
        serde_json::Value::Object(_) => "dict",
    }
}

fn validate_and_clone_meta_inner(
    key: &str,
    type_str: &str,
    val: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match type_str {
        "string" => match val.as_str() {
            Some(s) if s.len() > MAX_METADATA_VALUE_LENGTH => Err(format!(
                "String metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
            )),
            Some(s) => Ok(serde_json::Value::String(s.to_string())),
            None => Err(format!(
                "Metadata value for key '{key}' must be string for type='string', got {}",
                json_type_name(val)
            )),
        },
        "integer" => {
            if val.is_i64() || val.is_u64() {
                Ok(val.clone())
            } else {
                Err(format!(
                    "Metadata value for key '{key}' must be integer for type='integer', got {}",
                    json_type_name(val)
                ))
            }
        }
        "float" => {
            if val.is_number() {
                Ok(val.clone())
            } else {
                Err(format!(
                    "Metadata value for key '{key}' must be number for type='float', got {}",
                    json_type_name(val)
                ))
            }
        }
        "boolean" => {
            if val.is_boolean() {
                Ok(val.clone())
            } else {
                Err(format!(
                    "Metadata value for key '{key}' must be boolean for type='boolean', got {}",
                    json_type_name(val)
                ))
            }
        }
        "datetime" => match val.as_str() {
            Some(s) => {
                let normalized = s.replace('Z', "+00:00");
                if chrono::DateTime::parse_from_rfc3339(&normalized).is_ok() {
                    Ok(serde_json::Value::String(s.to_string()))
                } else {
                    Err(format!(
                        "Metadata value for key '{key}' must be a valid ISO 8601 datetime string, got '{s}'"
                    ))
                }
            }
            None => Err(format!(
                "Metadata value for key '{key}' must be string (ISO 8601) for type='datetime', got {}",
                json_type_name(val)
            )),
        },
        _ => unreachable!(),
    }
}

pub fn doc_to_payload(
    doc: &DocumentWithVectors,
    store_content: bool,
) -> Result<serde_json::Map<String, serde_json::Value>, DbError> {
    let mut val = serde_json::to_value(doc)
        .map_err(|e| DbError::Config(format!("Serialization error: {e}")))?;
    if let serde_json::Value::Object(ref mut map) = val {
        map.remove("vectors");
        map.remove("custom_vectors");
        if !store_content {
            map.remove("content");
        }
    }
    match val {
        serde_json::Value::Object(m) => Ok(m),
        _ => Err(DbError::Config("Expected object when serializing document".into())),
    }
}

pub fn doc_payload_only(
    doc: &crate::models::Document,
    store_content: bool,
) -> Result<serde_json::Map<String, serde_json::Value>, DbError> {
    let mut val = serde_json::to_value(doc)
        .map_err(|e| DbError::Config(format!("Serialization error: {e}")))?;
    if let serde_json::Value::Object(ref mut map) = val {
        map.remove("custom_vectors");
        if !store_content {
            map.remove("content");
        }
    }
    match val {
        serde_json::Value::Object(m) => Ok(m),
        _ => Err(DbError::Config("Expected object when serializing document".into())),
    }
}

pub fn search_result_from_point(
    point: &qdrant_client::qdrant::ScoredPoint,
    score: f64,
    vector_scores: Vec<VectorScore>,
) -> Result<SearchResult, DbError> {
    let mut map = serde_json::Map::new();
    for (k, v) in &point.payload {
        map.insert(k.clone(), qdrant_val_to_json(v));
    }
    map.insert("score".to_string(), serde_json::json!(score));
    map.insert(
        "vector_scores".to_string(),
        serde_json::to_value(&vector_scores)
            .map_err(|e| DbError::Config(format!("Serialization error: {e}")))?,
    );
    serde_json::from_value(serde_json::Value::Object(map))
        .map_err(|e| DbError::Config(format!("SearchResult deserialization error: {e}")))
}

pub fn qdrant_val_to_json(v: &qdrant_client::qdrant::Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;
    match &v.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::IntegerValue(i)) => serde_json::json!(*i),
        Some(Kind::DoubleValue(d)) => serde_json::json!(*d),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(l)) => {
            serde_json::Value::Array(l.values.iter().map(qdrant_val_to_json).collect())
        }
        Some(Kind::StructValue(s)) => serde_json::Value::Object(
            s.fields.iter().map(|(k, v)| (k.clone(), qdrant_val_to_json(v))).collect(),
        ),
    }
}

pub fn scored_point_id(point: &qdrant_client::qdrant::ScoredPoint) -> String {
    match &point.id {
        Some(pid) => match &pid.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u.clone(),
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n.to_string(),
            None => String::new(),
        },
        None => String::new(),
    }
}

/// Split `"field_vectorname"` on the first `_`: → `("field", "vectorname")`.
pub fn split_first_underscore(s: &str) -> (&str, &str) {
    match s.find('_') {
        Some(pos) => (&s[..pos], &s[pos + 1..]),
        None => (s, ""),
    }
}
