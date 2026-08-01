use std::collections::HashMap;

use murmurhash3::murmurhash3_x64_128;

use crate::common::MAX_METADATA_VALUE_LENGTH;
use crate::models::{Document, SearchResult, VectorScore};
use crate::qdrant::DbError;

/// Set `document.content_hash` from UTF-8 content (mmh3 128-bit hex).
/// Matches Python `format(mmh3.hash128(content.encode("utf-8"), signed=False), "032x")`.
/// Overwrites any client-supplied value.
pub fn set_content_hash(document: &mut Document) {
    let content = document.content.as_deref().unwrap_or("");
    let (h1, h2) = murmurhash3_x64_128(content.as_bytes(), 0);
    document.content_hash = Some(format!("{h2:016x}{h1:016x}"));
}

// ---------------------------------------------------------------------------
// Multi-retriever fusion — mirrors `amgix-server` `DatabaseBase`
// (`rrf_fuse`, `linear_weighted_score_fuse`).
// ---------------------------------------------------------------------------

/// Reciprocal rank fusion: `weight / (k + rank)` summed per id across arms.
/// `rank` here is **1-based** (`rank_idx + 1` matches Python `enumerate(..., start=1)`).
///
/// Tie-break (does not affect returned score): higher unweighted raw-score sum,
/// then lower fuse-key id. Mirrors `DatabaseBase.rrf_fuse`.
pub fn rrf_fuse(
    id_lists: &[Vec<String>],
    weights: &[f64],
    scored_lists: &[Vec<(String, f64)>],
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
    let mut raw_sums: HashMap<String, f64> = HashMap::new();
    for arm in scored_lists {
        for (item_id, score) in arm {
            *raw_sums.entry(item_id.clone()).or_insert(0.0) += score;
        }
    }
    let mut items: Vec<(String, f64, f64)> = fused
        .into_iter()
        .filter(|(_, s)| score_threshold.map_or(true, |t| *s >= t))
        .map(|(id, rrf)| {
            let raw = raw_sums.get(&id).copied().unwrap_or(0.0);
            (id, rrf, raw)
        })
        .collect();
    items.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.0.cmp(&b.0))
    });
    items.truncate(limit);
    items.into_iter().map(|(id, rrf, _)| (id, rrf)).collect()
}

/// Min-max normalizes each arm's scores, then sums `weight * normalized_score`.
/// If min == max for an arm, normalized score is **1.0** for each id.
/// Equal fused scores are broken by lower fuse-key id (determinism only).
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
    items.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    items.truncate(limit);
    items
}

// ---------------------------------------------------------------------------
// Metadata normalization — mirrors `document.py` `Document.validate_metadata`
// converting primitives + `{value,type}` dicts → flat values for storage.
// Qdrant / API JSON must match Python `Document.model_dump()` shape.
// ---------------------------------------------------------------------------

/// Mirrors Python `validate_metadata`: every entry is stored as a flat JSON value.
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
            if map.len() == 2
                && map.contains_key("value")
                && map.get("type").and_then(|t| t.as_str()).is_some()
            {
                let type_str = map.get("type").and_then(|t| t.as_str()).unwrap();
                let allowed = ["string", "integer", "float", "boolean", "datetime", "array", "object"];
                if !allowed.contains(&type_str) {
                    return Err(format!(
                        "Invalid metadata type '{type_str}' for key '{key}'. Allowed types: {allowed:?}"
                    ));
                }
                let val = map.get("value").unwrap();
                return validate_and_clone_meta_inner(key, type_str, val);
            }
            Ok(serde_json::Value::Object(map))
        }
        serde_json::Value::String(s) => {
            if s.chars().count() > MAX_METADATA_VALUE_LENGTH {
                return Err(format!(
                    "String metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit"
                ));
            }
            Ok(serde_json::Value::String(s))
        }
        serde_json::Value::Bool(b) => Ok(serde_json::Value::Bool(b)),
        serde_json::Value::Number(n) => Ok(serde_json::Value::Number(n)),
        serde_json::Value::Array(arr) => Ok(serde_json::Value::Array(arr)),
        serde_json::Value::Null => Ok(serde_json::Value::Null),
        other => Err(format!(
            "Metadata value for key '{key}' must be string, int, float, bool, array, null, or MetaValue (required for datetime and object), got {}",
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
            Some(s) if s.chars().count() > MAX_METADATA_VALUE_LENGTH => Err(format!(
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
                if crate::datetime_parse::is_valid_datetime_string(s) {
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
        "array" => {
            if val.is_array() {
                Ok(val.clone())
            } else {
                Err(format!(
                    "Metadata value for key '{key}' must be array for type='array', got {}",
                    json_type_name(val)
                ))
            }
        }
        "object" => {
            if val.is_object() || val.is_null() {
                Ok(val.clone())
            } else {
                Err(format!(
                    "Metadata value for key '{key}' must be object or null for type='object', got {}",
                    json_type_name(val)
                ))
            }
        }
        _ => unreachable!(),
    }
}

pub fn doc_to_payload(
    doc: &Document,
    store_content: bool,
) -> Result<serde_json::Map<String, serde_json::Value>, DbError> {
    let mut val = serde_json::to_value(doc)
        .map_err(|e| DbError::Config(format!("Serialization error: {e}")))?;
    if let serde_json::Value::Object(ref mut map) = val {
        map.remove("vectors");
        map.remove("token_lengths");
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

#[cfg(test)]
mod rrf_fuse_tests {
    use super::rrf_fuse;

    #[test]
    fn tie_breaks_on_raw_sum_then_id() {
        // Same RRF: both appear at rank 1 in one arm and rank 2 in the other (k=2).
        // weight/(2+1) + weight/(2+2) = 1/3 + 1/4 for both.
        let id_lists = vec![
            vec!["b".to_string(), "a".to_string()],
            vec!["a".to_string(), "b".to_string()],
        ];
        let weights = vec![1.0, 1.0];
        let scored_lists = vec![
            vec![("b".to_string(), 0.9), ("a".to_string(), 0.1)],
            vec![("a".to_string(), 0.5), ("b".to_string(), 0.4)],
        ];
        // raw sums: a=0.6, b=1.3 → b wins on raw sum
        let out = rrf_fuse(&id_lists, &weights, &scored_lists, 10, None, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0].1 - out[1].1).abs() < 1e-12);
        assert_eq!(out[0].0, "b");
        assert_eq!(out[1].0, "a");

        // Equal RRF and equal raw sums → lower id wins
        let scored_equal = vec![
            vec![("b".to_string(), 0.5), ("a".to_string(), 0.5)],
            vec![("a".to_string(), 0.5), ("b".to_string(), 0.5)],
        ];
        let out2 = rrf_fuse(&id_lists, &weights, &scored_equal, 10, None, 2);
        assert_eq!(out2[0].0, "a");
        assert_eq!(out2[1].0, "b");
    }
}

#[cfg(test)]
mod content_hash_tests {
    use super::set_content_hash;
    use crate::models::Document;
    use chrono::Utc;

    fn doc_with_content(content: Option<&str>) -> Document {
        Document {
            id: "t".into(),
            timestamp: Utc::now(),
            tags: None,
            name: None,
            description: None,
            content: content.map(str::to_owned),
            content_hash: None,
            metadata: None,
            custom_vectors: None,
            joined: None,
            vectors: None,
            token_lengths: Default::default(),
        }
    }

    /// Values from Python: format(mmh3.hash128(s.encode("utf-8"), signed=False), "032x")
    #[test]
    fn matches_python_mmh3_hash128() {
        let cases = [
            ("", "00000000000000000000000000000000"),
            ("hello world", "ab97467d60eb63b1533f6046eb7f610e"),
            ("hello world!", "edf56b1420cea7e75aa80377fe21bbe3"),
            ("café", "0acaaa4789576479a2e7c22a053364dd"),
        ];
        for (content, expected) in cases {
            let mut d = doc_with_content(Some(content));
            set_content_hash(&mut d);
            assert_eq!(d.content_hash.as_deref(), Some(expected), "content={content:?}");
        }
        let mut none_doc = doc_with_content(None);
        let mut empty_doc = doc_with_content(Some(""));
        set_content_hash(&mut none_doc);
        set_content_hash(&mut empty_doc);
        assert_eq!(none_doc.content_hash, empty_doc.content_hash);
    }
}
