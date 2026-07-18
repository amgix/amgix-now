//! Faceting helpers for search. Mirrors `amgix-server/src/core/database/search_facet.py`.
//!
//! Facet counts are computed in Rust over the Qdrant candidate pool
//! (`point_lookup`) — the same pool the Python Qdrant path aggregates over.

use std::collections::{BTreeMap, HashMap, HashSet};

use qdrant_client::qdrant::value::Kind;

use crate::common::SearchExcludeField;

/// Metadata payload must be fetched whenever faceting is enabled so the
/// candidate pool carries the values we count over.
pub fn required_fields_for_facets(facets_enabled: bool) -> HashSet<SearchExcludeField> {
    let mut required = HashSet::new();
    if facets_enabled {
        required.insert(SearchExcludeField::Metadata);
    }
    required
}

/// Canonical string key for a facet value, normalized by declared index type so
/// the Rust (Qdrant) and Python (Qdrant/SQL) backends agree. Returns `None` for
/// null/missing/mismatched kinds (the value is skipped).
pub fn facet_value_key(kind: &Kind, idx_type: &str) -> Option<String> {
    match idx_type {
        "integer" => match kind {
            Kind::IntegerValue(i) => Some(i.to_string()),
            Kind::DoubleValue(d) => Some((*d as i64).to_string()),
            _ => None,
        },
        "float" => match kind {
            Kind::DoubleValue(d) => Some(format!("{:?}", d)),
            Kind::IntegerValue(i) => Some(format!("{:?}", *i as f64)),
            _ => None,
        },
        "boolean" => match kind {
            Kind::BoolValue(b) => Some(b.to_string()),
            _ => None,
        },
        // Datetimes are stored as ISO-8601 strings in Qdrant payloads.
        "datetime" => match kind {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        },
        _ => match kind {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        },
    }
}

/// Compute per-field value counts over a candidate pool of scored points.
///
/// `indexed_fields` is a list of `(key, value_type)` pairs taken from the
/// collection's `metadata_indexes`. Each field is truncated to its top
/// `max_values` values by descending count (ties broken by value ascending).
pub fn compute_facet_counts<'a>(
    points: impl Iterator<Item = &'a qdrant_client::qdrant::ScoredPoint>,
    indexed_fields: &[(String, String)],
    max_values: usize,
) -> BTreeMap<String, BTreeMap<String, u64>> {
    let mut per_field: Vec<(String, String, HashMap<String, u64>)> = indexed_fields
        .iter()
        .map(|(k, t)| (k.clone(), t.clone(), HashMap::new()))
        .collect();

    for point in points {
        let Some(Kind::StructValue(meta)) = point.payload.get("metadata").and_then(|v| v.kind.as_ref()) else {
            continue;
        };
        for (key, vtype, counts) in per_field.iter_mut() {
            if let Some(field_val) = meta.fields.get(key) {
                if let Some(k) = field_val.kind.as_ref() {
                    if let Some(skey) = facet_value_key(k, vtype) {
                        *counts.entry(skey).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    let mut out = BTreeMap::new();
    for (key, _vtype, counts) in per_field {
        let mut entries: Vec<(String, u64)> = counts.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let capped: BTreeMap<String, u64> = entries.into_iter().take(max_values).collect();
        out.insert(key, capped);
    }
    out
}
