//! Grouping support for search: capping results per metadata field value, with
//! helpers for building the refetch filter used to exclude already-saturated
//! group values on subsequent rounds. Mirrors
//! `amgix-server/src/core/database/search_group.py`.

use std::collections::{HashMap, HashSet};

use crate::common::SearchExcludeField;
use crate::models::{CollectionConfigInternal, MetadataFilter};

/// Fields that must be fetched regardless of `exclude`, because grouping needs
/// the document's metadata to compute its group value.
pub fn required_fields_for_group(group_field: &Option<String>) -> HashSet<SearchExcludeField> {
    let mut required = HashSet::new();
    if group_field.is_some() {
        required.insert(SearchExcludeField::Metadata);
    }
    required
}

/// Validate that `group_field` is a key declared in collection `metadata_indexes`.
pub fn validate_group_field(
    collection_config: &CollectionConfigInternal,
    group_field: &str,
) -> Result<(), String> {
    let indexed = collection_config.metadata_indexes.as_deref().unwrap_or(&[]);
    if !indexed.iter().any(|idx| idx.key == group_field) {
        return Err(format!(
            "group_field '{group_field}' is not indexed in collection metadata_indexes"
        ));
    }
    Ok(())
}

/// Walk `fused_results` in rank order, keeping at most `group_max` items per
/// distinct group value (documents missing the group field share a single
/// `None` group value), until `limit` items are selected.
///
/// Returns `(selected, saturated_values, null_saturated, pool_exhausted)`:
/// - `selected`: capped, rank-ordered results, at most `limit` items.
/// - `saturated_values`: non-null group values that hit `group_max` among selected.
/// - `null_saturated`: whether the null (missing group field) bucket hit `group_max`.
/// - `pool_exhausted`: whether the whole `fused_results` list was scanned without
///   reaching `limit` selected items, meaning a refetch with an even tighter
///   filter cannot add anything a fresh fetch wouldn't also miss.
pub fn apply_group_cap(
    fused_results: &[(String, f64)],
    group_value_fn: impl Fn(&str) -> Option<serde_json::Value>,
    group_max: usize,
    limit: usize,
) -> (Vec<(String, f64)>, HashSet<serde_json::Value>, bool, bool) {
    let mut selected: Vec<(String, f64)> = Vec::new();
    let mut group_counts: HashMap<Option<serde_json::Value>, usize> = HashMap::new();

    for (item_id, score) in fused_results {
        if selected.len() >= limit {
            break;
        }
        let group_value = group_value_fn(item_id);
        let count = group_counts.get(&group_value).copied().unwrap_or(0);
        if count >= group_max {
            continue;
        }
        group_counts.insert(group_value, count + 1);
        selected.push((item_id.clone(), *score));
    }

    let pool_exhausted = selected.len() < limit;

    let saturated_values: HashSet<serde_json::Value> = group_counts
        .iter()
        .filter(|(value, count)| value.is_some() && **count >= group_max)
        .map(|(value, _)| value.clone().unwrap())
        .collect();
    let null_saturated = group_counts.get(&None).copied().unwrap_or(0) >= group_max;

    (selected, saturated_values, null_saturated, pool_exhausted)
}

/// AND the existing metadata filter with conditions excluding already-saturated
/// `group_field` values, so the next fetch round surfaces fresh candidates.
pub fn build_group_exclusion_filter(
    existing_filter: Option<MetadataFilter>,
    group_field: &str,
    saturated_values: &HashSet<serde_json::Value>,
    null_saturated: bool,
) -> Option<MetadataFilter> {
    let mut exclusions: Vec<MetadataFilter> = saturated_values
        .iter()
        .map(|value| MetadataFilter {
            not_: Some(Box::new(MetadataFilter {
                key: Some(group_field.to_string()),
                op: Some("eq".to_string()),
                value: Some(value.clone()),
                ..Default::default()
            })),
            ..Default::default()
        })
        .collect();

    if null_saturated {
        exclusions.push(MetadataFilter {
            not_: Some(Box::new(MetadataFilter {
                key: Some(group_field.to_string()),
                op: Some("is_null".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        });
    }

    if exclusions.is_empty() {
        return existing_filter;
    }
    if let Some(existing) = existing_filter {
        exclusions.push(existing);
    }
    if exclusions.len() == 1 {
        return Some(exclusions.remove(0));
    }
    Some(MetadataFilter { and_: Some(exclusions), ..Default::default() })
}
