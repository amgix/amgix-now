use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::common::SearchExcludeField;
use crate::encoder::{get_collection_info_cached, validate_metadata_filter, CollectionConfigCache, SearchError};
use crate::join_parser::{JoinField, JoinRefKind, JoinSideRef, JoinSpec};
use crate::models::{CollectionConfigInternal, Document, MetadataFilter, SearchResult};
use crate::qdrant::{DbError, QdrantDb};

/// Parse a join expression, wrapping syntax errors as `SearchError::InvalidFilter`.
pub fn parse_join_field_validated(join: &JoinField) -> Result<Vec<JoinSpec>, SearchError> {
    join.clone().into_specs().map_err(SearchError::InvalidFilter)
}

/// Fields that must be fetched for the parent document regardless of `exclude`,
/// because a join needs them to compute the parent-side join value.
///
/// Currently the only such dependency is a parent ref that reads a value off the
/// parent document's metadata (e.g. `collection[$.meta.foo=$$id]`).
pub fn required_fields_for_joins(specs: &[JoinSpec]) -> HashSet<SearchExcludeField> {
    let mut required = HashSet::new();
    if specs
        .iter()
        .any(|spec| matches!(spec.parent_ref.kind, JoinRefKind::Meta(_)))
    {
        required.insert(SearchExcludeField::Metadata);
    }
    required
}

trait JoinParent {
    fn join_id(&self) -> &str;
    fn join_metadata(&self) -> Option<&HashMap<String, Value>>;
    fn joined_mut(&mut self) -> &mut Option<HashMap<String, Vec<Document>>>;
}

impl JoinParent for Document {
    fn join_id(&self) -> &str {
        &self.id
    }

    fn join_metadata(&self) -> Option<&HashMap<String, Value>> {
        self.metadata.as_ref()
    }

    fn joined_mut(&mut self) -> &mut Option<HashMap<String, Vec<Document>>> {
        &mut self.joined
    }
}

impl JoinParent for SearchResult {
    fn join_id(&self) -> &str {
        &self.document.id
    }

    fn join_metadata(&self) -> Option<&HashMap<String, Value>> {
        self.document.metadata.as_ref()
    }

    fn joined_mut(&mut self) -> &mut Option<HashMap<String, Vec<Document>>> {
        &mut self.document.joined
    }
}

pub async fn enrich_documents_with_joins<P: JoinParent>(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    documents: &mut [P],
    join: &JoinField,
    limit: u32,
) -> Result<(), SearchError> {
    let specs = parse_join_field_validated(join)?;
    enrich_documents_with_parsed_joins(db, cache, documents, &specs, limit).await
}

/// Same as [`enrich_documents_with_joins`], but takes already-parsed `JoinSpec`s so
/// callers that also need the specs (e.g. to compute `required_fields_for_joins`)
/// don't have to parse the join expression twice.
pub async fn enrich_documents_with_parsed_joins<P: JoinParent>(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    documents: &mut [P],
    specs: &[JoinSpec],
    limit: u32,
) -> Result<(), SearchError> {
    for spec in specs {
        enrich_with_spec(db, cache, documents, spec, limit).await?;
    }
    Ok(())
}

async fn enrich_with_spec<P: JoinParent>(
    db: &QdrantDb,
    cache: &CollectionConfigCache,
    documents: &mut [P],
    spec: &JoinSpec,
    limit: u32,
) -> Result<(), SearchError> {
    let real_child = crate::common::get_real_collection_name(&spec.collection_name);
    let child_config = match get_collection_info_cached(db, cache, &real_child).await {
        Ok((cfg, _)) => cfg,
        Err(DbError::NotFound(_)) => {
            return Err(SearchError::InvalidFilter(format!(
                "Join collection '{}' not found",
                spec.collection_name
            )));
        }
        Err(e) => return Err(SearchError::Db(e)),
    };

    validate_join_spec(spec, &child_config)?;

    let mut join_values: Vec<Value> = Vec::new();
    let mut seen: HashMap<String, ()> = HashMap::new();
    for document in documents.iter() {
        let pv = parent_join_value(document, &spec.parent_ref);
        if pv.is_none() || pv.as_ref().is_some_and(|v| v.is_null()) {
            continue;
        }
        let pv = pv.unwrap();
        let key = join_value_key(&pv);
        if seen.insert(key, ()).is_none() {
            join_values.push(pv);
        }
    }

    let max_documents = limit.saturating_mul(documents.len() as u32) as usize;
    let children = fetch_children_for_join(
        db,
        spec,
        &join_values,
        &child_config,
        &real_child,
        max_documents,
    )
    .await?;

    let by_key = group_children_by_join_key(&children, &spec.child_ref);
    for document in documents.iter_mut() {
        let pv = parent_join_value(document, &spec.parent_ref);
        let children_for_parent = match pv {
            None => vec![],
            Some(v) if v.is_null() => vec![],
            Some(v) => by_key.get(&join_value_key(&v)).cloned().unwrap_or_default(),
        };
        if document.joined_mut().is_none() {
            *document.joined_mut() = Some(HashMap::new());
        }
        document
            .joined_mut()
            .as_mut()
            .unwrap()
            .insert(spec.collection_name.clone(), children_for_parent);
    }

    Ok(())
}

fn validate_join_spec(
    spec: &JoinSpec,
    child_config: &CollectionConfigInternal,
) -> Result<(), SearchError> {
    if let JoinRefKind::Meta(ref key) = spec.child_ref.kind {
        let indexes = child_config.metadata_indexes.as_deref().unwrap_or(&[]);
        if indexes.is_empty() {
            return Err(SearchError::InvalidFilter(format!(
                "Join child collection '{}' has no metadata_indexes; cannot join on metadata key '{key}'",
                spec.collection_name
            )));
        }
        if !indexes.iter().any(|idx| idx.key == *key) {
            return Err(SearchError::InvalidFilter(format!(
                "Join child metadata key '{key}' is not indexed in collection '{}'",
                spec.collection_name
            )));
        }
    }
    if let Some(ref filter) = spec.metadata_filter {
        validate_metadata_filter(child_config, filter)
            .map_err(|e| SearchError::InvalidFilter(e.0))?;
    }
    Ok(())
}

async fn fetch_children_for_join(
    db: &QdrantDb,
    spec: &JoinSpec,
    join_values: &[Value],
    child_config: &CollectionConfigInternal,
    real_child: &str,
    max_documents: usize,
) -> Result<Vec<Document>, SearchError> {
    if join_values.is_empty() || max_documents == 0 {
        return Ok(vec![]);
    }

    if matches!(spec.child_ref.kind, JoinRefKind::Id) {
        let owned_ids: Vec<String> = join_values
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                _ => v.to_string(),
            })
            .collect();
        let id_refs: Vec<&str> = owned_ids.iter().map(String::as_str).collect();
        let fetched = db
            .get_documents(real_child, &id_refs, true, false, None)
            .await
            .map_err(SearchError::Db)?;

        let mut docs = Vec::new();
        for doc in fetched.into_iter().flatten() {
            if let Some(ref filter) = spec.metadata_filter {
                if !document_matches_metadata_filter(&doc, filter) {
                    continue;
                }
            }
            docs.push(doc);
        }
        return Ok(docs);
    }

    let JoinRefKind::Meta(ref key) = spec.child_ref.kind else {
        return Ok(vec![]);
    };

    db.fetch_documents_by_metadata_values(
        real_child,
        key,
        join_values,
        spec.metadata_filter.as_ref(),
        child_config,
        max_documents,
    )
    .await
    .map_err(SearchError::Db)
}

fn group_children_by_join_key(
    children: &[Document],
    child_ref: &JoinSideRef,
) -> HashMap<String, Vec<Document>> {
    let mut groups: HashMap<String, Vec<Document>> = HashMap::new();
    for doc in children {
        let jv = child_join_value(doc, child_ref);
        if jv.is_none() || jv.as_ref().is_some_and(|v| v.is_null()) {
            continue;
        }
        let key = join_value_key(&jv.unwrap());
        groups.entry(key).or_default().push(doc.clone());
    }
    groups
}

fn parent_join_value<P: JoinParent>(document: &P, side: &JoinSideRef) -> Option<Value> {
    match &side.kind {
        JoinRefKind::Id => Some(Value::String(document.join_id().to_string())),
        JoinRefKind::Meta(key) => document
            .join_metadata()
            .and_then(|m| m.get(key).cloned()),
    }
}

fn child_join_value(doc: &Document, side: &JoinSideRef) -> Option<Value> {
    match &side.kind {
        JoinRefKind::Id => Some(Value::String(doc.id.clone())),
        JoinRefKind::Meta(key) => doc.metadata.as_ref().and_then(|m| m.get(key).cloned()),
    }
}

fn join_value_key(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

fn values_equal(a: &Value, b: &Value) -> bool {
    if a.is_null() || b.is_null() {
        return a.is_null() && b.is_null();
    }
    a == b
}

pub fn document_matches_metadata_filter(document: &Document, metadata_filter: &MetadataFilter) -> bool {
    let metadata = document.metadata.as_ref();
    eval_filter_node(metadata_filter, metadata)
}

fn eval_filter_node(node: &MetadataFilter, metadata: Option<&HashMap<String, Value>>) -> bool {
    if let Some(ref key) = node.key {
        let value = metadata.and_then(|m| m.get(key));
        let cmp = node.value.as_ref();
        let op = node.op.as_deref().unwrap_or("");
        return match op {
            "eq" => values_equal(value.unwrap_or(&Value::Null), cmp.unwrap_or(&Value::Null)),
            "neq" => !values_equal(value.unwrap_or(&Value::Null), cmp.unwrap_or(&Value::Null)),
            "gt" | "gte" | "lt" | "lte" => {
                let Some(v) = value else { return false };
                let Some(c) = cmp else { return false };
                compare_ordered(v, c, op)
            }
            _ => false,
        };
    }
    if let Some(ref not_) = node.not_ {
        return !eval_filter_node(not_, metadata);
    }
    if let Some(ref and_) = node.and_ {
        return and_.iter().all(|c| eval_filter_node(c, metadata));
    }
    if let Some(ref or_) = node.or_ {
        return or_.iter().any(|c| eval_filter_node(c, metadata));
    }
    true
}

fn compare_ordered(value: &Value, cmp: &Value, op: &str) -> bool {
    use std::cmp::Ordering;
    let ord = match (value, cmp) {
        (Value::Number(a), Value::Number(b)) => a
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&b.as_f64().unwrap_or(0.0))
            .unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => return false,
    };
    match op {
        "gt" => ord == Ordering::Greater,
        "gte" => ord != Ordering::Less,
        "lt" => ord == Ordering::Less,
        "lte" => ord != Ordering::Greater,
        _ => false,
    }
}
