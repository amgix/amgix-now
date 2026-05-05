use std::collections::HashMap;

use crate::models::VectorConfigInternal;

// TOKEN_HASH_RANGE mirrors VectorBase.TOKEN_HASH_RANGE = 2^31 - 1 (Mersenne prime)
pub const TOKEN_HASH_RANGE: i32 = 2147483647;

/// Normalize text: lowercase and collapse multiple spaces to one.
/// Mirrors VectorBase.preprocess_text (keep_case=False path).
pub fn preprocess_text(text: &str) -> String {
    let lowered = text.to_lowercase();
    let trimmed = lowered.trim();
    // collapse runs of whitespace to a single space
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}

/// Mirrors VectorBase.preprocess_text with keep_case=True.
pub fn preprocess_text_keep_case(text: &str) -> String {
    let trimmed = text.trim();
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}

/// Mirrors VectorBase.get_token — MurmurHash3 signed, mapped to [0, TOKEN_HASH_RANGE).
pub fn get_token(feature: &str) -> u32 {
    let hash = murmurhash3::murmurhash3_x86_32(feature.as_bytes(), 0) as i32;
    (hash % TOKEN_HASH_RANGE).unsigned_abs()
}

/// Mirrors VectorBase._validate_text.
pub fn validate_text(text: &str) -> bool {
    !text.is_empty()
}

/// Mirrors VectorBase.get_count_weights — log(1 + count) weighted token ids.
pub fn get_count_weights(tokens: &[String], base_weight: f32) -> Vec<(u32, f32)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        *counts.entry(t.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(token, count)| {
            let token_id = get_token(token);
            let weight = base_weight * (1.0 + count as f32).ln();
            (token_id, weight)
        })
        .collect()
}

/// Mirrors VectorBase.top_k — sort descending by weight, take top k, return (indices, values).
pub fn top_k(mut token_weights: Vec<(u32, f32)>, k: usize) -> (Vec<u32>, Vec<f32>) {
    token_weights.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    token_weights.truncate(k);
    let indices = token_weights.iter().map(|(i, _)| *i).collect();
    let values = token_weights.iter().map(|(_, v)| *v).collect();
    (indices, values)
}

/// Mirrors VectorBase.dedup_sparse — sum weights for duplicate indices.
pub fn dedup_sparse(indices: Vec<u32>, values: Vec<f32>) -> (Vec<u32>, Vec<f32>) {
    let mut map: HashMap<u32, f32> = HashMap::new();
    for (idx, val) in indices.into_iter().zip(values) {
        *map.entry(idx).or_insert(0.0) += val;
    }
    let indices = map.keys().copied().collect();
    let values = map.values().copied().collect();
    (indices, values)
}

/// Mirrors VectorBase.l2_norm — L2-normalize a vector; returns original if norm is zero.
pub fn l2_norm(vector: Vec<f32>) -> Vec<f32> {
    if vector.is_empty() {
        return vector;
    }
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector;
    }
    vector.into_iter().map(|x| x / norm).collect()
}

/// Mirrors VectorBase.get_language_code — detect or fall back to default.
/// Returns Err if neither detection nor default yields a code.
pub fn get_language_code(config: &VectorConfigInternal, text: &str) -> Result<String, String> {
    if config.language_detect {
        if let Some((lang_code, confidence)) = detect_language(text) {
            if confidence >= config.language_confidence as f32 {
                return Ok(lang_code);
            }
        }
    }
    if !config.language_default_code.is_empty() {
        return Ok(config.language_default_code.clone());
    }
    Err(format!(
        "No language code could be determined for vector config '{}'. \
         Either enable language_detect=True or specify language_default_code.",
        config.name
    ))
}

/// Language detection — not yet implemented; always returns None so callers
/// fall back to language_default_code, matching Python behaviour when confidence
/// is below the threshold. Will be wired to whatlang once the dependency is added.
fn detect_language(_text: &str) -> Option<(String, f32)> {
    None
}

// ---------------------------------------------------------------------------
// Trait mirroring VectorBase ABC
// ---------------------------------------------------------------------------

pub trait VectorBase {
    /// Mirrors VectorBase._get_sparse_vector (called per-doc by get_sparse_vector).
    fn get_sparse_vector_single(
        &self,
        config: &VectorConfigInternal,
        text: &str,
        avgdl: f64,
        trigram_weight: f64,
    ) -> Result<(Vec<u32>, Vec<f32>), String>;

    /// Mirrors VectorBase.get_sparse_vector — preprocess then call _get_sparse_vector per doc.
    fn get_sparse_vector(
        &self,
        config: &VectorConfigInternal,
        docs: &[String],
        avgdls: &[f64],
        trigram_weight: f64,
    ) -> Result<Vec<(Vec<u32>, Vec<f32>)>, String> {
        docs.iter()
            .zip(avgdls.iter())
            .map(|(doc, &avgdl)| {
                let processed = preprocess_text(doc);
                self.get_sparse_vector_single(config, &processed, avgdl, trigram_weight)
            })
            .collect()
    }

    /// Mirrors VectorBase.get_dense_vector.
    fn get_dense_vector(
        &self,
        config: &VectorConfigInternal,
        docs: &[String],
    ) -> Result<Vec<Vec<f32>>, String>;
}
