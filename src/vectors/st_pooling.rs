//! Sentence-Transformers `1_Pooling` config and on-device pooling.
//!
//! Mirrors `sentence_transformers.sentence_transformer.modules.Pooling` so dense vectors
//! match Amgix Server (`SentenceTransformer.encode`) while keeping the custom candle path.

use candle_core::{DType, Tensor};

/// Pooling mode from `1_Pooling/config.json` (Sentence-Transformers layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StPoolingMode {
    Cls,
    Max,
    Mean,
    MeanSqrtLen,
    WeightedMean,
    LastToken,
}

/// Ordered pooling modes — concatenated along the hidden dimension when multiple are enabled.
#[derive(Debug, Clone)]
pub struct StPoolingConfig {
    pub modes: Vec<StPoolingMode>,
}

impl StPoolingConfig {
    pub fn mean_only() -> Self {
        Self {
            modes: vec![StPoolingMode::Mean],
        }
    }

    pub fn cls_only() -> Self {
        Self {
            modes: vec![StPoolingMode::Cls],
        }
    }

    pub fn from_json_value(value: &serde_json::Value) -> Self {
        if let Some(mode) = value.get("pooling_mode") {
            if let Some(s) = mode.as_str() {
                return Self::from_mode_names(std::iter::once(s));
            }
            if let Some(arr) = mode.as_array() {
                let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                if !names.is_empty() {
                    return Self::from_mode_names(names.into_iter());
                }
            }
        }

        let legacy_order: [(&str, StPoolingMode); 6] = [
            ("pooling_mode_cls_token", StPoolingMode::Cls),
            ("pooling_mode_max_tokens", StPoolingMode::Max),
            ("pooling_mode_mean_tokens", StPoolingMode::Mean),
            (
                "pooling_mode_mean_sqrt_len_tokens",
                StPoolingMode::MeanSqrtLen,
            ),
            ("pooling_mode_weightedmean_tokens", StPoolingMode::WeightedMean),
            ("pooling_mode_lasttoken", StPoolingMode::LastToken),
        ];

        let modes: Vec<StPoolingMode> = legacy_order
            .iter()
            .filter(|(key, _)| value.get(*key).and_then(|v| v.as_bool()).unwrap_or(false))
            .map(|(_, mode)| *mode)
            .collect();

        if modes.is_empty() {
            Self::mean_only()
        } else {
            Self { modes }
        }
    }

    fn from_mode_names<'a, I: IntoIterator<Item = &'a str>>(names: I) -> Self {
        let modes: Vec<StPoolingMode> = names
            .into_iter()
            .filter_map(parse_mode_name)
            .collect();
        if modes.is_empty() {
            Self::mean_only()
        } else {
            Self { modes }
        }
    }
}

fn parse_mode_name(name: &str) -> Option<StPoolingMode> {
    match name.to_ascii_lowercase().as_str() {
        "cls" => Some(StPoolingMode::Cls),
        "max" => Some(StPoolingMode::Max),
        "mean" => Some(StPoolingMode::Mean),
        "mean_sqrt_len_tokens" => Some(StPoolingMode::MeanSqrtLen),
        "weightedmean" => Some(StPoolingMode::WeightedMean),
        "lasttoken" => Some(StPoolingMode::LastToken),
        _ => None,
    }
}

/// Apply Sentence-Transformers pooling to token hidden states on device.
/// `hidden`: [batch, seq_len, hidden_dim], `attention_mask`: [batch, seq_len]
pub fn pool_sentence_embeddings(
    hidden: &Tensor,
    attention_mask: &Tensor,
    config: &StPoolingConfig,
) -> Result<Tensor, String> {
    let mut output_vectors: Vec<Tensor> = Vec::with_capacity(config.modes.len());

    let mut mean_sum: Option<Tensor> = None;
    let mut mean_mask: Option<Tensor> = None;

    for mode in &config.modes {
        match mode {
            StPoolingMode::Cls => output_vectors.push(pool_cls(hidden, attention_mask)?),
            StPoolingMode::Max => output_vectors.push(pool_max(hidden, attention_mask)?),
            StPoolingMode::Mean | StPoolingMode::MeanSqrtLen => {
                if mean_sum.is_none() {
                    let (sum, mask) = mean_pooling_state(hidden, attention_mask)?;
                    mean_sum = Some(sum);
                    mean_mask = Some(mask);
                }
                let sum = mean_sum.as_ref().unwrap();
                let mask = mean_mask.as_ref().unwrap();
                if *mode == StPoolingMode::Mean {
                    output_vectors.push(
                        sum.div(mask)
                            .map_err(|e| format!("mean div failed: {e}"))?,
                    );
                } else {
                    let sqrt_mask = mask
                        .sqrt()
                        .map_err(|e| format!("mean_sqrt sqrt failed: {e}"))?;
                    output_vectors.push(
                        sum.div(&sqrt_mask)
                            .map_err(|e| format!("mean_sqrt div failed: {e}"))?,
                    );
                }
            }
            StPoolingMode::WeightedMean => {
                output_vectors.push(pool_weighted_mean(hidden, attention_mask)?);
            }
            StPoolingMode::LastToken => {
                output_vectors.push(pool_last_token(hidden, attention_mask)?);
            }
        }
    }

    if output_vectors.len() == 1 {
        Ok(output_vectors.remove(0))
    } else {
        Tensor::cat(&output_vectors, 1).map_err(|e| format!("pool cat failed: {e}"))
    }
}

/// Returns a contiguous `[batch, seq, hidden]` float mask with the same shape as `hidden`.
fn expand_attention_mask(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, String> {
    attention_mask
        .to_dtype(hidden.dtype())
        .map_err(|e| format!("mask to_dtype failed: {e}"))?
        .unsqueeze(2)
        .map_err(|e| format!("mask unsqueeze failed: {e}"))?
        .expand(hidden.shape())
        .map_err(|e| format!("mask expand failed: {e}"))?
        .contiguous()
        .map_err(|e| format!("mask contiguous failed: {e}"))
}

/// CLS pooling: first real token (handles left padding via mask argmax).
fn pool_cls(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, String> {
    let batch_size = hidden.dim(0).map_err(|e| format!("cls batch: {e}"))?;
    let hidden_dim = hidden.dim(2).map_err(|e| format!("cls hidden: {e}"))?;

    let mask_i64 = attention_mask
        .to_dtype(DType::I64)
        .map_err(|e| format!("cls mask dtype: {e}"))?;
    let first_indices = mask_i64
        .argmax(1)
        .map_err(|e| format!("cls argmax: {e}"))?;

    gather_rows(hidden, &first_indices, batch_size, hidden_dim)
}

fn pool_max(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, String> {
    let mask = expand_attention_mask(hidden, attention_mask)?;
    let neg = Tensor::full(f32::NEG_INFINITY, hidden.dims(), hidden.device())
        .map_err(|e| format!("max fill failed: {e}"))?
        .to_dtype(hidden.dtype())
        .map_err(|e| format!("max neg dtype failed: {e}"))?;
    let masked = mask
        .where_cond(hidden, &neg)
        .map_err(|e| format!("max where failed: {e}"))?;
    masked
        .max(1)
        .map_err(|e| format!("max pool max failed: {e}"))
}

fn mean_pooling_state(
    hidden: &Tensor,
    attention_mask: &Tensor,
) -> Result<(Tensor, Tensor), String> {
    let mask = expand_attention_mask(hidden, attention_mask)?;
    let sum_embeddings = hidden
        .mul(&mask)
        .map_err(|e| format!("mean mul failed: {e}"))?
        .sum(1)
        .map_err(|e| format!("mean sum failed: {e}"))?;
    // sum_mask: [batch, hidden] — per-row token counts, used as divisor
    let sum_mask = mask
        .sum(1)
        .map_err(|e| format!("mean sum_mask failed: {e}"))?
        .clamp(1e-9, f64::MAX)
        .map_err(|e| format!("mean clamp failed: {e}"))?;
    Ok((sum_embeddings, sum_mask))
}

fn pool_weighted_mean(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, String> {
    let seq_len = hidden
        .dim(1)
        .map_err(|e| format!("weighted_mean seq_len failed: {e}"))?;
    let mask = expand_attention_mask(hidden, attention_mask)?;
    // weights: [1, seq, 1] broadcast-expanded to [batch, seq, hidden]
    let weights = position_weights(hidden.device(), seq_len, hidden.dtype())?
        .expand(hidden.shape())
        .map_err(|e| format!("weighted_mean weights expand: {e}"))?
        .contiguous()
        .map_err(|e| format!("weighted_mean weights contiguous: {e}"))?;
    let weighted_mask = mask
        .mul(&weights)
        .map_err(|e| format!("weighted_mean mul failed: {e}"))?;
    let sum_embeddings = hidden
        .mul(&weighted_mask)
        .map_err(|e| format!("weighted_mean sum mul failed: {e}"))?
        .sum(1)
        .map_err(|e| format!("weighted_mean sum failed: {e}"))?;
    let sum_mask = weighted_mask
        .sum(1)
        .map_err(|e| format!("weighted_mean sum_mask failed: {e}"))?
        .clamp(1e-9, f64::MAX)
        .map_err(|e| format!("weighted_mean clamp failed: {e}"))?;
    sum_embeddings
        .div(&sum_mask)
        .map_err(|e| format!("weighted_mean div failed: {e}"))
}

fn position_weights(
    device: &candle_core::Device,
    seq_len: usize,
    dtype: DType,
) -> Result<Tensor, String> {
    let weights: Vec<f32> = (1..=seq_len).map(|i| i as f32).collect();
    Tensor::from_vec(weights, (1, seq_len, 1), device)
        .map_err(|e| format!("position weights tensor failed: {e}"))?
        .to_dtype(dtype)
        .map_err(|e| format!("position weights dtype failed: {e}"))
}

/// Last-token pooling with left-padding support (mirrors ST gather path).
fn pool_last_token(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, String> {
    let seq_len = hidden
        .dim(1)
        .map_err(|e| format!("lasttoken seq_len: {e}"))?;
    let batch_size = hidden.dim(0).map_err(|e| format!("lasttoken batch: {e}"))?;
    let hidden_dim = hidden.dim(2).map_err(|e| format!("lasttoken hidden: {e}"))?;

    let mask = attention_mask
        .to_dtype(DType::I64)
        .map_err(|e| format!("lasttoken mask dtype: {e}"))?;

    let flipped = mask
        .flip(&[1])
        .map_err(|e| format!("lasttoken flip: {e}"))?;
    let rev_values = flipped
        .max(1)
        .map_err(|e| format!("lasttoken max: {e}"))?;
    let rev_indices = flipped
        .argmax(1)
        .map_err(|e| format!("lasttoken argmax: {e}"))?;

    let zero = Tensor::zeros(rev_values.dims(), rev_values.dtype(), hidden.device())
        .map_err(|e| format!("lasttoken zeros: {e}"))?;
    let seq_len_m1 = (seq_len.saturating_sub(1)) as i64;
    let fallback = Tensor::full(seq_len_m1, rev_indices.dims(), hidden.device())
        .map_err(|e| format!("lasttoken fallback: {e}"))?
        .to_dtype(rev_indices.dtype())
        .map_err(|e| format!("lasttoken fallback dtype: {e}"))?;

    let rev_indices = rev_values
        .eq(&zero)
        .map_err(|e| format!("lasttoken eq: {e}"))?
        .where_cond(&fallback, &rev_indices)
        .map_err(|e| format!("lasttoken where: {e}"))?;

    let one = Tensor::full(1i64, rev_indices.dims(), hidden.device())
        .map_err(|e| format!("lasttoken one: {e}"))?
        .to_dtype(rev_indices.dtype())
        .map_err(|e| format!("lasttoken one dtype: {e}"))?;
    let token_indices = (Tensor::full(seq_len as i64, rev_indices.dims(), hidden.device())
        .map_err(|e| format!("lasttoken seq_len tensor: {e}"))?
        .to_dtype(rev_indices.dtype())
        .map_err(|e| format!("lasttoken seq_len dtype: {e}"))?
        - rev_indices
        - one)
        .map_err(|e| format!("lasttoken index calc: {e}"))?;

    let expanded_mask = expand_attention_mask(hidden, attention_mask)?;
    let masked_hidden = hidden
        .mul(&expanded_mask)
        .map_err(|e| format!("lasttoken mask mul: {e}"))?;

    gather_rows(&masked_hidden, &token_indices, batch_size, hidden_dim)
}

fn gather_rows(
    hidden: &Tensor,
    indices: &Tensor,
    batch_size: usize,
    hidden_dim: usize,
) -> Result<Tensor, String> {
    let idx_vec = indices
        .to_dtype(candle_core::DType::I64)
        .map_err(|e| format!("gather indices dtype: {e}"))?
        .to_vec1::<i64>()
        .map_err(|e| format!("gather indices vec: {e}"))?;

    let mut rows = Vec::with_capacity(batch_size);
    for (batch_idx, &token_idx) in idx_vec.iter().enumerate().take(batch_size) {
        let row = hidden
            .narrow(0, batch_idx, 1)
            .map_err(|e| format!("gather row narrow: {e}"))?
            .narrow(1, token_idx.max(0) as usize, 1)
            .map_err(|e| format!("gather tok narrow: {e}"))?
            .reshape((1, hidden_dim))
            .map_err(|e| format!("gather reshape: {e}"))?;
        rows.push(row);
    }

    if batch_size == 1 {
        Ok(rows.remove(0))
    } else {
        Tensor::cat(&rows, 0).map_err(|e| format!("gather cat: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn parses_pooling_mode_array() {
        let value = serde_json::json!({
            "pooling_mode": ["cls", "mean"]
        });
        let cfg = StPoolingConfig::from_json_value(&value);
        assert_eq!(cfg.modes, vec![StPoolingMode::Cls, StPoolingMode::Mean]);
    }

    #[test]
    fn legacy_bool_order_matches_st() {
        let value = serde_json::json!({
            "pooling_mode_cls_token": true,
            "pooling_mode_mean_tokens": true
        });
        let cfg = StPoolingConfig::from_json_value(&value);
        assert_eq!(cfg.modes, vec![StPoolingMode::Cls, StPoolingMode::Mean]);
    }

    #[test]
    fn mean_pooling_per_row() {
        // batch=2, seq=3, hidden=2
        // hidden[0] = [[1,2], [10,20], [100,200]]
        // hidden[1] = [[3,4], [30,40], [300,400]]
        // mask[0]   = [1, 1, 0]  → mean of first 2 tokens: ([1,2]+[10,20])/2 = [5.5, 11]
        // mask[1]   = [1, 1, 1]  → mean of all 3:           ([3,4]+[30,40]+[300,400])/3 = [111, 148]
        let device = Device::Cpu;
        #[rustfmt::skip]
        let hidden = Tensor::from_vec(
            vec![1.0f32, 2.0, 10.0, 20.0, 100.0, 200.0,
                 3.0f32, 4.0, 30.0, 40.0, 300.0, 400.0],
            (2, 3, 2),
            &device,
        )
        .unwrap();
        let mask = Tensor::from_vec(vec![1u8, 1, 0, 1, 1, 1], (2, 3), &device).unwrap();
        let cfg = StPoolingConfig::mean_only();
        let pooled = pool_sentence_embeddings(&hidden, &mask, &cfg).unwrap();
        let out = pooled.to_vec2::<f32>().unwrap();
        assert!((out[0][0] - 5.5).abs() < 1e-4, "out[0][0]={}", out[0][0]);
        assert!((out[0][1] - 11.0).abs() < 1e-4, "out[0][1]={}", out[0][1]);
        assert!((out[1][0] - 111.0).abs() < 1e-3, "out[1][0]={}", out[1][0]);
        assert!((out[1][1] - 148.0).abs() < 1e-3, "out[1][1]={}", out[1][1]);
    }
}
