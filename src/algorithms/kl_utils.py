"""
KL divergence and precision-sensitive utilities for RLHF.

This module contains all precision-critical computations that must be
handled carefully to avoid numerical issues.

CRITICAL REQUIREMENTS:
- Use log_softmax, NEVER log(softmax(x))
- All reductions in FP32
- Proper masking for padding tokens
- Clamp denominators to prevent divide-by-zero
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


def get_logprobs_for_tokens(
    logits: torch.Tensor,      # [B, T, V] - model outputs
    token_ids: torch.Tensor,   # [B, T] - input sequence (prompt + generated)
    attention_mask: torch.Tensor,  # [B, T] - 1 for real tokens, 0 for padding
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract log-probs for the actual tokens that were sampled.

    CRITICAL: This is the object we measure precision on.
    Using full-distribution mean/sum over vocab would measure the wrong thing.

    Args:
        logits: Model output logits [B, T, V]
        token_ids: Input token IDs [B, T]
        attention_mask: Attention mask [B, T], 1 for real tokens

    Returns:
        logp_taken: Log-prob of taken tokens [B, T-1]
        token_mask: Mask for valid tokens [B, T-1]

    Note:
        Next-token prediction: logits[t] predicts token_ids[t+1]
        So we slice logits[:, :-1] and targets = token_ids[:, 1:]
    """
    # PRECISION-CRITICAL: Use log_softmax, NEVER log(softmax(x))
    log_probs_all = F.log_softmax(logits, dim=-1)  # [B, T, V]

    # Next-token prediction: targets are input_ids shifted by 1
    # logits[t] predicts token_ids[t+1]
    log_probs_all = log_probs_all[:, :-1, :]  # [B, T-1, V]
    targets = token_ids[:, 1:]                 # [B, T-1]
    mask = attention_mask[:, 1:]               # [B, T-1]

    # Gather log-prob of the token that was actually taken
    logp_taken = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # Mask out padding tokens
    logp_taken = logp_taken * mask

    return logp_taken, mask


def compute_kl_with_diagnostics(
    logp_policy: torch.Tensor,      # [B, T-1] log-probs of taken tokens
    logp_ref: torch.Tensor,         # [B, T-1] log-probs from reference
    token_mask: torch.Tensor,       # [B, T-1] mask for valid tokens
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    KL computation with precision diagnostics.

    NOTE ON SHAPES: All inputs are [B, T-1], aligned to target token positions.
    This function receives token_mask (not attention_mask), which is already
    sliced to match logp_taken output from get_logprobs_for_tokens().

    IMPORTANT DISTINCTION:
    - kl_mc: Monte Carlo estimate E_pi[log pi - log pi_ref] = mean(log_ratio)
             This is the ACTUAL KL divergence estimate used in theory.
    - kl_penalty: exp(log_ratio) - 1 - log_ratio
             This is a nonnegative convex SURROGATE used as a penalty term.
             It's what most PPO implementations actually optimize against.

    CRITICAL REQUIREMENTS:
    - token_mask MUST be provided to exclude padding tokens
    - All reductions performed in FP32 regardless of input dtype
    - Log-ratio clamped to prevent exp() overflow from outliers
    - Divide-by-zero protection on valid_tokens

    Args:
        logp_policy: Policy log-probs for taken tokens [B, T-1]
        logp_ref: Reference log-probs for taken tokens [B, T-1]
        token_mask: Mask for valid tokens [B, T-1]
        compute_dtype: Dtype for computation (default FP32)

    Returns:
        kl_mc: Monte Carlo KL estimate (scalar)
        kl_penalty: Surrogate penalty (scalar)
        diagnostics: Dict of precision diagnostics
    """
    # Upcast for computation
    lp_policy = logp_policy.to(compute_dtype)
    lp_ref = logp_ref.to(compute_dtype)
    mask = token_mask.to(compute_dtype)

    log_ratio = lp_policy - lp_ref

    # Count valid tokens for proper averaging
    # CRITICAL: Protect against divide-by-zero
    valid_tokens = mask.sum()
    valid_tokens_safe = valid_tokens.clamp(min=1.0)

    # Early exit if no valid tokens (bad batch or mask bug)
    # CRITICAL: Use .item() for tensor comparison, return device-consistent tensors
    if valid_tokens.item() == 0:
        zero = torch.zeros((), device=logp_policy.device, dtype=compute_dtype)
        return zero, zero, {
            "log_ratio_mean": 0.0,
            "log_ratio_std": 0.0,
            "log_ratio_abs_max": 0.0,
            "kl_mc_mean": 0.0,
            "kl_penalty_mean": 0.0,
            "kl_penalty_p99": 0.0,
            "kl_penalty_max": 0.0,
            "log_ratio_near_zero_frac": 0.0,
            "kl_penalty_near_zero_frac": 0.0,
            "log_ratio_clamped_frac": 0.0,
            "valid_token_count": 0,
            "WARNING": "no_valid_tokens",
        }

    # Monte Carlo KL estimate (the actual KL divergence)
    # CRITICAL: Only average over valid (non-padding) tokens
    kl_mc = (log_ratio * mask).sum() / valid_tokens_safe

    # Surrogate penalty (what PPO typically uses)
    # CLAMP log_ratio to prevent exp() overflow from outliers
    # This prevents a single numerical outlier from nuking the batch gradient
    log_ratio_clamped = log_ratio.clamp(min=-10, max=10)
    kl_penalty_per_token = log_ratio_clamped.exp() - 1 - log_ratio_clamped

    # Masked mean for penalty (FP32 reduction)
    kl_penalty = (kl_penalty_per_token * mask).sum() / valid_tokens_safe

    # Diagnostics - ONLY over valid tokens
    valid_log_ratios = log_ratio[mask.bool()]
    valid_penalties = kl_penalty_per_token[mask.bool()]

    diagnostics = {
        # Log ratio (the fundamental quantity)
        "log_ratio_mean": valid_log_ratios.mean().item(),
        "log_ratio_std": valid_log_ratios.std().item() if len(valid_log_ratios) > 1 else 0.0,
        "log_ratio_abs_max": valid_log_ratios.abs().max().item(),

        # Monte Carlo KL estimate
        "kl_mc_mean": kl_mc.item(),

        # Surrogate penalty (what we optimize)
        "kl_penalty_mean": kl_penalty.item(),
        "kl_penalty_p99": torch.quantile(valid_penalties.float(), 0.99).item() if len(valid_penalties) > 0 else 0.0,
        "kl_penalty_max": valid_penalties.max().item() if len(valid_penalties) > 0 else 0.0,

        # Precision diagnostics
        "log_ratio_near_zero_frac": (valid_log_ratios.abs() < 1e-6).float().mean().item(),
        "kl_penalty_near_zero_frac": (valid_penalties < 1e-8).float().mean().item() if len(valid_penalties) > 0 else 0.0,

        # How many log_ratios were clamped? (indicates outliers)
        "log_ratio_clamped_frac": ((log_ratio.abs() > 10) * mask).sum().item() / valid_tokens_safe.item(),

        # Valid token count for debugging mask issues
        "valid_token_count": int(valid_tokens.item()),
    }

    return kl_mc, kl_penalty, diagnostics


def safe_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean reduction in FP32 for numerical stability.

    Args:
        tensor: Input tensor (any dtype)
        mask: Optional mask (1 for valid, 0 for invalid)

    Returns:
        Mean in FP32
    """
    t = tensor.float()
    if mask is not None:
        # CRITICAL: Clamp denominator to prevent divide-by-zero
        den = mask.float().sum().clamp_min(1.0)
        return (t * mask.float()).sum() / den
    return t.mean()


def safe_norm(tensor: torch.Tensor) -> torch.Tensor:
    """L2 norm in FP32."""
    return tensor.float().norm()


def compute_precision_error(x_low: torch.Tensor, x_high: torch.Tensor) -> Dict[str, float]:
    """
    Compare low-precision tensor against high-precision reference.

    Args:
        x_low: Tensor in lower precision (e.g., BF16)
        x_high: Tensor in higher precision (e.g., FP32), same values

    Returns:
        Dictionary of precision error metrics
    """
    # Cast both to FP32 for comparison
    low_f32 = x_low.float()
    high_f32 = x_high.float()

    abs_err = (low_f32 - high_f32).abs()
    rel_err = abs_err / (high_f32.abs() + 1e-12)

    return {
        # Absolute error stats
        "abs_err_mean": abs_err.mean().item(),
        "abs_err_max": abs_err.max().item(),
        "abs_err_p99": torch.quantile(abs_err.flatten(), 0.99).item(),

        # Relative error stats
        "rel_err_mean": rel_err.mean().item(),
        "rel_err_max": rel_err.max().item(),
        "rel_err_p99": torch.quantile(rel_err.flatten(), 0.99).item(),

        # Zero-rounding rate: how often does BF16 round to zero when FP32 didn't?
        "zero_rounding_rate": ((x_low == 0) & (x_high != 0)).float().mean().item(),

        # Sign flip rate: catastrophic errors
        "sign_flip_rate": ((x_low * x_high) < 0).float().mean().item(),
    }


def compute_reward_resolution(rewards: torch.Tensor) -> Dict[str, float]:
    """
    Measure effective resolution of reward signal.

    NOTE: Don't use unique().count() - it measures batch diversity, not precision.
    Instead, measure the gap distribution between distinct values.

    WARNING: min_gap is extremely sensitive to outliers/duplicates.
    Use median and p10 as primary metrics.

    Args:
        rewards: Reward tensor [B] or any shape

    Returns:
        Dictionary of resolution metrics
    """
    sorted_rewards, _ = rewards.flatten().float().sort()  # FP32 for analysis

    # Remove exact duplicates, then compute gaps
    unique_sorted = sorted_rewards.unique()

    if len(unique_sorted) < 2:
        return {
            "reward_resolution_median": float('inf'),
            "reward_resolution_p10": float('inf'),
            "reward_resolution_min": float('inf'),  # Diagnostic only
            "reward_unique_count": 1,
            "reward_range": 0.0,
            "reward_gap_count": 0,
        }

    # Gaps between consecutive unique values
    gaps = unique_sorted[1:] - unique_sorted[:-1]

    # Filter out exact-zero gaps (shouldn't happen after unique(), but defensive)
    gaps = gaps[gaps > 0]

    if len(gaps) == 0:
        return {
            "reward_resolution_median": float('inf'),
            "reward_resolution_p10": float('inf'),
            "reward_resolution_min": float('inf'),
            "reward_unique_count": len(unique_sorted),
            "reward_range": (unique_sorted[-1] - unique_sorted[0]).item(),
            "reward_gap_count": 0,
        }

    return {
        # PRIMARY METRICS (robust to outliers)
        "reward_resolution_median": gaps.median().item(),
        "reward_resolution_p10": torch.quantile(gaps, 0.1).item(),  # 10th percentile

        # DIAGNOSTIC ONLY (sensitive to outliers - don't rely on this)
        "reward_resolution_min": gaps.min().item(),

        # Context
        "reward_unique_count": len(unique_sorted),
        "reward_range": (unique_sorted[-1] - unique_sorted[0]).item(),
        "reward_gap_count": len(gaps),
    }


def kahan_cumsum(x: torch.Tensor) -> torch.Tensor:
    """
    Kahan compensated cumulative sum - near-exact baseline.

    This is for H4 analysis only - measuring accumulation error against
    a near-exact baseline. DO NOT use in the main PPO loop.

    Args:
        x: Input tensor [batch, seq_len]

    Returns:
        Cumulative sum with compensation [batch, seq_len]
    """
    result = torch.zeros_like(x)
    c = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # Compensation
    running_sum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    for i in range(x.shape[1]):
        y = x[:, i] - c
        t = running_sum + y
        c = (t - running_sum) - y  # Recovers lost low-order bits
        running_sum = t
        result[:, i] = running_sum

    return result


def measure_accumulation_error(
    log_probs_bf16: torch.Tensor,
    log_probs_fp32: torch.Tensor
) -> Dict[str, Optional[float]]:
    """
    Measure how rounding errors accumulate across sequence length.

    For H4 analysis - comparing BF16 vs FP32 log-prob accumulation.

    Args:
        log_probs_bf16: Log probs in BF16 [batch, seq_len]
        log_probs_fp32: Same values computed in FP32 [batch, seq_len]

    Returns:
        Dictionary of accumulation error metrics
    """
    batch_size, seq_len = log_probs_bf16.shape

    # Cumulative sums at each position
    cumsum_bf16 = log_probs_bf16.float().cumsum(dim=1)
    cumsum_fp32 = log_probs_fp32.cumsum(dim=1)

    # Kahan compensated sum as "ground truth" baseline
    cumsum_kahan = kahan_cumsum(log_probs_fp32)

    # Error vs position in sequence
    error_vs_position = (cumsum_bf16 - cumsum_kahan).abs()

    return {
        # Does error grow with sequence length?
        "accum_err_at_64": error_vs_position[:, min(63, seq_len-1)].mean().item(),
        "accum_err_at_256": error_vs_position[:, min(255, seq_len-1)].mean().item() if seq_len > 255 else None,
        "accum_err_at_512": error_vs_position[:, min(511, seq_len-1)].mean().item() if seq_len > 511 else None,
        "accum_err_final": error_vs_position[:, -1].mean().item(),

        # Additional diagnostic: error from FP32 cumsum (no Kahan)
        "fp32_cumsum_err_final": (cumsum_fp32[:, -1] - cumsum_kahan[:, -1]).abs().mean().item(),
    }
