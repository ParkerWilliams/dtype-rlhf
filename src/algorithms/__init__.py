"""RLHF algorithms with precision instrumentation."""

from .ppo import PPOTrainer, TokenBatch, compute_returns_and_advantages, normalize_advantages
from .kl_utils import (
    compute_kl_with_diagnostics,
    get_logprobs_for_tokens,
    safe_mean,
    safe_norm,
    compute_precision_error,
    compute_reward_resolution,
    kahan_cumsum,
)

__all__ = [
    "PPOTrainer",
    "TokenBatch",
    "compute_returns_and_advantages",
    "normalize_advantages",
    "compute_kl_with_diagnostics",
    "get_logprobs_for_tokens",
    "safe_mean",
    "safe_norm",
    "compute_precision_error",
    "compute_reward_resolution",
    "kahan_cumsum",
]
