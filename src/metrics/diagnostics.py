"""
Centralized diagnostics interface for RLHF precision experiments.

Defines the single source of truth for all metrics logged during training.
Every component writes to these dataclasses to prevent interface drift.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class FailureType(Enum):
    """Types of failures that can occur during training."""
    COMPLETED = "completed"
    NAN_LOSS = "nan_loss"
    NAN_GRADIENT = "nan_gradient"
    INF_VALUE = "inf_value_head"
    VALUE_DANGER = "value_mantissa_starvation"  # Not overflow, but precision collapse
    KL_COLLAPSE = "kl_penalty_collapsed"
    MEMORY_OOM = "out_of_memory"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown_error"


@dataclass
class StepDiagnostics:
    """
    All metrics for a single training step. Every component writes to this.

    This is the SINGLE SOURCE OF TRUTH for per-step metrics.
    Fields are organized by component for clarity.
    """
    step: int

    # === Losses ===
    policy_loss: float
    value_loss: float
    total_loss: float

    # === KL metrics (BOTH MC estimate and penalty - CRITICAL distinction) ===
    # kl_mc: Monte Carlo estimate E_pi[log pi - log pi_ref] = mean(log_ratio)
    #        This is the ACTUAL KL divergence estimate used in theory.
    # kl_penalty: exp(log_ratio) - 1 - log_ratio
    #        This is a nonnegative convex SURROGATE used as a penalty term.
    kl_mc_mean: float
    kl_penalty_mean: float
    kl_penalty_p99: float
    kl_penalty_max: float
    log_ratio_abs_max: float
    log_ratio_near_zero_frac: float
    log_ratio_clamped_frac: float  # Fraction of outliers that were clamped
    valid_token_count: int  # For debugging mask issues

    # === Value head (with mantissa starvation tracking) ===
    value_max: float
    value_headroom: float
    value_quant_step_median: float  # Quantization coarseness
    value_stagnation_rate: float    # How often value doesn't change

    # === PPO clipping (empirical discretization measurement) ===
    clip_fraction: float
    ratio_unique_in_range: int  # Low = discretized landscape

    # === Reward (robust resolution metrics) ===
    reward_mean: float
    reward_resolution_median: float  # PRIMARY
    reward_resolution_p10: float     # PRIMARY

    # === Gradients ===
    policy_grad_norm: float
    value_grad_norm: float

    # === Precision errors (if computing reference comparisons) ===
    precision_errors: Optional[Dict[str, float]] = None

    # === System ===
    memory_gb: float = 0.0
    step_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSONL serialization."""
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        # Flatten precision_errors into main dict
        if self.precision_errors:
            d.update(self.precision_errors)
            del d['precision_errors']
        return d


@dataclass
class RunSummary:
    """
    Final summary for a complete run. Written even on failure.

    This ensures we always have data for post-hoc analysis.
    """
    run_id: str
    config_name: str
    seed: int

    # === Completion status ===
    completed: bool
    failure_reason: Optional[str] = None
    failure_step: Optional[int] = None

    # === Final metrics ===
    final_reward: Optional[float] = None
    final_kl_mc: Optional[float] = None
    final_kl_penalty: Optional[float] = None
    final_policy_loss: Optional[float] = None
    final_value_loss: Optional[float] = None

    # === Stability counts ===
    nan_count: int = 0
    inf_count: int = 0
    value_danger_count: int = 0  # Mantissa starvation events
    kl_collapse_count: int = 0   # Steps where kl_penalty < threshold
    log_ratio_clamped_count: int = 0  # Steps with clamped outliers
    grad_spike_count: int = 0  # Steps with unusually large gradients

    # === Performance ===
    total_steps: int = 0
    wall_time_sec: float = 0.0
    peak_memory_gb: float = 0.0
    mean_step_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {k: v for k, v in self.__dict__.items()}

    def to_structured_dict(self) -> Dict[str, Any]:
        """Convert to nested dict matching the JSON schema."""
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "seed": self.seed,
            "status": {
                "completed": self.completed,
                "failure_reason": self.failure_reason,
                "failure_step": self.failure_step,
            },
            "final_metrics": {
                "final_reward": self.final_reward,
                "final_kl_mc": self.final_kl_mc,
                "final_kl_penalty": self.final_kl_penalty,
                "final_policy_loss": self.final_policy_loss,
                "final_value_loss": self.final_value_loss,
            },
            "stability_counts": {
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "value_danger_count": self.value_danger_count,
                "kl_collapse_count": self.kl_collapse_count,
                "log_ratio_clamped_count": self.log_ratio_clamped_count,
                "grad_spike_count": self.grad_spike_count,
            },
            "performance": {
                "total_steps": self.total_steps,
                "wall_time_sec": self.wall_time_sec,
                "peak_memory_gb": self.peak_memory_gb,
                "mean_step_time_ms": self.mean_step_time_ms,
            },
        }


# Threshold constants for failure detection
KL_COLLAPSE_THRESHOLD = 1e-8
VALUE_DANGER_THRESHOLD_BF16 = 1e4  # At this magnitude, BF16 spacing is ~78
VALUE_DANGER_THRESHOLD_FP16 = 6e4  # Near FP16 max of 65504
VALUE_DANGER_THRESHOLD_FP32 = 1e6  # Arbitrary "something is wrong" threshold
GRAD_SPIKE_THRESHOLD = 100.0  # Gradient norm above this is suspicious
