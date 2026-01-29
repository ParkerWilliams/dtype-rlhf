#!/usr/bin/env python3
"""Quick import verification test."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    # Core utilities
    from src.utils.dtype_context import precision_scope, DTYPE_EPS, get_torch_dtype
    print("  src.utils.dtype_context")

    from src.utils.determinism import enforce_determinism
    print("  src.utils.determinism")

    # Metrics
    from src.metrics.diagnostics import StepDiagnostics, RunSummary, FailureType
    print("  src.metrics.diagnostics")

    # Algorithms
    from src.algorithms.kl_utils import (
        get_logprobs_for_tokens,
        compute_kl_with_diagnostics,
        safe_mean,
        compute_precision_error,
    )
    print("  src.algorithms.kl_utils")

    from src.algorithms.ppo import (
        TokenBatch,
        compute_returns_and_advantages,
        normalize_advantages,
        PPOTrainer,
    )
    print("  src.algorithms.ppo")

    # Models
    from src.models.policy import PolicyWrapper
    from src.models.value_head import InstrumentedValueHead
    from src.models.reward import InstrumentedRewardModel, verify_frozen
    print("  src.models.*")

    # Data
    from src.data.synthetic import SyntheticPreferenceTask, get_synthetic_prompts
    print("  src.data.synthetic")

    # Reporting
    from src.reporting.logger import ExperimentLogger, ExperimentManifest
    from src.reporting.loader import load_all_runs, load_summaries
    print("  src.reporting.*")

    # Configs
    from configs.precision_configs import (
        RLHFPrecisionConfig,
        get_precision_config,
        ALL_CONFIGS,
    )
    print("  configs.precision_configs")

    print("\nAll imports successful!")


def test_basic_functionality():
    """Test basic functionality without GPU."""
    import torch

    print("\nTesting basic functionality...")

    # Test DTYPE_EPS
    from src.utils.dtype_context import DTYPE_EPS
    assert torch.bfloat16 in DTYPE_EPS
    print("  DTYPE_EPS constants")

    # Test synthetic task
    from src.data.synthetic import SyntheticPreferenceTask
    task = SyntheticPreferenceTask("length")
    reward = task.get_reward("This is a test sentence with several words.")
    assert reward > 0
    print(f"  Synthetic reward: {reward:.3f}")

    # Test precision config
    from configs.precision_configs import get_precision_config
    config = get_precision_config("bf16_pure")
    assert config.policy_dtype == "bfloat16"
    print(f"  Precision config: {config.name}")

    # Test dataclass creation
    from src.metrics.diagnostics import StepDiagnostics
    diag = StepDiagnostics(
        step=0,
        policy_loss=0.5,
        value_loss=0.1,
        total_loss=0.6,
        kl_mc_mean=0.001,
        kl_penalty_mean=0.0005,
        kl_penalty_p99=0.002,
        kl_penalty_max=0.01,
        log_ratio_abs_max=0.05,
        log_ratio_near_zero_frac=0.1,
        log_ratio_clamped_frac=0.0,
        valid_token_count=100,
        value_max=10.0,
        value_headroom=9990.0,
        value_quant_step_median=0.01,
        value_stagnation_rate=0.0,
        clip_fraction=0.02,
        ratio_unique_in_range=500,
        reward_mean=0.5,
        reward_resolution_median=0.001,
        reward_resolution_p10=0.0005,
        policy_grad_norm=1.5,
        value_grad_norm=0.8,
    )
    d = diag.to_dict()
    assert d["step"] == 0
    print("  StepDiagnostics.to_dict()")

    print("\nBasic functionality tests passed!")


if __name__ == "__main__":
    test_imports()
    test_basic_functionality()
