"""
Precision configuration matrix for RLHF experiments.

Defines composable precision configs that control dtype for each component.
Be explicit about what each config is testing:

| Aspect | "Storage Dtype" | "Compute Dtype" | "Reduction Dtype" |
|--------|-----------------|-----------------|-------------------|
| What it affects | Weight tensors on GPU | Matmuls, activations | Sums, means, norms |
| Where set | `model.to(dtype)` | `autocast(dtype=)` | Manual `.float()` calls |
| BF16 risk | Weight quantization | Matmul precision | Accumulation error |

For this study, we are primarily testing:
1. Weight storage dtype (policy/ref/reward in BF16 vs FP32)
2. Specific compute paths (KL, advantages) in different precisions
3. Accumulation errors in reductions

We are NOT testing autocast/AMP deeply - that conflates too many variables.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RLHFPrecisionConfig:
    """
    Precision configuration for RLHF experiments.

    Separates storage dtype (weights) from compute dtype (specific operations).
    """
    name: str

    # Model storage dtypes
    policy_dtype: str = "bfloat16"
    reference_dtype: str = "bfloat16"
    reward_dtype: str = "bfloat16"
    value_head_dtype: str = "bfloat16"

    # Computation dtypes (can differ from storage)
    kl_compute_dtype: str = "float32"      # Critical for H1
    advantage_compute_dtype: str = "float32"

    # AMP settings
    policy_use_amp: bool = False
    reward_use_amp: bool = False

    # Optimizer
    optimizer_dtype: str = "float32"       # Optimizer state dtype (Adam moments in FP32)
    use_8bit_adam: bool = False

    def __post_init__(self):
        """Validate config after initialization."""
        valid_dtypes = {"float32", "bfloat16", "float16"}
        for field_name in ["policy_dtype", "reference_dtype", "reward_dtype",
                          "value_head_dtype", "kl_compute_dtype", "advantage_compute_dtype",
                          "optimizer_dtype"]:
            value = getattr(self, field_name)
            if value not in valid_dtypes:
                raise ValueError(f"{field_name}={value} not in {valid_dtypes}")


# === Predefined Configurations ===

# Baseline "fast" config - everything in BF16
BF16_PURE = RLHFPrecisionConfig(
    name="bf16_pure",
    policy_dtype="bfloat16",
    reference_dtype="bfloat16",
    reward_dtype="bfloat16",
    value_head_dtype="bfloat16",
    kl_compute_dtype="bfloat16",
    advantage_compute_dtype="bfloat16",
)

# Test H1 (KL collapse) in isolation
BF16_FP32_KL = RLHFPrecisionConfig(
    name="bf16_fp32_kl",
    policy_dtype="bfloat16",
    reference_dtype="bfloat16",
    reward_dtype="bfloat16",
    value_head_dtype="bfloat16",
    kl_compute_dtype="float32",  # Only change: KL in FP32
    advantage_compute_dtype="float32",
)

# Test H3 (value head instability) in isolation
BF16_FP32_VALUE = RLHFPrecisionConfig(
    name="bf16_fp32_value",
    policy_dtype="bfloat16",
    reference_dtype="bfloat16",
    reward_dtype="bfloat16",
    value_head_dtype="float32",  # Only change: value head in FP32
    kl_compute_dtype="bfloat16",
    advantage_compute_dtype="float32",
)

# Test H2 (reward quantization) in isolation
BF16_FP32_REWARD = RLHFPrecisionConfig(
    name="bf16_fp32_reward",
    policy_dtype="bfloat16",
    reference_dtype="bfloat16",
    reward_dtype="float32",  # Only change: reward model in FP32
    value_head_dtype="bfloat16",
    kl_compute_dtype="bfloat16",
    advantage_compute_dtype="float32",
)

# Test H4 (reference model drift) in isolation
BF16_FP32_REF = RLHFPrecisionConfig(
    name="bf16_fp32_ref",
    policy_dtype="bfloat16",
    reference_dtype="float32",  # Only change: reference in FP32
    reward_dtype="bfloat16",
    value_head_dtype="bfloat16",
    kl_compute_dtype="bfloat16",
    advantage_compute_dtype="float32",
)

# Expected "best" config based on hypotheses
MIXED_RECOMMENDED = RLHFPrecisionConfig(
    name="mixed_recommended",
    policy_dtype="bfloat16",
    reference_dtype="bfloat16",
    reward_dtype="bfloat16",
    value_head_dtype="float32",  # H3: value head in FP32
    kl_compute_dtype="float32",  # H1: KL compute in FP32
    advantage_compute_dtype="float32",
)

# Golden reference - everything in FP32
FP32_BASELINE = RLHFPrecisionConfig(
    name="fp32_baseline",
    policy_dtype="float32",
    reference_dtype="float32",
    reward_dtype="float32",
    value_head_dtype="float32",
    kl_compute_dtype="float32",
    advantage_compute_dtype="float32",
)

# All configs for sweep
ALL_CONFIGS: Dict[str, RLHFPrecisionConfig] = {
    "bf16_pure": BF16_PURE,
    "bf16_fp32_kl": BF16_FP32_KL,
    "bf16_fp32_value": BF16_FP32_VALUE,
    "bf16_fp32_reward": BF16_FP32_REWARD,
    "bf16_fp32_ref": BF16_FP32_REF,
    "mixed_recommended": MIXED_RECOMMENDED,
    "fp32_baseline": FP32_BASELINE,
}


def get_precision_config(name: str) -> RLHFPrecisionConfig:
    """
    Get a precision config by name.

    Args:
        name: Config name (e.g., "bf16_pure", "fp32_baseline")

    Returns:
        RLHFPrecisionConfig instance

    Raises:
        ValueError: If config name not found
    """
    if name not in ALL_CONFIGS:
        raise ValueError(
            f"Unknown precision config: {name}. "
            f"Available: {list(ALL_CONFIGS.keys())}"
        )
    return ALL_CONFIGS[name]


def list_configs() -> list:
    """Get list of available config names."""
    return list(ALL_CONFIGS.keys())
