"""
Instrumented value head with precision monitoring.

The value head predicts expected cumulative reward. Unlike classification:
- Outputs are unbounded (can be any real number)
- Scale depends on reward normalization
- Gradients flow back through entire policy
- If it overflows, the entire policy gradient becomes NaN

Note on BF16 risk: The primary danger isn't overflow (BF16 max is ~3.4e38),
but mantissa starvation - coarse quantization as magnitude grows, causing
gradient noise amplification.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class InstrumentedValueHead(nn.Module):
    """
    Value head with precision monitoring and safety checks.

    Tracks:
    - Maximum output magnitude (for overflow detection)
    - Quantization step size (for mantissa starvation)
    - Stagnation rate (values not changing due to precision loss)
    """

    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        fail_fast: bool = True,
    ):
        """
        Initialize value head.

        Args:
            hidden_size: Input hidden dimension from policy
            dtype: Dtype for value head weights
            fail_fast: Whether to raise on danger conditions
        """
        super().__init__()
        self.head = nn.Linear(hidden_size, 1, dtype=dtype)
        self.output_dtype = dtype
        self.fail_fast = fail_fast

        # Tracking state
        self.max_output_seen = 0.0
        self.danger_count = 0
        self.prev_values: Optional[torch.Tensor] = None

        # Dtype-specific thresholds (mantissa starvation, not overflow)
        self.danger_threshold = {
            torch.bfloat16: 1e4,   # At this magnitude, BF16 spacing is ~78
            torch.float16: 6e4,    # Near FP16 max of 65504
            torch.float32: 1e6,    # Arbitrary "something is wrong" threshold
        }.get(dtype, 1e4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with safety monitoring.

        Args:
            hidden_states: Policy hidden states [B, T, H]

        Returns:
            values: Value predictions [B, T]
        """
        values = self.head(hidden_states).squeeze(-1)

        # Track max for diagnostics
        max_val = values.abs().max().item()
        self.max_output_seen = max(self.max_output_seen, max_val)

        # Check for NaN/Inf IMMEDIATELY
        if torch.isnan(values).any() or torch.isinf(values).any():
            self.danger_count += 1
            if self.fail_fast:
                raise ValueError(
                    f"Value head produced NaN/Inf! "
                    f"max_output_seen={self.max_output_seen:.2f}, "
                    f"dtype={self.output_dtype}"
                )

        # Check approaching danger zone (mantissa starvation)
        if max_val > self.danger_threshold:
            self.danger_count += 1
            if self.fail_fast:
                raise ValueError(
                    f"Value head in danger zone (mantissa starvation)! "
                    f"max={max_val:.2f} > threshold={self.danger_threshold:.2f}, "
                    f"dtype={self.output_dtype}"
                )

        return values

    def get_diagnostics(self, values: torch.Tensor) -> Dict[str, float]:
        """
        Compute value head precision diagnostics.

        Args:
            values: Value predictions from forward pass [B, T-1]

        Returns:
            Dict of diagnostic metrics
        """
        values_flat = values.flatten().float()  # FP32 for analysis

        # Quantization analysis (like reward resolution)
        unique_sorted = values_flat.unique().sort()[0]
        if len(unique_sorted) > 1:
            gaps = unique_sorted[1:] - unique_sorted[:-1]
            gaps = gaps[gaps > 0]
            quant_step_median = gaps.median().item() if len(gaps) > 0 else float('inf')
        else:
            quant_step_median = float('inf')

        # Stagnation detection: how often does value not change?
        stagnation_rate = 0.0
        if self.prev_values is not None and self.prev_values.shape == values.shape:
            # After dtype cast, how many values are identical to previous step?
            if self.output_dtype == torch.bfloat16:
                prev_cast = self.prev_values.bfloat16()
                curr_cast = values.bfloat16()
                stagnation_rate = (prev_cast == curr_cast).float().mean().item()
            elif self.output_dtype == torch.float16:
                prev_cast = self.prev_values.half()
                curr_cast = values.half()
                stagnation_rate = (prev_cast == curr_cast).float().mean().item()
            else:
                stagnation_rate = (self.prev_values == values).float().mean().item()

        self.prev_values = values.detach().clone()

        return {
            "value_max": self.max_output_seen,
            "value_headroom": self.danger_threshold - self.max_output_seen,
            "value_danger_count": self.danger_count,
            "value_quant_step_median": quant_step_median,
            "value_stagnation_rate": stagnation_rate,
            "value_unique_count": len(unique_sorted),
        }

    def reset_tracking(self):
        """Reset tracking state (call between runs)."""
        self.max_output_seen = 0.0
        self.danger_count = 0
        self.prev_values = None
