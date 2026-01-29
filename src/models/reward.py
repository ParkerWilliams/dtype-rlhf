"""
Instrumented reward model with precision monitoring.

Tracks effective precision of reward signals.
CRITICAL:
1. Instrument pre-activation values (tanh/sigmoid compress dynamic range)
2. Measure resolution via min-gap, NOT unique count (which measures batch diversity)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class InstrumentedRewardModel(nn.Module):
    """
    Reward model wrapper with precision instrumentation.

    Provides:
    - Pre-activation value tracking (before compression)
    - Post-activation resolution measurement
    - Compression ratio diagnostics
    """

    def __init__(
        self,
        model: nn.Module,
        head_activation: str = "none",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize reward model wrapper.

        Args:
            model: Base reward model
            head_activation: Activation on head output ("none", "tanh", "sigmoid")
            dtype: Model dtype
        """
        super().__init__()
        self.model = model
        self.head_activation = head_activation
        self.dtype = dtype

        # Track last batch for diagnostics
        self._last_pre_activation: Optional[torch.Tensor] = None
        self._last_rewards: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning reward scores.

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Attention mask [B, T]

        Returns:
            rewards: Reward scores [B]
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pre-activation rewards (CRITICAL: instrument here)
        if hasattr(outputs, 'logits'):
            pre_activation = outputs.logits.squeeze(-1)
        else:
            pre_activation = outputs[0].squeeze(-1)

        self._last_pre_activation = pre_activation.detach()

        # Apply activation (if any)
        if self.head_activation == "tanh":
            rewards = torch.tanh(pre_activation)
        elif self.head_activation == "sigmoid":
            rewards = torch.sigmoid(pre_activation)
        else:
            rewards = pre_activation

        self._last_rewards = rewards.detach()

        return rewards

    def get_diagnostics(
        self,
        pre_activation: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute precision diagnostics for reward outputs.

        Args:
            pre_activation: Optional pre-activation values (uses cached if None)
            rewards: Optional reward values (uses cached if None)

        Returns:
            Dict of precision diagnostics
        """
        if pre_activation is None:
            pre_activation = self._last_pre_activation
        if rewards is None:
            rewards = self._last_rewards

        if pre_activation is None or rewards is None:
            return {}

        result = {}

        # Pre-activation resolution (before compression)
        pre_act_res = self._compute_resolution(pre_activation)
        for k, v in pre_act_res.items():
            result[f"pre_act_{k}"] = v

        # Post-activation resolution
        reward_res = self._compute_resolution(rewards)
        for k, v in reward_res.items():
            result[f"reward_{k}"] = v

        # Compression ratio
        pre_range = (pre_activation.max() - pre_activation.min()).item()
        post_range = (rewards.max() - rewards.min()).item()
        result["activation_compression"] = pre_range / max(post_range, 1e-10)

        return result

    def _compute_resolution(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Measure effective resolution via min-gap (NOT unique count).

        Args:
            tensor: Input tensor

        Returns:
            Dict with resolution metrics
        """
        flat = tensor.flatten().float()
        sorted_unique = flat.unique().sort()[0]

        if len(sorted_unique) < 2:
            return {
                "resolution_min": float('inf'),
                "resolution_median": float('inf'),
                "range": 0.0,
                "unique_count": len(sorted_unique),
            }

        gaps = sorted_unique[1:] - sorted_unique[:-1]
        gaps = gaps[gaps > 0]

        if len(gaps) == 0:
            return {
                "resolution_min": float('inf'),
                "resolution_median": float('inf'),
                "range": (sorted_unique[-1] - sorted_unique[0]).item(),
                "unique_count": len(sorted_unique),
            }

        return {
            "resolution_min": gaps.min().item(),
            "resolution_median": gaps.median().item(),
            "range": (sorted_unique[-1] - sorted_unique[0]).item(),
            "unique_count": len(sorted_unique),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        head_activation: str = "none",
        device_map: str = "auto",
        num_labels: int = 1,
    ) -> "InstrumentedRewardModel":
        """
        Load reward model from pretrained.

        Args:
            model_name_or_path: Model name or path
            dtype: Target dtype
            head_activation: Activation function
            device_map: Device placement
            num_labels: Number of output labels (1 for regression)

        Returns:
            InstrumentedRewardModel instance
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            num_labels=num_labels,
        )

        return cls(model, head_activation=head_activation, dtype=dtype)


def verify_frozen(
    model: nn.Module,
    name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Verify model is properly frozen. Call at start of training.

    Checks:
    1. All params have requires_grad=False
    2. Model is in eval mode
    3. No params are in optimizer (if provided)

    Args:
        model: Model to verify
        name: Model name for error messages
        optimizer: Optional optimizer to check

    Raises:
        RuntimeError: If model is not properly frozen
    """
    # Check requires_grad
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"{name} has {len(trainable)} trainable params! "
            f"First few: {trainable[:3]}"
        )

    # Check eval mode
    if model.training:
        raise RuntimeError(f"{name} is in training mode, should be eval()")

    # Check not in optimizer
    if optimizer is not None:
        opt_params = set()
        for group in optimizer.param_groups:
            opt_params.update(id(p) for p in group['params'])

        model_params = set(id(p) for p in model.parameters())
        overlap = opt_params & model_params
        if overlap:
            raise RuntimeError(
                f"{name} has {len(overlap)} params in optimizer! "
                "This will corrupt precision experiments."
            )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Verified {name} frozen: {param_count:,} params")


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze a model for use as reference or reward model.

    Args:
        model: Model to freeze

    Returns:
        The frozen model (same object, modified in place)
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Disable running stats on BatchNorm layers
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False

    return model
