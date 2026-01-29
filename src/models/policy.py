"""
Policy model wrapper with precision controls.

Wraps a HuggingFace causal LM model with explicit dtype handling
and hidden state extraction for the value head.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class PolicyWrapper(nn.Module):
    """
    Wrapper around a causal LM policy with precision controls.

    Provides:
    - Explicit dtype control for weights
    - Hidden state extraction for value head
    - Consistent forward interface
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize policy wrapper.

        Args:
            model: Base causal LM model
            dtype: Target dtype for model weights
        """
        super().__init__()
        self.model = model
        self.dtype = dtype

        # Store hidden size for value head initialization
        self.hidden_size = self._get_hidden_size()

    def _get_hidden_size(self) -> int:
        """Extract hidden size from model config."""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
            elif hasattr(self.model.config, 'n_embd'):
                return self.model.config.n_embd
        raise ValueError("Cannot determine hidden size from model config")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Optional attention mask [B, T]

        Returns:
            logits: Output logits [B, T, V]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs.logits

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get final hidden states for value head.

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Optional attention mask [B, T]

        Returns:
            hidden_states: Final layer hidden states [B, T, H]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last hidden state from the final layer
        return outputs.hidden_states[-1]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dtype: Union[str, torch.dtype] = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> "PolicyWrapper":
        """
        Load policy from pretrained model.

        Args:
            model_name_or_path: HuggingFace model name or path
            dtype: Target dtype for weights
            device_map: Device placement strategy
            trust_remote_code: Whether to trust remote code

        Returns:
            PolicyWrapper instance
        """
        # Convert string dtype
        if isinstance(dtype, str):
            dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
            dtype = dtype_map.get(dtype, torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        return cls(model, dtype=dtype)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name_or_path: str,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> "PolicyWrapper":
        """
        Load FP32 checkpoint and cast to target dtype.

        CRITICAL: Load in FP32 first, THEN cast.
        Don't initialize directly in lower precision - you lose init precision.

        Args:
            checkpoint_path: Path to FP32 checkpoint
            model_name_or_path: Base model name for config
            target_dtype: Target dtype after loading

        Returns:
            PolicyWrapper instance
        """
        # Load state dict (FP32) - use map_location for CPU loading first
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Create model shell in FP32
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
        )
        model.load_state_dict(state_dict)

        # NOW cast to target dtype
        if target_dtype != torch.float32:
            model = model.to(target_dtype)
            print(f"Cast model to {target_dtype}")

        return cls(model, dtype=target_dtype)


def initialize_and_save_checkpoint(
    model_name_or_path: str,
    checkpoint_path: str,
    trust_remote_code: bool = False,
) -> nn.Module:
    """
    Initialize policy in FP32 and save checkpoint.

    All precision configs load THIS checkpoint, then cast.
    This ensures identical starting point for fair comparison.

    Args:
        model_name_or_path: HuggingFace model name
        checkpoint_path: Path to save checkpoint
        trust_remote_code: Whether to trust remote code

    Returns:
        The FP32 model
    """
    # Load/initialize in FP32 (full precision initialization)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,  # Always FP32 for init
        trust_remote_code=trust_remote_code,
    )

    # Save FP32 checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved FP32 init checkpoint: {checkpoint_path}")

    return model
