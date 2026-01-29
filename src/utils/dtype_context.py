"""
Precision context managers for RLHF experiments.

Provides explicit dtype control and auditable precision boundaries.
"""

from contextlib import contextmanager
from typing import Union
import torch

# Machine epsilon at magnitude 1.0 for each dtype
DTYPE_EPS = {
    torch.float32: 2**-23,    # ~1.19e-7 (23 mantissa bits)
    torch.float16: 2**-10,    # ~9.77e-4 (10 mantissa bits)
    torch.bfloat16: 2**-7,    # ~7.81e-3 (7 mantissa bits) -- NOT 2^-8!
}

# String to torch dtype mapping
DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def get_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Convert string dtype name to torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype}. Valid options: {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype]


@contextmanager
def precision_scope(dtype: Union[str, torch.dtype], include_autocast: bool = False):
    """
    Context manager for precision-controlled computation.

    Supports nesting - inner scope takes precedence.

    Args:
        dtype: "float32", "bfloat16", "float16" or torch.dtype
        include_autocast: If True, also set torch.autocast (for AMP)

    Yields:
        torch.dtype: The resolved dtype for manual casting within scope

    Example:
        with precision_scope("bfloat16") as dtype:
            logits = model(x.to(dtype))
            with precision_scope("float32") as compute_dtype:  # Nested
                kl = compute_kl(logits.to(compute_dtype), ref_logits.to(compute_dtype))
    """
    torch_dtype = get_torch_dtype(dtype)

    if include_autocast and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch_dtype):
            yield torch_dtype
    else:
        # Just provide the dtype for manual casting
        yield torch_dtype


def cast_for_computation(tensor: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """
    Cast tensor to compute dtype with explicit annotation.

    Use this instead of raw .to() for auditable precision boundaries.

    Args:
        tensor: Input tensor in any dtype
        compute_dtype: Target dtype for computation

    Returns:
        Tensor cast to compute_dtype (or original if already correct)
    """
    if tensor.dtype != compute_dtype:
        return tensor.to(compute_dtype)
    return tensor


def get_precision_at_magnitude(value: float, dtype: torch.dtype) -> float:
    """
    Get the spacing between representable values at a given magnitude.

    Args:
        value: The magnitude to query
        dtype: The floating point dtype

    Returns:
        The approximate spacing between representable values

    Note:
        For normalized numbers, spacing ~ |value| * epsilon
        For subnormals (very small numbers), spacing is fixed at ~2^-126 * epsilon
    """
    eps = DTYPE_EPS.get(dtype, DTYPE_EPS[torch.float32])
    # Spacing ~ |value| * epsilon (for normalized numbers)
    if abs(value) > 1e-38:
        return abs(value) * eps
    else:
        # Subnormal region - minimum spacing
        return eps * 2**-126


def verify_dtype(tensor: torch.Tensor, expected: torch.dtype, name: str = "tensor"):
    """
    Verify tensor has expected dtype, raise if not.

    Use at precision boundaries to catch accidental casts.
    """
    if tensor.dtype != expected:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}, expected {expected}. "
            f"Check for implicit casts at precision boundaries."
        )
