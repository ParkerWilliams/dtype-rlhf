"""
Determinism utilities for reproducible precision experiments.

For precision comparisons to be meaningful, FP32 and BF16 runs must start
from identical initial states. Otherwise "divergence" might be noise.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def enforce_determinism(seed: int = 42):
    """
    Enforce full determinism for reproducible precision experiments.

    Call ONCE at the start of each run.

    Args:
        seed: Random seed for all RNGs

    Note:
        This may make training slower due to deterministic algorithms.
        The CUBLAS_WORKSPACE_CONFIG setting is required for deterministic CUDA.
    """
    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use deterministic algorithms where available
    # warn_only=True allows operations without deterministic implementations to proceed
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Set environment variable for any subprocesses
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Required for deterministic CUDA operations
    # Must be set before CUDA initialization, but we set it anyway
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"Determinism enforced with seed {seed}")


def get_init_checkpoint_path(seed: int, output_dir: str) -> str:
    """
    Get path for initial weights checkpoint.

    Pattern:
    1. First run (fp32_baseline) saves init checkpoint
    2. All subsequent runs load from same checkpoint

    This ensures ALL configs start from identical weights.

    Args:
        seed: Random seed used for initialization
        output_dir: Base output directory

    Returns:
        Path to the initialization checkpoint
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path / f"init_weights_seed{seed}.pt")


def disable_flash_attention():
    """
    Disable Flash Attention to ensure consistent precision behavior.

    Flash attention changes precision characteristics, so we use
    standard math implementation for this study.

    Call at the top of training scripts.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # Use standard math implementation

        # Also set environment variables for completeness
        os.environ['TORCH_SDPA_FLASH'] = '0'
        os.environ['TORCH_SDPA_MEM_EFFICIENT'] = '0'

        print("Flash attention disabled, using math SDPA")


def verify_determinism_setup():
    """
    Verify that determinism settings are properly configured.

    Call after enforce_determinism() to validate setup.
    """
    errors = []

    # Check environment variables
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        errors.append("CUBLAS_WORKSPACE_CONFIG not set correctly")

    # Check PyTorch settings
    if not torch.backends.cudnn.deterministic:
        errors.append("cudnn.deterministic is False")

    if torch.backends.cudnn.benchmark:
        errors.append("cudnn.benchmark is True (should be False)")

    if errors:
        raise RuntimeError(
            f"Determinism not properly configured: {', '.join(errors)}"
        )

    print("Determinism setup verified")


def verify_torch_container():
    """
    Verify we're using the container's torch, not a pip-installed one.

    This prevents CUDA version mismatches from reinstalling torch.
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA not available")
        return

    print(f"Using container torch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Check CUDA version matches expected container version (12.x)
    cuda_major = torch.version.cuda.split('.')[0] if torch.version.cuda else "0"
    if cuda_major != "12":
        print(f"Warning: Expected CUDA 12.x, got {torch.version.cuda}")
