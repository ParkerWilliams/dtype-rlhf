#!/bin/bash
# Setup script for RLHF Precision Forensics
#
# Usage:
#   ./setup.sh              # Auto-detect environment
#   ./setup.sh --runpod     # RunPod container setup (uses container's torch)
#   ./setup.sh --local      # Local development setup (creates venv, installs torch)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect environment
detect_environment() {
    if [ -n "$RUNPOD_POD_ID" ] || [ -d "/workspace" ]; then
        echo "runpod"
    elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "cuda"
    else
        echo "cpu"
    fi
}

# RunPod setup - use container's torch, install deps into container's Python
setup_runpod() {
    echo "=== RunPod Container Setup ==="

    # Verify torch is available
    python -c "import torch; print(f'Container torch: {torch.__version__}, CUDA: {torch.version.cuda}')" || {
        echo "ERROR: torch not found in container. Are you using the correct base image?"
        echo "Expected: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime"
        exit 1
    }

    # Install dependencies (excluding torch)
    echo "Installing dependencies..."
    pip install --no-deps -e .
    pip install transformers>=4.36.0 datasets>=2.16.0 accelerate>=0.26.0 \
        peft>=0.7.0 numpy>=1.26.0 pandas>=2.1.0 matplotlib>=3.8.0 \
        seaborn>=0.13.0 plotly>=5.18.0 tqdm>=4.66.0 huggingface_hub>=0.20.0

    # Set environment variables
    export PYTHONHASHSEED=42
    export CUBLAS_WORKSPACE_CONFIG=':4096:8'
    export TOKENIZERS_PARALLELISM=false
    export TORCH_SDPA_FLASH=0
    export TORCH_SDPA_MEM_EFFICIENT=0

    # Verify installation
    echo ""
    echo "Verifying installation..."
    python -c "
from src.algorithms import PPOTrainer
from src.models import PolicyWrapper
from configs import get_precision_config
print('All imports successful!')
"

    echo ""
    echo "=== RunPod Setup Complete ==="
    echo "Run experiments with:"
    echo "  python scripts/static_kl_probe.py --num_prompts 10"
    echo "  python scripts/run_ppo_sweep.py --precision_config bf16_pure --max_steps 100"
}

# Local development setup - create venv with torch
setup_local() {
    echo "=== Local Development Setup ==="

    VENV_DIR="${SCRIPT_DIR}/.venv"

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Detect CUDA availability and install appropriate torch
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "CUDA detected, installing torch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "No CUDA detected, installing CPU-only torch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install package and dependencies
    echo "Installing package and dependencies..."
    pip install -e ".[dev]"

    # Verify installation
    echo ""
    echo "Verifying installation..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
from src.algorithms import PPOTrainer
from src.models import PolicyWrapper
from configs import get_precision_config
print('All imports successful!')
"

    echo ""
    echo "=== Local Setup Complete ==="
    echo ""
    echo "Activate the virtual environment with:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Run experiments with:"
    echo "  python scripts/static_kl_probe.py --num_prompts 10"
    echo "  python scripts/run_ppo_sweep.py --precision_config bf16_pure --max_steps 100"
}

# Main
main() {
    local mode="${1:-auto}"

    case "$mode" in
        --runpod)
            setup_runpod
            ;;
        --local)
            setup_local
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [--runpod | --local | --help]"
            echo ""
            echo "Options:"
            echo "  --runpod    Setup for RunPod container (uses container's torch)"
            echo "  --local     Setup for local development (creates venv, installs torch)"
            echo "  (default)   Auto-detect environment"
            ;;
        *)
            # Auto-detect
            local env=$(detect_environment)
            echo "Detected environment: $env"

            if [ "$env" = "runpod" ]; then
                setup_runpod
            else
                setup_local
            fi
            ;;
    esac
}

main "$@"
