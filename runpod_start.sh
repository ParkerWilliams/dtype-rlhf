#!/bin/bash
# RunPod deployment script
#
# Use with pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime base image
#
# Quick start on RunPod:
#   1. Create a GPU pod with the PyTorch container
#   2. Clone this repo: git clone <repo-url> /workspace/dtype-rlhf
#   3. Run: cd /workspace/dtype-rlhf && ./runpod_start.sh
#
# For quick sanity test only:
#   ./runpod_start.sh --quick

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# === Environment Setup ===
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export TORCH_HOME=/workspace/torch_cache

# Disable Flash Attention for consistent precision
export TORCH_SDPA_FLASH=0
export TORCH_SDPA_MEM_EFFICIENT=0

# Create cache directories
mkdir -p /workspace/hf_cache /workspace/torch_cache /workspace/results

echo "============================================================"
echo "RLHF Precision Forensics - RunPod Setup"
echo "============================================================"

# === System Check ===
echo ""
echo "=== System Check ==="
python -V
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Verify container torch
python - << 'PYCHECK'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

assert torch.cuda.is_available(), "CUDA not available!"
cuda_major = torch.version.cuda.split('.')[0]
if cuda_major != "12":
    print(f"WARNING: Expected CUDA 12.x, got {torch.version.cuda}")
print("Container torch verified!")
PYCHECK

# === HuggingFace Authentication ===
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "=== HuggingFace Authentication ==="
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
fi

# === Install Dependencies ===
echo ""
echo "=== Installing Dependencies ==="

# Install the package in editable mode (won't touch torch)
pip install --no-deps -e . --quiet

# Install other dependencies (excluding torch which is in the container)
pip install --quiet \
    "transformers>=4.36.0" \
    "datasets>=2.16.0" \
    "accelerate>=0.26.0" \
    "peft>=0.7.0" \
    "numpy>=1.26.0" \
    "pandas>=2.1.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "plotly>=5.18.0" \
    "tqdm>=4.66.0" \
    "huggingface_hub>=0.20.0"

# === Verify Installation ===
echo ""
echo "=== Verifying Installation ==="
python -c "
from src.algorithms import PPOTrainer
from src.models import PolicyWrapper
from configs import get_precision_config
print('All imports successful!')
"

# === Run Experiments ===
echo ""
echo "============================================================"

if [ "$1" = "--quick" ]; then
    echo "=== Running Quick Sanity Test ==="
    echo "============================================================"

    # Quick test: static KL probe
    python scripts/static_kl_probe.py \
        --num_prompts 5 \
        --seq_lengths 64 128 \
        --output_dir /workspace/results/static_kl_probe

    # Quick test: single PPO run
    python scripts/run_ppo_sweep.py \
        --precision_config bf16_pure \
        --use_synthetic_reward \
        --max_steps 50 \
        --batch_size 4 \
        --seed 0 \
        --output_dir /workspace/results

    echo ""
    echo "=== Quick Test Complete ==="
    cat /workspace/results/runs/bf16_pure_seed0/run_summary.json | python -m json.tool

else
    echo "=== Running Full Sweep ==="
    echo "============================================================"
    echo "This will take several hours. For a quick test, use: ./runpod_start.sh --quick"
    echo ""

    # Phase -1: Static KL probe (fast validation)
    echo ">>> Phase -1: Static KL Probe"
    python scripts/static_kl_probe.py \
        --num_prompts 20 \
        --seq_lengths 64 128 256 512 \
        --output_dir /workspace/results/static_kl_probe

    # Phase 2: Full PPO sweep
    echo ""
    echo ">>> Phase 2: PPO Precision Sweep"
    python scripts/run_full_sweep.py \
        --output_dir /workspace/results \
        --seeds 0 1 2 \
        --max_steps 1000

    # Analysis
    echo ""
    echo ">>> Generating Analysis"
    python scripts/analyze_results.py \
        --results_dir /workspace/results \
        --output_dir /workspace/results/plots
fi

echo ""
echo "============================================================"
echo "=== Complete ==="
echo "============================================================"
echo "Results saved to: /workspace/results/"
ls -la /workspace/results/
