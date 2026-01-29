# Dockerfile for RLHF Precision Forensics
#
# Build:
#   docker build -t rlhf-precision .
#
# Run:
#   docker run --gpus all -v $(pwd)/results:/workspace/results rlhf-precision
#
# Run with quick test:
#   docker run --gpus all -v $(pwd)/results:/workspace/results rlhf-precision --quick

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace/dtype-rlhf

# Set environment variables for determinism and precision
ENV PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_SDPA_FLASH=0 \
    TORCH_SDPA_MEM_EFFICIENT=0 \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    TORCH_HOME=/workspace/torch_cache

# Create cache directories
RUN mkdir -p /workspace/hf_cache /workspace/torch_cache /workspace/results

# Copy project files
COPY requirements.txt setup.py ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY experiments/ ./experiments/

# Install dependencies (NOT torch - use container's version)
RUN pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir \
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

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "from src.algorithms import PPOTrainer; from src.models import PolicyWrapper; print('Imports OK')"

# Copy remaining files
COPY runpod_start.sh setup.sh CLAUDE.md README.md ./

# Make scripts executable
RUN chmod +x runpod_start.sh setup.sh experiments/*.sh

# Default entrypoint
ENTRYPOINT ["./runpod_start.sh"]
CMD []
