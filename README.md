# RLHF Precision Forensics

Systematic characterization of floating-point precision failure modes in RLHF pipelines.

Unlike standard supervised training, RLHF has multiple interacting components (policy, reference, reward model, value head) where precision errors compound in unexplored ways. This project produces quantitative evidence for where precision matters in RLHF.

## Quick Start

### RunPod (Recommended)

1. Create a GPU pod with the **PyTorch container**:
   - Image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
   - GPU: H100 80GB recommended, A100 40GB minimum

2. Clone and run:
```bash
git clone https://github.com/yourusername/dtype-rlhf.git /workspace/dtype-rlhf
cd /workspace/dtype-rlhf
./runpod_start.sh --quick  # Quick sanity test (~5 min)
```

For the full experiment sweep (~8 hours on H100):
```bash
./runpod_start.sh
```

### Local Development

```bash
git clone https://github.com/yourusername/dtype-rlhf.git
cd dtype-rlhf
./setup.sh --local  # Creates venv, installs torch + dependencies
source .venv/bin/activate

# Run quick test
python scripts/static_kl_probe.py --num_prompts 10 --seq_lengths 64 128
```

### Docker

```bash
docker build -t rlhf-precision .
docker run --gpus all -v $(pwd)/results:/workspace/results rlhf-precision --quick
```

## Core Hypotheses

| Hypothesis | Description | Key Metric |
|------------|-------------|------------|
| **H1: KL Collapse** | BF16 KL divergence computation causes premature convergence due to catastrophic cancellation | `kl_penalty_mean`, `log_ratio_near_zero_frac` |
| **H2: Reward Quantization** | BF16 reward models produce discretized signals (8 mantissa bits = ~0.4% resolution) | `reward_resolution_median` |
| **H3: Value Head Instability** | Unbounded value predictions suffer from mantissa starvation at high magnitudes | `value_quant_step_median`, `value_stagnation_rate` |
| **H4: Reference Model Drift** | Per-token log-prob errors accumulate across sequence length | `accum_err_final` |

## Precision Configurations

| Config | Policy | Ref | Reward | Value | KL Compute | Notes |
|--------|--------|-----|--------|-------|------------|-------|
| `bf16_pure` | BF16 | BF16 | BF16 | BF16 | BF16 | Baseline "fast" |
| `bf16_fp32_kl` | BF16 | BF16 | BF16 | BF16 | FP32 | Tests H1 |
| `bf16_fp32_value` | BF16 | BF16 | BF16 | FP32 | BF16 | Tests H3 |
| `bf16_fp32_reward` | BF16 | BF16 | FP32 | BF16 | BF16 | Tests H2 |
| `bf16_fp32_ref` | BF16 | FP32 | BF16 | BF16 | BF16 | Tests H4 |
| `mixed_recommended` | BF16 | BF16 | BF16 | FP32 | FP32 | Expected best |
| `fp32_baseline` | FP32 | FP32 | FP32 | FP32 | FP32 | Golden reference |

## Project Structure

```
dtype-rlhf/
├── src/
│   ├── algorithms/      # PPO implementation, KL utils
│   ├── models/          # PolicyWrapper, ValueHead, RewardModel
│   ├── metrics/         # StepDiagnostics, RunSummary
│   ├── reporting/       # JSON logging, result loading
│   └── utils/           # Precision context, determinism
├── configs/             # Precision configuration matrix
├── scripts/
│   ├── static_kl_probe.py           # Phase -1: Quick validation
│   ├── generate_frozen_trajectories.py  # Phase 0: Trajectory freeze
│   ├── replay_trajectories.py       # Phase 0: Precision replay
│   ├── run_ppo_sweep.py             # Phase 2: PPO training
│   ├── run_full_sweep.py            # Full experiment orchestrator
│   └── analyze_results.py           # Generate plots
├── setup.sh             # Environment setup script
├── runpod_start.sh      # RunPod deployment script
└── Dockerfile           # Container build
```

## Running Experiments

### Phase -1: Static KL Probe (Cheapest Validation)
```bash
python scripts/static_kl_probe.py \
    --num_prompts 100 \
    --seq_lengths 64 128 256 512 \
    --output_dir results/static_kl_probe/
```

### Phase 0: Frozen Trajectory Replay
```bash
# Generate FP32 trajectories
python scripts/generate_frozen_trajectories.py \
    --num_batches 100 --batch_size 8 \
    --output_dir results/frozen_trajectories/

# Replay under different precisions
python scripts/replay_trajectories.py \
    --precision_config bf16_pure \
    --trajectories_dir results/frozen_trajectories/
```

### Phase 2: PPO Sweep
```bash
# Single config
python scripts/run_ppo_sweep.py \
    --precision_config bf16_pure \
    --max_steps 1000 --seed 0

# Full sweep (all configs, 3 seeds)
python scripts/run_full_sweep.py \
    --output_dir results/ \
    --seeds 0 1 2 \
    --max_steps 1000
```

### Analysis
```bash
python scripts/analyze_results.py \
    --results_dir results/ \
    --output_dir results/plots/
```

## Key Design Decisions

- **No trl**: Custom PPO implementation for full instrumentation control
- **No wandb**: Custom JSON reporting for precise data format control
- **No flash attention**: Disabled to ensure consistent precision behavior
- **FP32 init checkpoint**: All configs start from identical weights
- **Trajectory replay**: Isolates precision effects from sampling differences

## Critical Implementation Notes

See `CLAUDE.md` for the complete specification, including:
- Numerical grounding requirements (log_softmax, not log(softmax))
- Shape alignment contract ([B, T] vs [B, T-1])
- PPO implementation invariants
- Failure taxonomy and graceful exit patterns

## Budget

- ~$100 on H100 (RunPod)
- Full sweep: ~8 hours

## License

MIT
