#!/usr/bin/env python3
"""
Re-run the failed precision configs after dtype mismatch fix.

This script runs only bf16_fp32_value and mixed_recommended configs,
which failed due to a missing dtype cast between policy hidden states
and FP32 value head.

Usage:
    python scripts/run_failed_configs.py --output_dir results/

    # Or run specific config:
    python scripts/run_failed_configs.py --config bf16_fp32_value --seed 0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_ppo_sweep import run_training, TrainingConfig
from configs.precision_configs import get_precision_config
from src.utils.determinism import enforce_determinism, disable_flash_attention


FAILED_CONFIGS = ["bf16_fp32_value", "mixed_recommended"]


def main():
    parser = argparse.ArgumentParser(description="Re-run failed precision configs")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Specific config to run. If not specified, runs all: {FAILED_CONFIGS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specific seed to run. If not specified, runs seeds 0, 1, 2",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model name",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Determine which configs and seeds to run
    configs_to_run = [args.config] if args.config else FAILED_CONFIGS
    seeds_to_run = [args.seed] if args.seed is not None else [0, 1, 2]

    print("=" * 60)
    print("Re-running Failed Configs (dtype mismatch fix applied)")
    print("=" * 60)
    print(f"Configs: {configs_to_run}")
    print(f"Seeds: {seeds_to_run}")
    print()

    training_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )

    results = []

    for config_name in configs_to_run:
        for seed in seeds_to_run:
            print(f"\n{'=' * 60}")
            print(f"Running: {config_name} seed={seed}")
            print("=" * 60)

            # Setup determinism for this seed
            enforce_determinism(seed)
            disable_flash_attention()

            # Get precision config
            precision_config = get_precision_config(config_name)

            # Run training
            summary = run_training(
                precision_config=precision_config,
                training_config=training_config,
                base_model=args.base_model,
                seed=seed,
                output_dir=args.output_dir,
                use_synthetic_reward=True,
            )

            results.append({
                "config": config_name,
                "seed": seed,
                "completed": summary.completed,
                "failure_reason": summary.failure_reason,
                "final_reward": summary.final_reward,
            })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        status = "✓" if r["completed"] else f"✗ ({r['failure_reason']})"
        reward = f"{r['final_reward']:.3f}" if r['final_reward'] else "N/A"
        print(f"{r['config']} seed={r['seed']}: {status} (reward={reward})")

    completed = sum(1 for r in results if r["completed"])
    print(f"\nCompleted: {completed}/{len(results)}")


if __name__ == "__main__":
    main()
