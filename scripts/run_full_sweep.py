#!/usr/bin/env python3
"""
Full Experiment Sweep

Orchestrates running all precision configurations across multiple seeds.

Usage:
    python scripts/run_full_sweep.py \
        --output_dir results/ \
        --seeds 0 1 2 \
        --max_steps 1000

Quick test:
    python scripts/run_full_sweep.py \
        --output_dir results/ \
        --seeds 0 \
        --max_steps 50 \
        --configs bf16_pure fp32_baseline
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.precision_configs import list_configs


def main():
    parser = argparse.ArgumentParser(description="Run full precision sweep")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model name",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps per run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Configs to run (default: all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running",
    )

    args = parser.parse_args()

    # Get configs to run
    configs = args.configs if args.configs else list_configs()

    print("=" * 60)
    print("Full Precision Sweep")
    print("=" * 60)
    print(f"Configs: {configs}")
    print(f"Seeds: {args.seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Total runs: {len(configs) * len(args.seeds)}")
    print("=" * 60)

    # Run each config/seed combination
    completed = []
    failed = []

    for config in configs:
        for seed in args.seeds:
            run_id = f"{config}_seed{seed}"
            print(f"\n>>> Starting {run_id}")

            cmd = [
                sys.executable,
                "scripts/run_ppo_sweep.py",
                "--precision_config", config,
                "--seed", str(seed),
                "--max_steps", str(args.max_steps),
                "--batch_size", str(args.batch_size),
                "--output_dir", args.output_dir,
                "--base_model", args.base_model,
                "--use_synthetic_reward",
            ]

            if args.dry_run:
                print(f"Would run: {' '.join(cmd)}")
                continue

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=False,
                )
                completed.append(run_id)
                print(f">>> Completed {run_id}")

            except subprocess.CalledProcessError as e:
                failed.append((run_id, str(e)))
                print(f">>> Failed {run_id}: {e}")

            except KeyboardInterrupt:
                print("\n>>> Interrupted by user")
                break

    # Summary
    print("\n" + "=" * 60)
    print("Sweep Summary")
    print("=" * 60)
    print(f"Completed: {len(completed)}/{len(configs) * len(args.seeds)}")
    if completed:
        print(f"  {completed}")
    if failed:
        print(f"Failed: {len(failed)}")
        for run_id, error in failed:
            print(f"  {run_id}: {error}")


if __name__ == "__main__":
    main()
