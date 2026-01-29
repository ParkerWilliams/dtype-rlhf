#!/usr/bin/env python3
"""
Generate all plots and summary statistics from experiment results.

Usage:
    python scripts/analyze_results.py \
        --results_dir results/ \
        --output_dir results/plots/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting.loader import load_all_runs, load_summaries


# Color palette for consistent styling
PALETTE = {
    "bf16_pure": "#1f77b4",
    "bf16_fp32_kl": "#ff7f0e",
    "bf16_fp32_value": "#2ca02c",
    "bf16_fp32_reward": "#d62728",
    "bf16_fp32_ref": "#9467bd",
    "mixed_recommended": "#8c564b",
    "fp32_baseline": "#7f7f7f",
}


def plot_kl_trajectory(df: pd.DataFrame, output_path: Path):
    """Plot 1: KL penalty trajectory with collapse detection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MC estimate
    sns.lineplot(
        data=df, x="step", y="kl_mc_mean",
        hue="config_name", palette=PALETTE, ax=axes[0]
    )
    axes[0].set_title("KL Monte Carlo Estimate")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("E[log π - log π_ref]")
    axes[0].legend(title="Config", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Right: Penalty (what PPO optimizes)
    sns.lineplot(
        data=df, x="step", y="kl_penalty_mean",
        hue="config_name", palette=PALETTE, ax=axes[1]
    )
    axes[1].set_yscale("log")
    axes[1].axhline(1e-8, color="red", linestyle="--", alpha=0.7, label="Collapse threshold")
    axes[1].set_title("KL Penalty Surrogate (log scale)")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("exp(r) - 1 - r")
    axes[1].legend(title="Config", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / "kl_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: kl_trajectories.png")


def plot_reward_trajectory(df: pd.DataFrame, output_path: Path):
    """Plot reward trajectory over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df, x="step", y="reward_mean",
        hue="config_name", palette=PALETTE, ax=ax
    )
    ax.set_title("Reward Trajectory")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / "reward_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reward_trajectories.png")


def plot_value_magnitude(df: pd.DataFrame, output_path: Path):
    """Plot 3: Value head magnitude over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df, x="step", y="value_max",
        hue="config_name", palette=PALETTE, ax=ax
    )

    # Reference lines for danger zones
    ax.axhline(1e4, color="orange", linestyle="--", alpha=0.7, label="BF16 danger zone")
    ax.axhline(6e4, color="red", linestyle="--", alpha=0.7, label="FP16 danger zone")

    ax.set_title("Value Head Maximum Magnitude")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Max |value|")
    ax.set_yscale("log")
    ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / "value_magnitude.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: value_magnitude.png")


def plot_clip_discretization(df: pd.DataFrame, output_path: Path):
    """Plot 6: PPO clipping discretization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clip fraction
    sns.lineplot(
        data=df, x="step", y="clip_fraction",
        hue="config_name", palette=PALETTE, ax=axes[0]
    )
    axes[0].set_title("PPO Clip Fraction")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Fraction of Ratios Clipped")

    # Unique ratios in range
    sns.lineplot(
        data=df, x="step", y="ratio_unique_in_range",
        hue="config_name", palette=PALETTE, ax=axes[1]
    )
    axes[1].set_title("Ratio Discretization")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Unique Ratios in Clip Range")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / "clip_discretization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: clip_discretization.png")


def plot_gradient_norms(df: pd.DataFrame, output_path: Path):
    """Plot gradient norms over training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Policy gradient norm
    sns.lineplot(
        data=df, x="step", y="policy_grad_norm",
        hue="config_name", palette=PALETTE, ax=axes[0]
    )
    axes[0].set_title("Policy Gradient Norm")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Gradient Norm")
    axes[0].set_yscale("log")

    # Value gradient norm
    sns.lineplot(
        data=df, x="step", y="value_grad_norm",
        hue="config_name", palette=PALETTE, ax=axes[1]
    )
    axes[1].set_title("Value Head Gradient Norm")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / "gradient_norms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: gradient_norms.png")


def plot_stability_matrix(df_summaries: pd.DataFrame, output_path: Path):
    """Plot 5: Stability matrix heatmap."""
    if len(df_summaries) == 0:
        print("  Skipped: stability_matrix.png (no summaries)")
        return

    # Create stability score (higher = more stable)
    df_summaries["stability_score"] = (
        df_summaries["status_completed"].astype(float) * 100
        - df_summaries.get("stability_counts_nan_count", 0)
        - df_summaries.get("stability_counts_kl_collapse_count", 0) * 0.1
    )

    # Pivot to matrix form
    try:
        stability_matrix = df_summaries.pivot_table(
            index="config_name",
            columns="seed",
            values="stability_score",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            stability_matrix,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            center=50,
            ax=ax
        )
        ax.set_title("Stability Matrix (higher = more stable)")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Config")

        plt.tight_layout()
        plt.savefig(output_path / "stability_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: stability_matrix.png")
    except Exception as e:
        print(f"  Skipped: stability_matrix.png ({e})")


def generate_summary_table(df_summaries: pd.DataFrame, output_path: Path):
    """Generate markdown summary table."""
    if len(df_summaries) == 0:
        print("  Skipped: summary_table.md (no summaries)")
        return

    # Aggregate by config
    agg_cols = {
        "status_completed": ["mean", "sum"],
    }

    # Add columns if they exist
    for col in ["final_metrics_final_reward", "stability_counts_nan_count",
                "stability_counts_kl_collapse_count", "performance_peak_memory_gb",
                "performance_wall_time_sec"]:
        if col in df_summaries.columns:
            agg_cols[col] = "mean"

    try:
        summary = df_summaries.groupby("config_name").agg(agg_cols).round(3)

        # Save as markdown
        with open(output_path / "summary_table.md", "w") as f:
            f.write("# Experiment Summary\n\n")
            f.write(summary.to_markdown())

        # Also save as CSV for further analysis
        summary.to_csv(output_path / "summary_table.csv")
        print(f"  Saved: summary_table.md, summary_table.csv")
    except Exception as e:
        print(f"  Skipped: summary_table ({e})")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/plots)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analyzing Results")
    print("=" * 60)

    # Set consistent style
    sns.set_theme(style="whitegrid")

    # Load data
    print("\nLoading step metrics...")
    df_steps = load_all_runs(str(results_dir))

    if len(df_steps) == 0:
        print("No step metrics found!")
        return

    print(f"  Loaded {len(df_steps)} step records from {df_steps['config_name'].nunique()} configs")

    print("\nLoading run summaries...")
    df_summaries = load_summaries(str(results_dir))
    print(f"  Loaded {len(df_summaries)} run summaries")

    # Generate plots
    print("\nGenerating plots...")

    plot_kl_trajectory(df_steps, output_path)
    plot_reward_trajectory(df_steps, output_path)
    plot_value_magnitude(df_steps, output_path)
    plot_clip_discretization(df_steps, output_path)
    plot_gradient_norms(df_steps, output_path)
    plot_stability_matrix(df_summaries, output_path)
    generate_summary_table(df_summaries, output_path)

    print(f"\nAll plots saved to {output_path}")


if __name__ == "__main__":
    main()
