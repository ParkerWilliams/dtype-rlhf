#!/usr/bin/env python3
"""
Phase -1: Static KL Forensics (Pennies, Compelling Figure)

Before any training, run a cheap static diagnostic that directly measures
precision effects on KL computation.

Two measurements:
1. "Simulated BF16": FP32 logits cast to BF16 and back (isolates rounding)
2. "True BF16": Full BF16 forward pass (includes kernel differences)

The gap between them shows kernel/accumulation contributions beyond simple rounding.

Usage:
    python scripts/static_kl_probe.py --num_prompts 10 --seq_lengths 64 128

This should complete in <2 minutes and produce plots showing BF16 vs FP32
precision differences.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.determinism import enforce_determinism, disable_flash_attention


def probe_kl_precision(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    seq_lengths: List[int],
    temperatures: List[float],
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Static probe of KL precision without any training.

    Two measurements:
    1. "Simulated BF16": FP32 logits cast to BF16 and back (isolates rounding)
    2. "True BF16": Full BF16 forward pass (includes kernel differences)

    Args:
        model: Causal LM model
        tokenizer: Tokenizer for encoding prompts
        prompts: List of prompt strings
        seq_lengths: Sequence lengths to test
        temperatures: Temperature values to test
        device: Device to run on

    Returns:
        DataFrame with precision metrics
    """
    results = []

    model.eval()

    for prompt_text in tqdm(prompts, desc="Prompts"):
        # Tokenize prompt
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=max(seq_lengths),
            truncation=True,
            padding="max_length",
        )

        for seq_len in seq_lengths:
            # Truncate to seq_len
            input_ids = encoded["input_ids"][:, :seq_len].to(device)

            if input_ids.shape[1] < seq_len:
                # Skip if prompt is too short
                continue

            for temp in temperatures:
                with torch.no_grad():
                    # === Baseline: Pure FP32 forward ===
                    model.float()
                    logits_fp32 = model(input_ids).logits.float()
                    lp_fp32 = F.log_softmax(logits_fp32 / temp, dim=-1)

                    # === Simulated BF16: Isolate rounding effects ===
                    # Same logits, just cast through BF16
                    logits_sim_bf16 = logits_fp32.bfloat16().float()
                    lp_sim = F.log_softmax(logits_sim_bf16 / temp, dim=-1)

                    diff_sim = lp_sim - lp_fp32

                    # === True BF16: Full forward in BF16 ===
                    model.bfloat16()
                    logits_true_bf16 = model(input_ids).logits.float()
                    lp_true = F.log_softmax(logits_true_bf16 / temp, dim=-1)

                    diff_true = lp_true - lp_fp32

                    # === Metrics ===
                    # Subsample for quantile computation (avoids "tensor too large" error)
                    # With vocab_size ~50k and seq_len ~512, full tensor is 25M+ elements
                    max_samples = 100_000
                    diff_sim_flat = diff_sim.abs().flatten()
                    diff_true_flat = diff_true.abs().flatten()

                    if diff_sim_flat.numel() > max_samples:
                        # Random subsample for quantile
                        indices = torch.randperm(diff_sim_flat.numel(), device=diff_sim_flat.device)[:max_samples]
                        sim_p99 = torch.quantile(diff_sim_flat[indices], 0.99).item()
                        true_p99 = torch.quantile(diff_true_flat[indices], 0.99).item()
                    else:
                        sim_p99 = torch.quantile(diff_sim_flat, 0.99).item()
                        true_p99 = torch.quantile(diff_true_flat, 0.99).item()

                    results.append({
                        "seq_len": seq_len,
                        "temperature": temp,

                        # Simulated BF16 (rounding only)
                        "sim_near_zero_mass": (diff_sim.abs() < 1e-6).float().mean().item(),
                        "sim_zero_exact_rate": (diff_sim == 0).float().mean().item(),
                        "sim_abs_err_mean": diff_sim.abs().mean().item(),
                        "sim_abs_err_max": diff_sim.abs().max().item(),
                        "sim_abs_err_p99": sim_p99,

                        # True BF16 forward (end-to-end)
                        "true_near_zero_mass": (diff_true.abs() < 1e-6).float().mean().item(),
                        "true_zero_exact_rate": (diff_true == 0).float().mean().item(),
                        "true_abs_err_mean": diff_true.abs().mean().item(),
                        "true_abs_err_max": diff_true.abs().max().item(),
                        "true_abs_err_p99": true_p99,

                        # Compare: how much does kernel path add to error?
                        "kernel_contribution": (
                            diff_true.abs().mean().item() - diff_sim.abs().mean().item()
                        ),
                    })

                    model.float()  # Reset for next iteration

    return pd.DataFrame(results)


def generate_prompts(num_prompts: int) -> List[str]:
    """Generate diverse prompts for testing."""
    prompts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter.",
        "In a world where technology advances rapidly, we must consider the implications.",
        "Mathematics provides the foundation for understanding complex systems.",
        "The history of science reveals patterns of discovery and innovation.",
        "Programming languages evolve to meet the changing needs of developers.",
        "Climate change poses significant challenges for future generations.",
        "Artificial intelligence continues to transform various industries.",
        "The universe contains billions of galaxies, each with countless stars.",
        "Philosophy explores fundamental questions about existence and knowledge.",
        "Music has the power to evoke deep emotions and memories.",
        "Literature reflects the human experience across cultures and eras.",
        "Economic systems shape how societies allocate resources.",
        "Biology reveals the intricate mechanisms of living organisms.",
        "Architecture combines art and engineering to create functional spaces.",
        "Psychology helps us understand human behavior and cognition.",
        "Chemistry explains the composition and properties of matter.",
        "Physics describes the fundamental forces governing the universe.",
        "Sociology examines social structures and human interactions.",
        "Medicine advances through research and clinical practice.",
        "Education empowers individuals to reach their potential.",
    ]

    # Repeat if needed
    while len(prompts) < num_prompts:
        prompts = prompts + prompts

    return prompts[:num_prompts]


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Plot 1: Error vs Sequence Length
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate by seq_len
    seq_agg = df.groupby("seq_len").agg({
        "sim_abs_err_mean": "mean",
        "true_abs_err_mean": "mean",
        "sim_abs_err_p99": "mean",
        "true_abs_err_p99": "mean",
    }).reset_index()

    # Mean error
    axes[0].plot(seq_agg["seq_len"], seq_agg["sim_abs_err_mean"], 'b-o', label="Simulated BF16 (rounding)")
    axes[0].plot(seq_agg["seq_len"], seq_agg["true_abs_err_mean"], 'r-s', label="True BF16 (full forward)")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title("Log-Prob Error vs Sequence Length")
    axes[0].legend()
    axes[0].set_yscale("log")

    # P99 error
    axes[1].plot(seq_agg["seq_len"], seq_agg["sim_abs_err_p99"], 'b-o', label="Simulated BF16")
    axes[1].plot(seq_agg["seq_len"], seq_agg["true_abs_err_p99"], 'r-s', label="True BF16")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("P99 Absolute Error")
    axes[1].set_title("Log-Prob Error P99 vs Sequence Length")
    axes[1].legend()
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_seq_len.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Error vs Temperature
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate by temperature
    temp_agg = df.groupby("temperature").agg({
        "sim_abs_err_mean": "mean",
        "true_abs_err_mean": "mean",
        "sim_near_zero_mass": "mean",
        "true_near_zero_mass": "mean",
    }).reset_index()

    # Mean error
    axes[0].plot(temp_agg["temperature"], temp_agg["sim_abs_err_mean"], 'b-o', label="Simulated BF16")
    axes[0].plot(temp_agg["temperature"], temp_agg["true_abs_err_mean"], 'r-s', label="True BF16")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title("Log-Prob Error vs Temperature")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Near-zero mass
    axes[1].plot(temp_agg["temperature"], temp_agg["sim_near_zero_mass"], 'b-o', label="Simulated BF16")
    axes[1].plot(temp_agg["temperature"], temp_agg["true_near_zero_mass"], 'r-s', label="True BF16")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Fraction of Errors < 1e-6")
    axes[1].set_title("Near-Zero Error Fraction vs Temperature")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_temperature.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Kernel contribution
    fig, ax = plt.subplots(figsize=(8, 5))

    kernel_agg = df.groupby("seq_len")["kernel_contribution"].mean().reset_index()

    ax.bar(kernel_agg["seq_len"].astype(str), kernel_agg["kernel_contribution"])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Kernel Contribution (True - Simulated)")
    ax.set_title("Additional Error from BF16 Kernel Path")
    ax.axhline(0, color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "kernel_contribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase -1: Static KL precision probe"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=10,
        help="Number of prompts to test",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.7, 1.0, 1.5],
        help="Temperature values to test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/static_kl_probe",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup
    print("=" * 60)
    print("Static KL Precision Probe")
    print("=" * 60)

    enforce_determinism(args.seed)
    disable_flash_attention()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,  # Start in FP32
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # Generate prompts
    prompts = generate_prompts(args.num_prompts)
    print(f"\nTesting {len(prompts)} prompts")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Temperatures: {args.temperatures}")

    # Run probe
    print("\nRunning precision probe...")
    df = probe_kl_precision(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        seq_lengths=args.seq_lengths,
        temperatures=args.temperatures,
        device=device,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "probe_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'probe_results.csv'}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nSimulated BF16 (rounding only):")
    print(f"  Mean abs error: {df['sim_abs_err_mean'].mean():.6e}")
    print(f"  Max abs error:  {df['sim_abs_err_max'].max():.6e}")
    print(f"  Near-zero frac: {df['sim_near_zero_mass'].mean():.4f}")

    print("\nTrue BF16 (full forward):")
    print(f"  Mean abs error: {df['true_abs_err_mean'].mean():.6e}")
    print(f"  Max abs error:  {df['true_abs_err_max'].max():.6e}")
    print(f"  Near-zero frac: {df['true_near_zero_mass'].mean():.4f}")

    print(f"\nKernel contribution: {df['kernel_contribution'].mean():.6e}")

    # Create plots
    print("\nGenerating plots...")
    create_plots(df, output_dir)

    # Save summary
    summary = {
        "model": args.base_model,
        "num_prompts": args.num_prompts,
        "seq_lengths": args.seq_lengths,
        "temperatures": args.temperatures,
        "sim_abs_err_mean": float(df["sim_abs_err_mean"].mean()),
        "sim_abs_err_max": float(df["sim_abs_err_max"].max()),
        "true_abs_err_mean": float(df["true_abs_err_mean"].mean()),
        "true_abs_err_max": float(df["true_abs_err_max"].max()),
        "kernel_contribution_mean": float(df["kernel_contribution"].mean()),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Probe complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
