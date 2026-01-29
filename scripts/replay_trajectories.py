#!/usr/bin/env python3
"""
Phase 0: Replay Frozen Trajectories

Replay trajectories under each precision config to measure precision errors
against FP32 reference.

This gives clean precision forensics separate from "policy changed the world."

Usage:
    python scripts/replay_trajectories.py \
        --precision_config bf16_pure \
        --trajectories_dir results/frozen_trajectories/ \
        --output_dir results/replay/bf16_pure/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.determinism import enforce_determinism, disable_flash_attention
from src.utils.dtype_context import get_torch_dtype
from src.algorithms.kl_utils import (
    get_logprobs_for_tokens,
    compute_kl_with_diagnostics,
    compute_precision_error,
)
from configs.precision_configs import get_precision_config


def load_trajectory(path: Path) -> dict:
    """Load trajectory from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Convert lists back to tensors
    data["full_ids"] = torch.tensor(data["full_ids"])
    data["attention_mask_full"] = torch.tensor(data["attention_mask_full"])
    data["token_mask"] = torch.tensor(data["token_mask"])
    data["logp_policy_fp32"] = torch.tensor(data["logp_policy_fp32"])
    data["logp_ref_fp32"] = torch.tensor(data["logp_ref_fp32"])

    return data


def replay_batch(
    batch: dict,
    model,
    config,
    device: str,
) -> dict:
    """
    Replay frozen trajectory under specific precision config.

    Computes errors against FP32 reference for EACH component separately.

    Args:
        batch: Loaded trajectory dict
        model: Model in config's dtype
        config: Precision config
        device: Device

    Returns:
        Dict of precision error metrics
    """
    # Load inputs and FP32 ground truth
    full_ids = batch["full_ids"].to(device)
    attention_mask_full = batch["attention_mask_full"].to(device)
    token_mask = batch["token_mask"].to(device)
    logp_policy_fp32 = batch["logp_policy_fp32"].to(device)
    logp_ref_fp32 = batch["logp_ref_fp32"].to(device)

    # Get compute dtype
    kl_compute_dtype = get_torch_dtype(config.kl_compute_dtype)

    with torch.no_grad():
        # === Policy forward under this config ===
        logits_policy = model(full_ids).logits

        # CRITICAL: Pass attention_mask_full [B, T], not token_mask [B, T-1]
        logp_policy_cfg, token_mask_runtime = get_logprobs_for_tokens(
            logits_policy, full_ids, attention_mask_full
        )

        # Sanity check: runtime mask should match stored token_mask
        if not torch.allclose(token_mask_runtime.float(), token_mask.float()):
            print("Warning: Mask mismatch! Check trajectory generation.")

        # For replay, reference = same model (no separate reference)
        logp_ref_cfg = logp_policy_cfg.clone()

        # === KL penalty under this config ===
        _, kl_penalty_cfg, kl_diag = compute_kl_with_diagnostics(
            logp_policy_cfg, logp_ref_cfg, token_mask,
            compute_dtype=kl_compute_dtype
        )

    # === Precision errors vs FP32 for each component ===
    err_policy = compute_precision_error(logp_policy_cfg, logp_policy_fp32)
    err_ref = compute_precision_error(logp_ref_cfg, logp_ref_fp32)

    # For scalar KL penalty, expand to tensor for comparison
    kl_penalty_fp32 = batch["kl_penalty_fp32"]
    if isinstance(kl_penalty_cfg, torch.Tensor) and kl_penalty_cfg.dim() == 0:
        kl_penalty_cfg = kl_penalty_cfg.item()

    return {
        # KL diagnostics from this config
        **{f"kl_{k}": v for k, v in kl_diag.items()},

        # Policy precision errors
        **{f"policy_{k}": v for k, v in err_policy.items()},

        # Reference precision errors
        **{f"ref_{k}": v for k, v in err_ref.items()},

        # KL penalty error
        "kl_penalty_cfg": kl_penalty_cfg if isinstance(kl_penalty_cfg, float) else kl_penalty_cfg.item(),
        "kl_penalty_fp32": kl_penalty_fp32,
        "kl_penalty_abs_err": abs(
            (kl_penalty_cfg if isinstance(kl_penalty_cfg, float) else kl_penalty_cfg.item())
            - kl_penalty_fp32
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay frozen trajectories under precision config"
    )
    parser.add_argument(
        "--precision_config",
        type=str,
        required=True,
        help="Precision config name (e.g., bf16_pure, fp32_baseline)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="results/frozen_trajectories",
        help="Directory containing frozen trajectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/replay/<config>/)",
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
    print(f"Trajectory Replay: {args.precision_config}")
    print("=" * 60)

    enforce_determinism(args.seed)
    disable_flash_attention()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Get config
    config = get_precision_config(args.precision_config)
    policy_dtype = get_torch_dtype(config.policy_dtype)
    print(f"Policy dtype: {policy_dtype}")
    print(f"KL compute dtype: {config.kl_compute_dtype}")

    # Load model in config dtype
    print(f"\nLoading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=policy_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    # Load trajectories
    trajectories_dir = Path(args.trajectories_dir)
    manifest_path = trajectories_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: No manifest found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    num_batches = manifest["num_batches"]
    print(f"\nReplaying {num_batches} trajectory batches...")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/replay/{args.precision_config}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Replay all batches
    all_results = []

    for batch_id in tqdm(range(num_batches), desc="Replaying"):
        traj_path = trajectories_dir / f"batch_{batch_id:05d}.json"
        batch = load_trajectory(traj_path)

        results = replay_batch(batch, model, config, device)
        results["batch_id"] = batch_id
        all_results.append(results)

    # Save results
    with open(output_dir / "replay_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Compute summary statistics
    import numpy as np

    policy_errs = [r["policy_abs_err_mean"] for r in all_results]
    ref_errs = [r["ref_abs_err_mean"] for r in all_results]
    kl_errs = [r["kl_penalty_abs_err"] for r in all_results]

    summary = {
        "config": args.precision_config,
        "num_batches": num_batches,
        "policy_abs_err_mean": float(np.mean(policy_errs)),
        "policy_abs_err_max": float(np.max(policy_errs)),
        "ref_abs_err_mean": float(np.mean(ref_errs)),
        "ref_abs_err_max": float(np.max(ref_errs)),
        "kl_penalty_abs_err_mean": float(np.mean(kl_errs)),
        "kl_penalty_abs_err_max": float(np.max(kl_errs)),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Policy log-prob mean error: {summary['policy_abs_err_mean']:.6e}")
    print(f"Reference log-prob mean error: {summary['ref_abs_err_mean']:.6e}")
    print(f"KL penalty mean error: {summary['kl_penalty_abs_err_mean']:.6e}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
