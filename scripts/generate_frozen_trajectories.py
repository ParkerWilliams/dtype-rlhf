#!/usr/bin/env python3
"""
Phase 0: Generate Frozen Trajectories

Problem: Even with identical initial weights, BF16 and FP32 will sample different
tokens at step 3 due to arithmetic differences. Everything downstream changes,
making it impossible to attribute effects to precision vs. "different data."

Solution: Run FP32 baseline first, save trajectories, replay under all precision configs.

Usage:
    python scripts/generate_frozen_trajectories.py \
        --base_model EleutherAI/pythia-410m \
        --num_batches 100 \
        --batch_size 8 \
        --max_gen_length 128 \
        --output_dir results/frozen_trajectories/ \
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.determinism import enforce_determinism, disable_flash_attention
from src.algorithms.kl_utils import get_logprobs_for_tokens, compute_kl_with_diagnostics
from src.data.synthetic import get_synthetic_prompts


def generate_and_save_trajectories(
    model,
    tokenizer,
    prompts: list,
    num_batches: int,
    batch_size: int,
    max_gen_length: int,
    output_dir: Path,
    device: str,
):
    """
    Generate trajectories in FP32 and save for replay.

    Frozen trajectory format includes:
    - full_ids: Complete sequence (prompt + generated)
    - attention_mask_full: [B, T] mask
    - token_mask: [B, T-1] mask (pre-computed alignment)
    - logp_policy_fp32: Policy log-probs in FP32
    - logp_ref_fp32: Reference log-probs in FP32 (same as policy initially)
    - kl_penalty_fp32: KL penalty in FP32

    Args:
        model: FP32 model for generation
        tokenizer: Tokenizer
        prompts: List of prompt strings
        num_batches: Number of batches to generate
        batch_size: Batch size
        max_gen_length: Maximum generation length
        output_dir: Output directory
        device: Device to run on
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Use prompts cyclically
    prompt_idx = 0

    for batch_id in tqdm(range(num_batches), desc="Generating trajectories"):
        # Get prompts for this batch
        batch_prompts = []
        for _ in range(batch_size):
            batch_prompts.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1

        # Encode prompts
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        prompt_ids = encoded["input_ids"].to(device)
        prompt_mask = encoded["attention_mask"].to(device)
        prompt_len = prompt_ids.shape[1]

        # Generate continuations
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_gen_length,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Create full sequence and mask
        full_ids = generated
        full_len = full_ids.shape[1]

        # Create attention mask for full sequence
        # Pad tokens have attention_mask=0
        attention_mask_full = torch.ones_like(full_ids)
        attention_mask_full[full_ids == tokenizer.pad_token_id] = 0

        # Pre-compute token_mask (aligned to targets)
        token_mask = attention_mask_full[:, 1:].contiguous()

        # Compute FP32 log-probs
        with torch.no_grad():
            logits = model(full_ids).logits.float()
            logp_policy_fp32, _ = get_logprobs_for_tokens(
                logits, full_ids, attention_mask_full
            )

            # For initial trajectories, reference = policy
            logp_ref_fp32 = logp_policy_fp32.clone()

            # Compute KL penalty
            _, kl_penalty_fp32, _ = compute_kl_with_diagnostics(
                logp_policy_fp32, logp_ref_fp32, token_mask,
                compute_dtype=torch.float32
            )

        # Save trajectory
        trajectory = {
            "batch_id": batch_id,
            "prompt_len": prompt_len,
            "full_len": full_len,
            "prompt_ids": prompt_ids.cpu().tolist(),
            "full_ids": full_ids.cpu().tolist(),
            "attention_mask_full": attention_mask_full.cpu().tolist(),
            "token_mask": token_mask.cpu().tolist(),
            "logp_policy_fp32": logp_policy_fp32.cpu().tolist(),
            "logp_ref_fp32": logp_ref_fp32.cpu().tolist(),
            "kl_penalty_fp32": kl_penalty_fp32.cpu().item(),
        }

        # Save to file
        traj_path = output_dir / f"batch_{batch_id:05d}.json"
        with open(traj_path, "w") as f:
            json.dump(trajectory, f)

    # Save manifest
    manifest = {
        "num_batches": num_batches,
        "batch_size": batch_size,
        "max_gen_length": max_gen_length,
        "num_prompts": len(prompts),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {num_batches} trajectory batches to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate frozen trajectories for precision replay"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of trajectory batches to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=128,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/frozen_trajectories",
        help="Output directory",
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
    print("Frozen Trajectory Generation")
    print("=" * 60)

    enforce_determinism(args.seed)
    disable_flash_attention()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model in FP32
    print(f"\nLoading model in FP32: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # Get prompts
    prompts = get_synthetic_prompts(args.num_batches * args.batch_size)
    print(f"Using {len(prompts)} prompts")

    # Generate trajectories
    print(f"\nGenerating {args.num_batches} batches...")
    output_dir = Path(args.output_dir)

    generate_and_save_trajectories(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        max_gen_length=args.max_gen_length,
        output_dir=output_dir,
        device=device,
    )

    print("\n" + "=" * 60)
    print("Trajectory generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
