#!/usr/bin/env python3
"""
Phase 2: PPO Precision Sweep

Run PPO training under different precision configurations.

Usage:
    python scripts/run_ppo_sweep.py \
        --precision_config bf16_pure \
        --seed 0 \
        --max_steps 1000 \
        --output_dir results/

For synthetic task (fast iteration):
    python scripts/run_ppo_sweep.py \
        --precision_config bf16_pure \
        --use_synthetic_reward \
        --max_steps 100 \
        --seed 0
"""

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.determinism import enforce_determinism, disable_flash_attention
from src.utils.dtype_context import get_torch_dtype
from src.models.policy import PolicyWrapper, initialize_and_save_checkpoint
from src.models.value_head import InstrumentedValueHead
from src.models.reward import freeze_model, verify_frozen
from src.algorithms.ppo import PPOTrainer, TokenBatch
from src.algorithms.kl_utils import get_logprobs_for_tokens
from src.metrics.diagnostics import (
    RunSummary, FailureType,
    KL_COLLAPSE_THRESHOLD, GRAD_SPIKE_THRESHOLD,
)
from src.reporting.logger import ExperimentLogger, ExperimentManifest
from src.data.synthetic import SyntheticPreferenceTask, get_synthetic_prompts
from configs.precision_configs import get_precision_config, RLHFPrecisionConfig


class TrainingConfig:
    """Training hyperparameters."""
    def __init__(
        self,
        max_steps: int = 1000,
        batch_size: int = 8,
        max_seq_length: int = 256,
        max_gen_length: int = 128,
        learning_rate: float = 1e-5,
        kl_coef: float = 0.1,
        vf_coef: float = 0.5,
        clip_range: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 1.0,
    ):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_gen_length = max_gen_length
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm


def run_training(
    precision_config: RLHFPrecisionConfig,
    training_config: TrainingConfig,
    base_model: str,
    seed: int,
    output_dir: str,
    use_synthetic_reward: bool = True,
    checkpoint_path: str = None,
) -> RunSummary:
    """
    Run PPO training with specified precision config.

    Args:
        precision_config: Precision configuration
        training_config: Training hyperparameters
        base_model: Base model name
        seed: Random seed
        output_dir: Output directory
        use_synthetic_reward: Use synthetic reward (no RM needed)
        checkpoint_path: Optional FP32 init checkpoint

    Returns:
        RunSummary with results
    """
    run_id = f"{precision_config.name}_seed{seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize logging
    manifest = ExperimentManifest(output_dir, "rlhf_precision_forensics")
    manifest.register_run(run_id, precision_config.name, seed)

    logger = ExperimentLogger(output_dir, run_id)
    logger.log_config({
        "precision_config": asdict(precision_config),
        "training_config": vars(training_config),
        "base_model": base_model,
        "seed": seed,
        "use_synthetic_reward": use_synthetic_reward,
    })

    summary = RunSummary(
        run_id=run_id,
        config_name=precision_config.name,
        seed=seed,
        completed=False,
    )

    start_time = time.time()
    step = 0

    try:
        # Get dtypes
        policy_dtype = get_torch_dtype(precision_config.policy_dtype)
        value_dtype = get_torch_dtype(precision_config.value_head_dtype)
        kl_compute_dtype = get_torch_dtype(precision_config.kl_compute_dtype)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize or load policy
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading from checkpoint: {checkpoint_path}")
            policy = PolicyWrapper.from_checkpoint(
                checkpoint_path, base_model, target_dtype=policy_dtype
            )
        else:
            print(f"Initializing policy in {policy_dtype}")
            policy = PolicyWrapper.from_pretrained(
                base_model, dtype=policy_dtype
            )

        policy.model = policy.model.to(device)

        # Create reference (frozen copy of policy)
        reference = PolicyWrapper.from_pretrained(
            base_model, dtype=get_torch_dtype(precision_config.reference_dtype)
        )
        reference.model = freeze_model(reference.model.to(device))

        # Create value head
        value_head = InstrumentedValueHead(
            hidden_size=policy.hidden_size,
            dtype=value_dtype,
            fail_fast=True,
        ).to(device)

        # Verify frozen models
        verify_frozen(reference.model, "reference")

        # Create synthetic reward function
        reward_fn = SyntheticPreferenceTask(reward_fn="length")

        # Create PPO trainer
        class PPOConfig:
            def __init__(self, tc, pc):
                self.learning_rate = tc.learning_rate
                self.kl_coef = tc.kl_coef
                self.vf_coef = tc.vf_coef
                self.clip_range = tc.clip_range
                self.gamma = tc.gamma
                self.gae_lambda = tc.gae_lambda
                self.max_grad_norm = tc.max_grad_norm
                self.kl_compute_dtype = pc.kl_compute_dtype

        ppo_config = PPOConfig(training_config, precision_config)

        trainer = PPOTrainer(
            policy=policy,
            reference=reference,
            value_head=value_head,
            config=ppo_config,
        )

        # Get prompts
        prompts = get_synthetic_prompts(training_config.max_steps * training_config.batch_size)
        prompt_idx = 0

        # Memory guard after warmup
        print("Running warmup step...")
        warmup_batch = _create_batch(
            prompts[:training_config.batch_size],
            tokenizer,
            policy,
            reward_fn,
            training_config,
            device,
        )
        _, _ = trainer.ppo_step(warmup_batch)

        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            fraction = allocated / total
            print(f"Memory after warmup: {allocated:.1f}GB / {total:.1f}GB ({fraction*100:.1f}%)")

            if fraction > 0.85:
                raise MemoryError(f"Using {fraction*100:.1f}% of VRAM after warmup!")

        # Training loop
        print(f"\nStarting training for {training_config.max_steps} steps...")
        pbar = tqdm(range(training_config.max_steps), desc="Training")

        for step in pbar:
            # Get batch prompts
            batch_prompts = prompts[prompt_idx:prompt_idx + training_config.batch_size]
            prompt_idx = (prompt_idx + training_config.batch_size) % len(prompts)

            # Create batch with rollout
            raw_batch = _create_batch(
                batch_prompts,
                tokenizer,
                policy,
                reward_fn,
                training_config,
                device,
            )

            # PPO step
            _, diagnostics = trainer.ppo_step(raw_batch)

            # Log step metrics
            logger.log_step(step, diagnostics.to_dict())

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{diagnostics.total_loss:.3f}",
                "kl": f"{diagnostics.kl_penalty_mean:.2e}",
                "reward": f"{diagnostics.reward_mean:.3f}",
            })

            # Check failure conditions
            if math.isnan(diagnostics.total_loss):
                summary.failure_reason = FailureType.NAN_LOSS.value
                summary.failure_step = step
                summary.nan_count += 1
                break

            if diagnostics.kl_penalty_mean < KL_COLLAPSE_THRESHOLD:
                summary.kl_collapse_count += 1

            if diagnostics.policy_grad_norm > GRAD_SPIKE_THRESHOLD:
                summary.grad_spike_count += 1

            if diagnostics.log_ratio_clamped_frac > 0:
                summary.log_ratio_clamped_count += 1

        else:
            # Loop completed without break
            summary.completed = True

    except torch.cuda.OutOfMemoryError:
        summary.failure_reason = FailureType.MEMORY_OOM.value
        summary.failure_step = step

    except Exception as e:
        summary.failure_reason = FailureType.UNKNOWN.value
        summary.failure_step = step
        print(f"Unexpected error at step {step}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Finalize logging
        summary.total_steps = step + 1
        summary.wall_time_sec = time.time() - start_time
        summary.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        if step > 0:
            summary.final_reward = diagnostics.reward_mean
            summary.final_kl_mc = diagnostics.kl_mc_mean
            summary.final_kl_penalty = diagnostics.kl_penalty_mean
            summary.final_policy_loss = diagnostics.policy_loss
            summary.final_value_loss = diagnostics.value_loss
            summary.mean_step_time_ms = diagnostics.step_time_ms

        logger.finalize(summary)

        status = "completed" if summary.completed else f"failed:{summary.failure_reason}"
        manifest.complete_run(run_id, status, summary.failure_step)

        print(f"\nRun {run_id} finished: {status}")
        print(f"Summary written to {output_dir}/runs/{run_id}/run_summary.json")

    return summary


def _create_batch(
    prompts: list,
    tokenizer,
    policy: PolicyWrapper,
    reward_fn: SyntheticPreferenceTask,
    config: TrainingConfig,
    device: str,
) -> dict:
    """Create a training batch with generated completions."""
    # Encode prompts
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_seq_length,
    )

    prompt_ids = encoded["input_ids"].to(device)
    prompt_mask = encoded["attention_mask"].to(device)

    # Generate completions
    with torch.no_grad():
        generated = policy.model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=config.max_gen_length,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Create attention mask for full sequence
    attention_mask = torch.ones_like(generated)
    attention_mask[generated == tokenizer.pad_token_id] = 0

    # Get rewards
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    rewards = torch.tensor(
        reward_fn.get_batch_rewards(decoded),
        device=device,
        dtype=torch.float32,
    )

    # Get old_log_probs (CRITICAL: use same function as training)
    with torch.no_grad():
        logits = policy(generated)
        old_log_probs, _ = get_logprobs_for_tokens(
            logits, generated, attention_mask
        )

    return {
        "input_ids": generated,
        "attention_mask": attention_mask,
        "rewards": rewards,
        "old_log_probs": old_log_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="PPO Precision Sweep")
    parser.add_argument(
        "--precision_config",
        type=str,
        required=True,
        help="Precision config name",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
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
    parser.add_argument(
        "--use_synthetic_reward",
        action="store_true",
        help="Use synthetic reward function",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="FP32 init checkpoint path",
    )

    args = parser.parse_args()

    # Setup
    print("=" * 60)
    print(f"PPO Training: {args.precision_config}")
    print("=" * 60)

    enforce_determinism(args.seed)
    disable_flash_attention()

    # Get configs
    precision_config = get_precision_config(args.precision_config)
    training_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Precision config: {precision_config.name}")
    print(f"Policy dtype: {precision_config.policy_dtype}")
    print(f"KL compute dtype: {precision_config.kl_compute_dtype}")

    # Run training
    summary = run_training(
        precision_config=precision_config,
        training_config=training_config,
        base_model=args.base_model,
        seed=args.seed,
        output_dir=args.output_dir,
        use_synthetic_reward=True,  # Always use synthetic for now
        checkpoint_path=args.checkpoint_path,
    )

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Status: {'completed' if summary.completed else summary.failure_reason}")
    print(f"Steps: {summary.total_steps}")
    print(f"Wall time: {summary.wall_time_sec:.1f}s")
    print(f"Peak memory: {summary.peak_memory_gb:.1f}GB")


if __name__ == "__main__":
    main()
