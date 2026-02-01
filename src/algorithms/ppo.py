"""
Minimal PPO implementation with precision instrumentation.

This is a from-scratch implementation (~300 lines) designed for precision
forensics, NOT using trl.

PPO Implementation Contract:
1. 1 rollout batch -> 1 PPO update (no epochs/minibatching)
2. old_log_probs captured once per rollout, never recomputed
3. Rewards computed on-policy from reward model
4. Advantages normalized per-batch over all valid tokens (not per-sample)
5. Single optimizer for policy + value head
6. Gradient clipping on combined params (not per-module)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn

from .kl_utils import (
    get_logprobs_for_tokens,
    compute_kl_with_diagnostics,
    safe_mean,
    safe_norm,
    compute_reward_resolution,
)
from ..metrics.diagnostics import StepDiagnostics


@dataclass
class TokenBatch:
    """
    Single source of truth for aligned tensors.

    Create ONCE per batch. Every function takes this, not raw tensors.
    Prevents the #1 RLHF PPO bug: mask silently misaligned.

    CRITICAL MASK DISTINCTION:
    - attention_mask: [B, T] - Full sequence mask, needed for get_logprobs_for_tokens()
    - token_mask: [B, T-1] - Aligned to targets, for KL/loss/GAE computations
    """
    # Original inputs (shape [B, T])
    input_ids: torch.Tensor
    attention_mask: torch.Tensor  # [B, T] - for get_logprobs_for_tokens()

    # Aligned to target positions (shape [B, T-1])
    targets: torch.Tensor
    token_mask: torch.Tensor      # [B, T-1] - for all per-token computations

    # Rewards (sequence-level, shape [B])
    rewards: torch.Tensor

    @classmethod
    def from_batch(cls, input_ids: torch.Tensor, attention_mask: torch.Tensor, rewards: torch.Tensor):
        """Create aligned batch from raw inputs."""
        return cls(
            input_ids=input_ids,
            attention_mask=attention_mask,              # [B, T] - keep full mask
            targets=input_ids[:, 1:].contiguous(),      # [B, T-1]
            token_mask=attention_mask[:, 1:].contiguous(),  # [B, T-1]
            rewards=rewards,
        )

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def seq_len(self) -> int:
        """Length of target sequence (T-1)."""
        return self.targets.shape[1]


def compute_returns_and_advantages(
    rewards: torch.Tensor,      # [B] - sequence-level rewards
    values: torch.Tensor,       # [B, T-1] - token-level value predictions
    token_mask: torch.Tensor,   # [B, T-1]
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute returns and GAE advantages with terminal reward.

    Reward is applied only at the LAST valid token (end of generation).
    All other positions have reward=0.

    CRITICAL:
    - lastgaelam must be [B], not scalar
    - Must reset/zero at padding positions to prevent leakage
    - Only place terminal reward for samples that have valid tokens
    - next_value must be masked: values[:, t+1] * mask_f[:, t+1]

    Args:
        rewards: Sequence-level rewards [B]
        values: Token-level value predictions [B, T-1]
        token_mask: Mask for valid tokens [B, T-1]
        gamma: Discount factor (default 1.0 for RLHF)
        lam: GAE lambda parameter

    Returns:
        returns: Computed returns [B, T-1]
        advantages: GAE advantages [B, T-1]
    """
    B, T = values.shape
    device = values.device

    # Find last valid token position for each sequence
    lengths = token_mask.sum(dim=1).long()  # [B]
    has_tokens = lengths > 0                 # [B] - samples with at least one valid token
    terminal_idx = (lengths - 1).clamp(min=0)  # [B] - clamp for safety

    # Create per-token rewards: zero everywhere except terminal
    # CRITICAL: Only assign rewards for samples that have valid tokens
    # CRITICAL: Use FP32 for rewards_per_token since GAE is computed in FP32
    rewards_per_token = torch.zeros(B, T, device=device, dtype=torch.float32)
    valid_batch_idx = torch.arange(B, device=device)[has_tokens]
    rewards_per_token[valid_batch_idx, terminal_idx[has_tokens]] = rewards[has_tokens].float()

    # Standard GAE computation (in FP32, mask-aware)
    advantages = torch.zeros(B, T, device=device, dtype=torch.float32)
    mask_f = token_mask.float()

    # CRITICAL: lastgaelam must be per-sample vector, not scalar!
    lastgaelam = torch.zeros(B, device=device, dtype=torch.float32)

    for t in reversed(range(T)):
        m_t = mask_f[:, t]  # [B] mask for this timestep

        # Per-sample terminal handling: if next position is invalid, next_value=0
        # CRITICAL: Must mask next_value, not just multiply result by m_t
        if t < T - 1:
            next_value = values[:, t + 1].float() * mask_f[:, t + 1]
        else:
            next_value = torch.zeros(B, device=device, dtype=torch.float32)

        delta = rewards_per_token[:, t].float() + gamma * next_value - values[:, t].float()

        # CRITICAL: Zero out updates where mask is 0 to prevent padding leakage
        # This effectively resets lastgaelam at padding boundaries
        lastgaelam = (delta + gamma * lam * lastgaelam) * m_t
        advantages[:, t] = lastgaelam

    returns = advantages + values.float()

    return returns, advantages


def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize advantages to zero mean and unit variance.

    SCOPE: Per-batch normalization over ALL valid tokens (masked), NOT per-sample.

    CRITICAL: Must be done in FP32! In BF16, if advantages are small,
    std() loses precision and division by small epsilon amplifies noise.

    Args:
        advantages: Advantage tensor [B, T-1]
        mask: Token mask [B, T-1]

    Returns:
        Normalized advantages [B, T-1]
    """
    adv = advantages.float()
    mask_f = mask.float()

    # CRITICAL: Clamp to prevent divide-by-zero
    valid_count = mask_f.sum().clamp_min(1.0)

    # Masked mean
    mean = (adv * mask_f).sum() / valid_count

    # Masked std
    sq_diff = ((adv - mean) * mask_f) ** 2
    var = sq_diff.sum() / valid_count
    std = var.sqrt()

    # Normalize (FP32 throughout)
    normalized = (adv - mean) / (std + 1e-8)

    # Mask out invalid positions
    return normalized * mask_f


def compute_ppo_loss_with_clip_diagnostics(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,  # REQUIRED
    clip_range: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PPO clipped objective with precision diagnostics.

    Args:
        log_probs: Current policy log-probs [B, T-1]
        old_log_probs: Old policy log-probs [B, T-1]
        advantages: Normalized advantages [B, T-1]
        token_mask: Mask for valid tokens [B, T-1]
        clip_range: PPO clip epsilon (default 0.2)

    Returns:
        policy_loss: Scalar loss
        diagnostics: Dict of clip/ratio diagnostics
    """
    ratio = (log_probs - old_log_probs).exp()

    # Clipped and unclipped objectives
    obj_unclipped = ratio * advantages
    obj_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages

    # Take pessimistic bound (masked mean in FP32)
    policy_obj = torch.min(obj_unclipped, obj_clipped)
    policy_loss = -safe_mean(policy_obj, token_mask)

    # PRECISION DIAGNOSTICS (only over valid tokens)
    valid_ratios = ratio[token_mask.bool()]
    clipped_mask = (valid_ratios < 1 - clip_range) | (valid_ratios > 1 + clip_range)

    # Ratios within clip range
    in_range_mask = (valid_ratios >= 1 - clip_range) & (valid_ratios <= 1 + clip_range)
    in_range = valid_ratios[in_range_mask]

    diagnostics = {
        "clip_fraction": clipped_mask.float().mean().item() if len(valid_ratios) > 0 else 0.0,
        "ratio_mean": valid_ratios.mean().item() if len(valid_ratios) > 0 else 1.0,
        "ratio_std": valid_ratios.std().item() if len(valid_ratios) > 1 else 0.0,
        "ratio_min": valid_ratios.min().item() if len(valid_ratios) > 0 else 1.0,
        "ratio_max": valid_ratios.max().item() if len(valid_ratios) > 0 else 1.0,

        # EMPIRICAL discretization measurement
        # Low count indicates discretized optimization landscape
        # NOTE: .float() required because unique() not implemented for BF16
        "ratio_unique_in_range": in_range.float().unique().numel() if len(in_range) > 0 else 0,
        "ratio_in_range_count": len(in_range),
    }

    return policy_loss, diagnostics


class PPOTrainer:
    """
    Minimal PPO trainer with precision instrumentation.

    Key invariants:
    - 1 rollout batch -> 1 PPO update (no epochs/minibatching)
    - Single optimizer for policy + value head
    - Gradient clipping on combined params
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        value_head: nn.Module,
        reward_model: Optional[nn.Module] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: Policy model (trainable)
            reference: Reference model (frozen)
            value_head: Value head (trainable)
            reward_model: Optional reward model (frozen)
            config: Training configuration
        """
        self.policy = policy
        self.reference = reference
        self.value_head = value_head
        self.reward_model = reward_model

        # Default config values
        self.learning_rate = getattr(config, 'learning_rate', 1e-5)
        self.kl_coef = getattr(config, 'kl_coef', 0.1)
        self.vf_coef = getattr(config, 'vf_coef', 0.5)
        self.clip_range = getattr(config, 'clip_range', 0.2)
        self.gamma = getattr(config, 'gamma', 1.0)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        self.kl_compute_dtype = getattr(config, 'kl_compute_dtype', torch.float32)

        # Convert string dtype to torch.dtype if needed
        if isinstance(self.kl_compute_dtype, str):
            dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
            self.kl_compute_dtype = dtype_map.get(self.kl_compute_dtype, torch.float32)

        # Single optimizer for policy + value head
        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.step_count = 0

    def ppo_step(self, raw_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, StepDiagnostics]:
        """
        Single PPO update step with full precision instrumentation.

        Args:
            raw_batch: Dict with keys:
                - input_ids: [B, T]
                - attention_mask: [B, T]
                - rewards: [B] (sequence-level)
                - old_log_probs: [B, T-1] (from rollout)

        Returns:
            total_loss: Scalar loss
            diagnostics: StepDiagnostics dataclass
        """
        import time
        step_start = time.time()

        # === Gradient hygiene: zero gradients at start ===
        self.optimizer.zero_grad(set_to_none=True)

        # Create aligned batch (SINGLE SOURCE OF TRUTH for masks)
        batch = TokenBatch.from_batch(
            raw_batch["input_ids"],
            raw_batch["attention_mask"],
            raw_batch["rewards"]
        )

        # 1. Get current policy log probs
        #    CRITICAL: Pass batch.attention_mask [B, T] to get_logprobs_for_tokens()
        logits = self.policy(batch.input_ids)
        logp_taken, _ = get_logprobs_for_tokens(
            logits, batch.input_ids, batch.attention_mask  # [B, T] mask
        )
        # logp_taken is now [B, T-1], aligned with batch.token_mask

        # Get hidden states for value head
        hidden_states = self.policy.get_hidden_states(batch.input_ids)
        # Cast to value head dtype for mixed-precision configs (e.g., BF16 policy + FP32 value head)
        hidden_states = hidden_states.to(self.value_head.head.weight.dtype)
        values = self.value_head(hidden_states)[:, :-1]  # [B, T-1] - ALIGNED

        # 2. Get reference log probs (frozen, same token selection)
        with torch.no_grad():
            ref_logits = self.reference(batch.input_ids)
            logp_ref, _ = get_logprobs_for_tokens(
                ref_logits, batch.input_ids, batch.attention_mask  # [B, T] mask
            )

        # 3. KL computation
        #    CRITICAL: Pass batch.token_mask [B, T-1] for per-token computations
        kl_mc, kl_penalty, kl_diagnostics = compute_kl_with_diagnostics(
            logp_taken, logp_ref, batch.token_mask,  # [B, T-1] mask
            compute_dtype=self.kl_compute_dtype
        )

        # 4. Compute returns and advantages (FP32, uses batch.token_mask)
        returns, advantages = compute_returns_and_advantages(
            batch.rewards, values, batch.token_mask,
            gamma=self.gamma, lam=self.gae_lambda
        )

        # 5. Normalize advantages (MUST BE FP32 - precision trap!)
        advantages = normalize_advantages(advantages, batch.token_mask)

        # 6. PPO loss with clip diagnostics (uses batch.token_mask)
        # CRITICAL: old_log_probs must be gathered with same function as logp_taken
        policy_loss, clip_diagnostics = compute_ppo_loss_with_clip_diagnostics(
            logp_taken, raw_batch["old_log_probs"],
            advantages, batch.token_mask, clip_range=self.clip_range
        )

        # 7. Value loss (masked, FP32 reduction)
        value_loss = safe_mean((values.float() - returns) ** 2, batch.token_mask)

        # 8. Total loss with KL penalty (use penalty, not MC estimate)
        total_loss = (
            policy_loss
            + self.vf_coef * value_loss
            + self.kl_coef * kl_penalty
        )

        # 9. Collect value head diagnostics
        value_diagnostics = self.value_head.get_diagnostics(values)
        reward_diagnostics = compute_reward_resolution(batch.rewards)

        # 10. Backward and compute gradient norms (FP32)
        total_loss.backward()

        # Compute gradient norms BEFORE clipping (for diagnostics)
        policy_grads = [p.grad.flatten() for p in self.policy.parameters() if p.grad is not None]
        value_grads = [p.grad.flatten() for p in self.value_head.parameters() if p.grad is not None]

        policy_grad_norm = safe_norm(torch.cat(policy_grads)) if policy_grads else torch.tensor(0.0)
        value_grad_norm = safe_norm(torch.cat(value_grads)) if value_grads else torch.tensor(0.0)

        # === Gradient hygiene: clip and step ===
        # CRITICAL: Clip once over ALL params, not separately per module
        all_params = list(self.policy.parameters()) + list(self.value_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.max_grad_norm)
        self.optimizer.step()

        # 11. Build diagnostics
        step_time_ms = (time.time() - step_start) * 1000
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        diagnostics = StepDiagnostics(
            step=self.step_count,
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            total_loss=total_loss.item(),
            # KL metrics
            kl_mc_mean=kl_diagnostics.get("kl_mc_mean", 0.0),
            kl_penalty_mean=kl_diagnostics.get("kl_penalty_mean", 0.0),
            kl_penalty_p99=kl_diagnostics.get("kl_penalty_p99", 0.0),
            kl_penalty_max=kl_diagnostics.get("kl_penalty_max", 0.0),
            log_ratio_abs_max=kl_diagnostics.get("log_ratio_abs_max", 0.0),
            log_ratio_near_zero_frac=kl_diagnostics.get("log_ratio_near_zero_frac", 0.0),
            log_ratio_clamped_frac=kl_diagnostics.get("log_ratio_clamped_frac", 0.0),
            valid_token_count=kl_diagnostics.get("valid_token_count", 0),
            # Value head
            value_max=value_diagnostics.get("value_max", 0.0),
            value_headroom=value_diagnostics.get("value_headroom", 0.0),
            value_quant_step_median=value_diagnostics.get("value_quant_step_median", 0.0),
            value_stagnation_rate=value_diagnostics.get("value_stagnation_rate", 0.0),
            # PPO clip
            clip_fraction=clip_diagnostics.get("clip_fraction", 0.0),
            ratio_unique_in_range=clip_diagnostics.get("ratio_unique_in_range", 0),
            # Reward
            reward_mean=batch.rewards.mean().item(),
            reward_resolution_median=reward_diagnostics.get("reward_resolution_median", float('inf')),
            reward_resolution_p10=reward_diagnostics.get("reward_resolution_p10", float('inf')),
            # Gradients
            policy_grad_norm=policy_grad_norm.item(),
            value_grad_norm=value_grad_norm.item(),
            # System
            memory_gb=memory_gb,
            step_time_ms=step_time_ms,
        )

        self.step_count += 1

        return total_loss, diagnostics

    def collect_rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Collect rollout data including old_log_probs.

        CRITICAL: old_log_probs must be gathered with SAME function as training.

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Attention mask [B, T]

        Returns:
            Dict with old_log_probs and token_mask
        """
        with torch.no_grad():
            logits = self.policy(input_ids)
            # CRITICAL: Use SAME function as training
            old_log_probs, token_mask = get_logprobs_for_tokens(
                logits, input_ids, attention_mask
            )

        return {
            "old_log_probs": old_log_probs,  # [B, T-1], aligned to targets
            "token_mask": token_mask,         # [B, T-1], for verification
        }
