"""
Synthetic preference task for fast iteration.

Allows testing precision dynamics without conflating with reward model quality.
Uses simple, predictable reward functions.
"""

from typing import Callable, Optional


class SyntheticPreferenceTask:
    """
    Simple task: Generate continuations, reward based on
    measurable property (length, sentiment, keyword presence).

    Allows perfect reward model (no RM training needed).
    """

    def __init__(self, reward_fn: str = "length"):
        """
        Initialize synthetic task.

        Args:
            reward_fn: Reward function type:
                - "length": Reward longer responses
                - "keyword": Reward presence of target keyword
                - "custom": Use custom function
        """
        self.reward_fn_name = reward_fn
        self._reward_fn = self._get_reward_fn(reward_fn)

    def _get_reward_fn(self, name: str) -> Callable[[str], float]:
        """Get reward function by name."""
        if name == "length":
            return self._length_reward
        elif name == "keyword":
            return self._keyword_reward
        elif name == "short":
            return self._short_reward
        elif name == "diverse":
            return self._diverse_reward
        else:
            raise ValueError(f"Unknown reward function: {name}")

    def _length_reward(self, text: str) -> float:
        """Reward longer responses (simple, predictable)."""
        return len(text.split()) / 100.0

    def _keyword_reward(self, text: str) -> float:
        """Reward presence of target keyword."""
        keywords = ["therefore", "however", "consequently", "moreover"]
        text_lower = text.lower()
        return sum(1.0 for kw in keywords if kw in text_lower)

    def _short_reward(self, text: str) -> float:
        """Reward shorter responses (inverse of length)."""
        words = len(text.split())
        return max(0.0, 1.0 - words / 50.0)

    def _diverse_reward(self, text: str) -> float:
        """Reward vocabulary diversity."""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        unique = len(set(words))
        return unique / len(words)

    def get_reward(self, text: str) -> float:
        """
        Get reward for generated text.

        Args:
            text: Generated text

        Returns:
            Reward score
        """
        return self._reward_fn(text)

    def get_batch_rewards(self, texts: list) -> list:
        """
        Get rewards for a batch of texts.

        Args:
            texts: List of generated texts

        Returns:
            List of reward scores
        """
        return [self.get_reward(text) for text in texts]


def get_synthetic_prompts(num_prompts: int = 100) -> list:
    """
    Get a set of simple prompts for synthetic task.

    Args:
        num_prompts: Number of prompts to generate

    Returns:
        List of prompt strings
    """
    base_prompts = [
        "Explain the concept of",
        "Describe the process of",
        "What are the benefits of",
        "How does one go about",
        "The main reason for",
        "In order to understand",
        "The key insight is that",
        "When considering the topic of",
        "A common misconception about",
        "The fundamental principle behind",
    ]

    topics = [
        "machine learning",
        "software development",
        "data analysis",
        "system design",
        "problem solving",
        "team collaboration",
        "project management",
        "code review",
        "debugging",
        "optimization",
    ]

    prompts = []
    for i in range(num_prompts):
        base = base_prompts[i % len(base_prompts)]
        topic = topics[i % len(topics)]
        prompts.append(f"{base} {topic}:")

    return prompts
