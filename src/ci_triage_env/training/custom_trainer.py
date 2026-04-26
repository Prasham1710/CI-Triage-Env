"""GRPOTrainerForToolUse — multi-turn env rollout inside GRPO training loop.

Inherits from TRL's GRPOTrainer and overrides _generate_and_score to drive
full env episodes instead of single-turn completions.

All heavy imports (trl, torch) are lazy so the module is importable without GPU.
"""

from __future__ import annotations


class GRPOTrainerForToolUse:
    """GRPO trainer that uses multi-turn env rollouts instead of single-turn generate.

    Args:
        model: Loaded model (unsloth/PEFT).
        tokenizer: Matching tokenizer.
        rollout_fn: A TrainingRollout instance.
        args: GRPOConfig instance.
        **kwargs: Passed to GRPOTrainer.__init__.
    """

    def __init__(self, model, tokenizer, rollout_fn, args, **kwargs):
        from trl import GRPOTrainer  # type: ignore[import]

        # GRPOTrainer requires a reward_funcs argument; we provide a placeholder
        # since we override _generate_and_score below.
        super_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "args": args,
            **kwargs,
        }
        # Initialise without calling GRPOTrainer.train (our override replaces it).
        GRPOTrainer.__init__(self, **super_kwargs)
        self.rollout_fn = rollout_fn

    def _generate_and_score(self, batch):
        """Override: collect multi-turn rollouts; return (completions, rewards).

        For each prompt in the batch, run `num_generations` rollouts.
        The "completion" is the concatenation of all assistant turns in the episode.
        """
        rewards: list[float] = []
        completions: list[str] = []

        n_gens = self.args.num_generations
        n_prompts = len(batch.get("input_ids", [batch])) if isinstance(batch, dict) else 1
        total_rollouts = n_prompts * n_gens

        for _ in range(total_rollouts):
            result = self.rollout_fn(self.model, self.processing_class)
            rewards.append(result["reward"])
            assistant_text = "\n".join(
                m["content"]
                for m in result["messages"]
                if m["role"] == "assistant"
            )
            completions.append(assistant_text)

        return completions, rewards

    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """Delegate to GRPOTrainer.train after setting up rollout."""
        from trl import GRPOTrainer  # type: ignore[import]
        return GRPOTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def save_model(self, output_dir: str, **kwargs):
        from trl import GRPOTrainer  # type: ignore[import]
        return GRPOTrainer.save_model(self, output_dir, **kwargs)
