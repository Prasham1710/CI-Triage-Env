"""GRPO training script — wraps TRL's GRPOTrainer with multi-turn env rollout.

All GPU-heavy imports (trl, torch) are lazy so the module is importable without GPU.
"""

from __future__ import annotations

from pathlib import Path

from ci_triage_env.training.rollout import TrainingRollout
from ci_triage_env.training.sft import load_model_for_sft

GRPO_HYPERPARAMS: dict = {
    "learning_rate": 5e-6,
    "kl_coef": 0.04,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_prompt_length": 4096,
    "max_completion_length": 1024,
    "num_generations": 8,
    "temperature": 0.7,
    "top_p": 0.9,
    "logging_steps": 5,
    "save_steps": 200,
    "report_to": "wandb",
}


def run_grpo(
    sft_checkpoint_dir: str,
    output_dir: str,
    total_steps: int = 3000,
    env_client=None,
    env_url: str = "http://localhost:8000",
    scenarios_train_path: str = "data_artifacts/scenarios/train",
    hyperparams: dict | None = None,
    weights_override: dict | None = None,
) -> str:
    """Launch GRPO fine-tuning from an SFT checkpoint.

    Args:
        sft_checkpoint_dir: Path to the LoRA checkpoint produced by run_sft().
        output_dir: Where to write the GRPO-finetuned adapter.
        total_steps: Maximum training steps.
        env_client: Optional pre-built env client (MockEnvClient for testing).
            If None, an EnvClient is created pointing at env_url.
        env_url: URL of the running env server (used only when env_client is None).
        scenarios_train_path: Directory containing train split scenario JSON files.
        hyperparams: Override specific GRPO hyperparameters. Merged over GRPO_HYPERPARAMS.
        weights_override: Override reward component weights passed to CompositeReward.
            Used by ablation runs to zero out individual reward terms.

    Returns:
        output_dir path as a string.
    """
    from trl import GRPOConfig  # type: ignore[import]

    from ci_triage_env.training.custom_trainer import GRPOTrainerForToolUse

    if env_client is None:
        from ci_triage_env.training.mock_env_client import MockEnvClient
        env_client = MockEnvClient(scenarios_dir=scenarios_train_path)

    train_dir = Path(scenarios_train_path)
    scenario_ids = [p.stem for p in train_dir.rglob("*.json")] if train_dir.exists() else []

    max_turns = hp.pop("max_turns", 4)   # short episodes for faster GRPO
    rollout = TrainingRollout(
        env_client=env_client,
        scenarios_train=scenario_ids,
        weights=weights_override,
        max_turns=max_turns,
    )

    model, tokenizer = load_model_for_sft(model_name=sft_checkpoint_dir)

    hp = dict(GRPO_HYPERPARAMS)
    if hyperparams:
        hp.update(hyperparams)

    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=total_steps,
        **hp,
    )

    trainer = GRPOTrainerForToolUse(
        model=model,
        tokenizer=tokenizer,
        rollout_fn=rollout,
        args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
