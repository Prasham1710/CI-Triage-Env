"""TrainedPolicy — our GRPO-trained model. Same interface as ZeroShotPolicy."""

from __future__ import annotations

from ci_triage_env.training.baselines.zero_shot import ZeroShotPolicy


class TrainedPolicy(ZeroShotPolicy):
    """Our trained model. Same interface as zero-shot, loads from our checkpoint."""

    def __init__(self, checkpoint_path: str, system_prompt: str) -> None:
        super().__init__(checkpoint_path, system_prompt, name="trained_qwen3.5_4b")
