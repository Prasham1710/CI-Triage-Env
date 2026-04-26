"""Reward-layer ablation runs for CI-Triage-Env.

Each ablation zeroes one reward component weight, runs 1000-step GRPO from the
SFT checkpoint, then evaluates against the held-out set. Results are saved to
data_artifacts/results/ablations.csv.

All GPU-heavy imports are lazy; this module is importable without GPU.
"""

from __future__ import annotations

from ci_triage_env.rewards.weights import REWARD_WEIGHTS
from ci_triage_env.training.eval import Evaluator
from ci_triage_env.training.grpo import run_grpo

ABLATIONS: dict[str, dict[str, float]] = {
    "no_diagnosis": {"diagnosis": 0.0},
    "no_action_quality": {"action_quality": 0.0},
    "no_investigation": {"investigation": 0.0},
    "no_anti_gaming": {"anti_gaming": 0.0},
    # counterfactual ablation deferred to v2 — its weight is already 0.0 in v1.
}


def run_ablation(
    name: str,
    weight_overrides: dict[str, float],
    total_steps: int = 1000,
    base_sft_checkpoint: str = "checkpoints/sft/",
    env_client=None,
):
    """Run one ablation: train GRPO with modified weights then evaluate.

    Args:
        name: Ablation name (used as output directory suffix and CSV column).
        weight_overrides: Keys/values to merge over REWARD_WEIGHTS (zeroed components).
        total_steps: GRPO training steps for this ablation (default 1000, not 3000).
        base_sft_checkpoint: SFT warmstart checkpoint to train from.
        env_client: Optional env client override (MockEnvClient for testing).

    Returns:
        pandas DataFrame with eval results; has an extra "ablation" column.
    """

    new_weights = {**REWARD_WEIGHTS, **weight_overrides}
    output_dir = f"checkpoints/ablation_{name}/"

    run_grpo(
        sft_checkpoint_dir=base_sft_checkpoint,
        output_dir=output_dir,
        total_steps=total_steps,
        weights_override=new_weights,
        env_client=env_client,
    )

    evaluator = Evaluator(trained_checkpoint=output_dir, env_client=env_client)
    evaluator.BASELINES = ["random", "heuristic", "trained"]
    df = evaluator.run_all(seeds=[1])
    df["ablation"] = name
    return df


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    import pandas as pd

    from ci_triage_env.training.curves import plot_ablation_summary

    parser = argparse.ArgumentParser(description="Run reward-layer ablation matrix")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft/")
    parser.add_argument("--output", default="data_artifacts/results/")
    args = parser.parse_args(argv)

    all_results = []
    for name, overrides in ABLATIONS.items():
        print(f"=== Ablation: {name} ===")
        df = run_ablation(
            name, overrides,
            total_steps=args.steps,
            base_sft_checkpoint=args.sft_checkpoint,
        )
        all_results.append(df)
        print(df.groupby("baseline")["diagnosis_correct"].mean())

    full = pd.concat(all_results, ignore_index=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "ablations.csv", index=False)
    print(f"\nAblations saved to {out / 'ablations.csv'}")

    plot_ablation_summary(full, output_dir=out / "plots/")


if __name__ == "__main__":
    main()
