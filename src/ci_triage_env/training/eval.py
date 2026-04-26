"""Multi-baseline evaluation pipeline for CI-Triage-Env.

Run with:
    python -m ci_triage_env.training.eval --output data_artifacts/results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ci_triage_env.rewards.composite import compute_reward
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.training.baselines.heuristic_policy import HeuristicPolicy
from ci_triage_env.training.baselines.random_policy import RandomPolicy
from ci_triage_env.training.rollout import _SYSTEM_PROMPT_TEMPLATE

_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(tools="<see env for tool list>")


def load_scenario(
    scenario_id: str,
    scenarios_dir: str = "data_artifacts/scenarios",
) -> Scenario:
    """Load a Scenario by ID, searching common split subdirectories."""
    candidates = [
        Path(scenarios_dir) / f"{scenario_id}.json",
        Path(scenarios_dir) / "train" / f"{scenario_id}.json",
        Path(scenarios_dir) / "val" / f"{scenario_id}.json",
        Path(scenarios_dir) / "held_out" / f"{scenario_id}.json",
    ]
    for p in candidates:
        if p.exists():
            return Scenario.model_validate_json(p.read_text())

    # Fallback: generate a plausible mock from the scenario_id prefix
    family = scenario_id.split("-")[0]
    _valid_families = {
        "real_bug", "race_flake", "timing_flake",
        "infra_network", "infra_resource", "dependency_drift", "ambiguous",
    }
    from ci_triage_env.mock.scenario import make_mock_scenario
    return make_mock_scenario(family=family if family in _valid_families else "real_bug")


class Evaluator:
    """Run 5-baseline evaluation matrix over a held-out scenario set."""

    BASELINES = [
        "random",
        "heuristic",
        "qwen3.5_4b_zero_shot",
        "qwen3.5_9b_zero_shot",
        "trained",
    ]

    def __init__(
        self,
        eval_set_path: str = "data_artifacts/scenarios/held_out/",
        env_url: str = "http://localhost:8000",
        trained_checkpoint: str = "checkpoints/grpo_full/",
        env_client=None,
    ) -> None:
        if env_client is not None:
            self.env = env_client
        else:
            from ci_triage_env.training.env_client import EnvClient
            self.env = EnvClient(env_url)
        self.eval_scenarios = list(Path(eval_set_path).glob("*.json"))
        self.trained_checkpoint = trained_checkpoint

    def run_all(self, seeds: list[int] | None = None):
        import pandas as pd

        if seeds is None:
            seeds = [1, 2, 3]
        rows = []
        for baseline_name in self.BASELINES:
            policy = self._build(baseline_name)
            for scenario_path in self.eval_scenarios:
                scenario_id = scenario_path.stem
                for seed in seeds:
                    rows.append(self._run_one(policy, scenario_id, seed))
        return pd.DataFrame(rows)

    def _build(self, name: str):
        if name == "random":
            return RandomPolicy()
        if name == "heuristic":
            return HeuristicPolicy()
        if name == "qwen3.5_4b_zero_shot":
            from ci_triage_env.training.baselines.zero_shot import ZeroShotPolicy
            return ZeroShotPolicy("Qwen/Qwen3.5-4B", _SYSTEM_PROMPT)
        if name == "qwen3.5_9b_zero_shot":
            from ci_triage_env.training.baselines.zero_shot import ZeroShotPolicy
            return ZeroShotPolicy(
                "Qwen/Qwen3.5-9B", _SYSTEM_PROMPT, name="zero_shot_Qwen3.5-9B"
            )
        if name == "trained":
            from ci_triage_env.training.baselines.trained import TrainedPolicy
            return TrainedPolicy(self.trained_checkpoint, _SYSTEM_PROMPT)
        raise ValueError(f"Unknown baseline: {name}")

    def _run_one(self, policy, scenario_id: str, seed: int) -> dict:
        obs = self.env.reset(scenario_id=scenario_id, seed_override=seed)
        episode_id = obs.episode_id
        history: list = []
        for _ in range(12):
            action = policy.act(obs, history)
            try:
                obs = self.env.step(episode_id, action)
            except Exception:
                break
            history.append(action)
            if obs.is_terminal:
                break

        trace = self.env.get_trace(episode_id)
        scenario = load_scenario(scenario_id)
        reward = compute_reward(trace, scenario)

        final = trace.episode.final_action
        return {
            "baseline": policy.name,
            "scenario_id": scenario_id,
            "family": scenario.family,
            "difficulty": scenario.metadata.difficulty,
            "seed": seed,
            "total_reward": reward.total,
            "format_gate": reward.format_gate,
            "diagnosis_correct": (
                final.diagnosis == scenario.ground_truth.label if final else False
            ),
            "predicted_diagnosis": final.diagnosis if final else None,
            "true_diagnosis": scenario.ground_truth.label,
            "action_quality": reward.components["action_quality"].raw,
            "tool_call_count": sum(
                1 for r in trace.episode.history if isinstance(r.action, ToolCall)
            ),
            "total_cost": sum(r.cost_charged for r in trace.episode.history),
            "confidence": final.confidence if final else 0.0,
            "is_ambiguous_scenario": scenario.ground_truth.is_ambiguous,
            "brier_on_ambiguous": (
                (final.confidence - scenario.ground_truth.confidence_target) ** 2
                if scenario.ground_truth.is_ambiguous and final
                else None
            ),
        }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run multi-baseline CI-triage evaluation")
    parser.add_argument("--output", default="data_artifacts/results/")
    parser.add_argument("--eval-set", default="data_artifacts/scenarios/held_out/")
    parser.add_argument("--checkpoint", default="checkpoints/grpo_full/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args(argv)

    evaluator = Evaluator(
        eval_set_path=args.eval_set,
        trained_checkpoint=args.checkpoint,
    )
    df = evaluator.run_all(seeds=args.seeds)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eval.csv", index=False)
    print(
        df.groupby("baseline").agg({
            "diagnosis_correct": "mean",
            "action_quality": "mean",
            "tool_call_count": "mean",
            "total_cost": "mean",
            "total_reward": "mean",
        })
    )

    from ci_triage_env.training.plotting import plot_all_eval_metrics
    plot_all_eval_metrics(df, out / "plots/")


if __name__ == "__main__":
    main()
