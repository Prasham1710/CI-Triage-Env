"""Replay verifier — recomputes reward from persisted trace + scenario JSON.

Used for training-run auditing: given the files written to disk during a rollout,
reproduce the exact reward score that was computed at collection time.
"""

from __future__ import annotations

from pathlib import Path

from ci_triage_env.rewards.composite import compute_reward
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import RewardBreakdown
from ci_triage_env.schemas.scenario import Scenario


def replay_reward_from_disk(trace_path: Path, scenario_path: Path) -> RewardBreakdown:
    """Recompute reward from trace JSON + scenario JSON files."""
    trace = EpisodeTrace.model_validate_json(trace_path.read_text())
    scenario = Scenario.model_validate_json(scenario_path.read_text())
    return compute_reward(trace, scenario)


def assert_reward_reproducible(trace: EpisodeTrace, scenario: Scenario) -> None:
    """Compute reward twice and assert identical results. Catches non-determinism early."""
    r1 = compute_reward(trace, scenario)
    r2 = compute_reward(trace, scenario)
    if r1.total != r2.total:
        raise AssertionError(f"Reward not reproducible: {r1.total} != {r2.total}")
    for k in r1.components:
        if r1.components[k].raw != r2.components[k].raw:
            raise AssertionError(
                f"Component {k!r} not reproducible: "
                f"{r1.components[k].raw} != {r2.components[k].raw}"
            )
