"""ActionQualityReward — secondary action × failure-family matrix.

Raw score range: [-2.0, 1.5] (capped). Default weight: 0.20.
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario

# (action_name, ground_truth_family) → reward
ACTION_REWARD_MATRIX: dict[tuple[str, str], float] = {
    ("file_bug", "real_bug"): 1.0,
    ("file_bug", "dependency_drift"): 0.7,
    ("file_bug", "race_flake"): -0.5,
    ("file_bug", "timing_flake"): -0.3,
    ("file_bug", "infra_network"): -0.5,
    ("file_bug", "infra_resource"): -0.5,
    ("file_bug", "ambiguous"): -0.2,
    # Quarantine: ideal for flakes, catastrophic for real bugs
    ("quarantine_test", "race_flake"): 1.0,
    ("quarantine_test", "timing_flake"): 0.8,
    ("quarantine_test", "real_bug"): -1.5,
    ("quarantine_test", "infra_network"): -0.3,
    ("quarantine_test", "infra_resource"): -0.3,
    ("quarantine_test", "dependency_drift"): -0.5,
    ("quarantine_test", "ambiguous"): -0.3,
    # Rerun: right for transient failures, bad for bugs
    ("rerun_test", "race_flake"): 0.6,
    ("rerun_test", "timing_flake"): 0.6,
    ("rerun_test", "infra_network"): 0.8,
    ("rerun_test", "infra_resource"): 0.5,
    ("rerun_test", "real_bug"): -0.6,
    ("rerun_test", "dependency_drift"): -0.3,
    ("rerun_test", "ambiguous"): 0.2,
    # Ping owner: escalates to the right team
    ("ping_owner", "infra_resource"): 0.7,
    ("ping_owner", "infra_network"): 0.5,
    ("ping_owner", "real_bug"): 0.4,
    ("ping_owner", "dependency_drift"): 0.6,
    ("ping_owner", "race_flake"): 0.0,
    ("ping_owner", "timing_flake"): 0.0,
    ("ping_owner", "ambiguous"): 0.3,
}

_RAW_MIN = -2.0
_RAW_MAX = 1.5


class ActionQualityReward(RewardComponent):
    """Reward for secondary actions taken alongside the diagnosis.

    Multiple secondary actions are summed then capped to [-2.0, 1.5].
    No secondary actions → neutral (0.0). No terminal action → -0.5.
    """

    name = "action_quality"
    default_weight = 0.20

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        if trace.episode.final_action is None:
            raw = -0.5
            return ComponentScore(
                raw=raw,
                weighted=raw * self.default_weight,
                weight=self.default_weight,
                sub_scores={"no_action": -0.5},
            )

        true = scenario.ground_truth.label.value
        secondary = trace.episode.final_action.secondary_actions

        if not secondary:
            return ComponentScore(
                raw=0.0,
                weighted=0.0,
                weight=self.default_weight,
                sub_scores={"no_secondary": 0.0},
            )

        sub_scores: dict[str, float] = {}
        total = 0.0
        for sa in secondary:
            r = ACTION_REWARD_MATRIX.get((sa.name, true), 0.0)
            sub_scores[sa.name] = r
            total += r

        capped = max(min(total, _RAW_MAX), _RAW_MIN)
        return ComponentScore(
            raw=capped,
            weighted=capped * self.default_weight,
            weight=self.default_weight,
            sub_scores=sub_scores,
        )
