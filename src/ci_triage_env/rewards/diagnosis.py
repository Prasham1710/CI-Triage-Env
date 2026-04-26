"""DiagnosisReward — asymmetric confusion-matrix reward.

Raw score range: [-1.0, 1.0]. Default weight: 0.25.
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario

# (predicted, true) → reward
# Diagonal = 1.0; off-diagonal is asymmetric based on operational consequence.
# Worst: quarantine a real bug (ships to prod); best: correct identification.
DIAGNOSIS_REWARD_MATRIX: dict[tuple[str, str], float] = {
    ("real_bug", "real_bug"): 1.0,
    ("race_flake", "race_flake"): 1.0,
    ("timing_flake", "timing_flake"): 1.0,
    ("infra_network", "infra_network"): 1.0,
    ("infra_resource", "infra_resource"): 1.0,
    ("dependency_drift", "dependency_drift"): 1.0,
    ("ambiguous", "ambiguous"): 1.0,
    # Worst: predicting flake when it's a real bug (ships to prod)
    ("race_flake", "real_bug"): -1.0,
    ("timing_flake", "real_bug"): -1.0,
    ("ambiguous", "real_bug"): -0.7,
    # Bad: predicting infra when it's a real bug (file with wrong team)
    ("infra_network", "real_bug"): -0.5,
    ("infra_resource", "real_bug"): -0.5,
    ("dependency_drift", "real_bug"): -0.4,
    # Bad: predicting bug when it's a flake (false-alarm noise)
    ("real_bug", "race_flake"): -0.3,
    ("real_bug", "timing_flake"): -0.3,
    # Bad: predicting bug when it's infra (wastes engineering time)
    ("real_bug", "infra_network"): -0.4,
    ("real_bug", "infra_resource"): -0.4,
    ("real_bug", "dependency_drift"): -0.2,
    # Mild: confusing similar families
    ("race_flake", "timing_flake"): 0.2,
    ("timing_flake", "race_flake"): 0.2,
    ("infra_network", "infra_resource"): 0.1,
    ("infra_resource", "infra_network"): 0.1,
    # Abstaining on clear non-real-bug causes
    ("ambiguous", "race_flake"): 0.0,
    ("ambiguous", "timing_flake"): 0.0,
    ("ambiguous", "infra_network"): 0.0,
    ("ambiguous", "infra_resource"): 0.0,
    ("ambiguous", "dependency_drift"): 0.0,
}

_DEFAULT_OFF_DIAGONAL = -0.5


def lookup_reward(predicted: str, true: str) -> float:
    return DIAGNOSIS_REWARD_MATRIX.get((predicted, true), _DEFAULT_OFF_DIAGONAL)


class DiagnosisReward(RewardComponent):
    """Reward based on predicted vs. true failure family.

    Raw score range: [-1.0, 1.0]. No-terminal penalty: -1.0.
    """

    name = "diagnosis"
    default_weight = 0.25

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        if trace.episode.final_action is None:
            raw = -1.0
            return ComponentScore(
                raw=raw,
                weighted=raw * self.default_weight,
                weight=self.default_weight,
                sub_scores={"no_diagnosis": -1.0},
            )
        predicted = trace.episode.final_action.diagnosis.value
        true = scenario.ground_truth.label.value
        raw = lookup_reward(predicted, true)
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={
                "matrix_lookup": raw,
                "predicted": 1.0 if predicted == true else 0.0,
            },
        )
