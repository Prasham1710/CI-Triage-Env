"""CounterfactualPredictReward — DORMANT in v1.

Counterfactual probe is deferred to v2. In v1 the env never fires probes
(trace.counterfactual_replay is always None), so this component always returns
(raw=0.0, weight=0.0). The implementation is preserved so v2 re-enable is a
purely additive change: set default_weight to 0.10 in weights.py.

Raw score range: [-0.5, 1.0]. Default weight: 0.0 (dormant).
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class CounterfactualPredictReward(RewardComponent):
    """Rewards correct prediction of the counterfactual probe outcome.

    DORMANT in v1: default_weight=0.0 and trace.counterfactual_replay is always
    None, so score() always returns zero contribution.
    Raw score range: [-0.5, 1.0].
    """

    name = "counterfactual"
    default_weight = 0.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        # v1: probes never fire; replay list is None or empty
        if not trace.counterfactual_replay:
            return ComponentScore(
                raw=0.0,
                weighted=0.0,
                weight=self.default_weight,
                sub_scores={"fired": 0.0},
            )

        # v2 path (reachable only when probes are enabled):
        # The replay records encode the probe action and its observed outcome.
        # Compare the agent's predicted outcome (last record) vs actual terminal.
        predicted_record = trace.counterfactual_replay[-1]
        actual_record = trace.episode.history[-1] if trace.episode.history else None

        if actual_record is not None and predicted_record.action == actual_record.action:
            raw = 1.0
        else:
            raw = -0.5

        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={"fired": 1.0},
        )
