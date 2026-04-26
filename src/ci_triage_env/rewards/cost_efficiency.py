"""CostEfficiencyReward — penalises high tool-call cost spend.

Raw score range: [-1.0, 1.0]. Default weight: 0.15.
Mapping: 0 cost → 1.0; full BUDGET_REFERENCE spend → -1.0.
Over-budget episodes are not possible (env enforces budget), so ratio is clamped at 1.0.
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class CostEfficiencyReward(RewardComponent):
    """Linear reward inversely proportional to total cost spent.

    Raw score range: [-1.0, 1.0].
    """

    name = "cost_efficiency"
    default_weight = 0.15

    BUDGET_REFERENCE: float = 5.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        total_spent = sum(rec.cost_charged for rec in trace.episode.history)
        ratio = total_spent / self.BUDGET_REFERENCE
        raw = 1.0 - 2.0 * min(ratio, 1.0)
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={"total_cost": total_spent, "ratio": ratio},
        )
