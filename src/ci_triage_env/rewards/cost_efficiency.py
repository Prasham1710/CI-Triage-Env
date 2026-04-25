from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class CostEfficiencyReward(RewardComponent):
    """Penalises high tool-call spend relative to the episode budget reference.

    Maps spend to [-1.0, 1.0]: zero cost → 1.0, full BUDGET_REFERENCE → -1.0.
    Score range: [-1.0, 1.0].
    """

    name = "cost_efficiency"
    default_weight = 0.15

    BUDGET_REFERENCE = 5.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        total_spent = sum(
            rec.cost_charged for rec in trace.episode.history if isinstance(rec.action, ToolCall)
        )
        ratio = total_spent / self.BUDGET_REFERENCE
        raw = 1.0 - 2.0 * min(ratio, 1.0)
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={"total_cost": total_spent, "ratio": ratio},
        )
