from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class CounterfactualPredictReward(RewardComponent):
    """Counterfactual probe prediction reward.

    DORMANT in v1: the env never fires probes, so counterfactual_replay is always None
    and this component always returns zero. Preserved so v2 re-enable is purely additive.

    Score range: [-0.5, 1.0]. default_weight=0.0 (dormant).
    """

    name = "counterfactual"
    default_weight = 0.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        # Uses reward_breakdown.counterfactual (CounterfactualScore), not counterfactual_replay
        # (list[StepRecord]). In v1 this is always None.
        cf = trace.reward_breakdown.counterfactual
        if cf is None or not cf.fired:
            return ComponentScore(raw=0.0, weighted=0.0, weight=0.0, sub_scores={"fired": 0.0})

        # Reachable only in v2 when probe is enabled.
        raw = 1.0 if cf.predicted_outcome == cf.actual_outcome else -0.5
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={"fired": 1.0, "correct": 1.0 if cf.predicted_outcome == cf.actual_outcome else 0.0},
        )
