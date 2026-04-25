from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class MinimalEvidenceReward(RewardComponent):
    """Bonus for reaching the correct diagnosis using only the minimal evidence set.

    default_weight=0.0: not directly additive in the composite; its score is folded
    into InvestigationReward via a multiplier in C2.

    Score range: [-0.5, 1.0]. Empty min_set: 0.0.
    """

    name = "minimal_evidence"
    default_weight = 0.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        called = {rec.action.tool_name for rec in trace.episode.history if isinstance(rec.action, ToolCall)}
        min_set = set(scenario.minimal_evidence_set)
        if not min_set:
            return ComponentScore(raw=0.0, weighted=0.0, weight=0.0, sub_scores={})

        correct_diagnosis = (
            trace.episode.final_action is not None
            and str(trace.episode.final_action.diagnosis) == str(scenario.ground_truth.label)
        )
        if correct_diagnosis:
            min_used = called & min_set
            extra = called - min_set
            if min_used == min_set:
                bonus = max(1.0 - 0.1 * len(extra), -0.5)
            else:
                bonus = 0.3  # right answer but missing key evidence (lucky guess)
        else:
            bonus = 0.0

        bonus = max(min(bonus, 1.0), -0.5)
        return ComponentScore(
            raw=bonus,
            weighted=bonus * self.default_weight,
            weight=self.default_weight,
            sub_scores={
                "min_set_used_count": float(len(called & min_set)),
                "extras_count": float(len(called - min_set)),
            },
        )
