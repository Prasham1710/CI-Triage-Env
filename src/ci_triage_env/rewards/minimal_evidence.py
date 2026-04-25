"""MinimalEvidenceReward — bonus for diagnosing correctly with the minimal tool set.

Default weight: 0.0 — this component is NOT in the additive composite directly.
In Phase C2 its score modifies the InvestigationReward via a multiplier.
Raw score range: [-0.5, 1.0].
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class MinimalEvidenceReward(RewardComponent):
    """Bonus when the agent reaches the correct diagnosis using only the minimal evidence set.

    If minimal_evidence_set is empty (ambiguous scenarios), returns 0.0.
    Raw score range: [-0.5, 1.0]. Default weight: 0.0 (folded into InvestigationReward in C2).
    """

    name = "minimal_evidence"
    default_weight = 0.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        min_set = set(scenario.minimal_evidence_set)
        if not min_set:
            return ComponentScore(
                raw=0.0, weighted=0.0, weight=self.default_weight, sub_scores={}
            )

        called = {
            rec.action.tool_name
            for rec in trace.episode.history
            if isinstance(rec.action, ToolCall)
        }

        final = trace.episode.final_action
        correct_diagnosis = (
            final is not None
            and final.diagnosis.value == scenario.ground_truth.label.value
        )

        if correct_diagnosis:
            min_used = called & min_set
            extra = called - min_set
            if min_used == min_set:
                # All minimal evidence used; small penalty for extras
                bonus = max(min(1.0 - 0.1 * len(extra), 1.0), -0.5)
            else:
                bonus = 0.3  # correct answer but didn't use all key evidence
        else:
            bonus = 0.0

        return ComponentScore(
            raw=bonus,
            weighted=bonus * self.default_weight,
            weight=self.default_weight,
            sub_scores={
                "min_set_used": float(len(called & min_set)),
                "extras": float(len(called - min_set)),
            },
        )
