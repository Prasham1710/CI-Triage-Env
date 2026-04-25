"""TimePenaltyReward — penalises episodes that take more than REFERENCE_STEPS tool calls.

Raw score range: [-1.0, 0.0]. Default weight: 0.10.
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class TimePenaltyReward(RewardComponent):
    """Linear per-step penalty beyond REFERENCE_STEPS tool calls.

    0 to REFERENCE_STEPS calls → 0.0. Each extra step → -PER_STEP_PENALTY.
    Floor at -1.0. Raw score range: [-1.0, 0.0].
    """

    name = "time"
    default_weight = 0.10

    PER_STEP_PENALTY: float = 0.02
    REFERENCE_STEPS: int = 6

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        steps = sum(1 for r in trace.episode.history if isinstance(r.action, ToolCall))
        excess = max(0, steps - self.REFERENCE_STEPS)
        raw = max(-self.PER_STEP_PENALTY * excess, -1.0)
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={"steps": float(steps), "excess": float(excess)},
        )
