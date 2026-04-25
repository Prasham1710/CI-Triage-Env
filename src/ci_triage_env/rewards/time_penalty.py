from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class TimePenaltyReward(RewardComponent):
    """Per-step penalty for tool calls beyond the reference episode length.

    No penalty for <= REFERENCE_STEPS tool calls; then -PER_STEP_PENALTY per excess step.
    Score range: [-1.0, 0.0].
    """

    name = "time"
    default_weight = 0.10

    PER_STEP_PENALTY = 0.02
    REFERENCE_STEPS = 6

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        steps = sum(1 for r in trace.episode.history if isinstance(r.action, ToolCall))
        excess = max(0, steps - self.REFERENCE_STEPS)
        raw = -self.PER_STEP_PENALTY * excess
        raw = max(raw, -1.0)
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={"steps": float(steps), "excess": float(excess)},
        )
