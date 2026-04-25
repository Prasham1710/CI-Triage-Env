from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class AntiGamingReward(RewardComponent):
    """Guards against known reward-hacking patterns.

    Three sub-penalties:
    - no_info_action: final action without >= 2 prior tool calls → -0.5.
    - quarantine_rate: rate > 30 % in recent window → proportional penalty.
    - brier_calibration: on ambiguous scenarios, Brier-scored confidence penalty/bonus.

    Score range: [-1.5, 1.0].

    recent_episode_actions: injected by the trainer's rolling window (last N episode actions).
    In unit tests, pass an empty list (no quarantine-rate penalty).
    """

    name = "anti_gaming"
    default_weight = 0.15

    def __init__(self, recent_episode_actions: list[str] | None = None) -> None:
        self.recent_actions: list[str] = recent_episode_actions or []

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        sub: dict[str, float] = {}

        n_tool_calls = sum(1 for r in trace.episode.history if isinstance(r.action, ToolCall))
        no_info_penalty = -0.5 if (trace.episode.final_action is not None and n_tool_calls < 2) else 0.0
        sub["no_info_penalty"] = no_info_penalty

        quarantine_rate = self._compute_quarantine_rate()
        quarantine_penalty = -(quarantine_rate - 0.30) * 2.0 if quarantine_rate > 0.30 else 0.0
        sub["quarantine_rate"] = quarantine_rate
        sub["quarantine_penalty"] = quarantine_penalty

        brier_bonus = 0.0
        if scenario.ground_truth.is_ambiguous:
            target = scenario.ground_truth.confidence_target
            if trace.episode.final_action is not None:
                pred_conf = trace.episode.final_action.confidence
                brier = (pred_conf - target) ** 2
                brier_bonus = 0.5 * (1.0 - brier)
            else:
                brier_bonus = -0.5
        sub["brier_bonus"] = brier_bonus

        raw = no_info_penalty + quarantine_penalty + brier_bonus
        raw = max(min(raw, 1.0), -1.5)
        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores=sub,
        )

    def _compute_quarantine_rate(self) -> float:
        if not self.recent_actions:
            return 0.0
        return sum(1 for a in self.recent_actions if a == "quarantine_test") / len(self.recent_actions)
