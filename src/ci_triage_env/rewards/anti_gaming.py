"""AntiGamingReward — three guards against reward exploitation.

Guards:
  1. No-info-action: terminal with < 2 tool calls → -0.5
  2. Quarantine-rate: rolling-window over-use of quarantine_test → penalty
  3. Brier calibration: on ambiguous scenarios, penalises mis-calibrated confidence

Raw score range: [-1.5, 1.0]. Default weight: 0.15.
Quarantine-rate state is injected at construction; empty list → no penalty.
"""

from __future__ import annotations

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario

_QUARANTINE_THRESHOLD = 0.30
_RAW_MIN = -1.5
_RAW_MAX = 1.0


class AntiGamingReward(RewardComponent):
    """Guards against common reward-gaming strategies.

    Raw score range: [-1.5, 1.0].

    Args:
        recent_episode_actions: Names of the primary secondary actions taken in
            the last N episodes. Supplied by the trainer's rolling-window state.
            Pass an empty list for unit tests (no quarantine-rate pressure).
    """

    name = "anti_gaming"
    default_weight = 0.15

    def __init__(self, recent_episode_actions: list[str] | None = None) -> None:
        self.recent_actions: list[str] = recent_episode_actions or []

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        sub: dict[str, float] = {}

        # Guard 1: must gather at least 2 tool calls before diagnosing
        n_tool_calls = sum(1 for r in trace.episode.history if isinstance(r.action, ToolCall))
        if trace.episode.final_action is not None and n_tool_calls < 2:
            no_info_penalty = -0.5
        else:
            no_info_penalty = 0.0
        sub["no_info_penalty"] = no_info_penalty

        # Guard 2: quarantine over-use relative to a rolling window
        quarantine_rate = self._compute_quarantine_rate()
        if quarantine_rate > _QUARANTINE_THRESHOLD:
            quarantine_penalty = -(quarantine_rate - _QUARANTINE_THRESHOLD) * 2.0
        else:
            quarantine_penalty = 0.0
        sub["quarantine_rate"] = quarantine_rate
        sub["quarantine_penalty"] = quarantine_penalty

        # Guard 3: Brier calibration probe (ambiguous scenarios only)
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
        raw = max(min(raw, _RAW_MAX), _RAW_MIN)
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
