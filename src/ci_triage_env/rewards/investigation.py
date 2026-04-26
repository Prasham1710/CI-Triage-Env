"""InvestigationReward — shaping reward for evidence-gathering quality.

Combines:
  - coverage: fraction of informative_tools that were called (weight 0.6)
  - ordering: cheap-before-expensive bonus (weight 0.2)
  - redundancy_penalty: -0.1 per duplicate (tool_name, args) call

Raw score range: [-1.0, 1.0]. Default weight: 0.15.
"""

from __future__ import annotations

import json

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario

_CHEAP_TOOLS = frozenset({
    "read_logs", "query_flake_history", "recent_commits",
    "check_owner", "inspect_test_code", "cluster_metrics",
})
_EXPENSIVE_TOOLS = frozenset({
    "rerun_test", "run_diagnostic", "file_bug", "ping_owner", "quarantine_test",
})


class InvestigationReward(RewardComponent):
    """Shaping reward for how well the agent investigates the failure.

    Raw score range: [-1.0, 1.0].
    """

    name = "investigation"
    default_weight = 0.15

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        called_tools = [
            rec.action.tool_name
            for rec in trace.episode.history
            if isinstance(rec.action, ToolCall)
        ]

        # Coverage: fraction of informative_tools called
        informative = set(scenario.informative_tools)
        called_informative = sum(1 for t in called_tools if t in informative)
        coverage = called_informative / max(len(informative), 1)

        # Redundancy: duplicate (tool_name, sorted-args-json) calls
        seen_calls: set[tuple[str, str]] = set()
        redundancy_count = 0
        for rec in trace.episode.history:
            if isinstance(rec.action, ToolCall):
                key = (rec.action.tool_name, json.dumps(rec.action.args, sort_keys=True))
                if key in seen_calls:
                    redundancy_count += 1
                seen_calls.add(key)
        redundancy_penalty = -0.1 * redundancy_count

        # Ordering: cheap tools should precede expensive tools
        ordering = self._compute_ordering_score(called_tools)

        raw = 0.6 * coverage + 0.2 * ordering + redundancy_penalty
        raw = max(min(raw, 1.0), -1.0)

        return ComponentScore(
            raw=raw,
            weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={
                "coverage": coverage,
                "ordering": ordering,
                "redundancy_penalty": redundancy_penalty,
            },
        )

    def _compute_ordering_score(self, tools: list[str]) -> float:
        violations = 0
        seen_expensive = False
        for t in tools:
            if t in _EXPENSIVE_TOOLS:
                seen_expensive = True
            elif t in _CHEAP_TOOLS and seen_expensive:
                violations += 1
        return max(1.0 - 0.2 * violations, 0.0)
