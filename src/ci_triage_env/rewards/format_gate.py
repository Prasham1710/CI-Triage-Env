"""FormatGate — validates trajectory schema compliance.

Returns 1.0 (all records valid) or 0.0 (first violation found).
Raw score range: {0.0, 1.0}. Used as a multiplicative gate in composite.
"""

from __future__ import annotations

import jsonschema

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.schemas.tools import ALL_TOOLS

TOOL_DEF_BY_NAME: dict = {t.name: t for t in ALL_TOOLS}


class FormatGate(RewardComponent):
    """Validates every ToolCall args against the tool's args_schema and every
    TerminalAction against the DiagnosisLabel enum + confidence bounds.

    Returns 0.0 (gate fails) or 1.0 (passes). The composite uses this as a
    multiplicative gate: total = format_gate * weighted_sum.
    """

    name = "format_gate"
    default_weight = 1.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        for record in trace.episode.history:
            if isinstance(record.action, ToolCall):
                tool_def = TOOL_DEF_BY_NAME.get(record.action.tool_name)
                if tool_def is None:
                    return self._fail("unknown_tool")
                try:
                    jsonschema.validate(record.action.args, tool_def.args_schema)
                except jsonschema.ValidationError:
                    return self._fail("args_invalid")
            elif isinstance(record.action, TerminalAction):
                if record.action.diagnosis not in DiagnosisLabel:
                    return self._fail("invalid_diagnosis")
                if not (0.0 <= record.action.confidence <= 1.0):
                    return self._fail("confidence_oob")

        # v1: counterfactual_replay is a list of StepRecords or None; probes never fire
        if trace.counterfactual_replay is not None and len(trace.counterfactual_replay) > 0:
            # Any probe records must themselves contain valid actions
            for record in trace.counterfactual_replay:
                if isinstance(record.action, ToolCall):
                    tool_def = TOOL_DEF_BY_NAME.get(record.action.tool_name)
                    if tool_def is None:
                        return self._fail("probe_unknown_tool")

        return ComponentScore(
            raw=1.0,
            weighted=1.0,
            weight=self.default_weight,
            sub_scores={"valid": 1.0},
        )

    def _fail(self, reason: str) -> ComponentScore:
        return ComponentScore(
            raw=0.0,
            weighted=0.0,
            weight=self.default_weight,
            sub_scores={"reason": 0.0, "reason_code": 0.0},
        )
