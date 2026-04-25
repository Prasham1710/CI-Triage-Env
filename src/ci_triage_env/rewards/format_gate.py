import jsonschema

from ci_triage_env.rewards.base import RewardComponent
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.schemas.tools import ALL_TOOLS

TOOL_DEF_BY_NAME = {t.name: t for t in ALL_TOOLS}


class FormatGate(RewardComponent):
    """Validates every tool call and terminal action in the trajectory against MCP schemas.

    Returns 0.0 (gate fails) or 1.0 (passes). Used as a multiplicative gate in the composite.
    Score range: {0.0, 1.0}.
    """

    name = "format_gate"
    default_weight = 1.0

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        for record in trace.episode.history:
            if isinstance(record.action, ToolCall):
                tool_def = TOOL_DEF_BY_NAME.get(record.action.tool_name)
                if tool_def is None:
                    return self._fail()
                try:
                    jsonschema.validate(record.action.args, tool_def.args_schema)
                except jsonschema.ValidationError:
                    return self._fail()
            elif isinstance(record.action, TerminalAction):
                if not (0.0 <= record.action.confidence <= 1.0):
                    return self._fail()
        # Probe response check — uses reward_breakdown.counterfactual (CounterfactualScore)
        cf = trace.reward_breakdown.counterfactual
        if cf is not None and cf.fired and not cf.predicted_outcome:
            return self._fail()
        return ComponentScore(raw=1.0, weighted=1.0, weight=1.0, sub_scores={"valid": 1.0})

    def _fail(self) -> ComponentScore:
        return ComponentScore(raw=0.0, weighted=0.0, weight=1.0, sub_scores={"valid": 0.0})
