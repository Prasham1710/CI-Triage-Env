from typing import ClassVar

import jsonschema

from ci_triage_env.env.tools.base import ToolHandler
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.scenario import Scenario, ToolOutput
from ci_triage_env.schemas.tools import ALL_TOOLS

_TOOL_DEFS = {t.name: t for t in ALL_TOOLS}


class _StubToolHandler(ToolHandler):
    """Phase A1 stub. Validates args against MCPToolDef.args_schema, returns placeholder payload."""

    name: ClassVar[str] = ""
    cost_unit: ClassVar[float] = 0.0

    def validate_args(self, args: dict) -> None:
        spec = _TOOL_DEFS[self.name]
        try:
            jsonschema.validate(instance=args, schema=spec.args_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"invalid args for {self.name}: {exc.message}") from exc

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        return ToolOutput(
            tool_name=self.name,
            payload={"stub": True, "tool": self.name},
            cost_units=self.cost_unit,
        )


class ReadLogsHandler(_StubToolHandler):
    name: ClassVar[str] = "read_logs"
    cost_unit: ClassVar[float] = _TOOL_DEFS["read_logs"].cost_unit

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        return ToolOutput(
            tool_name=self.name,
            payload={"lines": ["[stub]"], "truncated": False},
            cost_units=self.cost_unit,
        )


class InspectTestCodeHandler(_StubToolHandler):
    name: ClassVar[str] = "inspect_test_code"
    cost_unit: ClassVar[float] = _TOOL_DEFS["inspect_test_code"].cost_unit


class RunDiagnosticHandler(_StubToolHandler):
    name: ClassVar[str] = "run_diagnostic"
    cost_unit: ClassVar[float] = _TOOL_DEFS["run_diagnostic"].cost_unit


class ClusterMetricsHandler(_StubToolHandler):
    name: ClassVar[str] = "cluster_metrics"
    cost_unit: ClassVar[float] = _TOOL_DEFS["cluster_metrics"].cost_unit
