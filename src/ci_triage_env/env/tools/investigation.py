"""Investigation tools: read_logs, inspect_test_code, run_diagnostic, cluster_metrics.

Each handler validates args against the frozen MCP schema (Phase 0) and looks up
its pre-computed payload from ``scenario.tool_outputs`` using a composite key
derived from the args. If the scenario doesn't carry a payload for the requested
arg combination, the handler returns an empty / "no signal" payload — that's
how families that don't depend on this tool are represented.

Cost values follow Phase A2's recalibration. ``cost_unit`` on the handler is
what the env actually charges; the value advertised on ``MCPToolDef`` in
``schemas/tools.py`` (frozen Phase 0) may differ — that's metadata.
"""

from __future__ import annotations

from typing import ClassVar

from ci_triage_env.env.tools.utils import SchemaValidatedHandler
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.scenario import Scenario, ToolOutput


def _payload_or_default(scenario: Scenario, key: str, default) -> dict | str:
    out = scenario.tool_outputs.get(key)
    if out is None:
        return default
    return out.payload


class ReadLogsHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "read_logs"
    cost_unit: ClassVar[float] = 0.001

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        scope = args["scope"]
        lines = int(args.get("lines", 200))

        raw = _payload_or_default(scenario, f"read_logs:{scope}", {"lines": [], "truncated": False})
        all_lines = list(raw.get("lines", [])) if isinstance(raw, dict) else []
        truncated = bool(raw.get("truncated", False)) if isinstance(raw, dict) else False
        if lines < len(all_lines):
            output = {"lines": all_lines[:lines], "truncated": True}
        else:
            output = {"lines": all_lines, "truncated": truncated}

        # Cost scales with information requested: 100 lines = 1 cost unit.
        cost = self.cost_unit * (lines / 100.0)
        return ToolOutput(tool_name=self.name, payload=output, cost_units=cost)


class InspectTestCodeHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "inspect_test_code"
    cost_unit: ClassVar[float] = 0.05

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        test_name = args["test_name"]
        include_fixtures = bool(args.get("include_fixtures", False))

        key = f"inspect_test_code:{test_name}"
        default = {"source": "", "fixtures": []}
        raw = _payload_or_default(scenario, key, default)
        if isinstance(raw, dict):
            output = {
                "source": raw.get("source", ""),
                "fixtures": raw.get("fixtures", []) if include_fixtures else [],
            }
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class RunDiagnosticHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "run_diagnostic"
    cost_unit: ClassVar[float] = 0.10

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        # Frozen Phase 0 schema enforces probe ∈ {network,disk,memory,cpu};
        # the A2 doc's alternate enum (cpu_profile/...) is not the live contract.
        self.validate_args(args)
        probe = args["probe"]
        key = f"run_diagnostic:{probe}"
        default = {"ok": True, "details": {}}
        raw = _payload_or_default(scenario, key, default)
        if isinstance(raw, dict):
            output = {
                "ok": bool(raw.get("ok", True)),
                "details": raw.get("details", {}),
            }
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class ClusterMetricsHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "cluster_metrics"
    cost_unit: ClassVar[float] = 0.02

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        metric = args["metric"]
        # ``window_minutes`` is part of the frozen schema; it doesn't affect
        # the lookup key (scenarios author one payload per metric) but we
        # echo it back so the agent can see what was queried.
        window_minutes = int(args.get("window_minutes", 30))

        key = f"cluster_metrics:{metric}"
        default = {"samples": []}
        raw = _payload_or_default(scenario, key, default)
        samples = raw.get("samples", []) if isinstance(raw, dict) else []
        output = {"samples": samples, "window_minutes": window_minutes}
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)
