"""Secondary-action tools: rerun_test, quarantine_test, file_bug, ping_owner.

These are *investigation-time* actions (the agent may invoke them mid-episode
to gather evidence or apply mitigations) — they're distinct from the terminal
``submit_diagnosis`` action that ends the episode.

Outputs are scenario-deterministic in A2: each scenario carries one payload per
tool keyed by the bare tool name. The pre-baked outcome captures whether the
mitigation succeeded under that scenario's failure mode.
"""

from __future__ import annotations

from typing import ClassVar

from ci_triage_env.env.tools.investigation import _payload_or_default
from ci_triage_env.env.tools.utils import SchemaValidatedHandler
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.scenario import Scenario, ToolOutput


class RerunTestHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "rerun_test"
    cost_unit: ClassVar[float] = 0.30

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        iterations = int(args.get("iterations", 1))
        default = {"results": []}
        raw = _payload_or_default(scenario, "rerun_test", default)
        results = raw.get("results", []) if isinstance(raw, dict) else []
        output = {"results": list(results)[:iterations]}
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class QuarantineTestHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "quarantine_test"
    # Cost is reputational / process — the anti-game guard in the reward layer
    # discourages quarantining a real bug. The env charges no budget for it.
    cost_unit: ClassVar[float] = 0.0

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        default = {"quarantined": True, "ticket": ""}
        raw = _payload_or_default(scenario, "quarantine_test", default)
        if isinstance(raw, dict):
            output = {
                "quarantined": bool(raw.get("quarantined", True)),
                "ticket": raw.get("ticket", ""),
            }
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class FileBugHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "file_bug"
    cost_unit: ClassVar[float] = 0.5  # ~30min of human triage time

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        default = {"ticket_id": "", "url": ""}
        raw = _payload_or_default(scenario, "file_bug", default)
        if isinstance(raw, dict):
            output = {
                "ticket_id": raw.get("ticket_id", ""),
                "url": raw.get("url", ""),
            }
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class PingOwnerHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "ping_owner"
    cost_unit: ClassVar[float] = 0.083  # ~5min of human attention

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        default = {"delivered": True}
        raw = _payload_or_default(scenario, "ping_owner", default)
        if isinstance(raw, dict):
            output = {"delivered": bool(raw.get("delivered", True))}
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)
