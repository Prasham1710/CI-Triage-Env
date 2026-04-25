"""Context tools: query_flake_history, recent_commits, check_owner.

See investigation.py for the routing pattern. Args validated against the
frozen MCP schemas; output read from scenario.tool_outputs[key]; missing key
returns an empty payload.
"""

from __future__ import annotations

from typing import ClassVar

from ci_triage_env.env.tools.investigation import _payload_or_default
from ci_triage_env.env.tools.utils import SchemaValidatedHandler
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.scenario import Scenario, ToolOutput


class QueryFlakeHistoryHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "query_flake_history"
    cost_unit: ClassVar[float] = 0.01

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        test_name = args["test_name"]
        key = f"query_flake_history:{test_name}"
        default = {"failure_count": 0, "pass_count": 0, "recent_failures": []}
        raw = _payload_or_default(scenario, key, default)
        if isinstance(raw, dict):
            output = {
                "failure_count": int(raw.get("failure_count", 0)),
                "pass_count": int(raw.get("pass_count", 0)),
                "recent_failures": list(raw.get("recent_failures", [])),
            }
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class RecentCommitsHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "recent_commits"
    cost_unit: ClassVar[float] = 0.01

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        # Frozen schema: {branch, limit}. The A2 doc's alternate {window, paths}
        # would conflict with the contract Branch B writes against, so we honor
        # the live schema and key on the branch.
        self.validate_args(args)
        branch = args["branch"]
        limit = int(args.get("limit", 10))

        key = f"recent_commits:{branch}"
        default = {"commits": []}
        raw = _payload_or_default(scenario, key, default)
        commits = raw.get("commits", []) if isinstance(raw, dict) else []
        output = {"commits": list(commits)[:limit]}
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)


class CheckOwnerHandler(SchemaValidatedHandler):
    name: ClassVar[str] = "check_owner"
    cost_unit: ClassVar[float] = 0.01

    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput:
        self.validate_args(args)
        target = args["target"]
        key = f"check_owner:{target}"
        default = {"owner": "", "team": "", "contact": ""}
        raw = _payload_or_default(scenario, key, default)
        if isinstance(raw, dict):
            output = {
                "owner": raw.get("owner", ""),
                "team": raw.get("team", ""),
                "contact": raw.get("contact", ""),
            }
        else:
            output = default
        return ToolOutput(tool_name=self.name, payload=output, cost_units=self.cost_unit)
