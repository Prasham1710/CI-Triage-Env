"""RandomPolicy — random tool calls + random terminal. Evaluation floor baseline."""

from __future__ import annotations

import random

from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.tools import ALL_TOOLS

ALL_TOOL_NAMES: list[str] = [t.name for t in ALL_TOOLS]

# Minimal valid args per tool — satisfies all required fields from each tool's args_schema.
ALL_TOOL_ARG_DEFAULTS: dict[str, dict] = {
    "read_logs": {"scope": "full"},
    "inspect_test_code": {"test_name": "failing_test"},
    "run_diagnostic": {"probe": "network"},
    "cluster_metrics": {"metric": "queue_depth"},
    "query_flake_history": {"test_name": "failing_test"},
    "recent_commits": {"branch": "main"},
    "check_owner": {"target": "failing_test"},
    "rerun_test": {"test_name": "failing_test"},
    "quarantine_test": {"test_name": "failing_test", "reason": "flaky"},
    "file_bug": {
        "title": "auto",
        "summary": "auto",
        "owner": "auto",
        "severity": "medium",
    },
    "ping_owner": {"owner": "auto", "message": "CI failure detected"},
}


class RandomPolicy:
    """Random tool calls + random terminal action. The evaluation floor."""

    name = "random"

    def __init__(self, max_turns: int = 8, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.max_turns = max_turns

    def act(self, obs, history: list) -> dict:
        if len(history) >= self.max_turns:
            return self._random_terminal()
        if self.rng.random() < 0.2:
            return self._random_terminal()
        return self._random_tool_call()

    def _random_tool_call(self) -> dict:
        tool = self.rng.choice(ALL_TOOL_NAMES)
        return {"tool_name": tool, "args": ALL_TOOL_ARG_DEFAULTS[tool]}

    def _random_terminal(self) -> dict:
        return {
            "action_type": "submit_diagnosis",
            "diagnosis": self.rng.choice(list(DiagnosisLabel)),
            "confidence": round(self.rng.random(), 3),
            "secondary_actions": [],
        }
