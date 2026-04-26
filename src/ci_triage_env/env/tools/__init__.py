from ci_triage_env.env.tools.actions import (
    FileBugHandler,
    PingOwnerHandler,
    QuarantineTestHandler,
    RerunTestHandler,
)
from ci_triage_env.env.tools.base import ToolHandler
from ci_triage_env.env.tools.context import (
    CheckOwnerHandler,
    QueryFlakeHistoryHandler,
    RecentCommitsHandler,
)
from ci_triage_env.env.tools.investigation import (
    ClusterMetricsHandler,
    InspectTestCodeHandler,
    ReadLogsHandler,
    RunDiagnosticHandler,
)

ALL_TOOL_HANDLERS: list[ToolHandler] = [
    ReadLogsHandler(),
    InspectTestCodeHandler(),
    RunDiagnosticHandler(),
    ClusterMetricsHandler(),
    QueryFlakeHistoryHandler(),
    RecentCommitsHandler(),
    CheckOwnerHandler(),
    RerunTestHandler(),
    QuarantineTestHandler(),
    FileBugHandler(),
    PingOwnerHandler(),
]

__all__ = [
    "ALL_TOOL_HANDLERS",
    "CheckOwnerHandler",
    "ClusterMetricsHandler",
    "FileBugHandler",
    "InspectTestCodeHandler",
    "PingOwnerHandler",
    "QuarantineTestHandler",
    "QueryFlakeHistoryHandler",
    "ReadLogsHandler",
    "RecentCommitsHandler",
    "RerunTestHandler",
    "RunDiagnosticHandler",
    "ToolHandler",
]
