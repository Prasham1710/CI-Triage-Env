from typing import ClassVar

from ci_triage_env.env.tools.investigation import _StubToolHandler
from ci_triage_env.schemas.tools import ALL_TOOLS

_TOOL_DEFS = {t.name: t for t in ALL_TOOLS}


class QueryFlakeHistoryHandler(_StubToolHandler):
    name: ClassVar[str] = "query_flake_history"
    cost_unit: ClassVar[float] = _TOOL_DEFS["query_flake_history"].cost_unit


class RecentCommitsHandler(_StubToolHandler):
    name: ClassVar[str] = "recent_commits"
    cost_unit: ClassVar[float] = _TOOL_DEFS["recent_commits"].cost_unit


class CheckOwnerHandler(_StubToolHandler):
    name: ClassVar[str] = "check_owner"
    cost_unit: ClassVar[float] = _TOOL_DEFS["check_owner"].cost_unit
