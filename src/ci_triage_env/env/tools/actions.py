from typing import ClassVar

from ci_triage_env.env.tools.investigation import _StubToolHandler
from ci_triage_env.schemas.tools import ALL_TOOLS

_TOOL_DEFS = {t.name: t for t in ALL_TOOLS}


class RerunTestHandler(_StubToolHandler):
    name: ClassVar[str] = "rerun_test"
    cost_unit: ClassVar[float] = _TOOL_DEFS["rerun_test"].cost_unit


class QuarantineTestHandler(_StubToolHandler):
    name: ClassVar[str] = "quarantine_test"
    cost_unit: ClassVar[float] = _TOOL_DEFS["quarantine_test"].cost_unit


class FileBugHandler(_StubToolHandler):
    name: ClassVar[str] = "file_bug"
    cost_unit: ClassVar[float] = _TOOL_DEFS["file_bug"].cost_unit


class PingOwnerHandler(_StubToolHandler):
    name: ClassVar[str] = "ping_owner"
    cost_unit: ClassVar[float] = _TOOL_DEFS["ping_owner"].cost_unit
