from pydantic import BaseModel, Field

RESERVED_NAMES: frozenset[str] = frozenset({"reset", "step", "state", "close"})


class MCPToolDef(BaseModel):
    name: str
    description: str
    args_schema: dict
    output_schema: dict
    cost_unit: float = Field(ge=0.0)


ALL_TOOLS: list[MCPToolDef] = [
    MCPToolDef(
        name="read_logs",
        description="Read log lines from the failed CI run. Use scope to narrow.",
        args_schema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["full", "test", "stderr", "kernel", "build"],
                },
                "lines": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 2000,
                    "default": 200,
                },
            },
            "required": ["scope"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "lines": {"type": "array", "items": {"type": "string"}},
                "truncated": {"type": "boolean"},
            },
        },
        cost_unit=0.001,
    ),
    MCPToolDef(
        name="inspect_test_code",
        description="Read source code of the failing test (and optionally fixtures it touches).",
        args_schema={
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "include_fixtures": {"type": "boolean", "default": False},
            },
            "required": ["test_name"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "fixtures": {"type": "array", "items": {"type": "string"}},
            },
        },
        cost_unit=0.002,
    ),
    MCPToolDef(
        name="run_diagnostic",
        description="Run a sandboxed diagnostic probe (network reachability, disk, memory).",
        args_schema={
            "type": "object",
            "properties": {
                "probe": {
                    "type": "string",
                    "enum": ["network", "disk", "memory", "cpu"],
                },
            },
            "required": ["probe"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "details": {"type": "object"},
            },
        },
        cost_unit=0.005,
    ),
    MCPToolDef(
        name="cluster_metrics",
        description="Fetch CI cluster-level health metrics around the failure timestamp.",
        args_schema={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["queue_depth", "node_health", "network_latency", "disk_io"],
                },
                "window_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 360,
                    "default": 30,
                },
            },
            "required": ["metric"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "samples": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
        cost_unit=0.003,
    ),
    MCPToolDef(
        name="query_flake_history",
        description="Query the flake-history index for prior failures of this test.",
        args_schema={
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "lookback_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90,
                    "default": 14,
                },
            },
            "required": ["test_name"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "failure_count": {"type": "integer"},
                "pass_count": {"type": "integer"},
                "recent_failures": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
        cost_unit=0.002,
    ),
    MCPToolDef(
        name="recent_commits",
        description="List recent commits on the branch that may have introduced a real bug.",
        args_schema={
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
            },
            "required": ["branch"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "commits": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
        cost_unit=0.002,
    ),
    MCPToolDef(
        name="check_owner",
        description="Look up the owning team or person for a test or module.",
        args_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
            },
            "required": ["target"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "team": {"type": "string"},
                "contact": {"type": "string"},
            },
        },
        cost_unit=0.001,
    ),
    MCPToolDef(
        name="rerun_test",
        description="Rerun the failing test (cheap heuristic for transient flakes).",
        args_schema={
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 1,
                },
            },
            "required": ["test_name"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
        cost_unit=0.01,
    ),
    MCPToolDef(
        name="quarantine_test",
        description="Mark the test as quarantined to stop blocking the pipeline.",
        args_schema={
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["test_name", "reason"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "quarantined": {"type": "boolean"},
                "ticket": {"type": "string"},
            },
        },
        cost_unit=0.005,
    ),
    MCPToolDef(
        name="file_bug",
        description="File a bug ticket against the owning team with structured context.",
        args_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "owner": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
            },
            "required": ["title", "summary", "owner", "severity"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "url": {"type": "string"},
            },
        },
        cost_unit=0.005,
    ),
    MCPToolDef(
        name="ping_owner",
        description="Notify the owning person/team about the failure (lightweight nudge).",
        args_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["owner", "message"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "delivered": {"type": "boolean"},
            },
        },
        cost_unit=0.002,
    ),
]
