from ci_triage_env.schemas.tools import ALL_TOOLS, RESERVED_NAMES

EXPECTED_TOOL_NAMES = {
    "read_logs",
    "inspect_test_code",
    "run_diagnostic",
    "cluster_metrics",
    "query_flake_history",
    "recent_commits",
    "check_owner",
    "rerun_test",
    "quarantine_test",
    "file_bug",
    "ping_owner",
}


def test_eleven_tools_present():
    assert len(ALL_TOOLS) == 11
    assert {t.name for t in ALL_TOOLS} == EXPECTED_TOOL_NAMES


def test_no_duplicate_names():
    names = [t.name for t in ALL_TOOLS]
    assert len(names) == len(set(names))


def test_no_reserved_names():
    for tool in ALL_TOOLS:
        assert tool.name not in RESERVED_NAMES


def test_each_tool_has_args_and_output_schema():
    for tool in ALL_TOOLS:
        assert isinstance(tool.args_schema, dict)
        assert tool.args_schema.get("type") == "object"
        assert isinstance(tool.output_schema, dict)
        assert tool.output_schema.get("type") == "object"


def test_each_tool_has_nonneg_cost_and_description():
    for tool in ALL_TOOLS:
        assert tool.cost_unit >= 0.0
        assert tool.description.strip()
