"""Phase A2 tool-handler tests.

Per the phase doc, each of the 11 tools must:
1. return a valid ``ToolOutput`` for valid args (with the correct cost charged)
2. raise ``ValueError`` for invalid args
3. return an empty / no-signal payload when ``scenario.tool_outputs`` lacks the
   expected key (rather than crashing)
4. return identical output on a repeated call with identical args

Plus two integration tests: a full 11-tool sweep and cumulative cost charging.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ci_triage_env.env.server import CITriageEnv, build_app
from ci_triage_env.env.tools import (
    CheckOwnerHandler,
    ClusterMetricsHandler,
    FileBugHandler,
    InspectTestCodeHandler,
    PingOwnerHandler,
    QuarantineTestHandler,
    QueryFlakeHistoryHandler,
    ReadLogsHandler,
    RecentCommitsHandler,
    RerunTestHandler,
    RunDiagnosticHandler,
)
from ci_triage_env.env.tools.utils import args_hash, deterministic_rng
from ci_triage_env.env.wire import CITriageAction
from ci_triage_env.schemas.scenario import Scenario, ToolOutput
from tests.env.conftest import make_a2_scenario

# (handler_factory, valid_args, invalid_args, missing_key_check, expected_cost_for_valid)
TOOL_MATRIX = [
    pytest.param(
        ReadLogsHandler,
        {"scope": "test", "lines": 100},
        {"scope": "bogus_scope"},
        "read_logs:test",
        0.001 * 100 / 100,
        id="read_logs",
    ),
    pytest.param(
        InspectTestCodeHandler,
        {"test_name": "tests/unit/test_widget.py::test_concurrent_update", "include_fixtures": True},
        {"include_fixtures": True},  # missing required test_name
        "inspect_test_code:tests/unit/test_widget.py::test_concurrent_update",
        0.05,
        id="inspect_test_code",
    ),
    pytest.param(
        RunDiagnosticHandler,
        {"probe": "network"},
        {"probe": "not_a_probe"},
        "run_diagnostic:network",
        0.10,
        id="run_diagnostic",
    ),
    pytest.param(
        ClusterMetricsHandler,
        {"metric": "queue_depth", "window_minutes": 15},
        {"metric": "no_such_metric"},
        "cluster_metrics:queue_depth",
        0.02,
        id="cluster_metrics",
    ),
    pytest.param(
        QueryFlakeHistoryHandler,
        {"test_name": "tests/unit/test_widget.py::test_concurrent_update"},
        {},  # missing test_name
        "query_flake_history:tests/unit/test_widget.py::test_concurrent_update",
        0.01,
        id="query_flake_history",
    ),
    pytest.param(
        RecentCommitsHandler,
        {"branch": "main", "limit": 5},
        {"limit": 5},  # missing branch
        "recent_commits:main",
        0.01,
        id="recent_commits",
    ),
    pytest.param(
        CheckOwnerHandler,
        {"target": "tests/unit/test_widget.py"},
        {},  # missing target
        "check_owner:tests/unit/test_widget.py",
        0.01,
        id="check_owner",
    ),
    pytest.param(
        RerunTestHandler,
        {"test_name": "tests/unit/test_widget.py::test_concurrent_update", "iterations": 2},
        {"iterations": 2},  # missing test_name
        "rerun_test",
        0.30,
        id="rerun_test",
    ),
    pytest.param(
        QuarantineTestHandler,
        {"test_name": "tests/unit/test_widget.py::test_concurrent_update", "reason": "flake"},
        {"test_name": "x"},  # missing reason
        "quarantine_test",
        0.0,
        id="quarantine_test",
    ),
    pytest.param(
        FileBugHandler,
        {"title": "t", "summary": "s", "owner": "alice", "severity": "high"},
        {"title": "t", "summary": "s", "owner": "alice"},  # missing severity
        "file_bug",
        0.5,
        id="file_bug",
    ),
    pytest.param(
        PingOwnerHandler,
        {"owner": "alice", "message": "hey"},
        {"owner": "alice"},  # missing message
        "ping_owner",
        0.083,
        id="ping_owner",
    ),
]


@pytest.fixture
def scenario() -> Scenario:
    return make_a2_scenario()


@pytest.mark.parametrize("handler_cls,valid_args,invalid_args,key,expected_cost", TOOL_MATRIX)
def test_tool_valid_args_returns_output(handler_cls, valid_args, invalid_args, key, expected_cost, scenario):
    handler = handler_cls()
    out = handler.call(valid_args, scenario, [])
    assert isinstance(out, ToolOutput)
    assert out.tool_name == handler.name
    assert out.cost_units == pytest.approx(expected_cost)


@pytest.mark.parametrize("handler_cls,valid_args,invalid_args,key,expected_cost", TOOL_MATRIX)
def test_tool_invalid_args_raises(handler_cls, valid_args, invalid_args, key, expected_cost, scenario):
    handler = handler_cls()
    with pytest.raises(ValueError):
        handler.call(invalid_args, scenario, [])


@pytest.mark.parametrize("handler_cls,valid_args,invalid_args,key,expected_cost", TOOL_MATRIX)
def test_tool_missing_scenario_data_returns_empty(handler_cls, valid_args, invalid_args, key, expected_cost, scenario):
    """Strip the relevant key from scenario.tool_outputs and confirm the
    handler returns a non-crashing empty payload."""
    stripped = {k: v for k, v in scenario.tool_outputs.items() if k != key}
    bare = scenario.model_copy(update={"tool_outputs": stripped})
    handler = handler_cls()
    out = handler.call(valid_args, bare, [])
    assert isinstance(out, ToolOutput)
    assert out.tool_name == handler.name
    # Cost is still charged even when the scenario doesn't carry data.
    assert out.cost_units >= 0.0


@pytest.mark.parametrize("handler_cls,valid_args,invalid_args,key,expected_cost", TOOL_MATRIX)
def test_tool_repeated_call_returns_same_output(handler_cls, valid_args, invalid_args, key, expected_cost, scenario):
    handler = handler_cls()
    first = handler.call(valid_args, scenario, [])
    second = handler.call(valid_args, scenario, [])
    assert first == second


# ---------------------------------------------------------------------------
# Read-logs cost scaling deserves a focused test (Phase A2 §implementation note)
# ---------------------------------------------------------------------------

def test_read_logs_cost_scales_with_lines(scenario):
    h = ReadLogsHandler()
    cheap = h.call({"scope": "test", "lines": 100}, scenario, [])
    pricey = h.call({"scope": "test", "lines": 200}, scenario, [])
    assert pricey.cost_units == pytest.approx(2.0 * cheap.cost_units)


def test_read_logs_truncates_when_lines_smaller_than_payload(scenario):
    out = ReadLogsHandler().call({"scope": "test", "lines": 10}, scenario, [])
    assert isinstance(out.payload, dict)
    assert len(out.payload["lines"]) == 10
    assert out.payload["truncated"] is True


# ---------------------------------------------------------------------------
# Integration: full 11-tool sweep + cumulative cost
# ---------------------------------------------------------------------------

def test_full_tool_loop_against_mock_scenario(scenario):
    expected = {p.id for p in TOOL_MATRIX}
    seen: set[str] = set()
    for handler_cls, valid_args, _inv, _key, _cost in [p.values for p in TOOL_MATRIX]:
        out = handler_cls().call(valid_args, scenario, [])
        assert isinstance(out, ToolOutput)
        seen.add(out.tool_name)
    assert seen == expected


def test_cost_charging_accumulates_correctly(client, a2_scenario):
    """Drive the env over WS with a sequence of tool calls; verify the budget
    deducts exactly the sum of each handler's reported ``cost_units``."""
    # Inject our richer scenario into a fresh CITriageEnv via build_app.
    factory = lambda: CITriageEnv(scenarios={a2_scenario.scenario_id: a2_scenario})  # noqa: E731
    app = build_app(env_factory=factory)
    c = TestClient(app)

    sequence = [
        CITriageAction.from_tool_call("read_logs", {"scope": "test", "lines": 100}),
        CITriageAction.from_tool_call("query_flake_history", {"test_name": a2_scenario.failure_summary.test_name}),
        CITriageAction.from_tool_call("recent_commits", {"branch": a2_scenario.failure_summary.branch, "limit": 5}),
        CITriageAction.from_tool_call("rerun_test", {"test_name": a2_scenario.failure_summary.test_name, "iterations": 2}),
        CITriageAction.from_tool_call("ping_owner", {"owner": "alice", "message": "hi"}),
    ]
    expected_costs = [0.001, 0.01, 0.01, 0.30, 0.083]

    with c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"scenario_id": a2_scenario.scenario_id}}))
        initial = json.loads(ws.receive_text())
        budget0 = initial["data"]["observation"]["payload"]["budget_remaining"]["cost_remaining"]

        running = budget0
        for action, expected in zip(sequence, expected_costs, strict=True):
            ws.send_text(json.dumps({"type": "step", "data": action.model_dump()}))
            resp = json.loads(ws.receive_text())
            charged = resp["data"]["observation"]["payload"]["tool_response"]["cost_charged"]
            assert charged == pytest.approx(expected)
            running -= charged
            assert resp["data"]["observation"]["payload"]["budget_remaining"]["cost_remaining"] == pytest.approx(running)


# ---------------------------------------------------------------------------
# utils smoke
# ---------------------------------------------------------------------------

def test_args_hash_is_stable_and_order_independent():
    a = args_hash({"x": 1, "y": [1, 2]})
    b = args_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert a != args_hash({"x": 2, "y": [1, 2]})


def test_deterministic_rng_is_reproducible():
    r1 = deterministic_rng(42, 3, "read_logs").random()
    r2 = deterministic_rng(42, 3, "read_logs").random()
    assert r1 == r2
    assert r1 != deterministic_rng(42, 4, "read_logs").random()
