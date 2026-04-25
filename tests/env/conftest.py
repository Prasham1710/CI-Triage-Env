import pytest
from fastapi.testclient import TestClient

from ci_triage_env.env.server import CITriageEnv, build_app
from ci_triage_env.mock.scenario import make_mock_scenario
from ci_triage_env.schemas.scenario import Scenario, ToolOutput


def _make_scenarios() -> dict:
    return {
        s.scenario_id: s
        for s in [
            make_mock_scenario("race_flake", seed=42),
            make_mock_scenario("real_bug", seed=7),
        ]
    }


def make_a2_scenario() -> Scenario:
    """Scenario with tool_outputs populated for every Phase A2 routing key.

    Used by ``tests/env/test_tools.py`` to exercise the real handlers against
    pre-baked payloads. Mirrors ``mock.scenario.make_mock_scenario`` for the
    non-tool fields so the same scenario can flow through ``CITriageEnv``.
    """

    base = make_mock_scenario("race_flake", seed=42)
    keyed_outputs: dict[str, ToolOutput] = {
        # read_logs: one per scope from the frozen schema
        **{
            f"read_logs:{scope}": ToolOutput(
                tool_name="read_logs",
                payload={"lines": [f"{scope.upper()} L{i}" for i in range(50)], "truncated": False},
                cost_units=0.001,
            )
            for scope in ("full", "test", "stderr", "kernel", "build")
        },
        # inspect_test_code keyed by test_name
        f"inspect_test_code:{base.failure_summary.test_name}": ToolOutput(
            tool_name="inspect_test_code",
            payload={"source": "def test():\n    assert race_condition()", "fixtures": ["fx_db"]},
            cost_units=0.05,
        ),
        # run_diagnostic: one per probe value (frozen enum)
        **{
            f"run_diagnostic:{probe}": ToolOutput(
                tool_name="run_diagnostic",
                payload={"ok": probe != "memory", "details": {"probe": probe}},
                cost_units=0.10,
            )
            for probe in ("network", "disk", "memory", "cpu")
        },
        # cluster_metrics: one per metric (frozen enum)
        **{
            f"cluster_metrics:{metric}": ToolOutput(
                tool_name="cluster_metrics",
                payload={"samples": [{"t": 0, "v": 1.0}, {"t": 1, "v": 1.2}]},
                cost_units=0.02,
            )
            for metric in ("queue_depth", "node_health", "network_latency", "disk_io")
        },
        f"query_flake_history:{base.failure_summary.test_name}": ToolOutput(
            tool_name="query_flake_history",
            payload={"failure_count": 7, "pass_count": 93, "recent_failures": [{"sha": "abc"}]},
            cost_units=0.01,
        ),
        f"recent_commits:{base.failure_summary.branch}": ToolOutput(
            tool_name="recent_commits",
            payload={"commits": [{"sha": f"c{i}", "msg": f"m{i}"} for i in range(15)]},
            cost_units=0.01,
        ),
        "check_owner:tests/unit/test_widget.py": ToolOutput(
            tool_name="check_owner",
            payload={"owner": "alice", "team": "infra", "contact": "alice@x"},
            cost_units=0.01,
        ),
        "rerun_test": ToolOutput(
            tool_name="rerun_test",
            payload={"results": [{"passed": True}, {"passed": False}, {"passed": True}]},
            cost_units=0.30,
        ),
        "quarantine_test": ToolOutput(
            tool_name="quarantine_test",
            payload={"quarantined": True, "ticket": "JIRA-42"},
            cost_units=0.0,
        ),
        "file_bug": ToolOutput(
            tool_name="file_bug",
            payload={"ticket_id": "BUG-7", "url": "https://example/BUG-7"},
            cost_units=0.5,
        ),
        "ping_owner": ToolOutput(
            tool_name="ping_owner",
            payload={"delivered": True},
            cost_units=0.083,
        ),
    }

    return base.model_copy(update={"tool_outputs": keyed_outputs})


@pytest.fixture
def a2_scenario() -> Scenario:
    return make_a2_scenario()


@pytest.fixture
def scenarios() -> dict:
    return _make_scenarios()


@pytest.fixture
def env_factory(scenarios):
    """Factory returning a fresh CITriageEnv per request, bound to fixed scenarios.

    Matches OpenEnv's per-session env model: each WebSocket session and each
    stateless /reset+/step request gets its own instance.
    """

    def _factory():
        return CITriageEnv(scenarios=scenarios)

    return _factory


@pytest.fixture
def app(env_factory):
    return build_app(env_factory=env_factory)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def known_scenario_id(scenarios) -> str:
    return next(iter(scenarios))
