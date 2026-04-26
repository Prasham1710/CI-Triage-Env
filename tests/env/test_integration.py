"""Phase A3 end-to-end integration: reset → 5 tool calls → submit_diagnosis → trace written.

Drives the canonical OpenEnv WebSocket session protocol; the env factory
points at a single A2-rich scenario so the tools route to real payloads.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ci_triage_env.env.server import CITriageEnv, build_app
from ci_triage_env.env.wire import CITriageAction
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from tests.env.conftest import make_a2_scenario


def test_full_episode_with_mock_scenario(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_TRIAGE_TRACE_DIR", str(tmp_path))

    scenario = make_a2_scenario()
    factory = lambda: CITriageEnv(scenarios={scenario.scenario_id: scenario})  # noqa: E731
    app = build_app(env_factory=factory)
    client = TestClient(app)

    tool_calls = [
        CITriageAction.from_tool_call("read_logs", {"scope": "full", "lines": 100}),
        CITriageAction.from_tool_call(
            "query_flake_history", {"test_name": scenario.failure_summary.test_name}
        ),
        CITriageAction.from_tool_call(
            "recent_commits", {"branch": scenario.failure_summary.branch, "limit": 3}
        ),
        CITriageAction.from_tool_call("check_owner", {"target": "tests/unit/test_widget.py"}),
        CITriageAction.from_tool_call("rerun_test", {"test_name": "x", "iterations": 1}),
    ]
    terminal = CITriageAction.from_terminal(
        DiagnosisLabel.REAL_BUG,
        confidence=0.85,
        secondary_actions=[],
    )

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"scenario_id": scenario.scenario_id}}))
        first = json.loads(ws.receive_text())
        episode_id = first["data"]["observation"]["payload"]["episode_id"]
        assert first["data"]["observation"]["payload"]["failure_summary"] is not None

        for action in tool_calls:
            ws.send_text(json.dumps({"type": "step", "data": action.model_dump()}))
            obs = json.loads(ws.receive_text())
            assert obs["data"]["done"] is False
            assert obs["data"]["observation"]["payload"]["tool_response"] is not None

        ws.send_text(json.dumps({"type": "step", "data": terminal.model_dump()}))
        final = json.loads(ws.receive_text())
        assert final["data"]["done"] is True
        assert final["data"]["observation"]["payload"]["is_terminal"] is True

        ws.send_text(json.dumps({"type": "state"}))
        state = json.loads(ws.receive_text())["data"]["payload"]
        assert state["is_terminated"] is True
        assert state["final_action"]["diagnosis"] == "real_bug"
        # 1 reset implicit + 5 tool calls + 1 terminal step recorded
        assert len(state["history"]) == 6

    trace_file = tmp_path / f"{episode_id}.json"
    assert trace_file.exists()
    payload = json.loads(trace_file.read_text())
    assert payload["episode"]["episode_id"] == episode_id
    assert payload["reward_breakdown"]["format_gate"] is False  # placeholder until Branch C
