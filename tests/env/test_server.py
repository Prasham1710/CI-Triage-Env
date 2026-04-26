"""Phase A1 server tests against the OpenEnv-canonical wire format.

OpenEnv's HTTP ``/reset`` and ``/step`` endpoints are stateless (a fresh env
instance is constructed per request). State-preserving multi-step flows go
over the WebSocket ``/ws`` endpoint, which is the canonical OpenEnv session
path used by ``EnvClient``. Stateful behavior is therefore validated over
``/ws``; HTTP endpoints are validated for shape and error handling.
"""

from __future__ import annotations

import json

from ci_triage_env.env.server import build_app
from ci_triage_env.env.wire import CITriageAction
from ci_triage_env.schemas.diagnosis import DiagnosisLabel


def _assert_observation_envelope(payload: dict) -> None:
    assert "observation" in payload
    assert "reward" in payload
    assert "done" in payload


# ---------------------------------------------------------------------------
# A1.1 Server boot
# ---------------------------------------------------------------------------

def test_server_boots(env_factory):
    app = build_app(env_factory=env_factory)
    assert app.title  # FastAPI app instantiated


# ---------------------------------------------------------------------------
# A1.2 / A1.3 HTTP /reset shape
# ---------------------------------------------------------------------------

def test_reset_returns_valid_observation(client):
    resp = client.post("/reset", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_observation_envelope(body)
    assert body["done"] is False
    obs_payload = body["observation"]["payload"]
    assert obs_payload["failure_summary"] is not None


def test_reset_with_specific_scenario_id(client, known_scenario_id):
    resp = client.post("/reset", json={"scenario_id": known_scenario_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    obs_payload = body["observation"]["payload"]
    assert obs_payload["episode_id"]


# ---------------------------------------------------------------------------
# A1.4 Unknown scenario surfaces as a server error
# ---------------------------------------------------------------------------

def test_reset_with_unknown_scenario_id_errors(app):
    from fastapi.testclient import TestClient

    nonraising = TestClient(app, raise_server_exceptions=False)
    resp = nonraising.post("/reset", json={"scenario_id": "does-not-exist"})
    assert resp.status_code >= 400


# ---------------------------------------------------------------------------
# A1.5–A1.8: Stateful flows over WebSocket /ws
# ---------------------------------------------------------------------------

def _ws_send_recv(ws, message: dict) -> dict:
    ws.send_text(json.dumps(message))
    return json.loads(ws.receive_text())


def test_step_with_tool_call_returns_observation(client, known_scenario_id):
    with client.websocket_connect("/ws") as ws:
        _ws_send_recv(ws, {"type": "reset", "data": {"scenario_id": known_scenario_id}})
        action = CITriageAction.from_tool_call("read_logs", {"scope": "test"})
        resp = _ws_send_recv(ws, {"type": "step", "data": action.model_dump()})
    assert resp["type"] == "observation"
    payload = resp["data"]["observation"]["payload"]
    assert payload["tool_response"] is not None
    assert payload["tool_response"]["tool_name"] == "read_logs"
    assert resp["data"]["done"] is False


def test_step_with_terminal_action_marks_done(client, known_scenario_id):
    with client.websocket_connect("/ws") as ws:
        _ws_send_recv(ws, {"type": "reset", "data": {"scenario_id": known_scenario_id}})
        terminal = CITriageAction.from_terminal(DiagnosisLabel.RACE_FLAKE, confidence=0.8)
        step = _ws_send_recv(ws, {"type": "step", "data": terminal.model_dump()})
        assert step["data"]["done"] is True
        assert step["data"]["observation"]["payload"]["is_terminal"] is True

        state = _ws_send_recv(ws, {"type": "state"})
    assert state["type"] == "state"
    assert state["data"]["payload"]["is_terminated"] is True
    assert state["data"]["payload"]["final_action"] is not None


def test_step_after_terminal_returns_error(client, known_scenario_id):
    with client.websocket_connect("/ws") as ws:
        _ws_send_recv(ws, {"type": "reset", "data": {"scenario_id": known_scenario_id}})
        terminal = CITriageAction.from_terminal(DiagnosisLabel.RACE_FLAKE, confidence=0.8)
        _ws_send_recv(ws, {"type": "step", "data": terminal.model_dump()})
        again = _ws_send_recv(ws, {"type": "step", "data": terminal.model_dump()})
    assert again["type"] == "error"


def test_state_endpoint_returns_episode_state(client, known_scenario_id):
    with client.websocket_connect("/ws") as ws:
        _ws_send_recv(ws, {"type": "reset", "data": {"scenario_id": known_scenario_id}})
        state = _ws_send_recv(ws, {"type": "state"})
    assert state["type"] == "state"
    payload = state["data"]["payload"]
    assert payload is not None
    assert payload["scenario_id"] == known_scenario_id
    assert payload["is_terminated"] is False


# ---------------------------------------------------------------------------
# A1.9 Distinct episode_ids across concurrent sessions
# ---------------------------------------------------------------------------

def test_concurrent_ws_sessions_get_distinct_episode_ids(client, known_scenario_id):
    episode_ids: list[str] = []
    with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        for ws in (ws_a, ws_b):
            obs = _ws_send_recv(ws, {"type": "reset", "data": {"scenario_id": known_scenario_id}})
            episode_ids.append(obs["data"]["observation"]["payload"]["episode_id"])
    assert episode_ids[0] != episode_ids[1]


# ---------------------------------------------------------------------------
# A1.10 MCP /mcp tools/list — canonical OpenEnv MCP discovery
# ---------------------------------------------------------------------------

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


def test_mcp_endpoint_lists_all_11_tools(client):
    # Establish an MCP session first so subsequent tools/list shares it.
    create = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "1", "method": "openenv/session/create", "params": {}},
    )
    assert create.status_code == 200, create.text
    session_id = create.json()["result"]["session_id"]

    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "2", "method": "tools/list", "params": {}},
        headers={"X-OpenEnv-Session-Id": session_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "result" in body, body
    tools = body["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == EXPECTED_TOOL_NAMES
    assert len(tools) == 11


# ---------------------------------------------------------------------------
# A1.11 Deterministic seeding
# ---------------------------------------------------------------------------

def test_episode_seeding_deterministic(client, known_scenario_id):
    def run_one() -> dict:
        with client.websocket_connect("/ws") as ws:
            _ws_send_recv(
                ws,
                {
                    "type": "reset",
                    "data": {"scenario_id": known_scenario_id, "seed": 12345},
                },
            )
            action = CITriageAction.from_tool_call("read_logs", {"scope": "test"})
            _ws_send_recv(ws, {"type": "step", "data": action.model_dump()})
            terminal = CITriageAction.from_terminal(DiagnosisLabel.RACE_FLAKE, confidence=0.6)
            _ws_send_recv(ws, {"type": "step", "data": terminal.model_dump()})
            return _ws_send_recv(ws, {"type": "state"})["data"]["payload"]

    a = run_one()
    b = run_one()
    assert a["seed"] == b["seed"] == 12345
    assert a["step"] == b["step"]
    assert a["is_terminated"] and b["is_terminated"]
    assert [r["action"] for r in a["history"]] == [r["action"] for r in b["history"]]
    assert [r["cost_charged"] for r in a["history"]] == [r["cost_charged"] for r in b["history"]]


# ---------------------------------------------------------------------------
# A1.12 /schema is exposed (OpenEnv standard surface)
# ---------------------------------------------------------------------------

def test_schema_endpoint_returns_action_observation_state(client):
    resp = client.get("/schema")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "action" in body
    assert "observation" in body
