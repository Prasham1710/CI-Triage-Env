from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from ci_triage_env.env.server import CITriageEnv, create_app
from ci_triage_env.mock.scenario import make_mock_scenario
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeState
from ci_triage_env.schemas.observation import Observation


def test_server_boots():
    env = CITriageEnv(scenarios={make_mock_scenario().scenario_id: make_mock_scenario()})
    app = create_app(env)
    assert app.title == "CI Triage Env"


def test_reset_returns_valid_observation(client: TestClient):
    resp = client.post("/reset", json={})
    assert resp.status_code == 200
    obs = Observation.model_validate(resp.json())
    assert obs.failure_summary is not None
    assert obs.step == 0
    assert obs.is_terminal is False


def test_reset_with_specific_scenario_id(client: TestClient, known_scenario_id: str):
    resp = client.post("/reset", json={"scenario_id": known_scenario_id})
    assert resp.status_code == 200
    obs = Observation.model_validate(resp.json())
    assert obs.episode_id


def test_reset_with_unknown_scenario_id_404(client: TestClient):
    resp = client.post("/reset", json={"scenario_id": "does-not-exist"})
    assert resp.status_code == 404


def test_step_with_tool_call_returns_observation(client: TestClient, known_scenario_id: str):
    reset = client.post("/reset", json={"scenario_id": known_scenario_id}).json()
    episode_id = reset["episode_id"]
    call = ToolCall(tool_name="read_logs", args={"scope": "test"})
    resp = client.post("/step", json={"episode_id": episode_id, "action": call.model_dump()})
    assert resp.status_code == 200, resp.text
    obs = Observation.model_validate(resp.json())
    assert obs.tool_response is not None
    assert obs.tool_response.tool_name == "read_logs"
    assert obs.is_terminal is False


def test_step_with_terminal_action_marks_done(client: TestClient, known_scenario_id: str):
    reset = client.post("/reset", json={"scenario_id": known_scenario_id}).json()
    episode_id = reset["episode_id"]
    terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=DiagnosisLabel.RACE_FLAKE,
        confidence=0.8,
    )
    resp = client.post("/step", json={"episode_id": episode_id, "action": terminal.model_dump()})
    assert resp.status_code == 200, resp.text
    obs = Observation.model_validate(resp.json())
    assert obs.is_terminal is True

    state_resp = client.get(f"/state/{episode_id}")
    state = EpisodeState.model_validate(state_resp.json())
    assert state.is_terminated is True
    assert state.final_action is not None


def test_step_after_terminal_returns_400(client: TestClient, known_scenario_id: str):
    reset = client.post("/reset", json={"scenario_id": known_scenario_id}).json()
    episode_id = reset["episode_id"]
    terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=DiagnosisLabel.RACE_FLAKE,
        confidence=0.8,
    )
    client.post("/step", json={"episode_id": episode_id, "action": terminal.model_dump()})
    again = client.post("/step", json={"episode_id": episode_id, "action": terminal.model_dump()})
    assert again.status_code == 400


def test_state_endpoint_returns_episode_state(client: TestClient, known_scenario_id: str):
    reset = client.post("/reset", json={"scenario_id": known_scenario_id}).json()
    episode_id = reset["episode_id"]
    resp = client.get(f"/state/{episode_id}")
    assert resp.status_code == 200
    state = EpisodeState.model_validate(resp.json())
    assert state.episode_id == episode_id
    assert state.scenario_id == known_scenario_id
    assert state.step == 0
    assert state.is_terminated is False


def test_state_unknown_episode_404(client: TestClient):
    resp = client.get("/state/not-a-real-episode-id")
    assert resp.status_code == 404


def test_concurrent_resets_get_distinct_episode_ids(client: TestClient, known_scenario_id: str):
    def do_reset() -> str:
        return client.post("/reset", json={"scenario_id": known_scenario_id}).json()["episode_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: do_reset(), range(8)))

    assert len(set(ids)) == len(ids)


def test_mcp_endpoint_lists_all_11_tools(client: TestClient):
    resp = client.get("/mcp/tools")
    assert resp.status_code == 200
    tools = resp.json()
    names = {t["name"] for t in tools}
    assert names == {
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
    assert len(tools) == 11


def test_episode_seeding_deterministic(client: TestClient, known_scenario_id: str):
    def run_one() -> EpisodeState:
        reset = client.post(
            "/reset",
            json={"scenario_id": known_scenario_id, "seed_override": 12345},
        ).json()
        episode_id = reset["episode_id"]
        call = ToolCall(tool_name="read_logs", args={"scope": "test"})
        client.post("/step", json={"episode_id": episode_id, "action": call.model_dump()})
        terminal = TerminalAction(
            action_type="submit_diagnosis",
            diagnosis=DiagnosisLabel.RACE_FLAKE,
            confidence=0.6,
        )
        client.post("/step", json={"episode_id": episode_id, "action": terminal.model_dump()})
        return EpisodeState.model_validate(client.get(f"/state/{episode_id}").json())

    a = run_one()
    b = run_one()
    assert a.seed == b.seed == 12345
    assert a.step == b.step
    assert a.is_terminated and b.is_terminated
    assert [r.action for r in a.history] == [r.action for r in b.history]
    assert [r.cost_charged for r in a.history] == [r.cost_charged for r in b.history]
