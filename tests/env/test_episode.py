"""Phase A3 episode-lifecycle unit tests.

Most cases hit ``EpisodeManager`` directly so we can exercise budget gates and
truncation without spinning up the full WS layer; a couple drive over WS for
the post-terminal 400 / first-observation shape behaviors.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ci_triage_env.env.episode import (
    DEFAULT_TOOL_CALL_BUDGET,
    INVALID_ARGS_PENALTY_COST,
    OBSERVATION_PAYLOAD_CAP,
    EpisodeManager,
    EpisodeTerminatedError,
)
from ci_triage_env.env.server import CITriageEnv, build_app
from ci_triage_env.env.tools import ALL_TOOL_HANDLERS
from ci_triage_env.env.trace import build_trace, write_trace
from ci_triage_env.env.wire import CITriageAction
from ci_triage_env.schemas.action import SecondaryAction, TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.scenario import ToolOutput
from tests.env.conftest import make_a2_scenario

HANDLERS = {h.name: h for h in ALL_TOOL_HANDLERS}


def _new_manager(**overrides) -> EpisodeManager:
    scenario = make_a2_scenario()
    kwargs = {"scenario": scenario, "episode_id": "ep-test", "seed": 42}
    kwargs.update(overrides)
    return EpisodeManager(**kwargs)


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def test_budget_exhaustion_forces_terminal():
    """After 12 valid tool calls (default budget), the 13th force-terminates."""
    mgr = _new_manager()
    call = ToolCall(tool_name="query_flake_history", args={"test_name": mgr.scenario.failure_summary.test_name})
    h = HANDLERS["query_flake_history"]
    for _ in range(DEFAULT_TOOL_CALL_BUDGET):
        obs = mgr.apply_tool_call(call, h)
        assert obs.is_terminal is False
    final = mgr.apply_tool_call(call, h)
    assert final.is_terminal is True
    assert mgr.is_terminated is True
    assert mgr.budget_exhausted is True
    assert mgr.final_action is None


def test_budget_exhaustion_via_cost_remaining():
    mgr = _new_manager(max_tool_calls=100, cost_budget=0.01)
    call = ToolCall(tool_name="rerun_test", args={"test_name": "x"})  # cost 0.30
    h = HANDLERS["rerun_test"]
    first = mgr.apply_tool_call(call, h)
    assert first.is_terminal is False
    # BudgetState.cost_remaining is clamped to >= 0 (frozen schema constraint),
    # so the overrun lives on the raw ledger that the gate actually reads.
    assert mgr.budget.cost_remaining == 0.0
    assert mgr.raw_cost_remaining == pytest.approx(0.01 - 0.30)
    second = mgr.apply_tool_call(call, h)
    assert second.is_terminal is True
    assert mgr.budget_exhausted is True


# ---------------------------------------------------------------------------
# Terminal action
# ---------------------------------------------------------------------------

def test_terminal_action_records_final():
    mgr = _new_manager()
    action = TerminalAction(diagnosis=DiagnosisLabel.RACE_FLAKE, confidence=0.8)
    obs = mgr.apply_terminal(action)
    assert obs.is_terminal is True
    assert mgr.is_terminated is True
    assert mgr.final_action == action
    assert mgr.budget_exhausted is False


def test_terminal_action_with_invalid_diagnosis_rejected_at_wire(client, known_scenario_id):
    """An unknown diagnosis label fails Pydantic validation on the wire (422)."""
    bad_terminal = {
        "kind": "submit_diagnosis",
        "terminal": {
            "action_type": "submit_diagnosis",
            "diagnosis": "definitely_not_a_label",
            "confidence": 0.9,
        },
    }
    resp = client.post("/step", json={"action": bad_terminal})
    assert resp.status_code == 422


def test_terminal_action_with_secondary_actions():
    mgr = _new_manager()
    action = TerminalAction(
        diagnosis=DiagnosisLabel.REAL_BUG,
        confidence=0.9,
        secondary_actions=[
            SecondaryAction(name="file_bug", args={"title": "x", "summary": "y"}),
            SecondaryAction(name="ping_owner", args={"owner": "alice", "message": "fyi"}),
        ],
    )
    mgr.apply_terminal(action)
    state = mgr.to_state()
    assert state.final_action is not None
    assert len(state.final_action.secondary_actions) == 2
    names = {sa.name for sa in state.final_action.secondary_actions}
    assert names == {"file_bug", "ping_owner"}


def test_step_after_terminal_raises():
    mgr = _new_manager()
    mgr.apply_terminal(TerminalAction(diagnosis=DiagnosisLabel.AMBIGUOUS, confidence=0.5))
    with pytest.raises(EpisodeTerminatedError):
        mgr.apply_terminal(TerminalAction(diagnosis=DiagnosisLabel.AMBIGUOUS, confidence=0.5))


# ---------------------------------------------------------------------------
# Invalid args = cheap penalty
# ---------------------------------------------------------------------------

def test_invalid_tool_args_charges_cheap_penalty():
    mgr = _new_manager()
    bad = ToolCall(tool_name="read_logs", args={"scope": "totally_bogus_scope"})
    obs = mgr.apply_tool_call(bad, HANDLERS["read_logs"])
    assert obs.is_terminal is False
    assert obs.tool_response is not None
    assert "error" in obs.tool_response.output
    assert obs.tool_response.cost_charged == pytest.approx(INVALID_ARGS_PENALTY_COST)
    # exactly one tool-call slot consumed; cost_remaining lightly debited
    assert mgr.budget.tool_calls_remaining == DEFAULT_TOOL_CALL_BUDGET - 1
    assert mgr.budget.cost_remaining == pytest.approx(5.0 - INVALID_ARGS_PENALTY_COST)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def test_first_observation_has_failure_summary():
    mgr = _new_manager()
    initial = mgr.initial_observation()
    assert initial.failure_summary is not None
    assert initial.tool_response is None
    assert initial.step == 0


def test_subsequent_observations_have_tool_response_only():
    mgr = _new_manager()
    call = ToolCall(tool_name="check_owner", args={"target": "tests/unit/test_widget.py"})
    obs = mgr.apply_tool_call(call, HANDLERS["check_owner"])
    assert obs.failure_summary is None
    assert obs.tool_response is not None


# ---------------------------------------------------------------------------
# Payload truncation
# ---------------------------------------------------------------------------

def test_long_payload_truncated():
    """Stuff a 100k-char log into the scenario; verify truncation kicks in."""
    scenario = make_a2_scenario()
    huge_lines = [f"L{i:05d} " + "x" * 200 for i in range(500)]  # ~100k chars
    scenario = scenario.model_copy(
        update={
            "tool_outputs": {
                **scenario.tool_outputs,
                "read_logs:full": ToolOutput(
                    tool_name="read_logs",
                    payload={"lines": huge_lines, "truncated": False},
                    cost_units=0.001,
                ),
            }
        }
    )
    mgr = EpisodeManager(scenario=scenario, episode_id="ep-trunc", seed=1)
    call = ToolCall(tool_name="read_logs", args={"scope": "full", "lines": 2000})
    obs = mgr.apply_tool_call(call, HANDLERS["read_logs"])
    out = obs.tool_response.output
    assert isinstance(out, dict)
    serialized = json.dumps(out)
    # Cap is per-tool-response payload; we leave a little headroom for the
    # truncation marker line itself.
    assert len(serialized) <= OBSERVATION_PAYLOAD_CAP * 2
    assert out["truncated"] is True
    # Head + tail kept (truncation marker line is in the middle).
    line_strs = [line for line in out["lines"] if "lines truncated" in line]
    assert len(line_strs) == 1


def test_recent_commits_truncates_long_messages():
    scenario = make_a2_scenario()
    scenario = scenario.model_copy(
        update={
            "tool_outputs": {
                **scenario.tool_outputs,
                "recent_commits:main": ToolOutput(
                    tool_name="recent_commits",
                    payload={"commits": [{"sha": "a", "msg": "z" * 1000}]},
                    cost_units=0.01,
                ),
            }
        }
    )
    mgr = EpisodeManager(scenario=scenario, episode_id="ep-rc", seed=1)
    obs = mgr.apply_tool_call(
        ToolCall(tool_name="recent_commits", args={"branch": "main", "limit": 5}),
        HANDLERS["recent_commits"],
    )
    msg = obs.tool_response.output["commits"][0]["msg"]
    assert "...[truncated]" in msg
    assert len(msg) < 1000


# ---------------------------------------------------------------------------
# Trace writing + round-trip
# ---------------------------------------------------------------------------

def test_trace_written_on_termination(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_TRIAGE_TRACE_DIR", str(tmp_path))
    scenarios = {s.scenario_id: s for s in [make_a2_scenario()]}
    factory = lambda: CITriageEnv(scenarios=scenarios)  # noqa: E731
    app = build_app(env_factory=factory)
    sid = next(iter(scenarios))

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"scenario_id": sid}}))
        first = json.loads(ws.receive_text())
        episode_id = first["data"]["observation"]["payload"]["episode_id"]
        ws.send_text(
            json.dumps(
                {
                    "type": "step",
                    "data": CITriageAction.from_terminal(
                        DiagnosisLabel.RACE_FLAKE, confidence=0.7
                    ).model_dump(),
                }
            )
        )
        ws.receive_text()

    written = tmp_path / f"{episode_id}.json"
    assert written.exists()


def test_trace_round_trips_via_schema():
    mgr = _new_manager()
    mgr.apply_tool_call(
        ToolCall(tool_name="check_owner", args={"target": "tests/unit/test_widget.py"}),
        HANDLERS["check_owner"],
    )
    mgr.apply_terminal(TerminalAction(diagnosis=DiagnosisLabel.REAL_BUG, confidence=0.9))
    trace = build_trace(mgr)
    rebuilt = EpisodeTrace.model_validate_json(trace.model_dump_json())
    assert rebuilt == trace


def test_write_trace_creates_directory(tmp_path):
    mgr = _new_manager()
    mgr.apply_terminal(TerminalAction(diagnosis=DiagnosisLabel.AMBIGUOUS, confidence=0.5))
    sub = tmp_path / "deep" / "nested"
    written = write_trace(mgr, sub)
    assert written.exists()
    assert written.parent == sub
