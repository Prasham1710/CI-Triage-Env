"""Tests for FormatGate reward component."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.format_gate import FormatGate
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation


def _dummy_obs() -> Observation:
    return Observation(
        episode_id="test",
        step=0,
        failure_summary=None,
        tool_response=None,
        budget_remaining=BudgetState(tool_calls_remaining=10, cost_remaining=1.0),
        is_terminal=False,
        probe_question=None,
    )


def test_format_gate_correct_case_returns_high_score() -> None:
    # Build a trajectory with only valid read_logs calls (scope is required)
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Replace history with a single valid tool call
    valid_record = StepRecord(
        step=0,
        action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
        observation=_dummy_obs(),
        cost_charged=0.001,
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [valid_record]})}
    )
    score = FormatGate().score(patched, scenario)
    assert score.raw == 1.0


def test_format_gate_wrong_case_returns_low_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")

    # Inject an unknown tool name
    bad_action = ToolCall(tool_name="__nonexistent_tool__", args={})
    bad_record = StepRecord(step=99, action=bad_action, observation=_dummy_obs(), cost_charged=0.0)
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": trace.episode.history + [bad_record]}
        )}
    )
    score = FormatGate().score(patched, scenario)
    assert score.raw == 0.0


def test_format_gate_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Build trajectory with only valid tool calls and no terminal action
    valid_record = StepRecord(
        step=0,
        action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
        observation=_dummy_obs(),
        cost_charged=0.001,
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": [valid_record], "final_action": None}
        )}
    )
    score = FormatGate().score(patched, scenario)
    # No terminal action → still valid if all tool calls are valid
    assert score.raw == 1.0


def test_format_gate_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    gate = FormatGate()
    s1 = gate.score(trace, scenario)
    s2 = gate.score(trace, scenario)
    assert s1.raw == s2.raw


def test_format_gate_score_is_in_documented_range() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Valid trajectory (single valid tool call)
    valid_record = StepRecord(
        step=0,
        action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
        observation=_dummy_obs(),
        cost_charged=0.001,
    )
    valid_trace = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [valid_record]})}
    )
    # Invalid trajectory (unknown tool)
    bad_record = StepRecord(
        step=0,
        action=ToolCall(tool_name="__bad__", args={}),
        observation=_dummy_obs(),
        cost_charged=0.0,
    )
    invalid_trace = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [bad_record]})}
    )
    assert FormatGate().score(valid_trace, scenario).raw == 1.0
    assert FormatGate().score(invalid_trace, scenario).raw == 0.0


def test_format_gate_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Use a valid trajectory to get the "valid" key
    valid_record = StepRecord(
        step=0,
        action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
        observation=_dummy_obs(),
        cost_charged=0.001,
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [valid_record]})}
    )
    score = FormatGate().score(patched, scenario)
    assert "valid" in score.sub_scores


def test_format_gate_invalid_args_fails() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")

    # read_logs requires "scope" arg — inject one without it
    bad_action = ToolCall(tool_name="read_logs", args={})  # missing required "scope"
    bad_record = StepRecord(step=99, action=bad_action, observation=_dummy_obs(), cost_charged=0.0)
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": [bad_record]}
        )}
    )
    score = FormatGate().score(patched, scenario)
    assert score.raw == 0.0
