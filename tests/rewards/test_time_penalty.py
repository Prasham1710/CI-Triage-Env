"""Tests for TimePenaltyReward component."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.time_penalty import TimePenaltyReward
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation


def _make_tool_records(n: int) -> list[StepRecord]:
    obs = Observation(
        episode_id="test",
        step=0,
        failure_summary=None,
        tool_response=None,
        budget_remaining=BudgetState(tool_calls_remaining=10, cost_remaining=1.0),
        is_terminal=False,
        probe_question=None,
    )
    return [
        StepRecord(step=i, action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
                   observation=obs, cost_charged=0.001)
        for i in range(n)
    ]


def test_time_penalty_correct_case_returns_high_score() -> None:
    # <= REFERENCE_STEPS tool calls → no penalty
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": _make_tool_records(4)})}
    )
    score = TimePenaltyReward().score(patched, scenario)
    assert score.raw == 0.0


def test_time_penalty_wrong_case_returns_low_score() -> None:
    # Many tool calls → large penalty
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": _make_tool_records(56)})}
    )
    score = TimePenaltyReward().score(patched, scenario)
    assert score.raw == -1.0  # capped at floor


def test_time_penalty_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = TimePenaltyReward().score(no_terminal, scenario)
    assert -1.0 <= score.raw <= 0.0


def test_time_penalty_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = TimePenaltyReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_time_penalty_score_is_in_documented_range() -> None:
    scenario = make_mock_scenario("real_bug")
    for n in [0, 3, 6, 10, 60]:
        trace = make_mock_trajectory(scenario, outcome="good")
        patched = trace.model_copy(
            update={"episode": trace.episode.model_copy(update={"history": _make_tool_records(n)})}
        )
        score = TimePenaltyReward().score(patched, scenario)
        assert -1.0 <= score.raw <= 0.0


def test_time_penalty_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = TimePenaltyReward().score(trace, scenario)
    assert "steps" in score.sub_scores
    assert "excess" in score.sub_scores
    assert score.sub_scores["steps"] >= 0
