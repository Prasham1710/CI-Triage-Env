"""Tests for AntiGamingReward component."""

from __future__ import annotations

import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.anti_gaming import AntiGamingReward
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


def test_anti_gaming_correct_case_returns_high_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = AntiGamingReward().score(trace, scenario)
    # Good trajectory: ≥ 2 tool calls, no quarantine abuse, non-ambiguous
    assert score.raw >= 0.0


def test_anti_gaming_wrong_case_returns_low_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Simulate no-info-action: only 1 tool call before terminal
    one_tool = [
        StepRecord(step=0, action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
                   observation=_dummy_obs(), cost_charged=0.001)
    ]
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": one_tool})}
    )
    score = AntiGamingReward().score(patched, scenario)
    assert score.raw <= -0.5


def test_anti_gaming_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = AntiGamingReward().score(no_terminal, scenario)
    # No terminal → no no-info-action penalty; result depends on other guards
    assert -1.5 <= score.raw <= 1.0


def test_anti_gaming_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = AntiGamingReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_anti_gaming_score_is_in_documented_range() -> None:
    for family in ["real_bug", "race_flake", "ambiguous"]:
        scenario = make_mock_scenario(family)
        for outcome in ["good", "bad"]:
            trace = make_mock_trajectory(scenario, outcome=outcome)
            score = AntiGamingReward().score(trace, scenario)
            assert -1.5 <= score.raw <= 1.0


def test_anti_gaming_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = AntiGamingReward().score(trace, scenario)
    assert "no_info_penalty" in score.sub_scores
    assert "quarantine_rate" in score.sub_scores
    assert "brier_bonus" in score.sub_scores


def test_brier_calibration_perfect_match_bonus() -> None:
    scenario = make_mock_scenario("ambiguous")
    # confidence_target=0.5 for ambiguous mock; abstain trajectory uses confidence=0.5
    trace = make_mock_trajectory(scenario, outcome="abstain")
    score = AntiGamingReward().score(trace, scenario)
    # Perfect match: brier=(0.5-0.5)^2=0; bonus=0.5*(1-0)=0.5
    assert score.sub_scores["brier_bonus"] == pytest.approx(0.5, abs=1e-6)


def test_quarantine_rate_above_threshold_penalizes() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # 100% quarantine rate → well above 30% threshold
    comp = AntiGamingReward(recent_episode_actions=["quarantine_test"] * 50)
    score = comp.score(trace, scenario)
    assert score.sub_scores["quarantine_penalty"] < 0.0
