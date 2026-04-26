"""Tests for InvestigationReward component."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.investigation import InvestigationReward
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation


def _dummy_obs(step: int = 0) -> Observation:
    return Observation(
        episode_id="test",
        step=step,
        failure_summary=None,
        tool_response=None,
        budget_remaining=BudgetState(tool_calls_remaining=10, cost_remaining=1.0),
        is_terminal=False,
        probe_question=None,
    )


def _make_tool_record(tool_name: str, args: dict, step: int = 0) -> StepRecord:
    return StepRecord(
        step=step,
        action=ToolCall(tool_name=tool_name, args=args),
        observation=_dummy_obs(step),
        cost_charged=0.001,
    )


def test_investigation_correct_case_returns_high_score() -> None:
    # Trajectory that calls all informative tools in cheap-before-expensive order
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = InvestigationReward().score(trace, scenario)
    # Mock trajectory calls read_logs, query_flake_history, recent_commits, rerun_test
    # informative tools for real_bug mock: read_logs, query_flake_history, rerun_test
    assert score.raw > 0.0


def test_investigation_wrong_case_returns_low_score() -> None:
    # Trajectory with no informative tools called (mock informative_tools = read_logs, query_flake_history, rerun_test)
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Use only tools NOT in informative_tools and in wrong order (expensive first, cheap second)
    no_informative = [
        _make_tool_record("ping_owner", {}, step=0),     # expensive, not informative
        _make_tool_record("recent_commits", {}, step=1),  # cheap after expensive = ordering violation
    ]
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": no_informative})}
    )
    score = InvestigationReward().score(patched, scenario)
    # coverage=0.0, ordering=0.8 (1 violation), redundancy=0 → raw=0.6*0+0.2*0.8=0.16
    assert score.raw <= 0.2


def test_investigation_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = InvestigationReward().score(no_terminal, scenario)
    assert -1.0 <= score.raw <= 1.0


def test_investigation_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = InvestigationReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_investigation_score_is_in_documented_range() -> None:
    for family in ["real_bug", "race_flake", "timing_flake", "ambiguous"]:
        scenario = make_mock_scenario(family)
        for outcome in ["good", "bad"]:
            trace = make_mock_trajectory(scenario, outcome=outcome)
            score = InvestigationReward().score(trace, scenario)
            assert -1.0 <= score.raw <= 1.0


def test_investigation_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = InvestigationReward().score(trace, scenario)
    assert "coverage" in score.sub_scores
    assert "ordering" in score.sub_scores
    assert "redundancy_penalty" in score.sub_scores


def test_investigation_redundancy_penalised() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Duplicate tool call with same args
    dup = _make_tool_record("read_logs", {"scope": "full"}, step=0)
    dup2 = _make_tool_record("read_logs", {"scope": "full"}, step=1)
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [dup, dup2]})}
    )
    score_dup = InvestigationReward().score(patched, scenario)
    only_one = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [dup]})}
    )
    score_single = InvestigationReward().score(only_one, scenario)
    assert score_dup.sub_scores["redundancy_penalty"] < score_single.sub_scores["redundancy_penalty"]
