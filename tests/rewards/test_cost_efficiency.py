"""Tests for CostEfficiencyReward component."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.cost_efficiency import CostEfficiencyReward


def test_cost_efficiency_correct_case_returns_high_score() -> None:
    # Low-cost trajectory (mock costs ~0.016) → should be positive
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CostEfficiencyReward().score(trace, scenario)
    assert score.raw > 0.0


def test_cost_efficiency_wrong_case_returns_low_score() -> None:
    # Simulate a trajectory that spent the full budget
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Patch each step to have spent the full BUDGET_REFERENCE
    from ci_triage_env.rewards.cost_efficiency import CostEfficiencyReward as CR
    budget_each = CR.BUDGET_REFERENCE / max(len(trace.episode.history), 1)
    patched_history = [
        r.model_copy(update={"cost_charged": budget_each})
        for r in trace.episode.history
    ]
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": patched_history})}
    )
    score = CR().score(patched, scenario)
    assert score.raw <= -0.9


def test_cost_efficiency_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    # Still scores based on cost; no terminal doesn't affect this component
    score = CostEfficiencyReward().score(no_terminal, scenario)
    assert -1.0 <= score.raw <= 1.0


def test_cost_efficiency_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = CostEfficiencyReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_cost_efficiency_score_is_in_documented_range() -> None:
    for family in ["real_bug", "race_flake", "ambiguous"]:
        scenario = make_mock_scenario(family)
        for outcome in ["good", "bad"]:
            trace = make_mock_trajectory(scenario, outcome=outcome)
            score = CostEfficiencyReward().score(trace, scenario)
            assert -1.0 <= score.raw <= 1.0


def test_cost_efficiency_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CostEfficiencyReward().score(trace, scenario)
    assert "total_cost" in score.sub_scores
    assert "ratio" in score.sub_scores
    assert score.sub_scores["total_cost"] >= 0.0
