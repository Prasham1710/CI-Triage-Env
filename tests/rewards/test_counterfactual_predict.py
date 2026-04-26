"""Tests for CounterfactualPredictReward component — dormant in v1."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.counterfactual_predict import CounterfactualPredictReward


def test_counterfactual_correct_case_returns_high_score() -> None:
    # In v1 probes never fire → always returns 0.0 even in "good" trajectory
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 0.0


def test_counterfactual_wrong_case_returns_low_score() -> None:
    # v1: still 0.0 since counterfactual_replay is always None
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="bad")
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 0.0


def test_no_probe_returns_zero() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    assert trace.counterfactual_replay is None
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 0.0
    assert score.weighted == 0.0


def test_counterfactual_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = CounterfactualPredictReward().score(no_terminal, scenario)
    assert score.raw == 0.0


def test_counterfactual_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = CounterfactualPredictReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_counterfactual_score_is_in_documented_range() -> None:
    for family in ["real_bug", "race_flake", "ambiguous"]:
        scenario = make_mock_scenario(family)
        trace = make_mock_trajectory(scenario, outcome="good")
        score = CounterfactualPredictReward().score(trace, scenario)
        # v1: always 0.0; generally in [-0.5, 1.0]
        assert -0.5 <= score.raw <= 1.0


def test_counterfactual_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CounterfactualPredictReward().score(trace, scenario)
    assert "fired" in score.sub_scores


def test_v1_default_weight_is_zero() -> None:
    assert CounterfactualPredictReward.default_weight == 0.0


def test_v1_weighted_always_zero() -> None:
    for family in ["real_bug", "ambiguous"]:
        scenario = make_mock_scenario(family)
        trace = make_mock_trajectory(scenario, outcome="good")
        score = CounterfactualPredictReward().score(trace, scenario)
        assert score.weighted == 0.0
