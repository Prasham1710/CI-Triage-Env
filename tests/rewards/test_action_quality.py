"""Tests for ActionQualityReward component."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.action_quality import ACTION_REWARD_MATRIX, ActionQualityReward
from ci_triage_env.schemas.action import SecondaryAction, TerminalAction


def _patch_secondary(trace, secondary_actions):
    new_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=trace.episode.final_action.diagnosis,
        confidence=trace.episode.final_action.confidence,
        secondary_actions=secondary_actions,
    )
    return trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": new_terminal})}
    )


def test_action_quality_correct_case_returns_high_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = _patch_secondary(trace, [SecondaryAction(name="file_bug", args={})])
    score = ActionQualityReward().score(patched, scenario)
    assert score.raw > 0.5


def test_action_quality_wrong_case_returns_low_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # quarantine_test on real_bug is catastrophically bad
    patched = _patch_secondary(trace, [SecondaryAction(name="quarantine_test", args={})])
    score = ActionQualityReward().score(patched, scenario)
    assert score.raw < 0.0


def test_action_quality_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = ActionQualityReward().score(no_terminal, scenario)
    assert score.raw == -0.5


def test_action_quality_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = ActionQualityReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_action_quality_score_is_in_documented_range() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Maximum stacking: multiple good actions
    patched = _patch_secondary(trace, [
        SecondaryAction(name="file_bug", args={}),
        SecondaryAction(name="ping_owner", args={}),
    ])
    score = ActionQualityReward().score(patched, scenario)
    assert -2.0 <= score.raw <= 1.5


def test_action_quality_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = _patch_secondary(trace, [SecondaryAction(name="file_bug", args={})])
    score = ActionQualityReward().score(patched, scenario)
    assert "file_bug" in score.sub_scores


def test_quarantine_real_bug_is_worst() -> None:
    worst = ACTION_REWARD_MATRIX[("quarantine_test", "real_bug")]
    assert worst == -1.5
    all_values = list(ACTION_REWARD_MATRIX.values())
    assert all(v >= worst for v in all_values)


def test_action_quality_no_secondary_neutral() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Default mock trajectory has no secondary actions
    score = ActionQualityReward().score(trace, scenario)
    assert score.raw == 0.0
