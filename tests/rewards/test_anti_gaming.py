import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.anti_gaming import AntiGamingReward
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


@pytest.fixture
def scenario():
    return make_mock_scenario("real_bug")


@pytest.fixture
def ambiguous_scenario():
    return make_mock_scenario("ambiguous")


def _trace_with_n_tools(n: int, final_action: TerminalAction | None = None) -> EpisodeTrace:
    budget = BudgetState(tool_calls_remaining=20, cost_remaining=10.0)
    history = [
        StepRecord(
            step=i,
            action=ToolCall(tool_name="read_logs", args={"scope": "test"}),
            observation=Observation(episode_id="e", step=i, budget_remaining=budget),
            cost_charged=0.001,
        )
        for i in range(n)
    ]
    if final_action is not None:
        history.append(
            StepRecord(
                step=n,
                action=final_action,
                observation=Observation(episode_id="e", step=n, budget_remaining=budget),
                cost_charged=0.0,
            )
        )
    return EpisodeTrace(
        episode=EpisodeState(
            episode_id="e",
            scenario_id="s",
            seed=1,
            step=n,
            history=history,
            budget=budget,
            is_terminated=True,
            final_action=final_action,
        ),
        reward_breakdown=RewardBreakdown(schema_version="1.0", total=0.0, format_gate=True, components={}),
    )


def test_anti_gaming_correct_case_returns_high_score(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = AntiGamingReward().score(trace, scenario)
    assert score.raw >= 0.0


def test_anti_gaming_wrong_case_returns_low_score(scenario):
    trace = make_mock_trajectory(scenario, outcome="bad")
    score = AntiGamingReward().score(trace, scenario)
    # wrong diagnosis but not ambiguous → brier_bonus=0.0; still passes no_info guard
    assert score.raw >= -1.5


def test_anti_gaming_handles_no_terminal_action(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = AntiGamingReward().score(trace, scenario)
    assert -1.5 <= score.raw <= 1.0


def test_anti_gaming_deterministic(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    component = AntiGamingReward()
    s1 = component.score(trace, scenario)
    s2 = component.score(trace, scenario)
    assert s1.raw == s2.raw


def test_anti_gaming_score_is_in_documented_range(scenario):
    component = AntiGamingReward()
    for outcome in ("good", "bad", "abstain"):
        trace = make_mock_trajectory(scenario, outcome=outcome)
        score = component.score(trace, scenario)
        assert -1.5 <= score.raw <= 1.0


def test_anti_gaming_subscores_are_meaningful(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = AntiGamingReward().score(trace, scenario)
    assert "no_info_penalty" in score.sub_scores
    assert "quarantine_rate" in score.sub_scores
    assert "quarantine_penalty" in score.sub_scores
    assert "brier_bonus" in score.sub_scores


def test_no_info_action_penalty_applied(scenario):
    # Final action with only 1 tool call → no_info_penalty = -0.5
    terminal = TerminalAction(diagnosis=DiagnosisLabel.REAL_BUG, confidence=0.9)
    trace = _trace_with_n_tools(1, final_action=terminal)
    score = AntiGamingReward().score(trace, scenario)
    assert score.sub_scores["no_info_penalty"] == pytest.approx(-0.5)


def test_no_info_action_no_penalty_with_enough_tools(scenario):
    # 2 tool calls → no_info_penalty = 0.0
    terminal = TerminalAction(diagnosis=DiagnosisLabel.REAL_BUG, confidence=0.9)
    trace = _trace_with_n_tools(2, final_action=terminal)
    score = AntiGamingReward().score(trace, scenario)
    assert score.sub_scores["no_info_penalty"] == pytest.approx(0.0)


def test_quarantine_rate_above_threshold_penalises():
    # 50 quarantine_test actions out of 50 → rate = 1.0 → penalty = -(1.0 - 0.3) * 2 = -1.4
    recent = ["quarantine_test"] * 50
    component = AntiGamingReward(recent_episode_actions=recent)
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = component.score(trace, scenario)
    assert score.sub_scores["quarantine_penalty"] < 0.0


def test_quarantine_rate_below_threshold_no_penalty():
    recent = ["file_bug"] * 50
    component = AntiGamingReward(recent_episode_actions=recent)
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = component.score(trace, scenario)
    assert score.sub_scores["quarantine_penalty"] == pytest.approx(0.0)


def test_brier_calibration_perfect_match_bonus(ambiguous_scenario):
    # confidence_target = 0.5, predicted = 0.5 → brier = 0.0, bonus = 0.5
    trace = make_mock_trajectory(ambiguous_scenario, outcome="abstain")
    # abstain trajectory sets confidence=0.5, which matches target=0.5
    score = AntiGamingReward().score(trace, ambiguous_scenario)
    assert score.sub_scores["brier_bonus"] == pytest.approx(0.5, abs=0.01)


def test_brier_calibration_worst_mismatch(ambiguous_scenario):
    # confidence_target=0.5, predicted=1.0 → brier=0.25, bonus=0.375
    terminal = TerminalAction(diagnosis=DiagnosisLabel.AMBIGUOUS, confidence=1.0)
    trace = _trace_with_n_tools(3, final_action=terminal)
    trace.episode.scenario_id = ambiguous_scenario.scenario_id
    score = AntiGamingReward().score(trace, ambiguous_scenario)
    assert score.sub_scores["brier_bonus"] < 0.5


def test_brier_calibration_not_applied_on_unambiguous(scenario):
    # scenario is not ambiguous → brier_bonus stays 0.0
    trace = make_mock_trajectory(scenario, outcome="good")
    score = AntiGamingReward().score(trace, scenario)
    assert score.sub_scores["brier_bonus"] == pytest.approx(0.0)


def test_anti_gaming_empty_recent_actions_no_quarantine_penalty(scenario):
    component = AntiGamingReward(recent_episode_actions=[])
    trace = make_mock_trajectory(scenario, outcome="good")
    score = component.score(trace, scenario)
    assert score.sub_scores["quarantine_rate"] == pytest.approx(0.0)
    assert score.sub_scores["quarantine_penalty"] == pytest.approx(0.0)
