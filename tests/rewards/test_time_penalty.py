import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.time_penalty import TimePenaltyReward
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


@pytest.fixture
def scenario():
    return make_mock_scenario("real_bug")


def _trace_with_n_tool_calls(n: int) -> EpisodeTrace:
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
    return EpisodeTrace(
        episode=EpisodeState(
            episode_id="e",
            scenario_id="s",
            seed=1,
            step=n,
            history=history,
            budget=budget,
            is_terminated=True,
        ),
        reward_breakdown=RewardBreakdown(schema_version="1.0", total=0.0, format_gate=True, components={}),
    )


def test_time_penalty_at_reference_steps_no_penalty(scenario):
    trace = _trace_with_n_tool_calls(TimePenaltyReward.REFERENCE_STEPS)
    score = TimePenaltyReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.0)


def test_time_penalty_below_reference_no_penalty(scenario):
    trace = _trace_with_n_tool_calls(2)
    score = TimePenaltyReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.0)


def test_time_penalty_above_reference_penalises(scenario):
    excess_steps = 5
    trace = _trace_with_n_tool_calls(TimePenaltyReward.REFERENCE_STEPS + excess_steps)
    score = TimePenaltyReward().score(trace, scenario)
    expected = -TimePenaltyReward.PER_STEP_PENALTY * excess_steps
    assert score.raw == pytest.approx(expected)


def test_time_penalty_correct_case_returns_high_score(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")  # 4 tool calls < 6
    score = TimePenaltyReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.0)


def test_time_penalty_wrong_case_same_steps(scenario):
    bad = make_mock_trajectory(scenario, outcome="bad")
    score = TimePenaltyReward().score(bad, scenario)
    assert score.raw == pytest.approx(0.0)  # same step count as good


def test_time_penalty_handles_no_terminal_action(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = TimePenaltyReward().score(trace, scenario)
    assert -1.0 <= score.raw <= 0.0


def test_time_penalty_deterministic(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    component = TimePenaltyReward()
    s1 = component.score(trace, scenario)
    s2 = component.score(trace, scenario)
    assert s1.raw == s2.raw


def test_time_penalty_score_is_in_documented_range(scenario):
    component = TimePenaltyReward()
    for n in (0, 1, 6, 10, 50, 100):
        trace = _trace_with_n_tool_calls(n)
        score = component.score(trace, scenario)
        assert -1.0 <= score.raw <= 0.0, f"out of range for n={n}: {score.raw}"


def test_time_penalty_subscores_are_meaningful(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = TimePenaltyReward().score(trace, scenario)
    assert "steps" in score.sub_scores
    assert "excess" in score.sub_scores


def test_time_penalty_capped_at_minus_one(scenario):
    trace = _trace_with_n_tool_calls(1000)
    score = TimePenaltyReward().score(trace, scenario)
    assert score.raw >= -1.0
