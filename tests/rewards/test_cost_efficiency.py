import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.cost_efficiency import CostEfficiencyReward
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


@pytest.fixture
def scenario():
    return make_mock_scenario("real_bug")


def _minimal_trace(cost_per_step: float, n_steps: int) -> EpisodeTrace:
    """Build a trace with n_steps tool calls each costing cost_per_step."""
    budget = BudgetState(tool_calls_remaining=20, cost_remaining=10.0)
    history = [
        StepRecord(
            step=i,
            action=ToolCall(tool_name="read_logs", args={"scope": "test"}),
            observation=Observation(episode_id="e", step=i, budget_remaining=budget),
            cost_charged=cost_per_step,
        )
        for i in range(n_steps)
    ]
    return EpisodeTrace(
        episode=EpisodeState(
            episode_id="e",
            scenario_id="s",
            seed=1,
            step=n_steps,
            history=history,
            budget=budget,
            is_terminated=True,
        ),
        reward_breakdown=RewardBreakdown(schema_version="1.0", total=0.0, format_gate=True, components={}),
    )


def test_cost_efficiency_zero_spend_returns_one(scenario):
    trace = _minimal_trace(cost_per_step=0.0, n_steps=3)
    score = CostEfficiencyReward().score(trace, scenario)
    assert score.raw == pytest.approx(1.0)


def test_cost_efficiency_full_budget_returns_minus_one(scenario):
    # BUDGET_REFERENCE = 5.0; spend 5.0
    trace = _minimal_trace(cost_per_step=5.0, n_steps=1)
    score = CostEfficiencyReward().score(trace, scenario)
    assert score.raw == pytest.approx(-1.0)


def test_cost_efficiency_correct_case_returns_high_score(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CostEfficiencyReward().score(trace, scenario)
    assert score.raw > 0.9  # mock spends ~0.015 << 5.0


def test_cost_efficiency_wrong_case_returns_lower_score_than_good(scenario):
    good = make_mock_trajectory(scenario, outcome="good")
    bad = make_mock_trajectory(scenario, outcome="bad")
    s_good = CostEfficiencyReward().score(good, scenario)
    s_bad = CostEfficiencyReward().score(bad, scenario)
    # Both have same cost (same tool calls), so should be equal
    assert s_good.raw == pytest.approx(s_bad.raw)


def test_cost_efficiency_handles_no_terminal_action(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = CostEfficiencyReward().score(trace, scenario)
    assert -1.0 <= score.raw <= 1.0


def test_cost_efficiency_deterministic(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    component = CostEfficiencyReward()
    s1 = component.score(trace, scenario)
    s2 = component.score(trace, scenario)
    assert s1.raw == s2.raw


def test_cost_efficiency_score_is_in_documented_range(scenario):
    component = CostEfficiencyReward()
    for cost in (0.0, 0.1, 1.0, 2.5, 5.0, 10.0):
        trace = _minimal_trace(cost, 1)
        score = component.score(trace, scenario)
        assert -1.0 <= score.raw <= 1.0, f"out of range for cost={cost}: {score.raw}"


def test_cost_efficiency_subscores_are_meaningful(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CostEfficiencyReward().score(trace, scenario)
    assert "total_cost" in score.sub_scores
    assert "ratio" in score.sub_scores
