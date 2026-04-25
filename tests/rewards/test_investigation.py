import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.investigation import InvestigationReward
from ci_triage_env.schemas.action import ToolCall
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


@pytest.fixture
def scenario():
    # informative_tools = ["read_logs", "query_flake_history", "rerun_test"]
    return make_mock_scenario("real_bug")


def _trace_with_tools(tool_names: list[str]) -> EpisodeTrace:
    budget = BudgetState(tool_calls_remaining=20, cost_remaining=10.0)
    history = [
        StepRecord(
            step=i,
            action=ToolCall(tool_name=t, args={}),
            observation=Observation(episode_id="e", step=i, budget_remaining=budget),
            cost_charged=0.001,
        )
        for i, t in enumerate(tool_names)
    ]
    return EpisodeTrace(
        episode=EpisodeState(
            episode_id="e",
            scenario_id="s",
            seed=1,
            step=len(tool_names),
            history=history,
            budget=budget,
            is_terminated=True,
        ),
        reward_breakdown=RewardBreakdown(schema_version="1.0", total=0.0, format_gate=True, components={}),
    )


def test_investigation_correct_case_returns_high_score(scenario):
    # Call all informative tools in correct order
    trace = _trace_with_tools(["read_logs", "query_flake_history", "rerun_test"])
    score = InvestigationReward().score(trace, scenario)
    assert score.raw > 0.5


def test_investigation_wrong_case_returns_low_score(scenario):
    # No informative tools called at all
    trace = _trace_with_tools([])
    score = InvestigationReward().score(trace, scenario)
    assert score.raw < 0.5


def test_investigation_handles_no_terminal_action(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = InvestigationReward().score(trace, scenario)
    assert -1.0 <= score.raw <= 1.0


def test_investigation_deterministic(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    component = InvestigationReward()
    s1 = component.score(trace, scenario)
    s2 = component.score(trace, scenario)
    assert s1.raw == s2.raw


def test_investigation_score_is_in_documented_range(scenario):
    component = InvestigationReward()
    for outcome in ("good", "bad", "abstain"):
        trace = make_mock_trajectory(scenario, outcome=outcome)
        score = component.score(trace, scenario)
        assert -1.0 <= score.raw <= 1.0


def test_investigation_subscores_are_meaningful(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = InvestigationReward().score(trace, scenario)
    assert "coverage" in score.sub_scores
    assert "ordering" in score.sub_scores
    assert "redundancy_penalty" in score.sub_scores


def test_investigation_redundancy_penalty_applied(scenario):
    # Calling the same tool twice with the same args → redundancy
    trace = _trace_with_tools(["read_logs", "read_logs"])
    score = InvestigationReward().score(trace, scenario)
    assert score.sub_scores["redundancy_penalty"] < 0.0


def test_investigation_no_redundancy_for_different_args(scenario):
    budget = BudgetState(tool_calls_remaining=20, cost_remaining=10.0)
    history = [
        StepRecord(
            step=0,
            action=ToolCall(tool_name="read_logs", args={"scope": "test"}),
            observation=Observation(episode_id="e", step=0, budget_remaining=budget),
            cost_charged=0.001,
        ),
        StepRecord(
            step=1,
            action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
            observation=Observation(episode_id="e", step=1, budget_remaining=budget),
            cost_charged=0.001,
        ),
    ]
    trace = EpisodeTrace(
        episode=EpisodeState(
            episode_id="e", scenario_id="s", seed=1, step=2, history=history,
            budget=budget, is_terminated=True,
        ),
        reward_breakdown=RewardBreakdown(schema_version="1.0", total=0.0, format_gate=True, components={}),
    )
    score = InvestigationReward().score(trace, scenario)
    assert score.sub_scores["redundancy_penalty"] == 0.0


def test_investigation_ordering_penalises_cheap_after_expensive(scenario):
    # expensive (rerun_test) before cheap (read_logs) → violation
    bad_order = _trace_with_tools(["rerun_test", "read_logs"])
    good_order = _trace_with_tools(["read_logs", "rerun_test"])
    s_bad = InvestigationReward().score(bad_order, scenario)
    s_good = InvestigationReward().score(good_order, scenario)
    assert s_good.sub_scores["ordering"] > s_bad.sub_scores["ordering"]


def test_investigation_full_coverage_bonus(scenario):
    # All informative tools called
    trace = _trace_with_tools(["read_logs", "query_flake_history", "rerun_test"])
    score = InvestigationReward().score(trace, scenario)
    assert score.sub_scores["coverage"] == pytest.approx(1.0)


def test_investigation_zero_coverage_for_no_informative_tools(scenario):
    trace = _trace_with_tools(["check_owner"])  # not informative for this scenario
    score = InvestigationReward().score(trace, scenario)
    assert score.sub_scores["coverage"] == pytest.approx(0.0)
