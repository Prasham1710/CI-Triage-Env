import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.minimal_evidence import MinimalEvidenceReward
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


@pytest.fixture
def scenario():
    # minimal_evidence_set = ["query_flake_history", "rerun_test"]
    # ground_truth.label = real_bug
    return make_mock_scenario("real_bug")


def _trace_with_tools_and_diagnosis(
    tool_names: list[str], diagnosis: DiagnosisLabel
) -> EpisodeTrace:
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
    terminal = TerminalAction(diagnosis=diagnosis, confidence=0.9)
    history.append(
        StepRecord(
            step=len(tool_names),
            action=terminal,
            observation=Observation(episode_id="e", step=len(tool_names), budget_remaining=budget),
            cost_charged=0.0,
        )
    )
    return EpisodeTrace(
        episode=EpisodeState(
            episode_id="e",
            scenario_id="s",
            seed=1,
            step=len(tool_names),
            history=history,
            budget=budget,
            is_terminated=True,
            final_action=terminal,
        ),
        reward_breakdown=RewardBreakdown(schema_version="1.0", total=0.0, format_gate=True, components={}),
    )


def test_minimal_evidence_correct_case_returns_high_score(scenario):
    # Use both min_set tools + correct diagnosis → bonus = 1.0 - 0.1*0 = 1.0
    trace = _trace_with_tools_and_diagnosis(
        ["query_flake_history", "rerun_test"], DiagnosisLabel.REAL_BUG
    )
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(1.0)


def test_minimal_evidence_wrong_case_returns_zero(scenario):
    # Wrong diagnosis → 0.0 regardless of tools
    trace = _trace_with_tools_and_diagnosis(
        ["query_flake_history", "rerun_test"], DiagnosisLabel.RACE_FLAKE
    )
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.0)


def test_minimal_evidence_handles_no_terminal_action(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.0)


def test_minimal_evidence_deterministic(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    component = MinimalEvidenceReward()
    s1 = component.score(trace, scenario)
    s2 = component.score(trace, scenario)
    assert s1.raw == s2.raw


def test_minimal_evidence_score_is_in_documented_range(scenario):
    component = MinimalEvidenceReward()
    for outcome in ("good", "bad", "abstain"):
        trace = make_mock_trajectory(scenario, outcome=outcome)
        score = component.score(trace, scenario)
        assert -0.5 <= score.raw <= 1.0, f"out of range for {outcome}: {score.raw}"


def test_minimal_evidence_subscores_are_meaningful(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    score = MinimalEvidenceReward().score(trace, scenario)
    assert "min_set_used_count" in score.sub_scores
    assert "extras_count" in score.sub_scores


def test_using_only_min_set_max_bonus(scenario):
    # Use exactly the minimal evidence set, no extras → bonus = 1.0
    trace = _trace_with_tools_and_diagnosis(
        ["query_flake_history", "rerun_test"], DiagnosisLabel.REAL_BUG
    )
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(1.0)


def test_extras_reduce_bonus(scenario):
    # min_set + 1 extra → bonus = 1.0 - 0.1 = 0.9
    trace = _trace_with_tools_and_diagnosis(
        ["query_flake_history", "rerun_test", "read_logs"], DiagnosisLabel.REAL_BUG
    )
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.9)


def test_lucky_guess_without_min_set(scenario):
    # Correct diagnosis but didn't use all min_set tools → bonus = 0.3
    trace = _trace_with_tools_and_diagnosis(["read_logs"], DiagnosisLabel.REAL_BUG)
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.3)


def test_default_weight_is_zero():
    assert MinimalEvidenceReward.default_weight == 0.0


def test_weighted_is_zero_due_to_default_weight(scenario):
    trace = _trace_with_tools_and_diagnosis(
        ["query_flake_history", "rerun_test"], DiagnosisLabel.REAL_BUG
    )
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.weighted == pytest.approx(0.0)


def test_empty_min_set_returns_zero():
    scenario = make_mock_scenario("real_bug")
    scenario = scenario.model_copy(update={"minimal_evidence_set": []})
    trace = make_mock_trajectory(scenario, outcome="good")
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == pytest.approx(0.0)
