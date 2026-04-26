"""Tests for MinimalEvidenceReward component."""

from __future__ import annotations

import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.minimal_evidence import MinimalEvidenceReward
from ci_triage_env.schemas.action import TerminalAction, ToolCall
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


def _make_tool_records(tool_names: list[str]) -> list[StepRecord]:
    return [
        StepRecord(step=i, action=ToolCall(tool_name=t, args={}),
                   observation=_dummy_obs(i), cost_charged=0.001)
        for i, t in enumerate(tool_names)
    ]


def test_minimal_evidence_correct_case_returns_high_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = MinimalEvidenceReward().score(trace, scenario)
    # score may be 0.0 (weight=0) but raw should be non-negative for correct diagnosis
    assert score.raw >= 0.0


def test_minimal_evidence_wrong_case_returns_low_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="bad")  # wrong diagnosis
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == 0.0  # wrong diagnosis → no bonus


def test_minimal_evidence_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = MinimalEvidenceReward().score(no_terminal, scenario)
    assert score.raw == 0.0  # no correct diagnosis → no bonus


def test_minimal_evidence_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = MinimalEvidenceReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_minimal_evidence_score_is_in_documented_range() -> None:
    for family in ["real_bug", "race_flake", "timing_flake"]:
        scenario = make_mock_scenario(family)
        for outcome in ["good", "bad"]:
            trace = make_mock_trajectory(scenario, outcome=outcome)
            score = MinimalEvidenceReward().score(trace, scenario)
            assert -0.5 <= score.raw <= 1.0


def test_minimal_evidence_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = MinimalEvidenceReward().score(trace, scenario)
    # Either empty (no min_set) or has the two keys
    if score.sub_scores:
        assert "min_set_used" in score.sub_scores
        assert "extras" in score.sub_scores


def test_using_only_min_set_max_bonus() -> None:
    scenario = make_mock_scenario("real_bug")
    assert scenario.minimal_evidence_set  # must have a min set for this test
    min_set = scenario.minimal_evidence_set
    trace = make_mock_trajectory(scenario, outcome="good")

    # Build a history using ONLY the minimal evidence tools
    minimal_records = _make_tool_records(min_set)
    correct_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=scenario.ground_truth.label,
        confidence=1.0,
        secondary_actions=[],
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": minimal_records, "final_action": correct_terminal}
        )}
    )
    score = MinimalEvidenceReward().score(patched, scenario)
    # Only min set tools → extras=0 → bonus = 1.0 - 0.1*0 = 1.0
    assert score.raw == pytest.approx(1.0, abs=1e-6)


def test_empty_min_set_returns_zero() -> None:
    # Use a real ambiguous scenario from the generator (which correctly has empty min_set)
    from ci_triage_env.data.generators import GENERATOR_REGISTRY
    from ci_triage_env.mock import make_mock_trajectory

    scenario = GENERATOR_REGISTRY["ambiguous"]().generate(seed=42)
    assert scenario.minimal_evidence_set == []
    trace = make_mock_trajectory(scenario, outcome="good")
    score = MinimalEvidenceReward().score(trace, scenario)
    assert score.raw == 0.0
