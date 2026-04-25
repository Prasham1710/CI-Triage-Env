"""Tests for DiagnosisReward component."""

from __future__ import annotations

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.diagnosis import DIAGNOSIS_REWARD_MATRIX, DiagnosisReward

ALL_FAMILIES = ["real_bug", "race_flake", "timing_flake", "infra_network", "infra_resource",
                "dependency_drift", "ambiguous"]


def test_diagnosis_correct_case_returns_high_score() -> None:
    for family in ALL_FAMILIES:
        scenario = make_mock_scenario(family)
        trace = make_mock_trajectory(scenario, outcome="good")
        score = DiagnosisReward().score(trace, scenario)
        assert score.raw == 1.0, f"family={family}: expected 1.0 got {score.raw}"


def test_diagnosis_wrong_case_returns_low_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="bad")
    score = DiagnosisReward().score(trace, scenario)
    assert score.raw < 0.0


def test_diagnosis_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": None})}
    )
    score = DiagnosisReward().score(no_terminal, scenario)
    assert score.raw == -1.0
    assert score.sub_scores.get("no_diagnosis") == -1.0


def test_diagnosis_deterministic() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    comp = DiagnosisReward()
    s1 = comp.score(trace, scenario)
    s2 = comp.score(trace, scenario)
    assert s1.raw == s2.raw


def test_diagnosis_score_is_in_documented_range() -> None:
    for family in ALL_FAMILIES:
        scenario = make_mock_scenario(family)
        for outcome in ["good", "bad"]:
            trace = make_mock_trajectory(scenario, outcome=outcome)
            score = DiagnosisReward().score(trace, scenario)
            assert -1.0 <= score.raw <= 1.0, f"out of range: family={family} outcome={outcome}"


def test_diagnosis_subscores_are_meaningful() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = DiagnosisReward().score(trace, scenario)
    assert "matrix_lookup" in score.sub_scores


def test_diagonal_matches_return_one() -> None:
    families = ALL_FAMILIES
    for f in families:
        assert DIAGNOSIS_REWARD_MATRIX.get((f, f)) == 1.0, f"diagonal {f} is not 1.0"


def test_quarantine_real_bug_is_most_negative_action() -> None:
    # The diagnosis matrix's worst entry for predicting flake on real_bug is -1.0
    flake_on_real = DIAGNOSIS_REWARD_MATRIX[("race_flake", "real_bug")]
    assert flake_on_real == -1.0
    # Every other (predicted, "real_bug") entry should be >= flake_on_real
    real_bug_penalties = [v for (p, t), v in DIAGNOSIS_REWARD_MATRIX.items() if t == "real_bug" and p != "real_bug"]
    assert all(v >= flake_on_real for v in real_bug_penalties)
