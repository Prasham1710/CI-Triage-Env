import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.diagnosis import DIAGNOSIS_REWARD_MATRIX, DiagnosisReward, lookup_reward
from ci_triage_env.schemas.diagnosis import DiagnosisLabel


@pytest.fixture
def scenario():
    return make_mock_scenario("real_bug")


@pytest.fixture
def good_trace(scenario):
    return make_mock_trajectory(scenario, outcome="good")


@pytest.fixture
def bad_trace(scenario):
    return make_mock_trajectory(scenario, outcome="bad")


def test_diagnosis_correct_case_returns_high_score(good_trace, scenario):
    score = DiagnosisReward().score(good_trace, scenario)
    assert score.raw == 1.0


def test_diagnosis_wrong_case_returns_low_score(bad_trace, scenario):
    score = DiagnosisReward().score(bad_trace, scenario)
    assert score.raw < 0.0


def test_diagnosis_handles_no_terminal_action(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = DiagnosisReward().score(trace, scenario)
    assert score.raw == -1.0


def test_diagnosis_deterministic(good_trace, scenario):
    component = DiagnosisReward()
    s1 = component.score(good_trace, scenario)
    s2 = component.score(good_trace, scenario)
    assert s1.raw == s2.raw


def test_diagnosis_score_is_in_documented_range(scenario):
    component = DiagnosisReward()
    for outcome in ("good", "bad", "abstain"):
        trace = make_mock_trajectory(scenario, outcome=outcome)
        score = component.score(trace, scenario)
        assert -1.0 <= score.raw <= 1.0, f"out of range for outcome={outcome}: {score.raw}"


def test_diagnosis_subscores_are_meaningful(good_trace, scenario):
    score = DiagnosisReward().score(good_trace, scenario)
    assert "matrix_lookup" in score.sub_scores


def test_diagonal_matches_return_one():
    labels = [e.value for e in DiagnosisLabel]
    for label in labels:
        assert lookup_reward(label, label) == 1.0, f"diagonal mismatch for {label}"


def test_all_diagonal_entries_in_matrix():
    for label in DiagnosisLabel:
        assert (label.value, label.value) in DIAGNOSIS_REWARD_MATRIX


def test_lookup_returns_default_for_unlisted_pair():
    assert lookup_reward("real_bug", "ambiguous") == -0.5


def test_no_terminal_subscores_key(scenario):
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = DiagnosisReward().score(trace, scenario)
    assert "no_diagnosis" in score.sub_scores
