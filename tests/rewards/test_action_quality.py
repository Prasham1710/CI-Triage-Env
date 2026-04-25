import pytest

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.action_quality import ACTION_REWARD_MATRIX, ActionQualityReward
from ci_triage_env.schemas.action import SecondaryAction, TerminalAction
from ci_triage_env.schemas.diagnosis import DiagnosisLabel


@pytest.fixture
def real_bug_scenario():
    return make_mock_scenario("real_bug")


@pytest.fixture
def race_flake_scenario():
    return make_mock_scenario("race_flake")


def _add_secondary(trace, actions: list[SecondaryAction]) -> None:
    ta = trace.episode.final_action
    assert ta is not None
    trace.episode.final_action = TerminalAction(
        action_type=ta.action_type,
        diagnosis=ta.diagnosis,
        confidence=ta.confidence,
        secondary_actions=actions,
    )


def test_action_quality_correct_case_returns_high_score(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    _add_secondary(trace, [SecondaryAction(name="file_bug", args={})])
    score = ActionQualityReward().score(trace, real_bug_scenario)
    assert score.raw > 0.5


def test_action_quality_wrong_case_returns_low_score(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    _add_secondary(trace, [SecondaryAction(name="quarantine_test", args={})])
    score = ActionQualityReward().score(trace, real_bug_scenario)
    assert score.raw < 0.0


def test_action_quality_handles_no_terminal_action(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    trace.episode.final_action = None
    score = ActionQualityReward().score(trace, real_bug_scenario)
    assert score.raw == -0.5


def test_action_quality_no_secondary_returns_zero(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    # make_mock_trajectory produces no secondary actions
    score = ActionQualityReward().score(trace, real_bug_scenario)
    assert score.raw == 0.0


def test_action_quality_deterministic(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    _add_secondary(trace, [SecondaryAction(name="file_bug", args={})])
    component = ActionQualityReward()
    s1 = component.score(trace, real_bug_scenario)
    s2 = component.score(trace, real_bug_scenario)
    assert s1.raw == s2.raw


def test_action_quality_score_is_in_documented_range(real_bug_scenario):
    component = ActionQualityReward()
    for sa_name in ("file_bug", "quarantine_test", "rerun_test", "ping_owner"):
        trace = make_mock_trajectory(real_bug_scenario, outcome="good")
        _add_secondary(trace, [SecondaryAction(name=sa_name, args={})])  # type: ignore[arg-type]
        score = component.score(trace, real_bug_scenario)
        assert -2.0 <= score.raw <= 1.5, f"out of range for {sa_name}: {score.raw}"


def test_action_quality_subscores_are_meaningful(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    _add_secondary(trace, [SecondaryAction(name="file_bug", args={})])
    score = ActionQualityReward().score(trace, real_bug_scenario)
    assert "file_bug" in score.sub_scores


def test_quarantine_real_bug_is_worst_entry():
    """quarantine_test on real_bug is the most negative entry in ACTION_REWARD_MATRIX."""
    quarantine_real_bug = ACTION_REWARD_MATRIX[("quarantine_test", "real_bug")]
    all_values = list(ACTION_REWARD_MATRIX.values())
    assert quarantine_real_bug == min(all_values)


def test_capping_prevents_stacking_exploit(real_bug_scenario):
    trace = make_mock_trajectory(real_bug_scenario, outcome="good")
    # Add many high-reward actions; total should be capped at 1.5
    _add_secondary(
        trace,
        [SecondaryAction(name="file_bug", args={})] * 5,
    )
    score = ActionQualityReward().score(trace, real_bug_scenario)
    assert score.raw <= 1.5


def test_action_quality_correct_for_race_flake(race_flake_scenario):
    trace = make_mock_trajectory(race_flake_scenario, outcome="good")
    _add_secondary(trace, [SecondaryAction(name="quarantine_test", args={})])
    score = ActionQualityReward().score(trace, race_flake_scenario)
    assert score.raw == ACTION_REWARD_MATRIX[("quarantine_test", "race_flake")]


def test_action_quality_uses_ground_truth_label(real_bug_scenario):
    """Reward is based on the scenario ground truth, not the predicted diagnosis."""
    trace = make_mock_trajectory(real_bug_scenario, outcome="bad")  # wrong diagnosis
    _add_secondary(trace, [SecondaryAction(name="file_bug", args={})])
    score = ActionQualityReward().score(trace, real_bug_scenario)
    # file_bug on real_bug → 1.0 regardless of wrong prediction
    assert score.raw == ACTION_REWARD_MATRIX[("file_bug", "real_bug")]


def test_action_quality_diagnosis_label_all_families():
    """file_bug on every family has a defined entry."""
    for label in DiagnosisLabel:
        assert ("file_bug", label.value) in ACTION_REWARD_MATRIX
