from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.counterfactual_predict import CounterfactualPredictReward
from ci_triage_env.schemas.reward import CounterfactualScore


def test_no_probe_returns_zero():
    """trace.reward_breakdown.counterfactual = None → score 0, weight 0."""
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    assert trace.reward_breakdown.counterfactual is None
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 0.0
    assert score.weighted == 0.0
    assert score.weight == 0.0


def test_v1_default_weight_is_zero():
    """default_weight is 0.0 in v1 — proves component is dormant."""
    assert CounterfactualPredictReward.default_weight == 0.0


def test_counterfactual_predict_handles_no_terminal_action():
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.episode.final_action = None
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 0.0


def test_counterfactual_predict_deterministic():
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    component = CounterfactualPredictReward()
    s1 = component.score(trace, scenario)
    s2 = component.score(trace, scenario)
    assert s1.raw == s2.raw


def test_counterfactual_predict_score_is_in_documented_range():
    scenario = make_mock_scenario("real_bug")
    component = CounterfactualPredictReward()
    for outcome in ("good", "bad", "abstain"):
        trace = make_mock_trajectory(scenario, outcome=outcome)
        score = component.score(trace, scenario)
        assert -0.5 <= score.raw <= 1.0


def test_counterfactual_predict_subscores_are_meaningful():
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    score = CounterfactualPredictReward().score(trace, scenario)
    assert "fired" in score.sub_scores


def test_probe_not_fired_returns_zero():
    """fired=False → score 0."""
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.reward_breakdown.counterfactual = CounterfactualScore(
        fired=False,
        probe_step=0,
        probe_action="rerun_test",
        predicted_outcome="pass",
        actual_outcome="pass",
        brier_score=0.0,
    )
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 0.0


def test_probe_fired_correct_prediction_returns_one():
    """In v2, correct prediction when probe fired → 1.0. Verifies dead-code path."""
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.reward_breakdown.counterfactual = CounterfactualScore(
        fired=True,
        probe_step=2,
        probe_action="rerun_test",
        predicted_outcome="pass",
        actual_outcome="pass",
        brier_score=0.0,
    )
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == 1.0
    # weight is 0.0 so weighted contribution is still 0 in v1
    assert score.weighted == 0.0


def test_probe_fired_wrong_prediction_returns_minus_half():
    """In v2, wrong prediction when probe fired → -0.5."""
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    trace.reward_breakdown.counterfactual = CounterfactualScore(
        fired=True,
        probe_step=2,
        probe_action="rerun_test",
        predicted_outcome="pass",
        actual_outcome="fail",
        brier_score=1.0,
    )
    score = CounterfactualPredictReward().score(trace, scenario)
    assert score.raw == -0.5
    assert score.weighted == 0.0
