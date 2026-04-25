from ci_triage_env.mock.scenario import make_mock_scenario
from ci_triage_env.mock.trajectory import make_mock_trajectory
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace


def test_episode_state_round_trip():
    trace = make_mock_trajectory(make_mock_scenario(), outcome="good")
    state = trace.episode
    restored = EpisodeState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_episode_trace_round_trip():
    trace = make_mock_trajectory(make_mock_scenario(), outcome="good")
    restored = EpisodeTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace


def test_step_record_history_preserved():
    trace = make_mock_trajectory(make_mock_scenario(), outcome="good")
    restored = EpisodeTrace.model_validate(trace.model_dump())
    assert len(restored.episode.history) == len(trace.episode.history)
    assert restored.episode.history[0].step == 0
