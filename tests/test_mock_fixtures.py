import json
from pathlib import Path

import pytest

from ci_triage_env.mock.scenario import make_mock_scenario
from ci_triage_env.mock.trajectory import make_mock_trajectory
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.scenario import Scenario

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_make_mock_scenario_returns_valid_scenario():
    scenario = make_mock_scenario("race_flake")
    assert isinstance(scenario, Scenario)
    assert scenario.family == "race_flake"
    assert scenario.tool_outputs


@pytest.mark.parametrize("outcome", ["good", "bad", "abstain"])
def test_make_mock_trajectory_all_outcomes(outcome):
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome=outcome)
    assert isinstance(trace, EpisodeTrace)
    assert trace.episode.is_terminated
    assert trace.episode.final_action is not None


def test_mock_scenario_fixture_file_validates():
    payload = json.loads((FIXTURES_DIR / "mock_scenario.json").read_text())
    Scenario.model_validate(payload)


def test_mock_trajectory_fixture_file_validates():
    payload = json.loads((FIXTURES_DIR / "mock_trajectory.json").read_text())
    EpisodeTrace.model_validate(payload)
