import pytest
from fastapi.testclient import TestClient

from ci_triage_env.env.server import CITriageEnv, create_app
from ci_triage_env.mock.scenario import make_mock_scenario


@pytest.fixture
def env() -> CITriageEnv:
    scenarios = {
        s.scenario_id: s
        for s in [
            make_mock_scenario("race_flake", seed=42),
            make_mock_scenario("real_bug", seed=7),
        ]
    }
    return CITriageEnv(scenarios=scenarios)


@pytest.fixture
def client(env: CITriageEnv) -> TestClient:
    return TestClient(create_app(env))


@pytest.fixture
def known_scenario_id(env: CITriageEnv) -> str:
    return next(iter(env.scenarios))
