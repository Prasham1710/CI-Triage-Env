import pytest
from fastapi.testclient import TestClient

from ci_triage_env.env.server import CITriageEnv, build_app
from ci_triage_env.mock.scenario import make_mock_scenario


def _make_scenarios() -> dict:
    return {
        s.scenario_id: s
        for s in [
            make_mock_scenario("race_flake", seed=42),
            make_mock_scenario("real_bug", seed=7),
        ]
    }


@pytest.fixture
def scenarios() -> dict:
    return _make_scenarios()


@pytest.fixture
def env_factory(scenarios):
    """Factory returning a fresh CITriageEnv per request, bound to fixed scenarios.

    Matches OpenEnv's per-session env model: each WebSocket session and each
    stateless /reset+/step request gets its own instance.
    """

    def _factory():
        return CITriageEnv(scenarios=scenarios)

    return _factory


@pytest.fixture
def app(env_factory):
    return build_app(env_factory=env_factory)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def known_scenario_id(scenarios) -> str:
    return next(iter(scenarios))
