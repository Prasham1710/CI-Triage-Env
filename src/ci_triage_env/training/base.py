from abc import ABC, abstractmethod

from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import RewardBreakdown
from ci_triage_env.schemas.scenario import Scenario


class RewardAggregator(ABC):
    """Composes per-component scores into a single RewardBreakdown for training."""

    @abstractmethod
    def compute(self, trace: EpisodeTrace, scenario: Scenario) -> RewardBreakdown: ...


class TrajectoryGenerator(ABC):
    """Produces an EpisodeTrace from a scenario (e.g. via a policy or scripted heuristic)."""

    @abstractmethod
    def rollout(self, scenario: Scenario, seed: int) -> EpisodeTrace: ...


class Trainer(ABC):
    """Drives an SFT or RL training loop over scenarios."""

    @abstractmethod
    def train(self, scenarios: list[Scenario]) -> None: ...
