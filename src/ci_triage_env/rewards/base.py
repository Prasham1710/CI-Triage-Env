from abc import ABC, abstractmethod

from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore
from ci_triage_env.schemas.scenario import Scenario


class RewardComponent(ABC):
    name: str
    default_weight: float

    @abstractmethod
    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore: ...
