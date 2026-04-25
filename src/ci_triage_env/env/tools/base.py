from abc import ABC, abstractmethod

from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.scenario import Scenario, ToolOutput


class ToolHandler(ABC):
    name: str
    cost_unit: float

    @abstractmethod
    def call(
        self,
        args: dict,
        scenario: Scenario,
        history: list[StepRecord],
    ) -> ToolOutput: ...

    @abstractmethod
    def validate_args(self, args: dict) -> None:
        """Raise ValueError if args don't match this tool's schema."""
