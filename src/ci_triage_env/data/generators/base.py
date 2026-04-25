from abc import ABC, abstractmethod

from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.scenario import Scenario


class ScenarioFamilyGenerator(ABC):
    family_name: str
    label: DiagnosisLabel

    @abstractmethod
    def generate(self, seed: int, source_log_hash: str | None = None) -> Scenario: ...

    @abstractmethod
    def informative_tools(self) -> list[str]:
        """Tools that are informative for this family by construction."""

    @abstractmethod
    def minimal_evidence_set(self) -> list[str]:
        """Smallest tool set that uniquely determines correct diagnosis."""
