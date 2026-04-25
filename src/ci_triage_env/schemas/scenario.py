from typing import Literal

from pydantic import BaseModel, Field

from ci_triage_env.schemas.diagnosis import DiagnosisLabel


class GroundTruth(BaseModel):
    label: DiagnosisLabel
    rationale: str
    is_ambiguous: bool = False
    confidence_target: float = Field(default=1.0, ge=0.0, le=1.0)


class FailureSummary(BaseModel):
    test_name: str
    suite: str
    branch: str
    last_passing_commit: str
    initial_log_excerpt: str
    timestamp: str


class ToolOutput(BaseModel):
    tool_name: str
    payload: dict | str
    cost_units: float = Field(ge=0.0)


class TerminalActionSpec(BaseModel):
    primary: str
    args: dict
    acceptable_alternatives: list[dict] = Field(default_factory=list)


class ScenarioMetadata(BaseModel):
    generator_version: str
    generated_at: str
    source_log_hash: str | None = None
    difficulty: Literal["easy", "medium", "hard"]


class Scenario(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    scenario_id: str
    family: str
    seed: int
    ground_truth: GroundTruth
    failure_summary: FailureSummary
    tool_outputs: dict[str, ToolOutput]
    informative_tools: list[str]
    minimal_evidence_set: list[str]
    correct_terminal_action: TerminalActionSpec
    metadata: ScenarioMetadata
