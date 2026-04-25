from typing import Literal

from pydantic import BaseModel, Field


class ComponentScore(BaseModel):
    raw: float
    weighted: float
    weight: float
    sub_scores: dict[str, float] = Field(default_factory=dict)


class CounterfactualScore(BaseModel):
    """v1: dormant — never populated. v2 path preserved as schema."""

    fired: bool
    probe_step: int
    probe_action: str
    predicted_outcome: str
    actual_outcome: str
    brier_score: float


class RewardBreakdown(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    total: float
    format_gate: bool
    components: dict[str, ComponentScore] = Field(default_factory=dict)
    counterfactual: CounterfactualScore | None = None
