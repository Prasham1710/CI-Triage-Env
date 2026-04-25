from pydantic import BaseModel, Field

from ci_triage_env.schemas.scenario import FailureSummary


class ProbeQuestion(BaseModel):
    """v1: dormant — env never emits this. v2 path preserved as schema."""

    step: int
    taken_action: dict | None = None
    alternate_action: dict


class ToolResponse(BaseModel):
    tool_name: str
    args: dict
    output: dict | str
    cost_charged: float = Field(ge=0.0)


class BudgetState(BaseModel):
    tool_calls_remaining: int = Field(ge=0)
    cost_remaining: float = Field(ge=0.0)


class Observation(BaseModel):
    episode_id: str
    step: int = Field(ge=0)
    failure_summary: FailureSummary | None = None
    tool_response: ToolResponse | None = None
    budget_remaining: BudgetState
    is_terminal: bool = False
    probe_question: ProbeQuestion | None = None
