from typing import Literal

from pydantic import BaseModel, Field

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


class StepRecord(BaseModel):
    step: int = Field(ge=0)
    action: ToolCall | TerminalAction
    observation: Observation
    cost_charged: float = Field(ge=0.0)


class EpisodeState(BaseModel):
    episode_id: str
    scenario_id: str
    seed: int
    step: int = Field(ge=0)
    history: list[StepRecord] = Field(default_factory=list)
    budget: BudgetState
    is_terminated: bool = False
    final_action: TerminalAction | None = None


class EpisodeTrace(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    episode: EpisodeState
    reward_breakdown: RewardBreakdown
    counterfactual_replay: list[StepRecord] | None = None
