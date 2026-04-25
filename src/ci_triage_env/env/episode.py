import random
from dataclasses import dataclass

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeState, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation, ToolResponse
from ci_triage_env.schemas.scenario import Scenario, ToolOutput

DEFAULT_TOOL_CALL_BUDGET = 12
DEFAULT_COST_BUDGET = 5.0


@dataclass
class EpisodeManager:
    """Owns state for a single in-flight episode.

    Phase A1 contract: validates lifecycle (initial obs, action stepping, termination,
    state export). Tool *output content* is the responsibility of the handlers wired in
    by the server. Budget enforcement and termination policies tighten in A3.
    """

    scenario: Scenario
    episode_id: str
    seed: int

    def __post_init__(self) -> None:
        self.step_idx: int = 0
        self.history: list[StepRecord] = []
        self.budget: BudgetState = BudgetState(
            tool_calls_remaining=DEFAULT_TOOL_CALL_BUDGET,
            cost_remaining=DEFAULT_COST_BUDGET,
        )
        self.is_terminated: bool = False
        self.final_action: TerminalAction | None = None
        self._rng = random.Random(self.seed)

    def initial_observation(self) -> Observation:
        return Observation(
            episode_id=self.episode_id,
            step=0,
            failure_summary=self.scenario.failure_summary,
            tool_response=None,
            budget_remaining=self.budget,
            is_terminal=False,
        )

    def derive_step_seed(self, tool_name: str) -> int:
        """Per-step seed derived from (episode seed, step_idx, tool_name).

        Tools that internally randomize must use this seed instead of a global RNG.
        """
        return hash((self.seed, self.step_idx, tool_name)) & 0xFFFFFFFF

    def apply_tool_call(
        self,
        action: ToolCall,
        output: ToolOutput,
    ) -> Observation:
        if self.is_terminated:
            raise RuntimeError("episode already terminated")

        cost_charged = output.cost_units
        self.budget = BudgetState(
            tool_calls_remaining=max(0, self.budget.tool_calls_remaining - 1),
            cost_remaining=max(0.0, self.budget.cost_remaining - cost_charged),
        )

        observation = Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=ToolResponse(
                tool_name=action.tool_name,
                args=action.args,
                output=output.payload,
                cost_charged=cost_charged,
            ),
            budget_remaining=self.budget,
            is_terminal=False,
        )

        self.history.append(
            StepRecord(
                step=self.step_idx,
                action=action,
                observation=observation,
                cost_charged=cost_charged,
            )
        )
        self.step_idx += 1
        return observation

    def apply_terminal(self, action: TerminalAction) -> Observation:
        if self.is_terminated:
            raise RuntimeError("episode already terminated")

        observation = Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=None,
            budget_remaining=self.budget,
            is_terminal=True,
        )

        self.history.append(
            StepRecord(
                step=self.step_idx,
                action=action,
                observation=observation,
                cost_charged=0.0,
            )
        )
        self.step_idx += 1
        self.is_terminated = True
        self.final_action = action
        return observation

    def to_state(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self.episode_id,
            scenario_id=self.scenario.scenario_id,
            seed=self.seed,
            step=self.step_idx,
            history=list(self.history),
            budget=self.budget,
            is_terminated=self.is_terminated,
            final_action=self.final_action,
        )
