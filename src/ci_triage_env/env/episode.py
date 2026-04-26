"""Episode lifecycle: budget, action dispatch, termination, payload truncation.

Phase A3 enforces the budget (tool-call slots and cost) on the env side. Once
exhausted, the episode self-terminates with ``final_action=None`` — that's the
signal Branch C's reward layer reads as a budget-failure case.

Tool-call args validation runs through ``ToolHandler.validate_args``. Bad args
are NOT a hard error: they consume one tool-call slot plus a tiny cost (cheap
penalty), the episode keeps going. Hard errors stay the responsibility of the
handler.
"""

from __future__ import annotations

import json
import random
from typing import Any

from ci_triage_env.env.tools.base import ToolHandler
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeState, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation, ToolResponse
from ci_triage_env.schemas.scenario import Scenario

DEFAULT_TOOL_CALL_BUDGET = 12
DEFAULT_COST_BUDGET = 5.0
OBSERVATION_PAYLOAD_CAP = 4000  # chars per tool response payload
INVALID_ARGS_PENALTY_COST = 0.001
COMMIT_MSG_CAP = 200


class EpisodeTerminatedError(RuntimeError):
    """Raised when an action is submitted after an episode has terminated."""


class EpisodeManager:
    """Owns state for a single in-flight episode."""

    def __init__(
        self,
        scenario: Scenario,
        episode_id: str,
        seed: int,
        max_tool_calls: int | None = None,
        cost_budget: float | None = None,
    ) -> None:
        self.scenario = scenario
        self.episode_id = episode_id
        self.seed = seed
        self.step_idx: int = 0
        self.history: list[StepRecord] = []
        # Raw cost ledger can go negative (one tool call may overshoot the
        # remaining budget). The frozen ``BudgetState`` schema constrains
        # ``cost_remaining >= 0`` so we clamp on every update and use the
        # raw value (plus tool-call slots) for the exhaustion gate.
        initial_calls = max_tool_calls if max_tool_calls is not None else DEFAULT_TOOL_CALL_BUDGET
        self._raw_cost_remaining: float = (
            cost_budget if cost_budget is not None else DEFAULT_COST_BUDGET
        )
        self.budget: BudgetState = BudgetState(
            tool_calls_remaining=initial_calls,
            cost_remaining=max(0.0, self._raw_cost_remaining),
        )
        self.is_terminated: bool = False
        self.final_action: TerminalAction | None = None
        self.budget_exhausted: bool = False
        self._rng = random.Random(self.seed)

    @property
    def raw_cost_remaining(self) -> float:
        return self._raw_cost_remaining

    def _update_budget(self, cost_charged: float) -> None:
        self._raw_cost_remaining -= cost_charged
        self.budget = BudgetState(
            tool_calls_remaining=self.budget.tool_calls_remaining - 1,
            cost_remaining=max(0.0, self._raw_cost_remaining),
        )

    # ------------------------------------------------------------------ obs
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
        return hash((self.seed, self.step_idx, tool_name)) & 0xFFFFFFFF

    # ------------------------------------------------------------------ tool call
    def apply_tool_call(
        self,
        action: ToolCall,
        handler: ToolHandler,
    ) -> Observation:
        if self.is_terminated:
            raise EpisodeTerminatedError("episode already terminated")

        # Budget gate first — exhaustion forces termination on this very step.
        if self.budget.tool_calls_remaining <= 0 or self._raw_cost_remaining < 0:
            return self._force_terminate_budget_exhausted()

        # Validate args. Bad args = cheap penalty observation, episode continues.
        try:
            handler.validate_args(action.args)
        except ValueError as exc:
            return self._record_cheap_penalty(action, str(exc))

        output = handler.call(action.args, self.scenario, self.history)
        cost_charged = output.cost_units
        truncated_payload = self._truncate_payload(output.payload, action.tool_name)

        self._update_budget(cost_charged)

        observation = Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=ToolResponse(
                tool_name=output.tool_name,
                args=action.args,
                output=truncated_payload,
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

    def _record_cheap_penalty(self, action: ToolCall, error_message: str) -> Observation:
        cost = INVALID_ARGS_PENALTY_COST
        self._update_budget(cost)
        observation = Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=ToolResponse(
                tool_name=action.tool_name,
                args=action.args,
                output={"error": error_message},
                cost_charged=cost,
            ),
            budget_remaining=self.budget,
            is_terminal=False,
        )
        self.history.append(
            StepRecord(
                step=self.step_idx,
                action=action,
                observation=observation,
                cost_charged=cost,
            )
        )
        self.step_idx += 1
        return observation

    def _force_terminate_budget_exhausted(self) -> Observation:
        """Budget out without an explicit terminal action — the reward layer
        reads ``final_action is None`` together with ``is_terminated=True`` as
        a budget-failure case."""
        self.is_terminated = True
        self.budget_exhausted = True
        self.final_action = None
        return Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=None,
            budget_remaining=self.budget,
            is_terminal=True,
        )

    # ------------------------------------------------------------------ terminal
    def apply_terminal(self, action: TerminalAction) -> Observation:
        if self.is_terminated:
            raise EpisodeTerminatedError("episode already terminated")

        # Pydantic already validated the diagnosis enum, confidence range, and
        # secondary-action shapes by the time we get here, but re-affirm the
        # invariants the reward layer relies on.
        if not (0.0 <= action.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {action.confidence}")

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

    # ------------------------------------------------------------------ truncation
    def _truncate_payload(self, payload: Any, tool_name: str) -> Any:
        """Cap the serialized size of a tool response.

        - ``read_logs`` keeps head + tail of ``lines`` so the agent retains
          context from both ends of a long log.
        - ``recent_commits`` keeps every commit but trims each ``msg`` to
          ``COMMIT_MSG_CAP`` chars (commit identity > prose).
        - All other payloads: serialize, take the first ``OBSERVATION_PAYLOAD_CAP``
          chars, append a marker.
        """
        if not isinstance(payload, (dict, str)):
            return payload

        if isinstance(payload, dict) and tool_name == "read_logs" and "lines" in payload:
            lines = list(payload.get("lines", []))
            joined = "\n".join(lines)
            if len(joined) <= OBSERVATION_PAYLOAD_CAP:
                return payload
            half = OBSERVATION_PAYLOAD_CAP // 2
            head, tail = [], []
            running = 0
            for line in lines:
                if running + len(line) + 1 > half:
                    break
                head.append(line)
                running += len(line) + 1
            running = 0
            for line in reversed(lines):
                if running + len(line) + 1 > half:
                    break
                tail.append(line)
                running += len(line) + 1
            tail.reverse()
            dropped = len(lines) - len(head) - len(tail)
            kept = head + [f"...[{dropped} lines truncated]..."] + tail
            return {**payload, "lines": kept, "truncated": True}

        if isinstance(payload, dict) and tool_name == "recent_commits" and "commits" in payload:
            new_commits = []
            for c in payload.get("commits", []):
                if isinstance(c, dict):
                    msg = c.get("msg", "")
                    if isinstance(msg, str) and len(msg) > COMMIT_MSG_CAP:
                        c = {**c, "msg": msg[:COMMIT_MSG_CAP] + "...[truncated]"}
                new_commits.append(c)
            return {**payload, "commits": new_commits}

        # Generic fallback.
        if isinstance(payload, str):
            if len(payload) <= OBSERVATION_PAYLOAD_CAP:
                return payload
            return payload[:OBSERVATION_PAYLOAD_CAP] + " ...[truncated]"

        serialized = json.dumps(payload, default=str)
        if len(serialized) <= OBSERVATION_PAYLOAD_CAP:
            return payload
        return {
            "_truncated": True,
            "_serialized_preview": serialized[:OBSERVATION_PAYLOAD_CAP] + " ...[truncated]",
        }

    # ------------------------------------------------------------------ export
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
