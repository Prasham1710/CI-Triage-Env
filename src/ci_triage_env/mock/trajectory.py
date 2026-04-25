from typing import Literal

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation, ToolResponse
from ci_triage_env.schemas.reward import ComponentScore, RewardBreakdown
from ci_triage_env.schemas.scenario import Scenario


def _make_observation(
    episode_id: str,
    step: int,
    tool_response: ToolResponse | None,
    budget: BudgetState,
    is_terminal: bool,
) -> Observation:
    return Observation(
        episode_id=episode_id,
        step=step,
        failure_summary=None,
        tool_response=tool_response,
        budget_remaining=budget,
        is_terminal=is_terminal,
        probe_question=None,
    )


def make_mock_trajectory(
    scenario: Scenario,
    outcome: Literal["good", "bad", "abstain"] = "good",
) -> EpisodeTrace:
    """Generate a deterministic trajectory of 4 tool calls + terminal.

    Three variants (good, bad, abstain) for testing different reward paths.
    """
    if outcome not in {"good", "bad", "abstain"}:
        raise ValueError(f"unknown outcome: {outcome}")

    episode_id = f"mock-episode-{scenario.scenario_id}-{outcome}"
    initial_budget = BudgetState(tool_calls_remaining=10, cost_remaining=1.0)

    tool_call_names = ["read_logs", "query_flake_history", "recent_commits", "rerun_test"]
    history: list[StepRecord] = []
    budget = initial_budget

    for step_idx, tool_name in enumerate(tool_call_names):
        tool_output = scenario.tool_outputs.get(tool_name)
        cost_charged = tool_output.cost_units if tool_output else 0.001
        budget = BudgetState(
            tool_calls_remaining=budget.tool_calls_remaining - 1,
            cost_remaining=max(0.0, budget.cost_remaining - cost_charged),
        )
        action = ToolCall(tool_name=tool_name, args={"scope": "test"} if tool_name == "read_logs" else {})
        observation = _make_observation(
            episode_id=episode_id,
            step=step_idx,
            tool_response=ToolResponse(
                tool_name=tool_name,
                args=action.args,
                output=tool_output.payload if tool_output else {},
                cost_charged=cost_charged,
            ),
            budget=budget,
            is_terminal=False,
        )
        history.append(
            StepRecord(
                step=step_idx,
                action=action,
                observation=observation,
                cost_charged=cost_charged,
            )
        )

    if outcome == "good":
        diagnosis = scenario.ground_truth.label
        confidence = scenario.ground_truth.confidence_target
        format_gate = True
        total = 1.0
    elif outcome == "abstain":
        diagnosis = DiagnosisLabel.AMBIGUOUS
        confidence = 0.5
        format_gate = True
        total = 0.3
    else:
        wrong = (
            DiagnosisLabel.REAL_BUG
            if scenario.ground_truth.label is not DiagnosisLabel.REAL_BUG
            else DiagnosisLabel.INFRA_NETWORK
        )
        diagnosis = wrong
        confidence = 0.9
        format_gate = True
        total = 0.0

    terminal_action = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=diagnosis,
        confidence=confidence,
        secondary_actions=[],
    )

    terminal_step = len(tool_call_names)
    terminal_observation = _make_observation(
        episode_id=episode_id,
        step=terminal_step,
        tool_response=None,
        budget=budget,
        is_terminal=True,
    )
    history.append(
        StepRecord(
            step=terminal_step,
            action=terminal_action,
            observation=terminal_observation,
            cost_charged=0.0,
        )
    )

    episode_state = EpisodeState(
        episode_id=episode_id,
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        step=terminal_step,
        history=history,
        budget=budget,
        is_terminated=True,
        final_action=terminal_action,
    )

    reward = RewardBreakdown(
        schema_version="1.0",
        total=total if format_gate else 0.0,
        format_gate=format_gate,
        components={
            "diagnosis_correctness": ComponentScore(
                raw=1.0 if outcome == "good" else (0.3 if outcome == "abstain" else 0.0),
                weighted=total,
                weight=1.0,
                sub_scores={},
            ),
        },
        counterfactual=None,
    )

    return EpisodeTrace(
        schema_version="1.0",
        episode=episode_state,
        reward_breakdown=reward,
        counterfactual_replay=None,
    )
