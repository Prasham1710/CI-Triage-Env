import pytest

from ci_triage_env.rewards.format_gate import FormatGate
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import ComponentScore, RewardBreakdown
from ci_triage_env.schemas.scenario import Scenario

# ── helpers ──────────────────────────────────────────────────────────────────


def _budget() -> BudgetState:
    return BudgetState(tool_calls_remaining=10, cost_remaining=5.0)


def _obs(episode_id: str, step: int) -> Observation:
    return Observation(
        episode_id=episode_id,
        step=step,
        budget_remaining=_budget(),
    )


def _reward_breakdown(gate: bool = True) -> RewardBreakdown:
    return RewardBreakdown(schema_version="1.0", total=0.0, format_gate=gate, components={})


def _make_trace(history: list[StepRecord], final_action: TerminalAction | None = None) -> EpisodeTrace:
    return EpisodeTrace(
        episode=EpisodeState(
            episode_id="test-ep",
            scenario_id="test-sc",
            seed=1,
            step=len(history),
            history=history,
            budget=_budget(),
            is_terminated=True,
            final_action=final_action,
        ),
        reward_breakdown=_reward_breakdown(),
        counterfactual_replay=None,
    )


def _valid_history() -> tuple[list[StepRecord], TerminalAction]:
    """History with correct required args for each tool call."""
    terminal = TerminalAction(diagnosis=DiagnosisLabel.REAL_BUG, confidence=0.9)
    history = [
        StepRecord(
            step=0,
            action=ToolCall(tool_name="read_logs", args={"scope": "test"}),
            observation=_obs("test-ep", 0),
            cost_charged=0.001,
        ),
        StepRecord(
            step=1,
            action=ToolCall(tool_name="query_flake_history", args={"test_name": "test_widget"}),
            observation=_obs("test-ep", 1),
            cost_charged=0.002,
        ),
        StepRecord(
            step=2,
            action=terminal,
            observation=_obs("test-ep", 2),
            cost_charged=0.0,
        ),
    ]
    return history, terminal


# ── tests ────────────────────────────────────────────────────────────────────


def test_format_gate_correct_case_returns_high_score(make_mock_scenario):
    history, terminal = _valid_history()
    trace = _make_trace(history, terminal)
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw == 1.0


def test_format_gate_unknown_tool_returns_zero(make_mock_scenario):
    history = [
        StepRecord(
            step=0,
            action=ToolCall(tool_name="nonexistent_tool", args={}),
            observation=_obs("test-ep", 0),
            cost_charged=0.0,
        )
    ]
    trace = _make_trace(history)
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw == 0.0


def test_format_gate_invalid_args_returns_zero(make_mock_scenario):
    # read_logs requires scope to be in enum; passing an invalid value fails
    history = [
        StepRecord(
            step=0,
            action=ToolCall(tool_name="read_logs", args={"scope": "INVALID_VALUE"}),
            observation=_obs("test-ep", 0),
            cost_charged=0.001,
        )
    ]
    trace = _make_trace(history)
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw == 0.0


def test_format_gate_missing_required_arg_returns_zero(make_mock_scenario):
    # query_flake_history requires test_name
    history = [
        StepRecord(
            step=0,
            action=ToolCall(tool_name="query_flake_history", args={}),
            observation=_obs("test-ep", 0),
            cost_charged=0.002,
        )
    ]
    trace = _make_trace(history)
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw == 0.0


def test_format_gate_handles_no_terminal_action(make_mock_scenario):
    # Budget-exhausted: no terminal action, only valid tool calls → gate passes
    history = [
        StepRecord(
            step=0,
            action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
            observation=_obs("test-ep", 0),
            cost_charged=0.001,
        )
    ]
    trace = _make_trace(history, final_action=None)
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw == 1.0


def test_format_gate_empty_history_passes(make_mock_scenario):
    trace = _make_trace([])
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw == 1.0


def test_format_gate_deterministic(make_mock_scenario):
    history, terminal = _valid_history()
    trace = _make_trace(history, terminal)
    component = FormatGate()
    s1 = component.score(trace, make_mock_scenario)
    s2 = component.score(trace, make_mock_scenario)
    assert s1.raw == s2.raw


def test_format_gate_score_is_in_documented_range(make_mock_scenario):
    history, terminal = _valid_history()
    trace = _make_trace(history, terminal)
    score = FormatGate().score(trace, make_mock_scenario)
    assert score.raw in (0.0, 1.0)


def test_format_gate_subscores_are_meaningful(make_mock_scenario):
    history, terminal = _valid_history()
    trace = _make_trace(history, terminal)
    score = FormatGate().score(trace, make_mock_scenario)
    assert isinstance(score, ComponentScore)
    assert "valid" in score.sub_scores


@pytest.fixture
def make_mock_scenario():
    from ci_triage_env.mock import make_mock_scenario as _make
    return _make("real_bug")
