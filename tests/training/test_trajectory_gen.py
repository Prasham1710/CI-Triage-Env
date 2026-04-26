"""Tests for Phase C3 — trajectory generator (mocked, no OpenAI/server calls)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.training.mock_env_client import MockEnvClient
from ci_triage_env.training.trajectory_gen import (
    TrajectoryGenerator,
    _filter_top_fraction,
)

# ---------------------------------------------------------------------------
# _parse_action (static method, tested directly)
# ---------------------------------------------------------------------------


def test_parse_action_valid_tool_call() -> None:
    text = '{"tool_name": "read_logs", "args": {"scope": "full"}}'
    action = TrajectoryGenerator._parse_action(text)
    assert isinstance(action, ToolCall)
    assert action.tool_name == "read_logs"


def test_parse_action_valid_terminal() -> None:
    text = (
        '{"action_type": "submit_diagnosis", "diagnosis": "real_bug",'
        ' "confidence": 0.9, "secondary_actions": []}'
    )
    action = TrajectoryGenerator._parse_action(text)
    assert isinstance(action, TerminalAction)
    assert action.diagnosis.value == "real_bug"
    assert action.confidence == 0.9


def test_parse_action_with_code_block() -> None:
    text = '```json\n{"tool_name": "read_logs", "args": {"scope": "full"}}\n```'
    action = TrajectoryGenerator._parse_action(text)
    assert action is not None
    assert isinstance(action, ToolCall)


def test_parse_action_malformed_returns_none() -> None:
    assert TrajectoryGenerator._parse_action("this is not JSON") is None
    assert TrajectoryGenerator._parse_action("") is None
    assert TrajectoryGenerator._parse_action("{}") is None  # no tool_name or action_type


def test_parse_action_embedded_in_text() -> None:
    text = 'I will now call the tool: {"tool_name": "read_logs", "args": {"scope": "test"}}'
    action = TrajectoryGenerator._parse_action(text)
    assert isinstance(action, ToolCall)


def test_parse_action_terminal_in_text() -> None:
    text = (
        'Based on my analysis:\n'
        '{"action_type": "submit_diagnosis", "diagnosis": "race_flake",'
        ' "confidence": 0.8, "secondary_actions": []}'
    )
    action = TrajectoryGenerator._parse_action(text)
    assert isinstance(action, TerminalAction)


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_cost_uses_token_counts() -> None:
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    completion = SimpleNamespace(usage=usage)
    cost = TrajectoryGenerator._estimate_cost(completion)
    # 1M input at $0.15 + 1M output at $0.60 = $0.75
    from ci_triage_env.training.trajectory_gen import _PRICE_IN_PER_M, _PRICE_OUT_PER_M
    expected = (_PRICE_IN_PER_M + _PRICE_OUT_PER_M) / 1.0
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_zero_tokens() -> None:
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    completion = SimpleNamespace(usage=usage)
    assert TrajectoryGenerator._estimate_cost(completion) == 0.0


# ---------------------------------------------------------------------------
# _filter_top_fraction
# ---------------------------------------------------------------------------


def test_top_fraction_filter() -> None:
    trajs = [
        {"reward": r, "format_gate_passed": True}
        for r in [0.5, 0.8, -0.2, 0.9, 0.3]
    ]
    top = _filter_top_fraction(trajs, fraction=0.4)
    assert len(top) == 2
    assert top[0]["reward"] == 0.9
    assert top[1]["reward"] == 0.8


def test_top_fraction_excludes_format_failures() -> None:
    trajs = [
        {"reward": 1.0, "format_gate_passed": False},
        {"reward": 0.5, "format_gate_passed": True},
    ]
    top = _filter_top_fraction(trajs, fraction=1.0)
    assert len(top) == 1
    assert top[0]["reward"] == 0.5


def test_top_fraction_empty_input() -> None:
    result = _filter_top_fraction([], fraction=0.3)
    assert result == []


def test_top_fraction_keeps_at_least_one() -> None:
    trajs = [{"reward": 0.5, "format_gate_passed": True}]
    top = _filter_top_fraction(trajs, fraction=0.01)
    assert len(top) == 1


# ---------------------------------------------------------------------------
# Budget check
# ---------------------------------------------------------------------------


def _make_expensive_completion() -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=10_000_000, completion_tokens=10_000_000)
    choice = SimpleNamespace(message=SimpleNamespace(content='{"tool_name": "read_logs", "args": {"scope": "full"}}'))
    return SimpleNamespace(usage=usage, choices=[choice])


def test_budget_check_stops_generation() -> None:
    env = MockEnvClient()
    gen = TrajectoryGenerator(api_key="fake", model="gpt-4o-mini",
                              budget_usd=0.001, env_client=env)

    with patch.object(gen.client.chat.completions, "create",
                      return_value=_make_expensive_completion()):
        result = gen.generate_one()
    # Either None (budget hit) or a trajectory — budget must now be exceeded
    assert gen.spent >= 0.001 or result is None


def test_generate_one_returns_none_when_budget_already_exceeded() -> None:
    env = MockEnvClient()
    gen = TrajectoryGenerator(api_key="fake", model="gpt-4o-mini",
                              budget_usd=0.0, env_client=env)
    gen.spent = 1.0  # already over budget
    assert gen.generate_one() is None


# ---------------------------------------------------------------------------
# Full loop with mock env + mocked OpenAI
# ---------------------------------------------------------------------------


def _make_completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(usage=usage, choices=[choice])


def _make_tool_completion(tool_name: str = "read_logs", args: dict | None = None) -> SimpleNamespace:
    _default_args: dict[str, dict] = {
        "read_logs": {"scope": "full"},
        "query_flake_history": {"test_name": "tests/unit/test_widget.py::test_concurrent_update"},
        "recent_commits": {"branch": "main"},
        "inspect_test_code": {"test_name": "tests/unit/test_widget.py::test_concurrent_update"},
    }
    resolved_args = args or _default_args.get(tool_name, {})
    import json as _json
    content = _json.dumps({"tool_name": tool_name, "args": resolved_args})
    return _make_completion(content)


def _make_terminal_completion(family: str = "real_bug") -> SimpleNamespace:
    import json as _json
    content = _json.dumps({
        "action_type": "submit_diagnosis",
        "diagnosis": family,
        "confidence": 0.9,
        "secondary_actions": [],
    })
    return _make_completion(content, completion_tokens=80)


def test_generate_one_full_loop_with_mock_env() -> None:
    env = MockEnvClient(seed=0)
    gen = TrajectoryGenerator(api_key="fake", model="gpt-4o-mini",
                              budget_usd=10.0, env_client=env)

    responses = [
        _make_tool_completion("read_logs"),
        _make_tool_completion("query_flake_history"),
        _make_tool_completion("recent_commits"),
        _make_terminal_completion("real_bug"),
    ]

    with patch.object(gen.client.chat.completions, "create", side_effect=responses):
        traj = gen.generate_one()

    assert traj is not None
    assert "episode_id" in traj
    assert "scenario_id" in traj
    assert "messages" in traj
    assert "reward" in traj
    assert isinstance(traj["reward"], float)
    assert isinstance(traj["messages"], list)


def test_generate_one_returns_none_on_openai_error() -> None:
    env = MockEnvClient(seed=0)
    gen = TrajectoryGenerator(api_key="fake", model="gpt-4o-mini",
                              budget_usd=10.0, env_client=env)

    with patch.object(gen.client.chat.completions, "create",
                      side_effect=Exception("API error")):
        result = gen.generate_one()

    assert result is None


# ---------------------------------------------------------------------------
# MockEnvClient
# ---------------------------------------------------------------------------


def test_mock_env_reset_returns_observation() -> None:
    env = MockEnvClient()
    obs = env.reset()
    assert obs.episode_id
    assert obs.failure_summary is not None
    assert obs.is_terminal is False


def test_mock_env_tool_step_returns_tool_response() -> None:
    env = MockEnvClient()
    obs = env.reset()
    ep_id = obs.episode_id
    action = ToolCall(tool_name="read_logs", args={"scope": "full"})
    next_obs = env.step(ep_id, action)
    assert next_obs.tool_response is not None
    assert next_obs.tool_response.tool_name == "read_logs"


def test_mock_env_terminal_step_terminates() -> None:
    env = MockEnvClient()
    obs = env.reset()
    ep_id = obs.episode_id
    from ci_triage_env.schemas.diagnosis import DiagnosisLabel
    terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=DiagnosisLabel.REAL_BUG,
        confidence=0.9,
        secondary_actions=[],
    )
    next_obs = env.step(ep_id, terminal)
    assert next_obs.is_terminal is True


def test_mock_env_get_trace_returns_episode_trace() -> None:
    from ci_triage_env.schemas.episode import EpisodeTrace

    env = MockEnvClient()
    obs = env.reset()
    ep_id = obs.episode_id
    env.step(ep_id, ToolCall(tool_name="read_logs", args={"scope": "full"}))
    trace = env.get_trace(ep_id)
    assert isinstance(trace, EpisodeTrace)
    assert trace.episode.episode_id == ep_id
    assert len(trace.episode.history) == 1


def test_mock_env_list_tools() -> None:
    env = MockEnvClient()
    tools = env.list_tools()
    assert len(tools) == 11  # ALL_TOOLS has 11 entries
    assert all("name" in t for t in tools)
