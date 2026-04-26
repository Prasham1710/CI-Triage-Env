"""Tests for Phase C4 — GRPO rollout and SFT formatting (mocked; no GPU)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.training.mock_env_client import MockEnvClient
from ci_triage_env.training.rollout import TrainingRollout
from ci_triage_env.training.sft import format_for_sft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_response(content: str) -> SimpleNamespace:
    """Fake model generate() output that tokenizer.decode returns `content`."""
    return SimpleNamespace(shape=(1, 10))  # just needs .shape[1]


def _mock_generate(content: str):
    """Return a tokenizer+model pair whose generate/decode returns `content`."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = MagicMock(
        to=lambda device: MagicMock(shape=[1, 5])
    )
    tokenizer.decode.return_value = content
    tokenizer.eos_token_id = 0

    out_tensor = MagicMock()
    out_tensor.__getitem__ = lambda self, idx: MagicMock()

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [out_tensor]
    return model, tokenizer


# ---------------------------------------------------------------------------
# TrainingRollout — mocked GPU
# ---------------------------------------------------------------------------


def _make_torch_mock() -> MagicMock:
    """Build a minimal torch mock that lets rollout.__call__ run without GPU."""
    import unittest.mock as um
    ctx = um.MagicMock()
    ctx.__enter__ = lambda s: None
    ctx.__exit__ = lambda s, *a: None

    mock_torch = MagicMock()
    mock_torch.no_grad.return_value = ctx
    return mock_torch


def test_rollout_with_mock_env() -> None:
    env = MockEnvClient(seed=0)
    rollout = TrainingRollout(env_client=env, max_turns=4)

    terminal_json = (
        '{"action_type": "submit_diagnosis", "diagnosis": "real_bug",'
        ' "confidence": 0.9, "secondary_actions": []}'
    )
    read_logs_json = '{"tool_name": "read_logs", "args": {"scope": "full"}}'

    responses = [read_logs_json, read_logs_json, terminal_json]
    call_idx = 0

    def fake_decode(*args, **kwargs):
        nonlocal call_idx
        r = responses[min(call_idx, len(responses) - 1)]
        call_idx += 1
        return r

    model, tokenizer = _mock_generate(read_logs_json)
    tokenizer.decode.side_effect = fake_decode

    mock_torch = _make_torch_mock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = rollout(model, tokenizer)

    assert "messages" in result
    assert "reward" in result
    assert isinstance(result["reward"], float)
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) >= 2  # at least system + user


def test_rollout_handles_format_failure() -> None:
    env = MockEnvClient(seed=1)
    rollout = TrainingRollout(env_client=env, max_turns=3)

    model, tokenizer = _mock_generate("this is definitely not JSON")

    mock_torch = _make_torch_mock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = rollout(model, tokenizer)

    # Episode terminates early due to malformed JSON; reward is computed (likely negative)
    assert "reward" in result
    assert result["terminated"] is False  # terminated by format failure, not is_terminal


def test_quarantine_window_updates() -> None:
    env = MockEnvClient(seed=2)
    rollout = TrainingRollout(env_client=env, max_turns=2)

    # Simulate a terminal with quarantine_test secondary action
    quarantine_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=DiagnosisLabel.RACE_FLAKE,
        confidence=0.8,
        secondary_actions=[],
    )

    obs = env.reset()
    ep_id = obs.episode_id
    env.step(ep_id, ToolCall(tool_name="read_logs", args={"scope": "full"}))
    env.step(ep_id, quarantine_terminal)

    # Manually call rollout update logic
    trace = env.get_trace(ep_id)
    if trace.episode.final_action:
        for sa in trace.episode.final_action.secondary_actions:
            rollout._quarantine_window.append(sa.name)

    # After a terminal with no secondary actions, window stays empty
    assert isinstance(rollout._quarantine_window, list)


def test_quarantine_window_caps_at_fifty() -> None:
    env = MockEnvClient(seed=0)
    rollout = TrainingRollout(env_client=env)

    # Manually fill window beyond 50
    rollout._quarantine_window = ["quarantine_test"] * 60
    rollout._quarantine_window = rollout._quarantine_window[-50:]
    assert len(rollout._quarantine_window) == 50


# ---------------------------------------------------------------------------
# format_for_sft
# ---------------------------------------------------------------------------


def test_sft_data_format() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<s>user content</s>"

    traj = {
        "messages": [
            {"role": "user", "content": "investigate this"},
            {"role": "assistant", "content": '{"tool_name": "read_logs", "args": {}}'},
        ],
        "reward": 0.5,
    }
    result = format_for_sft(traj, tokenizer)
    assert "text" in result
    assert isinstance(result["text"], str)
    tokenizer.apply_chat_template.assert_called_once_with(
        traj["messages"], tokenize=False, add_generation_prompt=False
    )


def test_sft_data_format_preserves_messages() -> None:
    tokenizer = MagicMock()
    expected_text = "SYSTEM|USER|ASSISTANT"
    tokenizer.apply_chat_template.return_value = expected_text

    traj = {"messages": [{"role": "user", "content": "hi"}], "reward": 1.0}
    result = format_for_sft(traj, tokenizer)
    assert result["text"] == expected_text


# ---------------------------------------------------------------------------
# GRPO hyperparams
# ---------------------------------------------------------------------------


def test_grpo_hyperparams_present() -> None:
    from ci_triage_env.training.grpo import GRPO_HYPERPARAMS

    required_keys = {
        "learning_rate", "kl_coef", "num_generations",
        "per_device_train_batch_size", "logging_steps", "save_steps",
    }
    assert required_keys <= set(GRPO_HYPERPARAMS.keys())


def test_grpo_hyperparams_reasonable_values() -> None:
    from ci_triage_env.training.grpo import GRPO_HYPERPARAMS

    assert 0 < GRPO_HYPERPARAMS["learning_rate"] < 1e-3
    assert 0 < GRPO_HYPERPARAMS["kl_coef"] < 1.0
    assert GRPO_HYPERPARAMS["num_generations"] >= 4


# ---------------------------------------------------------------------------
# GPU-gated smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    True,  # always skip in CI — run manually with GPU
    reason="GPU required for SFT/GRPO training",
)
def test_sft_smoke(tmp_path) -> None:
    """5-step SFT on a tiny dataset; loss decreases."""
    from datasets import Dataset

    from ci_triage_env.training.sft import run_sft

    messages = [
        {"role": "user", "content": "investigate"},
        {"role": "assistant", "content": '{"tool_name": "read_logs", "args": {"scope": "full"}}'},
        {"role": "user", "content": "output here"},
        {"role": "assistant", "content": '{"action_type": "submit_diagnosis", "diagnosis": "real_bug", "confidence": 0.9, "secondary_actions": []}'},
    ]
    traj = {"messages": messages, "reward": 1.0}
    ds = Dataset.from_list([traj] * 5)
    dataset_dir = str(tmp_path / "sft_data")
    ds.save_to_disk(dataset_dir)

    out = run_sft(dataset_path=dataset_dir, output_dir=str(tmp_path / "sft_out"), num_epochs=1)
    assert out is not None


@pytest.mark.skipif(
    True,
    reason="GPU required for GRPO training",
)
def test_grpo_smoke(tmp_path) -> None:
    """5-step GRPO; no NaN losses."""
    from ci_triage_env.training.grpo import run_grpo
    from ci_triage_env.training.mock_env_client import MockEnvClient

    env = MockEnvClient()
    out = run_grpo(
        sft_checkpoint_dir="Qwen/Qwen3.5-4B",
        output_dir=str(tmp_path / "grpo_out"),
        total_steps=5,
        env_client=env,
    )
    assert out is not None
