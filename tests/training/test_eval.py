"""Tests for Phase C5 — evaluation harness (no GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.training.baselines.heuristic_policy import HeuristicPolicy
from ci_triage_env.training.baselines.random_policy import ALL_TOOL_ARG_DEFAULTS, RandomPolicy
from ci_triage_env.training.eval import Evaluator
from ci_triage_env.training.readme_table import generate_results_table


def _mock_obs() -> Observation:
    return Observation(
        episode_id="ep-test",
        step=0,
        failure_summary=None,
        budget_remaining=BudgetState(tool_calls_remaining=12, cost_remaining=1.0),
    )


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


def test_random_policy_emits_valid_action() -> None:
    p = RandomPolicy(seed=42)
    action = p.act(_mock_obs(), [])
    assert "tool_name" in action or "action_type" in action


def test_random_policy_terminates_at_max_turns() -> None:
    p = RandomPolicy(max_turns=2, seed=0)
    action = p.act(_mock_obs(), [{}, {}])  # len == max_turns
    assert action["action_type"] == "submit_diagnosis"


def test_random_policy_terminal_is_valid() -> None:
    from ci_triage_env.schemas.diagnosis import DiagnosisLabel

    p = RandomPolicy(seed=7)
    action = p._random_terminal()
    assert action["diagnosis"] in list(DiagnosisLabel)
    assert 0.0 <= action["confidence"] <= 1.0
    assert action["secondary_actions"] == []


def test_random_tool_call_uses_valid_defaults() -> None:
    p = RandomPolicy(seed=0)
    for _ in range(30):
        action = p._random_tool_call()
        tool_name = action["tool_name"]
        assert tool_name in ALL_TOOL_ARG_DEFAULTS
        assert action["args"] == ALL_TOOL_ARG_DEFAULTS[tool_name]


# ---------------------------------------------------------------------------
# HeuristicPolicy
# ---------------------------------------------------------------------------


def test_heuristic_policy_completes_investigation_then_diagnoses() -> None:
    p = HeuristicPolicy()
    history: list = []
    for _ in range(4):
        action = p.act(_mock_obs(), history)
        assert "tool_name" in action
        history.append({"output": "ok"})
    final = p.act(_mock_obs(), history)
    assert final["action_type"] == "submit_diagnosis"


def test_heuristic_policy_plan_length() -> None:
    assert len(HeuristicPolicy.INVESTIGATION_PLAN) == 4


def test_heuristic_policy_defaults_to_ambiguous_on_no_match() -> None:
    p = HeuristicPolicy()
    # 4 "ok" outputs match no rules → ambiguous
    history = [{"output": "ok"}] * 4
    final = p._classify_from_history(history)
    assert final["diagnosis"] == "ambiguous"
    assert final["confidence"] == pytest.approx(0.4)


def test_heuristic_policy_secondary_for_real_bug() -> None:
    p = HeuristicPolicy()
    sa = p._secondary_for("real_bug")
    assert len(sa) == 1
    assert sa[0]["name"] == "file_bug"


def test_heuristic_policy_secondary_for_flake() -> None:
    p = HeuristicPolicy()
    for family in ("race_flake", "timing_flake"):
        sa = p._secondary_for(family)
        assert sa[0]["name"] == "quarantine_test"


# ---------------------------------------------------------------------------
# Evaluator._run_one
# ---------------------------------------------------------------------------


def test_evaluator_run_one_returns_row() -> None:
    from ci_triage_env.training.mock_env_client import MockEnvClient

    env = MockEnvClient(seed=0)
    evaluator = Evaluator(env_client=env)
    policy = RandomPolicy(seed=0)
    # "real_bug-00" prefix causes load_scenario to fall back to make_mock_scenario("real_bug")
    row = evaluator._run_one(policy, "real_bug-00", seed=1)

    expected_keys = {
        "baseline", "scenario_id", "family", "difficulty", "seed",
        "total_reward", "format_gate", "diagnosis_correct",
        "predicted_diagnosis", "true_diagnosis", "action_quality",
        "tool_call_count", "total_cost", "confidence",
        "is_ambiguous_scenario", "brier_on_ambiguous",
    }
    assert expected_keys <= set(row.keys())
    assert isinstance(row["total_reward"], float)
    assert isinstance(row["tool_call_count"], int)
    assert row["scenario_id"] == "real_bug-00"
    assert row["seed"] == 1


def test_evaluator_run_one_heuristic() -> None:
    from ci_triage_env.training.mock_env_client import MockEnvClient

    env = MockEnvClient(seed=1)
    evaluator = Evaluator(env_client=env)
    row = evaluator._run_one(HeuristicPolicy(), "race_flake-00", seed=2)
    assert 0.0 <= row["total_cost"]
    assert isinstance(row["diagnosis_correct"], bool)


# ---------------------------------------------------------------------------
# generate_results_table
# ---------------------------------------------------------------------------


def test_results_table_markdown() -> None:
    df = pd.DataFrame({
        "baseline": ["random", "heuristic", "random", "heuristic"],
        "diagnosis_correct": [0.2, 0.6, 0.3, 0.7],
        "action_quality": [0.1, 0.5, 0.2, 0.55],
        "total_cost": [0.05, 0.04, 0.06, 0.03],
        "tool_call_count": [4, 4, 5, 4],
        "total_reward": [0.1, 0.5, 0.2, 0.6],
    })
    table = generate_results_table(df)
    assert isinstance(table, str)
    assert "|" in table
    assert "baseline" in table
    assert "diagnosis_acc" in table
    assert "heuristic" in table
    assert "random" in table


def test_results_table_float_format() -> None:
    df = pd.DataFrame({
        "baseline": ["a"],
        "diagnosis_correct": [0.123456],
        "action_quality": [0.5],
        "total_cost": [0.01],
        "tool_call_count": [3],
        "total_reward": [0.25],
    })
    table = generate_results_table(df)
    assert "0.123" in table
    assert "0.124" not in table  # 3-decimal truncation, not rounding beyond


# ---------------------------------------------------------------------------
# plot_all_eval_metrics (matplotlib mocked)
# ---------------------------------------------------------------------------


def test_plotting_writes_pngs(tmp_path: Path) -> None:
    """plot_all_eval_metrics records ≥ 5 savefig calls (matplotlib mocked)."""
    df = pd.DataFrame({
        "baseline": ["random", "heuristic", "random", "heuristic"],
        "diagnosis_correct": [True, False, True, True],
        "total_reward": [0.1, 0.5, 0.2, 0.6],
        "family": ["real_bug", "race_flake", "real_bug", "race_flake"],
        "total_cost": [0.05, 0.04, 0.06, 0.03],
        "tool_call_count": [4, 4, 5, 3],
        "confidence": [0.9, 0.6, 0.8, 0.7],
        "is_ambiguous_scenario": [False, False, False, False],
        "brier_on_ambiguous": [None, None, None, None],
    })

    saved: list[str] = []
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_sns = MagicMock()

    def _record_savefig(path, **kwargs):
        saved.append(str(path))

    mock_fig.savefig.side_effect = _record_savefig

    # Patch the module-level plt/sns so the function uses our mocks without
    # needing sys.modules tricks (plotting.py uses try/except module-level imports).
    with patch("ci_triage_env.training.plotting.plt", mock_plt), \
         patch("ci_triage_env.training.plotting.sns", mock_sns):
        from ci_triage_env.training.plotting import plot_all_eval_metrics
        plot_all_eval_metrics(df, tmp_path / "plots")

    assert len(saved) >= 5, f"Expected ≥ 5 savefig calls, got {len(saved)}: {saved}"
