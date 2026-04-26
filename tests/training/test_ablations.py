"""Tests for Phase C6 — ablations, curves, and readme finalization (no GPU)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from ci_triage_env.training.ablations import ABLATIONS, run_ablation
from ci_triage_env.training.finalize_readme import populate_readme

# ---------------------------------------------------------------------------
# ABLATIONS dict
# ---------------------------------------------------------------------------


def test_ablations_dict_has_4_entries() -> None:
    assert len(ABLATIONS) == 4  # counterfactual deferred to v2


def test_each_ablation_zeros_exactly_one_weight() -> None:
    for name, overrides in ABLATIONS.items():
        zeroed = [k for k, v in overrides.items() if v == 0.0]
        assert len(zeroed) == 1, f"Ablation '{name}' should zero exactly 1 weight, got {zeroed}"


def test_ablation_names_reference_valid_reward_keys() -> None:
    from ci_triage_env.rewards.weights import REWARD_WEIGHTS

    for name, overrides in ABLATIONS.items():
        for key in overrides:
            assert key in REWARD_WEIGHTS, (
                f"Ablation '{name}' references unknown reward key '{key}'"
            )


# ---------------------------------------------------------------------------
# run_ablation smoke (mock run_grpo + Evaluator)
# ---------------------------------------------------------------------------


def test_run_ablation_smoke() -> None:
    """Mock run_grpo and Evaluator; verify run_ablation returns a DataFrame."""
    fake_df = pd.DataFrame({
        "baseline": ["random", "heuristic", "trained"],
        "scenario_id": ["s1", "s1", "s1"],
        "family": ["real_bug"] * 3,
        "difficulty": ["easy"] * 3,
        "seed": [1, 1, 1],
        "total_reward": [0.1, 0.5, 0.7],
        "format_gate": [True] * 3,
        "diagnosis_correct": [False, True, True],
        "predicted_diagnosis": ["ambiguous", "real_bug", "real_bug"],
        "true_diagnosis": ["real_bug"] * 3,
        "action_quality": [0.0, 0.3, 0.5],
        "tool_call_count": [3, 4, 5],
        "total_cost": [0.03, 0.04, 0.05],
        "confidence": [0.5, 0.8, 0.9],
        "is_ambiguous_scenario": [False] * 3,
        "brier_on_ambiguous": [None] * 3,
    })

    mock_evaluator = MagicMock()
    mock_evaluator.run_all.return_value = fake_df

    with patch("ci_triage_env.training.ablations.run_grpo") as mock_grpo, \
         patch("ci_triage_env.training.ablations.Evaluator", return_value=mock_evaluator):
        mock_grpo.return_value = "checkpoints/ablation_test/"
        result = run_ablation(
            "no_diagnosis",
            {"diagnosis": 0.0},
            total_steps=10,
        )

    assert isinstance(result, pd.DataFrame)
    assert "ablation" in result.columns
    assert (result["ablation"] == "no_diagnosis").all()
    mock_grpo.assert_called_once()
    call_kwargs = mock_grpo.call_args.kwargs
    assert call_kwargs["weights_override"]["diagnosis"] == 0.0


def test_run_ablation_passes_weights_to_grpo() -> None:
    """Confirm the merged weights dict reaches run_grpo."""
    from ci_triage_env.rewards.weights import REWARD_WEIGHTS

    fake_df = pd.DataFrame({"baseline": [], "total_reward": [], "diagnosis_correct": [],
                             "scenario_id": [], "family": [], "difficulty": [], "seed": [],
                             "format_gate": [], "predicted_diagnosis": [], "true_diagnosis": [],
                             "action_quality": [], "tool_call_count": [], "total_cost": [],
                             "confidence": [], "is_ambiguous_scenario": [], "brier_on_ambiguous": []})
    mock_evaluator = MagicMock()
    mock_evaluator.run_all.return_value = fake_df

    with patch("ci_triage_env.training.ablations.run_grpo") as mock_grpo, \
         patch("ci_triage_env.training.ablations.Evaluator", return_value=mock_evaluator):
        mock_grpo.return_value = "checkpoints/ablation_no_anti_gaming/"
        run_ablation("no_anti_gaming", {"anti_gaming": 0.0}, total_steps=5)

    weights_sent = mock_grpo.call_args.kwargs["weights_override"]
    assert weights_sent["anti_gaming"] == 0.0
    # All other weights preserved from REWARD_WEIGHTS
    for k, v in REWARD_WEIGHTS.items():
        if k != "anti_gaming":
            assert weights_sent[k] == v


# ---------------------------------------------------------------------------
# plot_ablation_summary (matplotlib mocked)
# ---------------------------------------------------------------------------


def test_plot_ablation_summary_writes_png(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "ablation": ["no_diagnosis", "no_action_quality", "no_diagnosis", "no_action_quality"],
        "baseline": ["random", "random", "heuristic", "heuristic"],
        "diagnosis_correct": [0.2, 0.4, 0.5, 0.6],
        "total_reward": [0.1, 0.3, 0.4, 0.5],
        "action_quality": [0.0, 0.2, 0.3, 0.4],
    })

    saved: list[str] = []
    mock_fig = MagicMock()
    mock_axes = [MagicMock(), MagicMock()]
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    mock_sns = MagicMock()

    def _record(path, **kwargs):
        saved.append(str(path))

    mock_fig.savefig.side_effect = _record

    with patch("ci_triage_env.training.curves.plt", mock_plt), \
         patch("ci_triage_env.training.curves.sns", mock_sns):
        from ci_triage_env.training.curves import plot_ablation_summary
        plot_ablation_summary(df, output_dir=tmp_path / "plots")

    assert len(saved) >= 1
    assert any("ablation_summary" in s for s in saved)


# ---------------------------------------------------------------------------
# populate_readme
# ---------------------------------------------------------------------------


def test_finalize_readme_replaces_table_marker(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Results\n\n[FILL: 5-row × 6-metric table]\n\nMore text.\n"
    )
    eval_csv = tmp_path / "eval.csv"
    pd.DataFrame({
        "baseline": ["random", "heuristic"],
        "diagnosis_correct": [0.3, 0.6],
        "action_quality": [0.1, 0.4],
        "total_cost": [0.05, 0.03],
        "tool_call_count": [4, 4],
        "total_reward": [0.2, 0.5],
    }).to_csv(eval_csv, index=False)

    # No ablation csv; no plots dir
    n = populate_readme(
        template_path=readme,
        eval_csv=eval_csv,
        ablation_csv=tmp_path / "ablations.csv",
        plots_dir=tmp_path / "plots",
    )

    result = readme.read_text()
    assert "[FILL: 5-row × 6-metric table]" not in result
    assert "|" in result  # table was inserted
    assert n >= 1


def test_finalize_readme_embeds_plot_images(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Plots\n\n[FILL: diagnosis accuracy]\n")

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "diagnosis_accuracy.png").touch()

    n = populate_readme(
        template_path=readme,
        eval_csv=tmp_path / "eval.csv",
        ablation_csv=tmp_path / "ablations.csv",
        plots_dir=plots_dir,
    )

    result = readme.read_text()
    assert "[FILL: diagnosis accuracy]" not in result
    assert "diagnosis_accuracy.png" in result
    assert n >= 1


def test_finalize_readme_missing_csv_does_not_crash(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# CI Triage\n\nNo markers here.\n")

    n = populate_readme(
        template_path=readme,
        eval_csv=tmp_path / "nonexistent.csv",
        ablation_csv=tmp_path / "nonexistent2.csv",
        plots_dir=tmp_path / "no_plots",
    )
    assert n == 0
    assert readme.read_text() == "# CI Triage\n\nNo markers here.\n"
