"""Training curve plots and ablation summary charts for CI-Triage-Env.

Module-level try/except lets the module import without matplotlib; tests patch
`ci_triage_env.training.curves.plt` and `.sns` directly.
W&B imports are lazy (inside functions) since wandb is Colab-only.
"""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
    import seaborn as sns  # type: ignore[import]
except ImportError:
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]


def plot_training_curves_from_wandb(run_id: str, output_dir: Path) -> None:
    """Pull per-step metrics from W&B and write training curve PNGs.

    Args:
        run_id: W&B run path, e.g. "<entity>/ci-triage-env/<run_id>".
        output_dir: Directory to write PNG files.
    """
    if plt is None:
        raise ImportError("matplotlib required — install with: pip install matplotlib")

    import wandb  # type: ignore[import]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(run_id)
    history = run.history(samples=10000)

    # 1. Reward curve (raw + smoothed)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["step"], history["mean_reward"], alpha=0.4, label="raw")
    smooth = history["mean_reward"].rolling(window=20, min_periods=1).mean()
    ax.plot(history["step"], smooth, label="smoothed (window=20)", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("GRPO training: reward over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_reward_curve.png", dpi=120)
    plt.close(fig)

    # 2. Per-component reward breakdown
    component_keys = [
        "reward/diagnosis", "reward/action_quality", "reward/cost_efficiency",
        "reward/investigation", "reward/anti_gaming",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in component_keys:
        if k in history.columns:
            smooth = history[k].rolling(window=20, min_periods=1).mean()
            ax.plot(history["step"], smooth, label=k.replace("reward/", ""))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean component reward (smoothed)")
    ax.set_title("Per-component reward breakdown over training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_component_curve.png", dpi=120)
    plt.close(fig)

    # 3. KL divergence
    if "kl" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history["step"], history["kl"])
        ax.set_xlabel("Training step")
        ax.set_ylabel("KL(policy || ref)")
        ax.set_title("KL divergence to SFT reference")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "kl_divergence.png", dpi=120)
        plt.close(fig)

    # 4. Format-gate pass rate
    if "format_gate_pass_rate" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history["step"], history["format_gate_pass_rate"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Format gate pass rate")
        ax.set_title("Fraction of episodes passing format gate")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "format_gate.png", dpi=120)
        plt.close(fig)


def plot_ablation_summary(df, output_dir=None) -> None:
    """Bar chart comparing diagnosis accuracy and total reward across ablations.

    Args:
        df: DataFrame with columns [ablation, baseline, diagnosis_correct, total_reward, ...].
        output_dir: Directory to write ablation_summary.png. Defaults to
            data_artifacts/results/plots/.
    """
    if plt is None:
        raise ImportError("matplotlib required — install with: pip install matplotlib")

    if output_dir is None:
        output_dir = Path("data_artifacts/results/plots/")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = df.groupby("ablation").agg(
        diagnosis_acc=("diagnosis_correct", "mean"),
        total_reward=("total_reward", "mean"),
        action_qual=("action_quality", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(list(summary["ablation"]), list(summary["diagnosis_acc"]))
    axes[0].set_title("Diagnosis Accuracy by Ablation")
    axes[0].set_ylabel("Diagnosis Accuracy")
    axes[0].set_xlabel("Ablation")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(list(summary["ablation"]), list(summary["total_reward"]))
    axes[1].set_title("Mean Total Reward by Ablation")
    axes[1].set_ylabel("Mean Total Reward")
    axes[1].set_xlabel("Ablation")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_dir / "ablation_summary.png", dpi=120)
    plt.close(fig)
