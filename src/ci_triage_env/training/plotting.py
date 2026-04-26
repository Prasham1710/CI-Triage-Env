"""Evaluation metric plots for the CI-Triage-Env baseline comparison.

Module-level try/except lets the module import without matplotlib; tests patch
`ci_triage_env.training.plotting.plt` and `.sns` directly.
"""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
    import seaborn as sns  # type: ignore[import]
except ImportError:
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]


def plot_all_eval_metrics(df, output_dir) -> None:
    """Write ≥ 5 PNG plots summarising the evaluation results.

    Uses module-level plt/sns so tests can inject mocks without sys.modules tricks.
    Raises ImportError if matplotlib is not installed.
    """
    if plt is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Diagnosis accuracy by baseline
    fig, ax = plt.subplots(figsize=(8, 5))
    acc = df.groupby("baseline")["diagnosis_correct"].mean()
    ax.bar(list(acc.index), list(acc.values))
    ax.set_ylabel("Diagnosis Accuracy")
    ax.set_xlabel("Baseline")
    ax.set_title("Diagnosis Accuracy by Baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnosis_accuracy.png", dpi=120)
    plt.close(fig)

    # 2. Mean total reward ± std
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = df.groupby("baseline")["total_reward"].agg(["mean", "std"])
    ax.bar(list(agg.index), list(agg["mean"]), yerr=list(agg["std"].fillna(0)))
    ax.set_ylabel("Mean Total Reward")
    ax.set_xlabel("Baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "total_reward.png", dpi=120)
    plt.close(fig)

    # 3. Per-family accuracy heatmap
    pivot = df.pivot_table(
        index="baseline", columns="family",
        values="diagnosis_correct", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Greens", ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "per_family_accuracy.png", dpi=120)
    plt.close(fig)

    # 4. Reliability (calibration) on ambiguous scenarios
    amb = df[df["is_ambiguous_scenario"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for baseline in amb["baseline"].unique():
        sub = amb[amb["baseline"] == baseline]
        brier_vals = sub["brier_on_ambiguous"].fillna(0)
        ax.scatter(list(sub["confidence"]), list(1 - brier_vals), label=baseline, alpha=0.5)
    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Calibration score (1 − Brier)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_ambiguous.png", dpi=120)
    plt.close(fig)

    # 5. Tool-call cost distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="baseline", y="total_cost", ax=ax)
    ax.set_ylabel("Total cost ($)")
    fig.tight_layout()
    fig.savefig(output_dir / "cost_distribution.png", dpi=120)
    plt.close(fig)
