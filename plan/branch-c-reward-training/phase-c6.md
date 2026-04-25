# Phase C6 — Curves + Ablations

**Owner:** Branch C.
**Prerequisite:** C5 merged. Trained model + eval results exist.
**Estimated time:** 4–5 hours including ablation runs.
**Budget impact:** ~$15 (5 ablation runs × ~$3 each).

---

## Outcome

The reward-layer ablation matrix and final training curves. By end of phase:

1. Four ablation runs complete: each with one reward layer dropped (set weight to 0).
2. `data_artifacts/results/ablations.csv` — eval results for each ablation.
3. Per-ablation training curves saved as PNG.
4. Reward curve PNG (full training run, smoothed): the headline plot.
5. README's Results section is fully populated with embedded plots.
6. All planned `[FILL POST-TRAIN]` markers removed from README.

---

**Budget update.** With 4 ablations instead of 5, ablation phase is ~$12 instead of $15. Reserve the saved $3 for additional baseline-vs-trained comparison seeds.

## Files to create

### `src/ci_triage_env/training/ablations.py`

```python
ABLATIONS = {
    "no_diagnosis": {"diagnosis": 0.0},
    "no_action_quality": {"action_quality": 0.0},
    "no_investigation": {"investigation": 0.0},
    "no_anti_gaming": {"anti_gaming": 0.0},
    # counterfactual ablation is deferred to v2 — its weight is already 0 in v1.
}

def run_ablation(name: str, weight_overrides: dict, total_steps: int = 1000,
                 base_sft_checkpoint: str = "checkpoints/sft/",
                 base_grpo_checkpoint: str = "checkpoints/grpo_full/"):
    """Run one ablation: take the SFT checkpoint, train GRPO with modified weights, eval."""
    from .grpo import run_grpo

    # Compute custom weights
    new_weights = {**REWARD_WEIGHTS}
    for k, v in weight_overrides.items():
        if k in new_weights:
            new_weights[k] = v

    output_dir = f"checkpoints/ablation_{name}/"
    run_grpo(
        sft_checkpoint_dir=base_sft_checkpoint,
        output_dir=output_dir,
        total_steps=total_steps,
        weights_override=new_weights,    # add this kwarg to run_grpo
    )
    # Run eval
    from .eval import Evaluator
    evaluator = Evaluator(trained_checkpoint=output_dir)
    evaluator.BASELINES = ["random", "heuristic", "trained"]   # skip 9B zero-shot
    df = evaluator.run_all(seeds=[1])
    df["ablation"] = name
    return df

def main():
    all_results = []
    for name, overrides in ABLATIONS.items():
        print(f"=== Ablation: {name} ===")
        df = run_ablation(name, overrides, total_steps=1000)
        all_results.append(df)
    full = pd.concat(all_results)
    full.to_csv("data_artifacts/results/ablations.csv", index=False)
    plot_ablation_summary(full)
```

### `src/ci_triage_env/training/curves.py`

```python
def plot_training_curves_from_wandb(run_id: str, output_dir: Path):
    """Pull per-step metrics from W&B, plot, save."""
    import wandb
    api = wandb.Api()
    run = api.run(f"<entity>/ci-triage-env/{run_id}")
    history = run.history(samples=10000)

    # Reward curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["step"], history["mean_reward"], label="raw")
    smooth = history["mean_reward"].rolling(window=20, min_periods=1).mean()
    ax.plot(history["step"], smooth, label="smoothed (window=20)", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("GRPO training: reward over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_reward_curve.png", dpi=120)

    # Per-component breakdown over training
    component_keys = ["reward/diagnosis", "reward/action_quality", "reward/cost_efficiency",
                      "reward/investigation", "reward/anti_gaming"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in component_keys:
        if k in history.columns:
            smooth = history[k].rolling(window=20, min_periods=1).mean()
            ax.plot(history["step"], smooth, label=k.replace("reward/", ""))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean component reward (smoothed)")
    ax.legend()
    fig.savefig(output_dir / "per_component_curve.png", dpi=120)

    # KL divergence
    if "kl" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history["step"], history["kl"])
        ax.set_xlabel("Training step")
        ax.set_ylabel("KL(policy || ref)")
        ax.set_title("KL divergence to SFT reference")
        ax.grid(alpha=0.3)
        fig.savefig(output_dir / "kl_divergence.png", dpi=120)

    # Format-gate pass rate
    if "format_gate_pass_rate" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history["step"], history["format_gate_pass_rate"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Format gate pass rate")
        fig.savefig(output_dir / "format_gate.png", dpi=120)

def plot_ablation_summary(df: pd.DataFrame):
    """Bar chart: each ablation's diagnosis_acc and total_reward."""
    summary = df.groupby("ablation").agg(
        diagnosis_acc=("diagnosis_correct", "mean"),
        total_reward=("total_reward", "mean"),
        action_qual=("action_quality", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary.plot(x="ablation", y="diagnosis_acc", kind="bar", ax=axes[0])
    axes[0].set_title("Diagnosis Accuracy by Ablation")
    axes[0].axhline(y=summary.loc[summary["ablation"] == "full", "diagnosis_acc"].iloc[0] if "full" in summary["ablation"].values else 0.5,
                    color="r", linestyle="--", label="full reward")
    summary.plot(x="ablation", y="total_reward", kind="bar", ax=axes[1])
    axes[1].set_title("Mean Total Reward by Ablation")
    fig.tight_layout()
    fig.savefig("data_artifacts/results/plots/ablation_summary.png", dpi=120)
```

### `src/ci_triage_env/training/finalize_readme.py`

```python
def populate_readme(template_path: Path = Path("README.md"),
                    eval_csv: Path = Path("data_artifacts/results/eval.csv"),
                    ablation_csv: Path = Path("data_artifacts/results/ablations.csv"),
                    plots_dir: Path = Path("data_artifacts/results/plots/")):
    """Fill [FILL POST-TRAIN] markers in README."""
    template = template_path.read_text()
    df = pd.read_csv(eval_csv)
    abl = pd.read_csv(ablation_csv)

    # Generate the results table
    results_md = generate_results_table(df)
    template = template.replace("[FILL: 5-row × 6-metric table]", results_md)

    # Embed plot images
    for plot_name in plots_dir.glob("*.png"):
        marker = f"[FILL: {plot_name.stem.replace('_', ' ')}]"
        embed = f"![{plot_name.stem}]({plot_name.relative_to(template_path.parent)})"
        template = template.replace(marker, embed)

    # ... handle remaining [FILL] markers programmatically

    template_path.write_text(template)
```

### `notebooks/eval.ipynb`

A separate Colab-runnable notebook that:
1. Loads trained checkpoint.
2. Runs full eval.
3. Generates all plots.
4. Optionally runs ablations.
5. Writes results to disk.

This is what judges use to reproduce eval if they want.

---

## Implementation notes

- **Ablation runs are 1000 steps**, not 3000. Reason: budget. We're not aiming to match full-training quality on each ablation — we're showing the *delta* from removing a component. 1000 steps is enough to see whether the ablated model converges noticeably worse.
- **`run_grpo` needs a `weights_override` kwarg.** Add it in C4 if not already present. Goes through to `CompositeReward(weights=...)`.
- **Plot polish.** Per the keynote: label axes clearly, include units, save as PNG. Reviewers spend seconds. Make titles descriptive ("GRPO training: reward over time", not "Plot").
- **Same axes across ablations.** Plot all 4 ablation reward curves on the same axes (one figure with 4 lines) — easier to read than 4 separate figures.

---

## Tests required (`tests/training/test_ablations.py`)

```python
def test_ablations_dict_has_4_entries():
    assert len(ABLATIONS) == 4   # counterfactual ablation deferred to v2

def test_each_ablation_zeros_one_weight():
    for name, overrides in ABLATIONS.items():
        assert sum(1 for v in overrides.values() if v == 0.0) == 1

def test_run_ablation_smoke(monkeypatch):
    """Mock run_grpo + Evaluator; verify ablation flow returns DataFrame."""

def test_plot_ablation_summary_writes_png(tmp_path):
    """plot_ablation_summary on a fixture df writes the PNG."""

def test_finalize_readme_replaces_markers(tmp_path):
    """populate_readme replaces [FILL] markers."""
```

---

## Execution plan (the actual run)

```bash
# After full GRPO training is done and eval is complete, run ablations.

# 1. Run all 5 ablations sequentially (~5 hours total)
python -m ci_triage_env.training.ablations

# 2. Pull training curves from W&B
python -c "
from ci_triage_env.training.curves import plot_training_curves_from_wandb
plot_training_curves_from_wandb('<wandb_run_id>', Path('data_artifacts/results/plots/'))
"

# 3. Generate ablation summary plot
python -c "
from ci_triage_env.training.curves import plot_ablation_summary
import pandas as pd
df = pd.read_csv('data_artifacts/results/ablations.csv')
plot_ablation_summary(df)
"

# 4. Populate README
python -c "
from ci_triage_env.training.finalize_readme import populate_readme
populate_readme()
"

# 5. Verify no [FILL] markers remain
grep -n '\[FILL' README.md && echo 'WARNING: unfilled markers found' || echo 'README ready'
```

---

## What "done" looks like

- `data_artifacts/results/eval.csv` — the master eval table
- `data_artifacts/results/ablations.csv` — ablation results
- `data_artifacts/results/plots/`:
  - `training_reward_curve.png` — the headline plot
  - `per_component_curve.png`
  - `kl_divergence.png`
  - `format_gate.png`
  - `diagnosis_accuracy.png`
  - `total_reward.png`
  - `per_family_accuracy.png`
  - `calibration_ambiguous.png`
  - `cost_distribution.png`
  - `ablation_summary.png`
- README with all `[FILL POST-TRAIN]` blocks populated, plots embedded, table generated.

---

## Open questions

1. **W&B run ID retrieval.** Check whether your W&B run is `<entity>/ci-triage-env/<run_id>`. If running locally without W&B, save metrics to JSON during training and plot from JSON instead.
2. **What if an ablation run fails (NaN, crash)?** Log it, skip, move to next. Document in the README that this ablation's column is missing due to training instability — don't fabricate numbers.
3. **Should we ablate weights individually or also combinatorially?** Individual is enough for v1. Combinatorial is overkill for the scope.

---

## Final pre-submission checks

After C6 is done, run the Definition-of-Done checklist from `INSTRUCTION-MANUAL.md` section 10:

- [ ] `main` is green
- [ ] HF Space loads
- [ ] README has no `[FILL]` markers
- [ ] Reward curve PNG embedded
- [ ] Comparison table embedded
- [ ] At least one ablation plot embedded
- [ ] Demo video published, link in README
- [ ] Colab notebook runs end-to-end on fresh runtime
- [ ] `openenv.yaml` validates
- [ ] Submission URL posted

---

## What's NOT in this phase

- Recording the demo video (separate task, anyone on team can do this last)
- Writing the HuggingFace blog post (separate, if time)
- Final submission portal posting
