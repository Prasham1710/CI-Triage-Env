# Phase C5 — Evaluation Harness

**Owner:** Branch C.
**Prerequisite:** C4 merged. Trained model checkpoint exists.
**Estimated time:** 3–4 hours.
**Budget impact:** ~$5 (compute for running 5 baselines × 150 scenarios × 3 seeds).

---

## Outcome

Multi-baseline evaluation pipeline. By end of phase:

1. `python -m ci_triage_env.training.eval --output results/` runs the full 5-baseline matrix.
2. Output: `data_artifacts/results/eval.csv` — one row per (baseline, scenario, seed).
3. Output: per-metric plots embedded in `data_artifacts/results/plots/`.
4. Headline numbers logged to W&B and CSV: diagnosis accuracy (overall + per-class), action appropriateness, mean tool-call cost, time-to-resolution, Brier on ambiguous.
5. README's results table is auto-generated from `eval.csv`.
6. All C5 tests pass.

---

## Files to create

### `src/ci_triage_env/training/baselines/random_policy.py`

```python
class RandomPolicy:
    """Random tool calls + random terminal action. The floor."""
    name = "random"

    def __init__(self, max_turns: int = 8, seed: int = 0):
        self.rng = random.Random(seed)
        self.max_turns = max_turns

    def act(self, obs: Observation, history: list) -> dict:
        if len(history) >= self.max_turns:
            return self._random_terminal()
        if self.rng.random() < 0.2:   # 20% chance to terminate
            return self._random_terminal()
        return self._random_tool_call()

    def _random_tool_call(self) -> dict:
        tool = self.rng.choice(ALL_TOOL_NAMES)
        return {"tool_name": tool, "args": ALL_TOOL_ARG_DEFAULTS[tool]}

    def _random_terminal(self) -> dict:
        return {
            "action_type": "submit_diagnosis",
            "diagnosis": self.rng.choice(list(DiagnosisLabel)),
            "confidence": self.rng.random(),
            "secondary_actions": [],
        }
```

### `src/ci_triage_env/training/baselines/heuristic_policy.py`

```python
class HeuristicPolicy:
    """Hand-coded rule-based classifier — the strong simple baseline.
    Calls a fixed set of tools then classifies via keyword matching on logs."""
    name = "heuristic"

    INVESTIGATION_PLAN = [
        ("read_logs", {"scope": "full", "lines": 200}),
        ("query_flake_history", {}),
        ("recent_commits", {"window": "24h"}),
        ("cluster_metrics", {"window": "5m"}),
    ]

    def act(self, obs: Observation, history: list) -> dict:
        if len(history) < len(self.INVESTIGATION_PLAN):
            tool, args = self.INVESTIGATION_PLAN[len(history)]
            return {"tool_name": tool, "args": args}
        # Investigated everything; classify
        return self._classify_from_history(history)

    def _classify_from_history(self, history: list) -> dict:
        # Concatenate all observations
        all_text = " ".join(str(h.get("output", "")) for h in history)
        from ..data.clustering.classifier import RuleBasedClassifier
        record_proxy = type("R", (), {"log_text": all_text})()
        family, conf = RuleBasedClassifier().classify(record_proxy)
        if family == "unknown":
            family = "ambiguous"
            conf = 0.4
        # Map family to terminal action
        secondary = self._secondary_for(family)
        return {
            "action_type": "submit_diagnosis",
            "diagnosis": family,
            "confidence": conf,
            "secondary_actions": secondary,
        }

    def _secondary_for(self, family: str) -> list[dict]:
        if family == "real_bug":
            return [{"name": "file_bug", "args": {"severity": "high", "title": "auto", "summary": "auto"}}]
        if family in ("race_flake", "timing_flake"):
            return [{"name": "quarantine_test", "args": {"test_id": "auto", "reason": "auto"}}]
        if family.startswith("infra_"):
            return [{"name": "rerun_test", "args": {}}]
        if family == "dependency_drift":
            return [{"name": "ping_owner", "args": {"owner": "deps", "message": "auto"}}]
        return []
```

### `src/ci_triage_env/training/baselines/zero_shot.py`

```python
class ZeroShotPolicy:
    """Wrapper around any HF model used in zero-shot inference mode."""
    def __init__(self, model_name: str, system_prompt: str, name: str | None = None):
        self.name = name or f"zero_shot_{model_name.split('/')[-1]}"
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=8192, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.system_prompt = system_prompt

    def act(self, obs, history) -> dict:
        messages = self._build_messages(obs, history)
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(input_ids, max_new_tokens=400, do_sample=False, temperature=0.0)
        response = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        action = parse_action(response)
        return action.model_dump() if action else {"action_type": "submit_diagnosis", "diagnosis": "ambiguous", "confidence": 0.5, "secondary_actions": []}
```

### `src/ci_triage_env/training/baselines/trained.py`

```python
class TrainedPolicy(ZeroShotPolicy):
    """Our trained model. Same interface as zero-shot, loads from our checkpoint."""
    def __init__(self, checkpoint_path: str, system_prompt: str):
        super().__init__(checkpoint_path, system_prompt, name="trained_qwen3.5_4b")
```

### `src/ci_triage_env/training/eval.py`

```python
class Evaluator:
    BASELINES = ["random", "heuristic", "qwen3.5_4b_zero_shot", "qwen3.5_9b_zero_shot", "trained"]

    def __init__(self, eval_set_path: str = "data_artifacts/scenarios/held_out/",
                 env_url: str = "http://localhost:8000",
                 trained_checkpoint: str = "checkpoints/grpo_full/"):
        self.env = EnvClient(env_url)
        self.eval_scenarios = list(Path(eval_set_path).glob("*.json"))
        self.trained_checkpoint = trained_checkpoint

    def run_all(self, seeds: list[int] = [1, 2, 3]) -> pd.DataFrame:
        rows = []
        for baseline_name in self.BASELINES:
            policy = self._build(baseline_name)
            for scenario_path in self.eval_scenarios:
                scenario_id = scenario_path.stem
                for seed in seeds:
                    row = self._run_one(policy, scenario_id, seed)
                    rows.append(row)
        return pd.DataFrame(rows)

    def _build(self, name: str):
        if name == "random":
            return RandomPolicy()
        if name == "heuristic":
            return HeuristicPolicy()
        if name == "qwen3.5_4b_zero_shot":
            return ZeroShotPolicy("Qwen/Qwen3.5-4B", SYSTEM_PROMPT)
        if name == "qwen3.5_9b_zero_shot":
            return ZeroShotPolicy("Qwen/Qwen3.5-9B", SYSTEM_PROMPT)
        if name == "trained":
            return TrainedPolicy(self.trained_checkpoint, SYSTEM_PROMPT)

    def _run_one(self, policy, scenario_id: str, seed: int) -> dict:
        obs = self.env.reset(scenario_id=scenario_id, seed_override=seed)
        episode_id = obs.episode_id
        history = []
        for turn in range(12):
            action = policy.act(obs, history)
            try:
                obs = self.env.step(episode_id, action)
            except Exception:
                break
            history.append(action)
            if obs.is_terminal:
                # Counterfactual probe is deferred to v2 — env never populates probe_question in v1.
                break

        trace = self.env.get_trace(episode_id)
        scenario = load_scenario(scenario_id)
        reward = compute_reward(trace, scenario)

        return {
            "baseline": policy.name,
            "scenario_id": scenario_id,
            "family": scenario.family,
            "difficulty": scenario.metadata.difficulty,
            "seed": seed,
            "total_reward": reward.total,
            "format_gate": reward.format_gate,
            "diagnosis_correct": (
                trace.episode.final_action.diagnosis == scenario.ground_truth.label
                if trace.episode.final_action else False
            ),
            "predicted_diagnosis": trace.episode.final_action.diagnosis if trace.episode.final_action else None,
            "true_diagnosis": scenario.ground_truth.label,
            "action_quality": reward.components["action_quality"].raw,
            "tool_call_count": sum(1 for r in trace.episode.history if isinstance(r.action, ToolCall)),
            "total_cost": sum(r.cost_charged for r in trace.episode.history),
            "confidence": trace.episode.final_action.confidence if trace.episode.final_action else 0,
            "is_ambiguous_scenario": scenario.ground_truth.is_ambiguous,
            "brier_on_ambiguous": (
                (trace.episode.final_action.confidence - scenario.ground_truth.confidence_target) ** 2
                if scenario.ground_truth.is_ambiguous and trace.episode.final_action else None
            ),
        }

def main():
    evaluator = Evaluator()
    df = evaluator.run_all()
    out = Path("data_artifacts/results/")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eval.csv", index=False)
    print(df.groupby("baseline").agg({
        "diagnosis_correct": "mean",
        "action_quality": "mean",
        "tool_call_count": "mean",
        "total_cost": "mean",
        "total_reward": "mean",
    }))

    from .plotting import plot_all_eval_metrics
    plot_all_eval_metrics(df, out / "plots/")
```

### `src/ci_triage_env/training/plotting.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_eval_metrics(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Diagnosis accuracy by baseline
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("baseline")["diagnosis_correct"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Diagnosis Accuracy")
    ax.set_xlabel("Baseline")
    ax.set_title("Diagnosis Accuracy by Baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnosis_accuracy.png", dpi=120)

    # 2. Mean total reward
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("baseline")["total_reward"].agg(["mean", "std"]).plot(kind="bar", y="mean", yerr="std", ax=ax)
    ax.set_ylabel("Mean Total Reward")
    fig.savefig(output_dir / "total_reward.png", dpi=120)

    # 3. Per-family accuracy heatmap
    pivot = df.pivot_table(index="baseline", columns="family", values="diagnosis_correct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Greens", ax=ax)
    fig.savefig(output_dir / "per_family_accuracy.png", dpi=120)

    # 4. Reliability (calibration) on ambiguous
    amb = df[df["is_ambiguous_scenario"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for baseline in amb["baseline"].unique():
        sub = amb[amb["baseline"] == baseline]
        ax.scatter(sub["confidence"], 1 - sub["brier_on_ambiguous"], label=baseline, alpha=0.5)
    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Calibration score")
    ax.legend()
    fig.savefig(output_dir / "calibration_ambiguous.png", dpi=120)

    # 5. Tool-call cost distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="baseline", y="total_cost", ax=ax)
    fig.savefig(output_dir / "cost_distribution.png", dpi=120)

    # 6. Counterfactual prediction accuracy — deferred to v2.
    # When v2 ships, uncomment:
    # cf = df[df["counterfactual_correct"].notna()]
    # if len(cf) > 0:
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     cf.groupby("baseline")["counterfactual_correct"].mean().plot(kind="bar", ax=ax)
    #     ax.set_ylabel("Counterfactual Prediction Accuracy")
    #     fig.savefig(output_dir / "counterfactual_accuracy.png", dpi=120)
```

### `src/ci_triage_env/training/readme_table.py`

```python
def generate_results_table(df: pd.DataFrame) -> str:
    """Markdown table for README's Results section."""
    summary = df.groupby("baseline").agg(
        diagnosis_acc=("diagnosis_correct", "mean"),
        action_qual=("action_quality", "mean"),
        cost=("total_cost", "mean"),
        steps=("tool_call_count", "mean"),
        reward=("total_reward", "mean"),
    )
    return summary.to_markdown(floatfmt=".3f")
```

---

## Implementation notes

- **Random and heuristic baselines run on CPU.** Zero-shot and trained need GPU.
- **9B zero-shot baseline.** Needs more VRAM. If a single A100 can't hold both 4B-trained and 9B-zero-shot at once, run sequentially with model unload between.
- **Eval set composition.** Held-out has all the ambiguous scenarios + a fraction of the unambiguous. Do not run eval on val or train splits — those have leaked into training-time decisions.
- **3 seeds.** Random and ZeroShot have stochasticity. Heuristic and trained (with `do_sample=False`) are deterministic; running 3 seeds is wasted compute for them — collapse to 1.
- **Total eval cost.** ~150 scenarios × 5 baselines = 750 episode runs. Each ~5-8k tokens. With batched zero-shot evaluation, ~3 hours wall on A100. ~$3-5.

---

## Tests required (`tests/training/test_eval.py`)

```python
def test_random_policy_emits_valid_action():
    p = RandomPolicy(seed=42)
    action = p.act(mock_observation(), [])
    # Should be ToolCall or TerminalAction shape
    assert "tool_name" in action or "action_type" in action

def test_heuristic_policy_completes_investigation_then_diagnoses():
    p = HeuristicPolicy()
    history = []
    for _ in range(4):
        action = p.act(mock_observation(), history)
        history.append({"output": "ok"})
    final = p.act(mock_observation(), history)
    assert final["action_type"] == "submit_diagnosis"

def test_evaluator_run_one_returns_row(monkeypatch):
    """Mock env client; verify _run_one returns a dict with all expected fields."""

def test_results_table_markdown(tmp_path):
    """generate_results_table produces valid markdown."""

def test_plotting_writes_pngs(tmp_path):
    """plot_all_eval_metrics writes ≥ 5 PNGs."""
```

---

## Smoke test (manual)

```bash
# Smaller eval to verify pipeline (3 baselines × 10 scenarios × 1 seed)
python -c "
from ci_triage_env.training.eval import Evaluator
import pandas as pd
e = Evaluator()
e.eval_scenarios = e.eval_scenarios[:10]
e.BASELINES = ['random', 'heuristic']   # skip LLM ones for smoke
df = e.run_all(seeds=[1])
print(df.groupby('baseline')['diagnosis_correct'].mean())
"

# Check plots
ls data_artifacts/results/plots/
```

---

## Full eval (onsite, after training)

```bash
python -m ci_triage_env.training.eval --output data_artifacts/results/

# Verify results
cat data_artifacts/results/eval.csv | head
ls data_artifacts/results/plots/

# Generate README table
python -c "
from ci_triage_env.training.readme_table import generate_results_table
import pandas as pd
df = pd.read_csv('data_artifacts/results/eval.csv')
print(generate_results_table(df))
" > data_artifacts/results/results_table.md
```

---

## Open questions

1. **What if 9B zero-shot won't fit alongside trained-4B?** Run sequentially: load 9B, run all 9B evals, save results, unload, load 4B trained, run, save. Doubles eval wall time but stays within VRAM.
2. **Do we need temperature sampling for trained model at eval?** No — use greedy (`do_sample=False`) for stable comparison. Sampling only matters during training.
3. **Counterfactual probe** — deferred to v2; not relevant in v1 eval.

---

## What's NOT in this phase

- Reward layer ablations (C6)
- Final README population
