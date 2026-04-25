# Phase B5 — Scenario Instantiation + HF Publish

**Owner:** Branch B.
**Prerequisite:** B4 merged.
**Estimated time:** 2–3 hours.

---

## Outcome

Mass-generate the scenario corpus and publish to HF dataset hub. By end of phase:

1. `python -m ci_triage_env.data.cli generate --total 200 --split 70/15/15` produces 200 scenarios distributed across 7 families.
2. Each scenario JSON written to `data_artifacts/scenarios/<split>/<scenario_id>.json`.
3. Train/val/held-out split is deterministic given seed.
4. `python -m ci_triage_env.data.cli publish-hf --dataset-name <user>/ci-triage-scenarios` uploads to HF dataset hub.
5. The HF dataset has README, license note, and split metadata.
6. **All branches can now consume scenarios from the HF dataset URL.**
7. All B5 tests pass.

---

## Files to create

### `src/ci_triage_env/data/instantiation.py`

```python
class CorpusBuilder:
    DEFAULT_DISTRIBUTION = {
        "real_bug": 0.20,
        "race_flake": 0.15,
        "timing_flake": 0.10,
        "infra_network": 0.10,
        "infra_resource": 0.15,
        "dependency_drift": 0.10,
        "ambiguous": 0.20,   # over-represented for the calibration probe
    }

    def __init__(self, total: int = 200, distribution: dict | None = None,
                 split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
                 base_seed: int = 100000):
        self.total = total
        self.distribution = distribution or self.DEFAULT_DISTRIBUTION
        self.split_ratios = split_ratios
        self.base_seed = base_seed

    def build(self, output_dir: Path) -> dict:
        """Generates the full corpus; returns summary dict."""
        from .generators import GENERATOR_REGISTRY

        per_family = self._compute_per_family()
        all_scenarios: list[Scenario] = []
        for family, count in per_family.items():
            generator = GENERATOR_REGISTRY[family]()
            for i in range(count):
                seed = self.base_seed + sum(per_family[f] for f in list(per_family)[:list(per_family).index(family)]) + i
                scenario = generator.generate(seed=seed)
                all_scenarios.append(scenario)

        # Deterministic split based on scenario_id hash
        train, val, held_out = self._split(all_scenarios)
        for split_name, split in [("train", train), ("val", val), ("held_out", held_out)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for scenario in split:
                (split_dir / f"{scenario.scenario_id}.json").write_text(scenario.model_dump_json(indent=2))

        return {
            "total": len(all_scenarios),
            "train": len(train), "val": len(val), "held_out": len(held_out),
            "by_family": {f: count for f, count in per_family.items()},
        }

    def _compute_per_family(self) -> dict[str, int]:
        return {f: max(1, int(self.total * w)) for f, w in self.distribution.items()}

    def _split(self, scenarios: list[Scenario]) -> tuple[list, list, list]:
        rng = random.Random(self.base_seed)
        rng.shuffle(scenarios)
        n = len(scenarios)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])
        # Held-out: ALL ambiguous scenarios go here regardless of split rng
        held_out = [s for s in scenarios if s.family == "ambiguous"]
        rest = [s for s in scenarios if s.family != "ambiguous"]
        train = rest[:n_train]
        val = rest[n_train:n_train + n_val]
        held_out += rest[n_train + n_val:]
        return train, val, held_out
```

> **Critical detail:** ambiguous scenarios all live in held-out. They're the calibration probe set. Train and val have no ambiguous instances — the model trains on the 6 unambiguous families, then evaluated on its calibration when the held-out injects ambiguity.

### `src/ci_triage_env/data/publish.py`

```python
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, DatasetDict

def publish_to_hf(scenarios_dir: Path, dataset_name: str, token: str | None = None):
    """Upload corpus to HF dataset hub."""
    api = HfApi(token=token)
    create_repo(repo_id=dataset_name, repo_type="dataset", exist_ok=True, token=token)

    splits = {}
    for split_name in ["train", "val", "held_out"]:
        split_dir = scenarios_dir / split_name
        records = []
        for path in split_dir.glob("*.json"):
            scenario_dict = json.loads(path.read_text())
            records.append({
                "scenario_id": scenario_dict["scenario_id"],
                "family": scenario_dict["family"],
                "scenario_json": json.dumps(scenario_dict),  # full scenario as string
                "difficulty": scenario_dict["metadata"]["difficulty"],
            })
        splits[split_name] = Dataset.from_list(records)
    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(dataset_name, token=token)

    # Upload README
    readme = generate_dataset_readme(scenarios_dir)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
        token=token,
    )

def generate_dataset_readme(scenarios_dir: Path) -> str:
    """Auto-generated README for the HF dataset."""
    return f"""# CI-Triage Scenarios

A corpus of CI-failure scenarios for the CI-Triage-Env OpenEnv environment.

## Splits
- train: {len(list((scenarios_dir / 'train').glob('*.json')))} scenarios
- val: {len(list((scenarios_dir / 'val').glob('*.json')))} scenarios
- held_out: {len(list((scenarios_dir / 'held_out').glob('*.json')))} scenarios (includes ALL ambiguous instances)

## Schema
Each row contains:
- `scenario_id` (string): unique identifier
- `family` (string): one of real_bug, race_flake, timing_flake, infra_network, infra_resource, dependency_drift, ambiguous
- `scenario_json` (string): full Scenario JSON, validates against ci_triage_env.schemas.scenario.Scenario
- `difficulty` (string): easy / medium / hard

## License
CC-BY-4.0. Generated from public OSS CI logs (anonymized) and open-license datasets (DeFlaker, iDFlakies, FlakeFlagger, LogHub).

## Citation
If you use this corpus, cite the originating datasets and the Meta PyTorch OpenEnv hackathon submission.
"""
```

### Modify `src/ci_triage_env/data/cli.py`

Add `generate` and `publish-hf` subcommands:

```python
def cmd_generate(args):
    builder = CorpusBuilder(
        total=args.total,
        split_ratios=tuple(map(float, args.split.split("/"))) if "/" in args.split else (0.70, 0.15, 0.15),
        base_seed=args.seed,
    )
    summary = builder.build(Path(args.output_dir))
    print(json.dumps(summary, indent=2))

def cmd_publish_hf(args):
    publish_to_hf(
        scenarios_dir=Path(args.scenarios_dir),
        dataset_name=args.dataset_name,
        token=os.environ.get("HF_TOKEN"),
    )
```

CLI options:

```
python -m ci_triage_env.data.cli generate --total 200 --split 70/15/15 --seed 100000 --output-dir data_artifacts/scenarios/
python -m ci_triage_env.data.cli publish-hf --scenarios-dir data_artifacts/scenarios/ --dataset-name <user>/ci-triage-scenarios
```

### `src/ci_triage_env/data/annotations/informative_tools.py`

(Final pass to enrich annotations beyond what generators produce by default.)

```python
def enrich_annotations(scenario: Scenario) -> Scenario:
    """Re-derive informative_tools and minimal_evidence_set from the scenario's
    actual tool outputs by checking which outputs would let a human reach the
    correct diagnosis. Used as a sanity check on generator output."""
    ...
```

This is a defensive pass; if generators are correct, this is a no-op. If not, it surfaces the bug.

---

## Implementation notes

- **Distribution choice.** Ambiguous is over-represented (20%) because it's a small absolute number but the calibration probe needs many examples. Train sees zero ambiguous (they're all held-out); the model has never seen the ambiguous family during training, so eval-time is a pure generalization test for calibration.
- **Splitting determinism.** Use `base_seed` + `scenario_id` hash to assign splits if you want stronger determinism. Above implementation shuffles by RNG seeded from `base_seed` — equivalent.
- **HF dataset structure.** Single string column for the full Scenario JSON simplifies uploading. Branch A's loader parses it. Alternative is column-per-field (richer schema in HF) but more brittle. Recommend single JSON string for v1.
- **README generation.** Auto-generated per build keeps it accurate. Hand-edit only the static parts (license, citations).

---

## Tests required (`tests/data/test_instantiation.py`)

```python
def test_corpus_builder_produces_target_total(tmp_path):
    """Build corpus of size 50 — output has ≥ 49 (rounding tolerance)."""

def test_split_ratios_respected(tmp_path):
    """For 100 scenarios with (0.7, 0.15, 0.15), splits have ~70/15/15 within ±2."""

def test_all_ambiguous_in_held_out(tmp_path):
    """No ambiguous scenarios appear in train or val splits."""

def test_split_is_deterministic(tmp_path):
    """Same base_seed → identical scenario_id → split mapping."""

def test_per_family_count_respects_distribution(tmp_path):
    """Each family appears ≥ 1 in the output."""

def test_corpus_builder_outputs_validate(tmp_path):
    """All generated JSONs in output_dir validate against Scenario schema."""

def test_publish_hf_dry_run(monkeypatch, tmp_path):
    """Mock HF API; verify push_to_hub called with correct dataset_dict structure."""

def test_dataset_readme_includes_counts(tmp_path):
    """generate_dataset_readme(scenarios_dir) includes the actual split counts."""
```

---

## Smoke test (manual)

```bash
# Generate corpus
python -m ci_triage_env.data.cli generate --total 200 --output-dir data_artifacts/scenarios/

# Verify
ls data_artifacts/scenarios/train/ | wc -l
ls data_artifacts/scenarios/val/ | wc -l
ls data_artifacts/scenarios/held_out/ | wc -l

# Publish (only if HF_TOKEN set)
export HF_TOKEN=hf_...
python -m ci_triage_env.data.cli publish-hf \
  --scenarios-dir data_artifacts/scenarios/ \
  --dataset-name <user>/ci-triage-scenarios

# Verify on HF
echo "Open https://huggingface.co/datasets/<user>/ci-triage-scenarios"
```

---

## Realism check before publishing

**Before pushing to HF**, run a final manual review:

1. Pick 5 random scenarios from each family (35 total).
2. For each, read the scenario as if you're an SRE seeing this for the first time.
3. Check: would you reach the ground-truth diagnosis given the tool outputs?
4. Check: are the informative_tools actually informative?
5. Check: for ambiguous, is the case genuinely ambiguous or could a careful human pick the right answer?

If any scenario fails, fix the generator and re-build.

---

## Gate-1 entry

After B5 lands and CI passes, Branch B is fully done for Gate-1. The HF dataset URL becomes the canonical scenario source for both Branch A's runtime and Branch C's training/eval.

Update `plan/INSTRUCTION-MANUAL.md` with the actual dataset URL.

---

## Open questions

1. **Dataset name.** Will the team use a personal HF account or create a team account? Recommend team account (e.g., `ci-triage-team/ci-triage-scenarios`) so the URL is stable across team members.
2. **Versioning.** If you find a bug after Gate-1 and regenerate, bump dataset version (HF supports tags). Don't overwrite v1 silently.
3. **Total count.** Plan calls for 200; aim for 300 if there's spare time. Stop at 200 if Gate-1 is approaching.

---

## What's NOT in this phase

Anything related to training, rewards, or env runtime. Branch B is done after this phase.
