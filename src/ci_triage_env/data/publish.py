"""HuggingFace dataset publisher for the CI-Triage scenario corpus.

Imports ``huggingface_hub`` and ``datasets`` lazily so the rest of the package
remains usable without those dependencies installed.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_dataset_readme(scenarios_dir: Path) -> str:
    """Auto-generated README for the HF dataset, accurate to the actual counts."""

    def _count(split: str) -> int:
        d = scenarios_dir / split
        return len(list(d.glob("*.json"))) if d.exists() else 0

    train_n = _count("train")
    val_n = _count("val")
    held_out_n = _count("held_out")

    return f"""---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
tags:
  - ci-triage
  - openenv
  - rl-environment
  - failure-diagnosis
---

# CI-Triage Scenarios

A corpus of CI-failure scenarios for the **CI-Triage-Env** OpenEnv RL environment.
Generated from public OSS CI logs (anonymized) and open-license datasets
(DeFlaker, iDFlakies, FlakeFlagger, LogHub).

## Splits

| Split | Count | Notes |
|-------|-------|-------|
| train | {train_n} | Unambiguous families only |
| val | {val_n} | Unambiguous families only |
| held_out | {held_out_n} | Includes ALL ambiguous instances (calibration probe) |

## Schema

Each row contains:

- `scenario_id` (string): unique identifier (`<family>-s<seed>-<hash>`)
- `family` (string): one of `real_bug`, `race_flake`, `timing_flake`,
  `infra_network`, `infra_resource`, `dependency_drift`, `ambiguous`
- `scenario_json` (string): full `Scenario` JSON, validates against
  `ci_triage_env.schemas.scenario.Scenario`
- `difficulty` (string): `easy` / `medium` / `hard`

## Failure Families

| Family | Description |
|--------|-------------|
| `real_bug` | A genuine code defect introduced by a recent commit |
| `race_flake` | Non-deterministic failure from a data race / goroutine conflict |
| `timing_flake` | Intermittent timeout under CI scheduler load |
| `infra_network` | DNS / TLS / connectivity failure on the CI node |
| `infra_resource` | OOM-kill, disk full, or file-descriptor exhaustion |
| `dependency_drift` | Breaking change from a dependency version bump |
| `ambiguous` | Multiple plausible causes — correct response is low confidence |

## License

CC-BY-4.0. Generated from public OSS CI logs (anonymized) and open-license
datasets (DeFlaker, iDFlakies, FlakeFlagger, LogHub).

## Citation

If you use this corpus, please cite the originating datasets and the
Meta PyTorch OpenEnv hackathon submission (CI-Triage-Env).
"""


def publish_to_hf(
    scenarios_dir: Path,
    dataset_name: str,
    token: str | None = None,
) -> None:
    """Upload the generated corpus to the HuggingFace dataset hub.

    Args:
        scenarios_dir: Directory produced by ``CorpusBuilder.build()``,
            containing ``train/``, ``val/``, and ``held_out/`` subdirectories.
        dataset_name: HF repo id, e.g. ``"your-org/ci-triage-scenarios"``.
        token: HF API token. Falls back to ``HF_TOKEN`` env var if ``None``.
    """
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id=dataset_name, repo_type="dataset", exist_ok=True, token=token)

    splits: dict[str, Dataset] = {}
    for split_name in ("train", "val", "held_out"):
        split_dir = scenarios_dir / split_name
        if not split_dir.exists():
            continue
        records = []
        for path in sorted(split_dir.glob("*.json")):
            scenario_dict = json.loads(path.read_text())
            records.append(
                {
                    "scenario_id": scenario_dict["scenario_id"],
                    "family": scenario_dict["family"],
                    "scenario_json": json.dumps(scenario_dict),
                    "difficulty": scenario_dict["metadata"]["difficulty"],
                }
            )
        if records:
            splits[split_name] = Dataset.from_list(records)

    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(dataset_name, token=token)

    readme = generate_dataset_readme(scenarios_dir)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
        token=token,
    )
    print(
        f"Published {sum(len(ds) for ds in splits.values())} scenarios to "
        f"https://huggingface.co/datasets/{dataset_name}"
    )
