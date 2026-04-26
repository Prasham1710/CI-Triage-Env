import json
from pathlib import Path

from ci_triage_env.schemas.scenario import Scenario

DEFAULT_SCENARIO_DIR = Path("data_artifacts/scenarios")


def load_from_disk(path: Path) -> dict[str, Scenario]:
    """Load all *.json files under `path` as Scenario objects, keyed by scenario_id."""
    out: dict[str, Scenario] = {}
    for fp in sorted(path.glob("*.json")):
        scenario = Scenario.model_validate_json(fp.read_text())
        out[scenario.scenario_id] = scenario
    return out


def load_from_hf(dataset_name: str) -> dict[str, Scenario]:
    """Load all rows of an HF dataset as Scenario objects, keyed by scenario_id."""
    from datasets import load_dataset

    out: dict[str, Scenario] = {}
    ds = load_dataset(dataset_name, split="train")
    for row in ds:
        if isinstance(row, dict) and "scenario_json" in row:
            scenario = Scenario.model_validate_json(row["scenario_json"])
        else:
            scenario = Scenario.model_validate(json.loads(json.dumps(dict(row))))
        out[scenario.scenario_id] = scenario
    return out


def load_scenarios(source: str | None) -> dict[str, Scenario]:
    """Dispatch by source prefix.

    - None / "" → load from `data_artifacts/scenarios/`.
    - "hf://<name>" → load from HF dataset `<name>`.
    - any other string → treated as a filesystem path.
    """
    if not source:
        return load_from_disk(DEFAULT_SCENARIO_DIR)
    if source.startswith("hf://"):
        return load_from_hf(source[len("hf://") :])
    return load_from_disk(Path(source))
