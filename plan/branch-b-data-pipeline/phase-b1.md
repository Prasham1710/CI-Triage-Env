# Phase B1 — Public Dataset Ingest

**Owner:** Branch B.
**Prerequisite:** `phase-0-complete` on `main`.
**Estimated time:** 3–4 hours.
**Parallel with:** B2 (you can start either first).

---

## Outcome

Loaders for four public datasets, all producing a normalized internal `FailureRecord` shape that downstream phases (B3 clustering, B4 generators) consume. By end of phase:

1. Loaders for DeFlaker, iDFlakies, FlakeFlagger, LogHub working.
2. Each loader has a CLI subcommand: `python -m ci_triage_env.data.cli load <dataset>`.
3. Cached locally in `data_artifacts/datasets_cache/<dataset>/`.
4. All loaded records validate against `FailureRecord` schema.
5. All B1 unit tests pass.

---

## Files to create

### `src/ci_triage_env/data/datasets/_base.py`

```python
class FailureRecord(BaseModel):
    """Normalized representation of one failure across all source datasets."""
    record_id: str
    source_dataset: Literal["deflaker", "idflakies", "flakeflagger", "loghub", "github_actions"]
    project: str                      # e.g. "kubernetes/kubernetes"
    test_name: str | None
    failure_type_label: str | None    # source-dataset-specific label, e.g. "flaky", "real_bug"
    log_text: str                     # raw log content
    metadata: dict                    # source-specific extras

class DatasetLoader(ABC):
    name: str
    cache_dir: Path

    @abstractmethod
    def fetch(self) -> Iterable[FailureRecord]:
        """Yield FailureRecord for each item in the dataset."""

    @abstractmethod
    def info(self) -> dict:
        """Return summary statistics: count, label distribution, etc."""
```

### `src/ci_triage_env/data/datasets/deflaker.py`

DeFlaker (Bell et al., FSE 2018): commits labeled flaky vs. real-bug across 26 OSS Java projects. Source: published artifact link in the paper (verify exact URL in their GitHub).

```python
class DeFlakerLoader(DatasetLoader):
    name = "deflaker"
    DOWNLOAD_URL = "https://github.com/.../deflaker_artifact.tar.gz"  # verify exact URL

    def fetch(self):
        path = self._download_and_extract()
        for entry in self._iter_entries(path):
            yield FailureRecord(
                record_id=f"deflaker-{entry['commit_sha']}-{entry['test']}",
                source_dataset="deflaker",
                project=entry["project"],
                test_name=entry["test"],
                failure_type_label=entry["label"],   # "flaky" or "real"
                log_text=entry.get("log", ""),
                metadata={"commit_sha": entry["commit_sha"]},
            )
```

### `src/ci_triage_env/data/datasets/idflakies.py`

iDFlakies (Lam et al., ICSE 2019): order-dependent flaky tests. https://github.com/idflakies (verify; possibly under TestingResearchIllinois org).

```python
class IDFlakiesLoader(DatasetLoader):
    name = "idflakies"
    DOWNLOAD_URL = "..."  # verify

    def fetch(self):
        # Format: CSV with columns (project, test_name, type:OD|NOD|...)
        ...
```

Maps `OD` (order-dependent) to `race_flake` later in clustering; `NOD` (non-order-dependent) flaky to `timing_flake`.

### `src/ci_triage_env/data/datasets/flakeflagger.py`

FlakeFlagger (Alshammari et al., ICSE 2021): https://github.com/AlshammariA/FlakeFlagger. ~800 flakes with rich features (timing, IO, threading classifications).

```python
class FlakeFlaggerLoader(DatasetLoader):
    name = "flakeflagger"
    DOWNLOAD_URL = "https://github.com/AlshammariA/FlakeFlagger/raw/master/data/..."

    def fetch(self):
        # Format: CSV with timing/IO/threading flag columns
        ...
```

### `src/ci_triage_env/data/datasets/loghub.py`

LogHub (Zhu et al., ISSRE 2019): https://github.com/logpai/loghub. 19 system log datasets. We use a subset relevant to CI/build/test failures.

```python
class LogHubLoader(DatasetLoader):
    name = "loghub"
    DATASETS_TO_USE = ["HDFS_v1", "Hadoop", "Spark", "Linux", "BGL"]
    # GitHub: https://github.com/logpai/loghub

    def fetch(self):
        for dataset_name in self.DATASETS_TO_USE:
            # Each subset is structured logs + anomaly labels
            ...
            yield FailureRecord(
                record_id=...,
                source_dataset="loghub",
                project=dataset_name,
                test_name=None,
                failure_type_label="anomaly" if has_anomaly_label else "normal",
                log_text=...,
                metadata={"sub_dataset": dataset_name},
            )
```

LogHub provides log patterns relevant to **infra-resource** family especially (OOM patterns from Linux/BGL, memory pressure from Hadoop).

### `src/ci_triage_env/data/cli.py`

```python
import argparse
from .datasets import deflaker, idflakies, flakeflagger, loghub

def cmd_load(args):
    loader_cls = {
        "deflaker": deflaker.DeFlakerLoader,
        "idflakies": idflakies.IDFlakiesLoader,
        "flakeflagger": flakeflagger.FlakeFlaggerLoader,
        "loghub": loghub.LogHubLoader,
    }[args.dataset]
    loader = loader_cls()
    count = 0
    for record in loader.fetch():
        # Cache to disk
        cache_path = loader.cache_dir / f"{record.record_id}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(record.model_dump_json())
        count += 1
    print(f"Loaded {count} records into {loader.cache_dir}")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    p_load = sub.add_parser("load")
    p_load.add_argument("dataset", choices=["deflaker", "idflakies", "flakeflagger", "loghub"])
    p_load.set_defaults(func=cmd_load)
    # B5 will add `generate` and `publish-hf` subcommands
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
```

### `src/ci_triage_env/data/datasets/cache.py`

```python
def cache_dir_for(dataset_name: str) -> Path:
    return Path(os.environ.get("CI_TRIAGE_DATA_CACHE", "data_artifacts/datasets_cache")) / dataset_name

def is_cached(record_id: str, dataset_name: str) -> bool:
    return (cache_dir_for(dataset_name) / f"{record_id}.json").exists()

def load_cached(dataset_name: str) -> Iterable[FailureRecord]:
    for path in cache_dir_for(dataset_name).glob("*.json"):
        yield FailureRecord.model_validate_json(path.read_text())
```

---

## Implementation notes

- **Verify exact download URLs.** This doc lists guesses. Each loader's first task is to find the canonical artifact URL by reading the paper or the GitHub repo. If a URL has rotted (papers from 2018–2021 often do), document the alternative source in code comments.
- **Some artifacts require manual download.** DeFlaker's data may require accepting a license click-through or filling out a form. If so, the loader's `fetch` raises a clear `RuntimeError` with instructions on how to obtain the file manually, and where to place it.
- **Don't load everything into memory.** Use generator-based `fetch`. Some logs are huge.
- **Per-dataset rate limits / sizes.**
  - DeFlaker: ~5000 records, < 100MB total
  - iDFlakies: ~800 records, < 50MB
  - FlakeFlagger: ~800 records, < 30MB
  - LogHub (subset): variable, can be GB-scale; subsample to 100 records per dataset
- **Caching is content-addressable.** Same `record_id` → same path. Re-running is safe.
- **License check.** Each dataset's license matters for redistribution. **You don't need to redistribute the raw data** — only the *generated scenarios* (Phase B5). Raw data stays in local cache, in `data_artifacts/` which is gitignored.

---

## Tests required (`tests/data/test_dataset_loaders.py`)

```python
def test_failure_record_round_trip():
    """FailureRecord serializes and deserializes via Pydantic."""

def test_each_loader_has_correct_name():
    """Loader.name matches expected enum value."""

# For each loader (4 total):
def test_<loader>_fetches_at_least_one_record():
    """With a fixture file simulating the dataset structure, loader produces ≥ 1 record."""

def test_<loader>_records_validate_against_schema():
    """All produced records pass FailureRecord validation."""

def test_<loader>_caches_to_disk():
    """After fetch, records are written to cache_dir."""

def test_<loader>_repeat_fetch_uses_cache():
    """Calling fetch twice without --force doesn't re-download."""

# Fixtures
def test_fixtures_exist():
    """Each fixture file under tests/data/fixtures/<dataset>/ exists."""
```

For each dataset, ship a small fixture (e.g., 3-record CSV slice) under `tests/data/fixtures/<dataset>/` so tests run without internet.

---

## Smoke test (manual)

```bash
# Load each dataset (requires internet)
python -m ci_triage_env.data.cli load deflaker
python -m ci_triage_env.data.cli load idflakies
python -m ci_triage_env.data.cli load flakeflagger
python -m ci_triage_env.data.cli load loghub

# Verify counts
ls data_artifacts/datasets_cache/deflaker/ | wc -l    # expect 100s
ls data_artifacts/datasets_cache/loghub/ | wc -l      # expect ~500
```

---

## Open questions

1. **DeFlaker/iDFlakies download URLs.** Verify before starting. If artifacts are gone, find replacements or downgrade scope (drop that dataset, document in README).
2. **LogHub subset selection.** The full LogHub is large. Stick to the listed 5 subsets or expand if useful. If you expand, document why.
3. **Should we extract logs from FlakeFlagger feature CSVs or also fetch the raw test runs?** Feature CSVs are sufficient for clustering — features tell us "this is a timing flake" without needing raw logs. Raw logs come from B2.

---

## What's NOT in this phase

- Mining real GitHub Actions logs (B2)
- Clustering into the 7-family taxonomy (B3)
- Scenario generation (B4)
