"""Cache layout for B1 dataset loaders.

One subdir per dataset under ``CI_TRIAGE_DATA_CACHE`` (default
``data_artifacts/datasets_cache``); each ``FailureRecord`` is written as
``<record_id>.json``. Re-loading is content-addressable: same ``record_id``
overwrites the same file, so re-running ``cli load`` is idempotent.

The cache dir is gitignored (see ``.gitignore`` for ``data_artifacts/*``);
only the *generated scenarios* (Phase B5) are committed / published.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

DEFAULT_CACHE_ROOT = Path("data_artifacts/datasets_cache")


def cache_root() -> Path:
    return Path(os.environ.get("CI_TRIAGE_DATA_CACHE", str(DEFAULT_CACHE_ROOT)))


def cache_dir_for(dataset_name: str) -> Path:
    return cache_root() / dataset_name


def is_cached(record_id: str, dataset_name: str) -> bool:
    return (cache_dir_for(dataset_name) / f"{record_id}.json").exists()


def load_cached(dataset_name: str) -> Iterator:
    """Yield cached ``FailureRecord``s for a dataset, if any."""
    from ci_triage_env.data.datasets._base import FailureRecord  # circular guard

    target = cache_dir_for(dataset_name)
    if not target.exists():
        return
    for path in sorted(target.glob("*.json")):
        yield FailureRecord.model_validate_json(path.read_text())


def load_all_cached() -> list:
    """Return all cached ``FailureRecord``s across every dataset sub-directory."""
    from ci_triage_env.data.datasets._base import FailureRecord  # circular guard

    root = cache_root()
    records = []
    if not root.exists():
        return records
    for json_path in sorted(root.rglob("*.json")):
        try:
            records.append(FailureRecord.model_validate_json(json_path.read_text()))
        except Exception:
            pass
    return records
