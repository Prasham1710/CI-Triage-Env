"""FlakeFlagger (Alshammari et al., ICSE 2021) loader.

Source: https://github.com/AlshammariA/FlakeFlagger. ~800 flaky tests labeled
with rich features (timing, IO, threading, randomness, network, concurrency).
Their published feature CSVs are sufficient for clustering — we don't need
raw test runs (B2 mines raw logs).

Expected on-disk format: a CSV with at least these columns:
``project, test_name, flake_type, timing_flag, io_flag, threading_flag``
(boolean-ish flags as ``0/1`` or ``True/False``). Extra columns are kept on
``metadata`` so future B3 features can pick them up without a schema bump.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator

from ci_triage_env.data.datasets._base import DatasetLoader, FailureRecord

_REQUIRED_COLUMNS = {"project", "test_name"}


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "t"}


class FlakeFlaggerLoader(DatasetLoader):
    name = "flakeflagger"
    env_var = "FLAKEFLAGGER_DATA_PATH"
    download_instructions = (
        "Download FlakeFlagger's published feature CSV from "
        "https://github.com/AlshammariA/FlakeFlagger (under data/ in the repo) "
        "and point $FLAKEFLAGGER_DATA_PATH at the file. Expected columns "
        "include (project, test_name, flake_type, timing_flag, io_flag, "
        "threading_flag); extras are preserved in metadata."
    )

    def fetch(self) -> Iterator[FailureRecord]:
        path = self._require_data_path()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            missing = _REQUIRED_COLUMNS - cols
            if missing:
                raise ValueError(
                    f"flakeflagger CSV at {path} is missing required columns: {sorted(missing)}"
                )
            extra_cols = cols - {"project", "test_name", "flake_type"}
            for row in reader:
                project = (row.get("project") or "").strip()
                test_name = (row.get("test_name") or "").strip()
                if not project or not test_name:
                    continue
                flake_type = (row.get("flake_type") or "").strip() or None
                metadata = {
                    "timing_flag": _truthy(row.get("timing_flag")),
                    "io_flag": _truthy(row.get("io_flag")),
                    "threading_flag": _truthy(row.get("threading_flag")),
                }
                # Preserve any extra columns verbatim so B3 can use them.
                for c in extra_cols - {"timing_flag", "io_flag", "threading_flag"}:
                    metadata[c] = row.get(c, "")
                suffix = hashlib.sha1(f"{project}|{test_name}".encode()).hexdigest()[:8]
                yield FailureRecord(
                    record_id=f"flakeflagger-{suffix}",
                    source_dataset="flakeflagger",
                    project=project,
                    test_name=test_name,
                    failure_type_label=flake_type,
                    log_text="",
                    metadata=metadata,
                )
