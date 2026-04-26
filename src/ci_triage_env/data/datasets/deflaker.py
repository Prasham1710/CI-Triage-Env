"""DeFlaker (Bell et al., FSE 2018) loader.

Source paper: `DeFlaker: Automatically Detecting Flaky Tests
<https://www.jonbell.net/icse18-deflaker.pdf>`_.

DeFlaker labels test-failure events on 26 OSS Java projects as either ``flaky``
(test fails on a commit that didn't change code reachable from the test) or
``real`` (the test fails because the code under test changed). The published
artifact ships as a CSV-ish dump per project; the canonical mirror has rotted
in the past, so we read whatever local file the user points us at.

Expected on-disk format under ``data_path/`` is **one CSV** with columns
``project, test, label, commit_sha, log`` (``label`` ∈ {``flaky``, ``real``}).
This matches the reduced shape the published artifact uses after extracting
its per-project ``failures.csv`` files; concatenate them if you start from the
raw release. Document any deviation in the loader's caller.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator

from ci_triage_env.data.datasets._base import DatasetLoader, FailureRecord

_REQUIRED_COLUMNS = {"project", "test", "label", "commit_sha"}


class DeFlakerLoader(DatasetLoader):
    name = "deflaker"
    env_var = "DEFLAKER_DATA_PATH"
    download_instructions = (
        "Visit https://github.com/jonbell-/deflaker (or the FSE 2018 paper's "
        "supplementary materials) and download the per-project failures.csv "
        "files. Concatenate into one CSV with columns "
        "(project, test, label, commit_sha, log) and point "
        "$DEFLAKER_DATA_PATH at that file."
    )

    def fetch(self) -> Iterator[FailureRecord]:
        path = self._require_data_path()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"deflaker CSV at {path} is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                project = (row.get("project") or "").strip()
                test = (row.get("test") or "").strip()
                commit_sha = (row.get("commit_sha") or "").strip()
                label = (row.get("label") or "").strip().lower() or None
                if not project or not test or not commit_sha:
                    continue
                yield FailureRecord(
                    record_id=f"deflaker-{commit_sha[:12]}-{test}",
                    source_dataset="deflaker",
                    project=project,
                    test_name=test,
                    failure_type_label=label,
                    log_text=row.get("log", "") or "",
                    metadata={"commit_sha": commit_sha},
                )
