"""iDFlakies (Lam et al., ICSE 2019) loader.

Source: `iDFlakies: A Framework for Detecting and Partially Classifying Flaky
Tests <https://github.com/TestingResearchIllinois/idflakies>`_. Labels each
flaky test as ``OD`` (order-dependent — fails only under specific test
orderings) or ``NOD`` (non-order-dependent — fails non-deterministically even
in isolation).

B3 clustering maps:
- ``OD``  → ``race_flake`` family
- ``NOD`` → ``timing_flake`` family

Expected on-disk format: a CSV with columns ``project, test_name, type``
where ``type`` ∈ {``OD``, ``NOD``, ``OD-Brit``, ``OD-Vict`` ...} — only the
``OD``/``NOD`` prefix matters for normalization here; B3 reads the rest.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator

from ci_triage_env.data.datasets._base import DatasetLoader, FailureRecord

_REQUIRED_COLUMNS = {"project", "test_name", "type"}


class IDFlakiesLoader(DatasetLoader):
    name = "idflakies"
    env_var = "IDFLAKIES_DATA_PATH"
    download_instructions = (
        "Clone https://github.com/TestingResearchIllinois/idflakies and copy "
        "the dataset CSV (their published flaky-test list) to a local file. "
        "Point $IDFLAKIES_DATA_PATH at that CSV; expected columns: "
        "(project, test_name, type) where type ∈ {OD, NOD, ...}."
    )

    def fetch(self) -> Iterator[FailureRecord]:
        path = self._require_data_path()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"idflakies CSV at {path} is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                project = (row.get("project") or "").strip()
                test_name = (row.get("test_name") or "").strip()
                ftype = (row.get("type") or "").strip()
                if not project or not test_name or not ftype:
                    continue
                # Some rows share a (project, test_name) pair across orderings;
                # hash the type to keep record_ids unique without exposing it.
                suffix = hashlib.sha1(f"{project}|{test_name}|{ftype}".encode()).hexdigest()[:8]
                yield FailureRecord(
                    record_id=f"idflakies-{suffix}",
                    source_dataset="idflakies",
                    project=project,
                    test_name=test_name,
                    failure_type_label=ftype,
                    log_text="",  # iDFlakies dataset is metadata-only; raw logs come from B2
                    metadata={"flake_type_prefix": ftype.split("-", 1)[0]},
                )
