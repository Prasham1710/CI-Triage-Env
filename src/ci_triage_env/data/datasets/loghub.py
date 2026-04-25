"""LogHub (Zhu et al., ISSRE 2019) loader.

Source: https://github.com/logpai/loghub. 19 system-log datasets with anomaly
labels; we only consume a subset relevant to CI/build/test failures (memory
pressure, OOMs, IO errors). LogHub is large — we **subsample 100 records per
sub-dataset** by default (phase doc §performance).

Expected on-disk layout under ``data_path/``:

    <data_path>/
      <subset_name>/
        <subset>.log              # one log line per record
        <subset>_anomaly.csv      # rows: (line_id, label) — label ∈ {anomaly, normal}

If a subset's anomaly CSV is missing, every line is treated as ``unlabeled``
so the loader still works with partial mirrors.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from ci_triage_env.data.datasets._base import DatasetLoader, FailureRecord

DEFAULT_SUB_DATASETS = ["HDFS_v1", "Hadoop", "Spark", "Linux", "BGL"]
DEFAULT_PER_SUBSET_LIMIT = 100


class LogHubLoader(DatasetLoader):
    name = "loghub"
    env_var = "LOGHUB_DATA_PATH"
    download_instructions = (
        "Clone https://github.com/logpai/loghub and place each subset's .log "
        "file (and optionally its <subset>_anomaly.csv label file) into "
        "$LOGHUB_DATA_PATH/<subset>/. Use HDFS_v1, Hadoop, Spark, Linux, BGL "
        "(or override sub_datasets= when constructing the loader)."
    )

    def __init__(
        self,
        data_path=None,
        cache_dir=None,
        sub_datasets: list[str] | None = None,
        per_subset_limit: int = DEFAULT_PER_SUBSET_LIMIT,
    ) -> None:
        super().__init__(data_path=data_path, cache_dir=cache_dir)
        self.sub_datasets = sub_datasets if sub_datasets is not None else list(DEFAULT_SUB_DATASETS)
        self.per_subset_limit = per_subset_limit

    def fetch(self) -> Iterator[FailureRecord]:
        root = self._require_data_path()
        for subset in self.sub_datasets:
            subset_dir = root / subset
            log_path = subset_dir / f"{subset}.log"
            if not log_path.exists():
                # Stay silent on missing subsets — partial mirrors are fine,
                # the info() summary will reflect what's actually loaded.
                continue
            anomaly_csv = subset_dir / f"{subset}_anomaly.csv"
            anomalies: dict[int, str] = {}
            if anomaly_csv.exists():
                anomalies = _read_anomaly_csv(anomaly_csv)
            yielded = 0
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh):
                    if yielded >= self.per_subset_limit:
                        break
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    label = anomalies.get(line_no, "unlabeled" if not anomalies else "normal")
                    record_id = "loghub-{}-{}".format(
                        subset.lower(),
                        hashlib.sha1(f"{subset}|{line_no}".encode()).hexdigest()[:10],
                    )
                    yield FailureRecord(
                        record_id=record_id,
                        source_dataset="loghub",
                        project=subset,
                        test_name=None,
                        failure_type_label=label,
                        log_text=line,
                        metadata={"sub_dataset": subset, "line_no": line_no},
                    )
                    yielded += 1


def _read_anomaly_csv(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                line_id = int(row.get("line_id", "").strip())
            except (TypeError, ValueError):
                continue
            label = (row.get("label") or "").strip().lower() or "unlabeled"
            out[line_id] = label
    return out
