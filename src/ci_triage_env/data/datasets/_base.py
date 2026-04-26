"""Normalized failure record + DatasetLoader ABC shared by every B1 loader.

All four public datasets (DeFlaker, iDFlakies, FlakeFlagger, LogHub) flatten to
``FailureRecord``. Downstream phases (B3 clustering, B4 generators) consume
records by ``source_dataset`` and ``failure_type_label`` only — they don't
peek at source-specific schemas.

Each concrete loader is a thin adapter that:
1. Reads pre-downloaded artifacts from a local path (no network here — phase doc
   §implementation notes warns that some artifacts require manual download or
   click-through licenses).
2. Yields ``FailureRecord`` objects.
3. Optionally caches them under ``data_artifacts/datasets_cache/<name>/``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ci_triage_env.data.datasets.cache import cache_dir_for

SourceDataset = Literal[
    "deflaker",
    "idflakies",
    "flakeflagger",
    "loghub",
    "github_actions",
]


class FailureRecord(BaseModel):
    """Normalized representation of one failure across all source datasets."""

    record_id: str
    source_dataset: SourceDataset
    project: str
    test_name: str | None = None
    failure_type_label: str | None = None
    log_text: str = ""
    metadata: dict = Field(default_factory=dict)


class MissingArtifactError(RuntimeError):
    """Raised when a loader can't find its source data and the user must
    download it manually first. Carries the documented instructions so the
    error message points the user at the right place."""


class DatasetLoader(ABC):
    name: str
    env_var: str  # env var pointing at the local artifact path
    download_instructions: str = ""

    def __init__(
        self,
        data_path: Path | str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if data_path is None:
            env_value = os.environ.get(self.env_var)
            if env_value:
                data_path = Path(env_value)
        self.data_path: Path | None = Path(data_path) if data_path else None
        self.cache_dir: Path = cache_dir if cache_dir is not None else cache_dir_for(self.name)

    # ------------------------------------------------------------------ contract
    @abstractmethod
    def fetch(self) -> Iterator[FailureRecord]:
        """Yield one ``FailureRecord`` per source row.

        Implementations should:
        - Call ``self._require_data_path()`` first.
        - Stream rather than build the full list in memory (phase doc §performance).
        """

    # ------------------------------------------------------------------ derived
    def info(self) -> dict:
        records = list(self.fetch())
        return {
            "name": self.name,
            "count": len(records),
            "label_distribution": dict(Counter(r.failure_type_label for r in records)),
            "data_path": str(self.data_path) if self.data_path else None,
            "cache_dir": str(self.cache_dir),
        }

    def cache_records(self, records: Iterable[FailureRecord]) -> int:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for record in records:
            (self.cache_dir / f"{record.record_id}.json").write_text(record.model_dump_json())
            n += 1
        return n

    def load_or_fetch(self) -> Iterator[FailureRecord]:
        """Return cached records if any are on disk, otherwise fetch + cache.

        This is the path the CLI uses — re-running ``cli load <dataset>`` is a
        no-op when the cache is already populated.
        """
        if self.cache_dir.exists() and any(self.cache_dir.glob("*.json")):
            for path in sorted(self.cache_dir.glob("*.json")):
                yield FailureRecord.model_validate_json(path.read_text())
            return
        records = list(self.fetch())
        self.cache_records(records)
        yield from records

    # ------------------------------------------------------------------ helpers
    def _require_data_path(self) -> Path:
        if self.data_path is None or not Path(self.data_path).exists():
            raise MissingArtifactError(
                f"{self.name} dataset not found locally. "
                f"Set ${self.env_var}=<path> or pass data_path=...\n"
                f"How to obtain: {self.download_instructions}"
            )
        return Path(self.data_path)
