"""Cache directory for raw mined GitHub Actions logs.

Layout::

    <CI_TRIAGE_MINING_CACHE>/
      <owner>__<name>/
        <run_id>__<job_id>.txt   # raw (pre-anonymization) log, truncated to 200KB

The directory is gitignored (see ``.gitignore`` for ``data_artifacts/*``).
Anonymized FailureRecords land separately under
``data_artifacts/datasets_cache/github_actions/`` like every other Phase B1
loader's output.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MINING_CACHE = Path("data_artifacts/mined_logs")


def mining_cache_dir() -> Path:
    return Path(os.environ.get("CI_TRIAGE_MINING_CACHE", str(DEFAULT_MINING_CACHE)))
