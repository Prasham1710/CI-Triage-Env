"""GitHub Actions failure-log scraper using the ``gh`` CLI.

The scraper shells out to ``gh`` rather than calling the REST API directly so
the user's existing ``gh auth login`` credentials are picked up without any
extra config. All fetched logs are cached raw under
``CI_TRIAGE_MINING_CACHE``; anonymization is applied only when constructing
the ``FailureRecord``, so a future re-anonymizer pass can re-derive records
from the original cache.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ci_triage_env.data.datasets._base import FailureRecord
from ci_triage_env.data.mining.anonymizer import anonymize
from ci_triage_env.data.mining.cache import mining_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_REPOS = [
    "kubernetes/kubernetes",
    "facebook/react",
    "tensorflow/tensorflow",
    "rust-lang/rust",
    "golang/go",
    "apache/spark",
    "pytorch/pytorch",
    "nodejs/node",
]

# A single failed-build log can be 50MB+. Cap on cache + anonymization so
# downstream tokenization doesn't choke (phase doc §implementation notes).
DEFAULT_LOG_BYTE_CAP = 200_000

# GitHub authenticated REST limit is ~83/min; leave headroom.
DEFAULT_RATE_LIMIT_PER_MIN = 60

# Filter out workflows whose name suggests docs-only / lint failures —
# we want test/build failures (phase doc §implementation notes).
_SKIP_WORKFLOW_KEYWORDS = ("docs", "lint", "format", "style", "spelling", "typo")


class GhAuthError(RuntimeError):
    """Raised when ``gh auth status`` fails."""


def check_gh_auth() -> None:
    """Confirm the ``gh`` CLI is installed and authenticated.

    Designed to be called from the CLI's mine command, not from the scraper
    constructor — keeps unit tests offline (no need to monkeypatch auth in
    every test).
    """
    try:
        proc = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GhAuthError(
            "gh CLI not found on PATH. Install from https://cli.github.com/ "
            "and run `gh auth login`."
        ) from exc
    if proc.returncode != 0:
        raise GhAuthError(
            "gh CLI is not authenticated. Run `gh auth login` and retry.\n"
            f"gh stderr: {proc.stderr.strip()}"
        )


def _safe_repo_dirname(repo: str) -> str:
    return repo.replace("/", "__")


def _is_skippable_workflow(name: str | None) -> bool:
    if not name:
        return False
    lower = name.lower()
    return any(kw in lower for kw in _SKIP_WORKFLOW_KEYWORDS)


class GitHubActionsLogScraper:
    """Mine failed runs and jobs from public GitHub Actions logs.

    Args:
        rate_limit_per_min: Sliding-window cap on outbound ``gh`` calls.
            Defaults to 60 so we leave headroom under GitHub's ~83/min
            authenticated limit.
        max_log_bytes: Cap each cached log at this many bytes. Logs from
            large monorepos can be 50MB+; truncating keeps the cache and the
            downstream tokenizer manageable.
    """

    DEFAULT_REPOS: list[str] = DEFAULT_REPOS

    def __init__(
        self,
        rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
        max_log_bytes: int = DEFAULT_LOG_BYTE_CAP,
        cache_dir: Path | None = None,
    ) -> None:
        self.rate_limit = rate_limit_per_min
        self.max_log_bytes = max_log_bytes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else mining_cache_dir()
        self._last_calls: list[float] = []

    # ------------------------------------------------------------------ public
    def mine_repo(self, repo: str, count: int = 30) -> Iterator[FailureRecord]:
        runs = self._list_failed_runs(repo, count)
        if not runs:
            logger.warning("no failed runs returned for %s", repo)
            return
        for run in runs:
            if _is_skippable_workflow(run.get("workflowName")):
                continue
            run_id = run.get("databaseId")
            if run_id is None:
                continue
            for job in self._list_failed_jobs(repo, run_id):
                job_id = job.get("databaseId")
                if job_id is None:
                    continue
                log_text = self._fetch_log(repo, run_id, job_id)
                if not log_text:
                    continue
                anonymized = anonymize(log_text)
                yield FailureRecord(
                    record_id=f"gha-{_safe_repo_dirname(repo)}-{run_id}-{job_id}",
                    source_dataset="github_actions",
                    project=repo,
                    test_name=job.get("name"),
                    failure_type_label=None,  # B3 clustering decides
                    log_text=anonymized,
                    metadata={
                        "run_id": run_id,
                        "job_id": job_id,
                        "workflow": run.get("workflowName"),
                        "branch": run.get("headBranch"),
                        "started_at": run.get("createdAt"),
                    },
                )

    # ------------------------------------------------------------------ internals
    def _list_failed_runs(self, repo: str, count: int) -> list[dict[str, Any]]:
        self._throttle()
        result = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "-R",
                repo,
                "--status",
                "failure",
                "--limit",
                str(count),
                "--json",
                "databaseId,workflowName,headBranch,createdAt",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("gh run list failed for %s: %s", repo, result.stderr.strip())
            return []
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    def _list_failed_jobs(self, repo: str, run_id: int) -> list[dict[str, Any]]:
        self._throttle()
        result = subprocess.run(
            ["gh", "run", "view", str(run_id), "-R", repo, "--json", "jobs"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("gh run view failed for %s/%s: %s", repo, run_id, result.stderr.strip())
            return []
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        return [j for j in jobs if j.get("conclusion") == "failure"]

    def _cache_path(self, repo: str, run_id: int, job_id: int) -> Path:
        return self.cache_dir / _safe_repo_dirname(repo) / f"{run_id}__{job_id}.txt"

    def _fetch_log(self, repo: str, run_id: int, job_id: int) -> str:
        cache_path = self._cache_path(repo, run_id, job_id)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")
        self._throttle()
        result = subprocess.run(
            [
                "gh",
                "run",
                "view",
                str(run_id),
                "-R",
                repo,
                "--job",
                str(job_id),
                "--log",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "gh log fetch failed for %s run=%s job=%s: %s",
                repo,
                run_id,
                job_id,
                result.stderr.strip(),
            )
            return ""
        log = result.stdout
        if len(log.encode("utf-8")) > self.max_log_bytes:
            log = log.encode("utf-8")[: self.max_log_bytes].decode("utf-8", errors="replace")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(log, encoding="utf-8")
        return log

    def _throttle(self) -> None:
        now = time.time()
        # Drop calls older than 60s — sliding window.
        self._last_calls = [t for t in self._last_calls if now - t < 60]
        if len(self._last_calls) >= self.rate_limit:
            sleep_for = 60 - (now - self._last_calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            # After sleeping, the oldest call has aged out.
            now = time.time()
            self._last_calls = [t for t in self._last_calls if now - t < 60]
        self._last_calls.append(time.time())
