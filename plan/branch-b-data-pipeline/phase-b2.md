# Phase B2 — GitHub Actions Log Mining

**Owner:** Branch B.
**Prerequisite:** `phase-0-complete` on `main`. Can run in parallel with B1.
**Estimated time:** 3–4 hours.

---

## Outcome

A scraper that fetches real CI failure logs from public GitHub Actions runs across 8 OSS repos. By end of phase:

1. `python -m ci_triage_env.data.cli mine --repo <owner/name> --count 30` fetches up to 30 failed runs.
2. Output: `data_artifacts/mined_logs/<owner>__<name>/<run_id>__<job_id>.txt` (one file per failed job).
3. Anonymizer replaces project-specific identifiers (org names, internal hostnames, employee names) with neutral tokens.
4. Each mined log is also serialized as a `FailureRecord` for downstream consumption.
5. Rate limited: ≤ 100 API calls/min, falls back to cache when possible.
6. All B2 tests pass.

---

## Files to create

### `src/ci_triage_env/data/mining/github_actions.py`

```python
import subprocess
import json
import time
from pathlib import Path
from ..datasets._base import FailureRecord
from .anonymizer import anonymize
from .cache import mining_cache_dir

class GitHubActionsLogScraper:
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

    def __init__(self, rate_limit_per_min: int = 100):
        self.rate_limit = rate_limit_per_min
        self._last_calls: list[float] = []

    def mine_repo(self, repo: str, count: int = 30) -> Iterable[FailureRecord]:
        """Fetch up to `count` failed runs from the repo."""
        runs = self._list_failed_runs(repo, count)
        for run in runs:
            for job in self._list_failed_jobs(repo, run["databaseId"]):
                log_text = self._fetch_log(repo, run["databaseId"], job["databaseId"])
                if not log_text:
                    continue
                anonymized = anonymize(log_text)
                yield FailureRecord(
                    record_id=f"gha-{repo.replace('/', '__')}-{run['databaseId']}-{job['databaseId']}",
                    source_dataset="github_actions",
                    project=repo,
                    test_name=job.get("name"),
                    failure_type_label=None,  # unlabeled — clustering decides in B3
                    log_text=anonymized,
                    metadata={
                        "run_id": run["databaseId"],
                        "job_id": job["databaseId"],
                        "workflow": run.get("workflowName"),
                        "branch": run.get("headBranch"),
                        "started_at": run.get("createdAt"),
                    },
                )

    def _list_failed_runs(self, repo: str, count: int) -> list[dict]:
        self._throttle()
        result = subprocess.run(
            ["gh", "run", "list", "-R", repo, "--status", "failure",
             "--limit", str(count), "--json",
             "databaseId,workflowName,headBranch,createdAt"],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def _list_failed_jobs(self, repo: str, run_id: int) -> list[dict]:
        self._throttle()
        result = subprocess.run(
            ["gh", "run", "view", str(run_id), "-R", repo, "--json", "jobs"],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        return [j for j in data.get("jobs", []) if j.get("conclusion") == "failure"]

    def _fetch_log(self, repo: str, run_id: int, job_id: int) -> str:
        # Cache check
        cache_path = mining_cache_dir() / repo.replace("/", "__") / f"{run_id}__{job_id}.txt"
        if cache_path.exists():
            return cache_path.read_text()
        self._throttle()
        result = subprocess.run(
            ["gh", "run", "view", str(run_id), "-R", repo, "--job", str(job_id), "--log"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return ""
        log = result.stdout
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(log)
        return log

    def _throttle(self):
        now = time.time()
        # Drop calls older than 60s
        self._last_calls = [t for t in self._last_calls if now - t < 60]
        if len(self._last_calls) >= self.rate_limit:
            sleep = 60 - (now - self._last_calls[0])
            time.sleep(max(sleep, 0))
        self._last_calls.append(time.time())
```

### `src/ci_triage_env/data/mining/anonymizer.py`

```python
import re

# Patterns to anonymize:
# - GitHub usernames (commits-by, mentions): @user, by user
# - Email addresses
# - Internal hostnames matching common patterns (corp.*, internal.*)
# - Long random IDs (UUIDs, SHAs) — replaced with a stable hash to preserve uniqueness
# - File paths under /home/<user>/, /Users/<user>/

REPLACEMENTS = [
    (r'@[A-Za-z0-9_\-]+', '@USER'),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
    (r'\b[a-f0-9]{40}\b', lambda m: f"sha-{hash_short(m.group())}"),  # full SHA
    (r'\b[a-f0-9]{8}\b', lambda m: f"hex-{hash_short(m.group())}"),  # short SHA
    (r'/(home|Users)/[^/]+/', '/PATH/USER/'),
    (r'\b\d+\.\d+\.\d+\.\d+\b', 'IP'),  # IPv4
]

def anonymize(text: str) -> str:
    out = text
    for pattern, replacement in REPLACEMENTS:
        if callable(replacement):
            out = re.sub(pattern, replacement, out)
        else:
            out = re.sub(pattern, replacement, out)
    return out

def hash_short(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode()).hexdigest()[:8]
```

### `src/ci_triage_env/data/mining/cache.py`

```python
def mining_cache_dir() -> Path:
    return Path(os.environ.get("CI_TRIAGE_MINING_CACHE", "data_artifacts/mined_logs"))
```

### Modify `src/ci_triage_env/data/cli.py`

Add `mine` subcommand:

```python
def cmd_mine(args):
    scraper = GitHubActionsLogScraper(rate_limit_per_min=args.rate_limit)
    records = []
    if args.repo:
        records = list(scraper.mine_repo(args.repo, args.count))
    else:
        for repo in scraper.DEFAULT_REPOS:
            records.extend(scraper.mine_repo(repo, args.count))
    # Persist as FailureRecord
    out_dir = Path("data_artifacts/datasets_cache/github_actions")
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        (out_dir / f"{record.record_id}.json").write_text(record.model_dump_json())
    print(f"Mined {len(records)} records.")
```

---

## Implementation notes

- **`gh` CLI must be authenticated.** Check at startup: `gh auth status`. If not authenticated, fail with clear instructions.
- **Rate limit reality.** GitHub's authenticated API limit is 5000/hr (~83/min). Set our internal rate to 60/min to leave headroom for retries.
- **Logs can be huge.** A single Kubernetes failed-build log can be 50MB+. Truncate to first 200KB at fetch time. Document in code.
- **Skip workflow files / docs-only failures.** `gh run list` returns all failures; filter out workflows whose name suggests docs/style/lint (we want test/build failures).
- **Anonymization is best-effort.** Real-world logs have many forms of identifiers. Aim for 90% coverage; perfect anonymization is impossible. Document this in the README's Limitations section.
- **Cache aggressively.** First run is slow (network-bound). Subsequent runs use cache. Don't invalidate without `--force`.
- **Total target: ~250 mined logs.** 8 repos × ~30 each = 240. Some will be duplicates or empty — final yield ~200.

---

## Tests required (`tests/data/test_mining.py`)

```python
def test_anonymizer_replaces_emails():
    assert "EMAIL" in anonymize("contact bob@example.com for help")

def test_anonymizer_replaces_shas():
    assert "sha-" in anonymize("commit abc123def456abc123def456abc123def456abcd")

def test_anonymizer_idempotent():
    """Anonymizing twice produces the same result."""

def test_anonymizer_preserves_log_structure():
    """Line breaks, indentation, ANSI codes are preserved."""

def test_scraper_uses_cache(monkeypatch):
    """Mock subprocess; second call uses cache, no subprocess invocation."""

def test_scraper_throttles(monkeypatch):
    """Make 100 mock calls; assert at least one sleep happened."""

def test_scraper_handles_empty_log_gracefully():
    """Mock subprocess returning empty stdout — yields no record."""

def test_mine_repo_with_fixture_subprocess(monkeypatch):
    """Patch subprocess to return canned `gh` output; verify FailureRecord shape."""

def test_log_truncated_to_200kb():
    """A 5MB log is truncated to ≤ 200KB on cache write."""
```

Use `monkeypatch.setattr(subprocess, "run", ...)` to avoid network in unit tests.

---

## Smoke test (manual, requires `gh auth login`)

```bash
gh auth status   # should show authenticated

# Mine one repo
python -m ci_triage_env.data.cli mine --repo kubernetes/kubernetes --count 5

# Verify
ls data_artifacts/mined_logs/kubernetes__kubernetes/
ls data_artifacts/datasets_cache/github_actions/

# Inspect a record
cat data_artifacts/datasets_cache/github_actions/gha-kubernetes__kubernetes-*.json | jq .log_text | head
```

Expected: ~5 .txt files in mined_logs/, ~5 .json records in datasets_cache/github_actions/, all anonymized.

---

## Open questions

1. **What if a repo has < 30 recent failures?** Take what's available, log a warning, move on.
2. **Should the scraper deduplicate near-identical logs?** Optional — leave duplicates in B2; clustering in B3 will handle them. If we have time, add a hash-based dedup step.
3. **Are there licensing concerns about redistributing OSS logs?** Logs are public via the GitHub Actions UI. Generated *scenarios* (B5) don't redistribute the raw text — they use the structural patterns. Document this clearly.

---

## What's NOT in this phase

- Clustering mined logs into the 7 families (B3)
- Generating synthetic scenarios from mined logs (B4)
