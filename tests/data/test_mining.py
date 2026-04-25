"""Phase B2 mining tests.

All ``gh`` invocations are monkeypatched so the suite is fully offline. The
mining module itself never imports network libraries, so the only seam to
fake is ``subprocess.run`` — we route every call through a recording shim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from ci_triage_env.data.cli import build_parser
from ci_triage_env.data.cli import main as cli_main
from ci_triage_env.data.datasets._base import FailureRecord
from ci_triage_env.data.mining import github_actions as gha_mod
from ci_triage_env.data.mining.anonymizer import anonymize, hash_short
from ci_triage_env.data.mining.cache import DEFAULT_MINING_CACHE, mining_cache_dir
from ci_triage_env.data.mining.github_actions import (
    DEFAULT_REPOS,
    GhAuthError,
    GitHubActionsLogScraper,
    check_gh_auth,
)

# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

def test_anonymizer_replaces_emails():
    out = anonymize("contact bob@example.com for help")
    assert "EMAIL" in out
    assert "bob@example.com" not in out


def test_anonymizer_replaces_full_shas():
    full_sha = "abc123def456abc123def456abc123def456abcd"  # 40 hex
    out = anonymize(f"commit {full_sha} landed")
    assert "sha-" in out
    assert full_sha not in out


def test_anonymizer_replaces_short_hex():
    """8-char hex tokens get hashed to a stable short token."""
    out = anonymize("ref deadbeef in tree")
    assert "hex-" in out
    assert "deadbeef" not in out


def test_anonymizer_full_sha_takes_precedence_over_short_hex():
    """A 40-hex SHA must replace as one full token, not split into hex- segments."""
    full = "0" * 40
    out = anonymize(full)
    assert out.count("sha-") == 1
    assert "hex-" not in out


def test_anonymizer_replaces_user_paths():
    out = anonymize("Traceback in /home/alice/project/main.py")
    assert "/home/alice/" not in out
    assert "/PATH/USER/" in out


def test_anonymizer_replaces_user_mentions():
    out = anonymize("blame @alice and @bob-eng for the change")
    assert "@alice" not in out
    assert "@bob-eng" not in out
    assert "@USER" in out


def test_anonymizer_replaces_ipv4():
    out = anonymize("Connection refused: 192.168.1.10")
    assert "192.168.1.10" not in out
    assert "IP" in out


def test_anonymizer_idempotent():
    text = "user @alice (alice@example.com) on /home/alice/foo at 192.168.1.1"
    once = anonymize(text)
    twice = anonymize(once)
    assert once == twice


def test_anonymizer_preserves_log_structure():
    raw = "ERROR: failure\n  at line 12\n\tindented\nESC[31mred\nbye"
    out = anonymize(raw)
    assert out.count("\n") == raw.count("\n")
    assert "\t" in out
    assert "ESC[31m" in out


def test_hash_short_is_deterministic():
    assert hash_short("abc") == hash_short("abc")
    assert len(hash_short("abc")) == 8


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------

def test_mining_cache_dir_default(monkeypatch):
    monkeypatch.delenv("CI_TRIAGE_MINING_CACHE", raising=False)
    assert mining_cache_dir() == DEFAULT_MINING_CACHE


def test_mining_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_TRIAGE_MINING_CACHE", str(tmp_path / "alt"))
    assert mining_cache_dir() == tmp_path / "alt"


# ---------------------------------------------------------------------------
# Subprocess shim used by every scraper test below.
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_run_recorder(responses: dict[tuple[str, ...], _FakeProc]):
    """Return (fn, calls) where ``fn`` matches subprocess.run's signature.

    ``responses`` keys on the *tuple of relevant tokens* in the gh argv.
    Anything not matched returns an empty success.
    """
    calls: list[list[str]] = []

    def _run(argv, capture_output=True, text=True, check=False, **kw):
        calls.append(list(argv))
        for keys, proc in responses.items():
            if all(k in argv for k in keys):
                return proc
        return _FakeProc(returncode=0, stdout="", stderr="")

    return _run, calls


@pytest.fixture
def patched_subprocess(monkeypatch):
    def _install(responses):
        run, calls = _make_run_recorder(responses)
        monkeypatch.setattr(gha_mod.subprocess, "run", run)
        # Also disable real time.sleep so throttle tests don't actually wait.
        sleeps: list[float] = []
        monkeypatch.setattr(gha_mod.time, "sleep", lambda s: sleeps.append(s))
        return calls, sleeps

    return _install


# ---------------------------------------------------------------------------
# Scraper happy path
# ---------------------------------------------------------------------------

def _gh_run_list_payload(items: Iterable[dict]) -> str:
    return json.dumps(list(items))


def _gh_run_view_payload(jobs: Iterable[dict]) -> str:
    return json.dumps({"jobs": list(jobs)})


def test_mine_repo_with_fixture_subprocess_yields_failure_records(patched_subprocess, tmp_path):
    runs = _gh_run_list_payload([
        {
            "databaseId": 111,
            "workflowName": "test",
            "headBranch": "main",
            "createdAt": "2026-04-25T12:00:00Z",
        }
    ])
    jobs = _gh_run_view_payload([
        {"databaseId": 222, "name": "build/x", "conclusion": "failure"},
        {"databaseId": 333, "name": "skip/me", "conclusion": "success"},
    ])
    raw_log = "ERROR build failed at sha 0123456789abcdef0123456789abcdef01234567 by @alice (alice@x.com)"
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("run", "view", "111", "--json"): _FakeProc(stdout=jobs),
        ("run", "view", "111", "--log"): _FakeProc(stdout=raw_log),
    }
    patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(rate_limit_per_min=1000, cache_dir=tmp_path)
    records = list(scraper.mine_repo("kubernetes/kubernetes", count=5))

    assert len(records) == 1
    rec = records[0]
    assert rec.source_dataset == "github_actions"
    assert rec.project == "kubernetes/kubernetes"
    assert rec.test_name == "build/x"
    assert rec.metadata["run_id"] == 111
    assert rec.metadata["job_id"] == 222
    # log_text is the anonymized version, raw is in the cache.
    assert "@alice" not in rec.log_text
    assert "EMAIL" in rec.log_text
    assert "sha-" in rec.log_text
    cached = (tmp_path / "kubernetes__kubernetes" / "111__222.txt").read_text()
    assert "@alice" in cached  # raw cache stays un-anonymized for re-derivation


def test_scraper_uses_cache_on_repeat_fetch(patched_subprocess, tmp_path):
    """A second mine_repo call hits the cache for logs (no new --log subprocess)."""
    runs = _gh_run_list_payload([
        {"databaseId": 1, "workflowName": "test", "headBranch": "main", "createdAt": "t"}
    ])
    jobs = _gh_run_view_payload([{"databaseId": 2, "name": "j", "conclusion": "failure"}])
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("run", "view", "1", "--json"): _FakeProc(stdout=jobs),
        ("run", "view", "1", "--log"): _FakeProc(stdout="hello world"),
    }
    calls, _ = patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(rate_limit_per_min=1000, cache_dir=tmp_path)

    list(scraper.mine_repo("x/y", count=1))
    log_calls_first = sum(1 for c in calls if "--log" in c)
    assert log_calls_first == 1

    list(scraper.mine_repo("x/y", count=1))
    log_calls_second = sum(1 for c in calls if "--log" in c) - log_calls_first
    assert log_calls_second == 0  # second pass served from cache


def test_scraper_throttles_when_rate_limit_exceeded(patched_subprocess, tmp_path, monkeypatch):
    # Many failed jobs across runs, low rate limit → throttle must kick in.
    runs = _gh_run_list_payload([
        {"databaseId": i, "workflowName": "test", "headBranch": "main", "createdAt": "t"}
        for i in range(20)
    ])
    jobs = _gh_run_view_payload([{"databaseId": 99, "name": "j", "conclusion": "failure"}])
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("--json", "jobs"): _FakeProc(stdout=jobs),
        ("--log",): _FakeProc(stdout="log"),
    }
    _, sleeps = patched_subprocess(responses)
    # Pin time.time so the sliding window doesn't expire spontaneously.
    fake_now = [1000.0]
    monkeypatch.setattr(gha_mod.time, "time", lambda: fake_now[0])
    scraper = GitHubActionsLogScraper(rate_limit_per_min=3, cache_dir=tmp_path)
    list(scraper.mine_repo("x/y", count=20))
    assert any(s > 0 for s in sleeps), "expected at least one throttle sleep"


def test_scraper_skips_empty_log(patched_subprocess, tmp_path):
    runs = _gh_run_list_payload([
        {"databaseId": 1, "workflowName": "test", "headBranch": "main", "createdAt": "t"}
    ])
    jobs = _gh_run_view_payload([{"databaseId": 2, "name": "j", "conclusion": "failure"}])
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("--json", "jobs"): _FakeProc(stdout=jobs),
        ("--log",): _FakeProc(returncode=1, stdout="", stderr="boom"),
    }
    patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(rate_limit_per_min=1000, cache_dir=tmp_path)
    records = list(scraper.mine_repo("x/y", count=1))
    assert records == []


def test_scraper_skips_docs_and_lint_workflows(patched_subprocess, tmp_path):
    runs = _gh_run_list_payload([
        {"databaseId": 1, "workflowName": "Docs build", "headBranch": "main", "createdAt": "t"},
        {"databaseId": 2, "workflowName": "Spelling", "headBranch": "main", "createdAt": "t"},
        {"databaseId": 3, "workflowName": "test-suite", "headBranch": "main", "createdAt": "t"},
    ])
    jobs = _gh_run_view_payload([{"databaseId": 99, "name": "j", "conclusion": "failure"}])
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("--json", "jobs"): _FakeProc(stdout=jobs),
        ("--log",): _FakeProc(stdout="log"),
    }
    calls, _ = patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(rate_limit_per_min=1000, cache_dir=tmp_path)
    records = list(scraper.mine_repo("x/y", count=3))
    assert len(records) == 1
    # Only the test-suite run should have triggered jobs/log fetches.
    job_view_calls = [c for c in calls if "--json" in c and "jobs" in c]
    assert len(job_view_calls) == 1


def test_log_truncated_to_max_bytes(patched_subprocess, tmp_path):
    big = "x" * (5 * 1024 * 1024)  # 5MB
    runs = _gh_run_list_payload([
        {"databaseId": 1, "workflowName": "test", "headBranch": "main", "createdAt": "t"}
    ])
    jobs = _gh_run_view_payload([{"databaseId": 2, "name": "j", "conclusion": "failure"}])
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("--json", "jobs"): _FakeProc(stdout=jobs),
        ("--log",): _FakeProc(stdout=big),
    }
    patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(
        rate_limit_per_min=1000, cache_dir=tmp_path, max_log_bytes=200_000
    )
    list(scraper.mine_repo("x/y", count=1))
    cache_path = tmp_path / "x__y" / "1__2.txt"
    assert cache_path.exists()
    assert len(cache_path.read_bytes()) <= 200_000


def test_scraper_returns_empty_when_run_list_fails(patched_subprocess, tmp_path):
    responses = {
        ("run", "list"): _FakeProc(returncode=1, stderr="auth issue"),
    }
    patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(rate_limit_per_min=1000, cache_dir=tmp_path)
    assert list(scraper.mine_repo("x/y", count=1)) == []


def test_scraper_returns_empty_on_invalid_json(patched_subprocess, tmp_path):
    responses = {
        ("run", "list"): _FakeProc(stdout="not json"),
    }
    patched_subprocess(responses)
    scraper = GitHubActionsLogScraper(rate_limit_per_min=1000, cache_dir=tmp_path)
    assert list(scraper.mine_repo("x/y", count=1)) == []


# ---------------------------------------------------------------------------
# gh auth check
# ---------------------------------------------------------------------------

def test_check_gh_auth_raises_when_not_installed(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("no gh")

    monkeypatch.setattr(gha_mod.subprocess, "run", _raise)
    with pytest.raises(GhAuthError, match="gh CLI not found"):
        check_gh_auth()


def test_check_gh_auth_raises_when_not_authenticated(monkeypatch):
    monkeypatch.setattr(
        gha_mod.subprocess,
        "run",
        lambda *a, **k: _FakeProc(returncode=1, stderr="not logged in"),
    )
    with pytest.raises(GhAuthError, match="not authenticated"):
        check_gh_auth()


def test_check_gh_auth_passes_when_authenticated(monkeypatch):
    monkeypatch.setattr(
        gha_mod.subprocess,
        "run",
        lambda *a, **k: _FakeProc(returncode=0, stdout="ok"),
    )
    check_gh_auth()  # no exception


# ---------------------------------------------------------------------------
# CLI mine subcommand
# ---------------------------------------------------------------------------

def test_cli_mine_parser_lists_subcommand():
    parser = build_parser()
    parsed = parser.parse_args(["mine", "--repo", "x/y", "--count", "3", "--skip-auth-check"])
    assert parsed.cmd == "mine"
    assert parsed.repo == "x/y"
    assert parsed.count == 3
    assert parsed.skip_auth_check is True


def test_cli_mine_writes_failure_records(monkeypatch, tmp_path, capsys):
    runs = _gh_run_list_payload([
        {"databaseId": 5, "workflowName": "test", "headBranch": "main", "createdAt": "t"}
    ])
    jobs = _gh_run_view_payload([{"databaseId": 6, "name": "build", "conclusion": "failure"}])
    responses = {
        ("run", "list"): _FakeProc(stdout=runs),
        ("--json", "jobs"): _FakeProc(stdout=jobs),
        ("--log",): _FakeProc(stdout="failure log alice@example.com"),
    }
    run, _ = _make_run_recorder(responses)
    monkeypatch.setattr(gha_mod.subprocess, "run", run)
    monkeypatch.setattr(gha_mod.time, "sleep", lambda s: None)

    cache_dir = tmp_path / "raw"
    out_dir = tmp_path / "records"
    rc = cli_main(
        [
            "mine",
            "--repo",
            "k8s/k8s",
            "--count",
            "1",
            "--rate-limit",
            "1000",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--skip-auth-check",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "k8s/k8s: 1 records" in out
    assert "mined 1 records" in out
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1
    rec = FailureRecord.model_validate_json(written[0].read_text())
    assert rec.source_dataset == "github_actions"
    assert "EMAIL" in rec.log_text


def test_cli_mine_default_repos_constant_matches():
    """Sanity: the CLI's "no --repo" branch hits the same DEFAULT_REPOS list."""
    assert DEFAULT_REPOS  # non-empty
    assert "kubernetes/kubernetes" in DEFAULT_REPOS


def test_cli_mine_propagates_auth_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        gha_mod.subprocess,
        "run",
        lambda *a, **k: _FakeProc(returncode=1, stderr="please login"),
    )
    rc = cli_main(["mine", "--repo", "x/y", "--count", "1"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not authenticated" in captured.err


# ---------------------------------------------------------------------------
# Sanity: the default cache root is under the gitignored data_artifacts dir.
# ---------------------------------------------------------------------------

def test_default_mining_cache_under_data_artifacts():
    assert Path("data_artifacts") in DEFAULT_MINING_CACHE.parents or DEFAULT_MINING_CACHE.parts[0] == "data_artifacts"
