"""Shared utilities for all ScenarioFamilyGenerator subclasses.

Provides deterministic fake-data helpers, template filler, neutral tool-output
builder, and the ``ArchetypedGenerator`` intermediate base that wires in
archetype loading from ``data_artifacts/clustering/<family>/archetypes.json``.
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import abstractmethod
from pathlib import Path

from ci_triage_env.data.clustering.archetypes import Archetype
from ci_triage_env.data.generators.base import ScenarioFamilyGenerator
from ci_triage_env.schemas.scenario import (
    FailureSummary,
    ToolOutput,
)

# ---------------------------------------------------------------------------
# Sample pools — kept small but varied enough for realistic output
# ---------------------------------------------------------------------------

_FEATURES = [
    "auth", "payment", "search", "cache", "api-gateway",
    "notifications", "logging", "metrics", "rate-limiter",
]
_FIX_KEYWORDS = [
    "null-check", "race-condition", "timeout", "memory-leak",
    "index-bounds", "retry-logic", "deadlock",
]
_OWNERS = ["@alice", "@bob", "@carol", "@dave", "@eve", "@frank"]
_TEAMS = ["platform", "backend", "frontend", "infra", "data-eng", "reliability"]

_TEST_MODULES = [
    "tests/unit/test_auth.py",
    "tests/integration/test_api.py",
    "tests/unit/test_cache.py",
    "tests/e2e/test_checkout.py",
    "tests/unit/test_worker.py",
    "tests/unit/test_scheduler.py",
    "tests/integration/test_db.py",
]
_TEST_FUNCS = [
    "test_user_login",
    "test_concurrent_update",
    "test_api_response_code",
    "test_cache_miss",
    "test_checkout_flow",
    "test_worker_retry",
    "test_rate_limiter_burst",
    "test_scheduler_dedup",
    "test_db_transaction_rollback",
]

_BUGGY_CODE_SNIPPETS = [
    (
        "def compute_total(items):\n"
        "    total = None\n"
        "    for item in items:\n"
        "        total += item.price  # AttributeError when items is empty\n"
        "    return total\n"
    ),
    (
        "class Cache:\n"
        "    def get(self, key):\n"
        "        return self._store[key]  # KeyError not caught\n"
        "\n"
        "    def set(self, key, value):\n"
        "        self._store[key] = value\n"
    ),
    (
        "async def handle_request(req):\n"
        "    data = await fetch_user(req.user_id)\n"
        "    return process(data['profile'])  # KeyError: 'profile' after schema change\n"
    ),
    (
        "def merge_results(a, b):\n"
        "    merged = a\n"
        "    merged.update(b)  # mutates `a` — caller surprised by aliasing\n"
        "    return merged\n"
    ),
]

_NEUTRAL_CODE_TEMPLATE = (
    "def {func_name}(self):\n"
    "    result = self.client.fetch()\n"
    "    self.assertEqual(result.status, 200)\n"
    "    self.assertIsNotNone(result.data)\n"
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def fill_template(
    template: str,
    slot_distributions: dict[str, list[str]],
    rng: random.Random,
) -> str:
    """Replace ``{SLOT}`` placeholders with random values from the distribution."""
    out = template
    for slot, values in slot_distributions.items():
        placeholder = "{" + slot + "}"
        while placeholder in out:
            replacement = rng.choice(values) if values else f"<{slot}>"
            out = out.replace(placeholder, replacement, 1)
    return out


def fake_sha(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(40))


def fake_short_sha(rng: random.Random) -> str:
    return fake_sha(rng)[:8]


def fake_timestamp(rng: random.Random) -> str:
    year = rng.randint(2024, 2025)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"


def scenario_id_for(family: str, seed: int) -> str:
    """Deterministic scenario ID — does NOT consume rng state."""
    digest = hashlib.sha1(f"{family}:{seed}".encode()).hexdigest()[:8]
    return f"{family}-s{seed}-{digest}"


def infer_suite(test_name: str) -> str:
    if "integration" in test_name or "e2e" in test_name:
        return "integration"
    if "bench" in test_name or "perf" in test_name:
        return "benchmark"
    return "unit"


def pick_test_name(rng: random.Random) -> str:
    return f"{rng.choice(_TEST_MODULES)}::{rng.choice(_TEST_FUNCS)}"


def pick_module(test_name: str) -> str:
    parts = test_name.rsplit("::", 1)
    return parts[0] if parts else test_name


def make_failure_summary(
    family: str,
    rng: random.Random,
    *,
    test_name: str,
    log_excerpt: str,
) -> FailureSummary:
    branch = rng.choice(
        [
            "main",
            "develop",
            f"feature/{rng.choice(_FEATURES)}",
            f"fix/{rng.choice(_FIX_KEYWORDS)}",
        ]
    )
    return FailureSummary(
        test_name=test_name,
        suite=infer_suite(test_name),
        branch=branch,
        last_passing_commit=fake_sha(rng),
        initial_log_excerpt=log_excerpt[:400],
        timestamp=fake_timestamp(rng),
    )


def pick_owner(rng: random.Random) -> tuple[str, str]:
    """Return (owner_handle, team)."""
    return rng.choice(_OWNERS), rng.choice(_TEAMS)


# ---------------------------------------------------------------------------
# Metric sample helpers
# ---------------------------------------------------------------------------

def _metric_sample(rng: random.Random, metric: str, *, elevated: bool = False) -> dict:
    val = rng.uniform(0.5, 0.95) if elevated else rng.uniform(0.02, 0.25)
    return {"t": fake_timestamp(rng), metric: round(val, 3), "ok": not elevated}


def _metric_samples(
    rng: random.Random, metric: str, *, elevated: bool = False, n: int = 5
) -> list[dict]:
    return [_metric_sample(rng, metric, elevated=elevated) for _ in range(n)]


# ---------------------------------------------------------------------------
# Neutral tool-output builder — covers all 11 tools
# ---------------------------------------------------------------------------

def build_base_outputs(
    test_name: str,
    branch: str,
    rng: random.Random,
    *,
    log_lines: list[str] | None = None,
    rerun_passes: bool = True,
) -> dict[str, ToolOutput]:
    """Return a dict covering every tool with *neutral* (non-signal) content.

    Individual generators call this first, then merge in family-specific
    overrides for the informative tools.
    """
    if log_lines is None:
        log_lines = [f"[INFO] Running {test_name}", "PASS  1 test ran"]

    module = pick_module(test_name)
    owner, team = pick_owner(rng)
    func_name = test_name.rsplit("::", 1)[-1]
    neutral_code = _NEUTRAL_CODE_TEMPLATE.format(func_name=func_name)

    base: dict[str, ToolOutput] = {
        # read_logs — all 5 scopes
        "read_logs:full": ToolOutput(
            tool_name="read_logs",
            payload={"lines": log_lines, "truncated": False},
            cost_units=0.001,
        ),
        "read_logs:test": ToolOutput(
            tool_name="read_logs",
            payload={"lines": log_lines[-5:], "truncated": len(log_lines) > 5},
            cost_units=0.001,
        ),
        "read_logs:stderr": ToolOutput(
            tool_name="read_logs",
            payload={"lines": ["(empty)"], "truncated": False},
            cost_units=0.001,
        ),
        "read_logs:kernel": ToolOutput(
            tool_name="read_logs",
            payload={"lines": ["(no kernel messages)"], "truncated": False},
            cost_units=0.001,
        ),
        "read_logs:build": ToolOutput(
            tool_name="read_logs",
            payload={"lines": ["Build succeeded in 42s"], "truncated": False},
            cost_units=0.001,
        ),
        # inspect_test_code
        f"inspect_test_code:{test_name}": ToolOutput(
            tool_name="inspect_test_code",
            payload={"source": neutral_code, "fixtures": []},
            cost_units=0.002,
        ),
        # run_diagnostic — all 4 probes (healthy)
        "run_diagnostic:network": ToolOutput(
            tool_name="run_diagnostic",
            payload={"ok": True, "details": {"latency_ms": rng.randint(2, 15)}},
            cost_units=0.005,
        ),
        "run_diagnostic:disk": ToolOutput(
            tool_name="run_diagnostic",
            payload={"ok": True, "details": {"free_gb": round(rng.uniform(20, 200), 1)}},
            cost_units=0.005,
        ),
        "run_diagnostic:memory": ToolOutput(
            tool_name="run_diagnostic",
            payload={"ok": True, "details": {"available_gb": round(rng.uniform(4, 32), 1)}},
            cost_units=0.005,
        ),
        "run_diagnostic:cpu": ToolOutput(
            tool_name="run_diagnostic",
            payload={"ok": True, "details": {"load_avg_1m": round(rng.uniform(0.1, 1.5), 2)}},
            cost_units=0.005,
        ),
        # cluster_metrics — all 4 metrics (healthy)
        "cluster_metrics:queue_depth": ToolOutput(
            tool_name="cluster_metrics",
            payload={"samples": _metric_samples(rng, "queue_depth")},
            cost_units=0.003,
        ),
        "cluster_metrics:node_health": ToolOutput(
            tool_name="cluster_metrics",
            payload={"samples": _metric_samples(rng, "node_health")},
            cost_units=0.003,
        ),
        "cluster_metrics:network_latency": ToolOutput(
            tool_name="cluster_metrics",
            payload={"samples": _metric_samples(rng, "network_latency")},
            cost_units=0.003,
        ),
        "cluster_metrics:disk_io": ToolOutput(
            tool_name="cluster_metrics",
            payload={"samples": _metric_samples(rng, "disk_io")},
            cost_units=0.003,
        ),
        # query_flake_history — mostly passing (stable test)
        f"query_flake_history:{test_name}": ToolOutput(
            tool_name="query_flake_history",
            payload={
                "failure_count": 1,
                "pass_count": 99,
                "recent_failures": [{"run_id": fake_short_sha(rng), "at": fake_timestamp(rng)}],
            },
            cost_units=0.002,
        ),
        # recent_commits — innocuous changes
        f"recent_commits:{branch}": ToolOutput(
            tool_name="recent_commits",
            payload={
                "commits": [
                    {
                        "sha": fake_short_sha(rng),
                        "author": rng.choice(_OWNERS),
                        "msg": "docs: update README",
                        "files": ["README.md"],
                    },
                    {
                        "sha": fake_short_sha(rng),
                        "author": rng.choice(_OWNERS),
                        "msg": "chore: bump minor versions",
                        "files": ["pyproject.toml"],
                    },
                ]
            },
            cost_units=0.002,
        ),
        "recent_commits:main": ToolOutput(
            tool_name="recent_commits",
            payload={
                "commits": [
                    {
                        "sha": fake_short_sha(rng),
                        "author": rng.choice(_OWNERS),
                        "msg": "chore: ci fix",
                        "files": [".github/workflows/ci.yml"],
                    }
                ]
            },
            cost_units=0.002,
        ),
        # check_owner
        f"check_owner:{module}": ToolOutput(
            tool_name="check_owner",
            payload={
                "owner": owner,
                "team": team,
                "contact": f"{owner.lstrip('@')}@company.com",
            },
            cost_units=0.001,
        ),
        # actions — bare keys, pre-populated with defaults
        "rerun_test": ToolOutput(
            tool_name="rerun_test",
            payload={
                "results": [
                    {"passed": rerun_passes, "duration_s": round(rng.uniform(1, 30), 2)}
                ]
            },
            cost_units=0.01,
        ),
        "quarantine_test": ToolOutput(
            tool_name="quarantine_test",
            payload={"quarantined": False, "ticket": ""},
            cost_units=0.005,
        ),
        "file_bug": ToolOutput(
            tool_name="file_bug",
            payload={"ticket_id": "N/A", "url": ""},
            cost_units=0.005,
        ),
        "ping_owner": ToolOutput(
            tool_name="ping_owner",
            payload={"delivered": False},
            cost_units=0.002,
        ),
    }
    return base


# ---------------------------------------------------------------------------
# ArchetypedGenerator — intermediate base class
# ---------------------------------------------------------------------------

class ArchetypedGenerator(ScenarioFamilyGenerator):
    """Extends ScenarioFamilyGenerator with archetype-loading support.

    Subclasses must implement ``_default_archetypes()`` as the offline fallback.
    Archetypes are loaded lazily and cached for the lifetime of the instance.
    """

    def __init__(self, archetypes_dir: Path | None = None) -> None:
        self._archetypes_dir = archetypes_dir or Path("data_artifacts/clustering")
        self._loaded: list[Archetype] | None = None

    @abstractmethod
    def _default_archetypes(self) -> list[Archetype]:
        """Hardcoded fallback archetypes used when no clustering data is on disk."""

    def _get_archetypes(self) -> list[Archetype]:
        if self._loaded is not None:
            return self._loaded
        path = self._archetypes_dir / self.family_name / "archetypes.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                loaded = [Archetype.model_validate(d) for d in data]
                if loaded:
                    self._loaded = loaded
                    return self._loaded
            except Exception:
                pass
        self._loaded = self._default_archetypes()
        return self._loaded

    def _pick_archetype(self, rng: random.Random) -> Archetype:
        return rng.choice(self._get_archetypes())

    def _pick_buggy_code(self, rng: random.Random) -> str:
        return rng.choice(_BUGGY_CODE_SNIPPETS)
