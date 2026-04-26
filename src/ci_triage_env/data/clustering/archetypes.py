"""Archetype extraction for CI failure families.

An *archetype* is a templated log pattern with ``{SLOT}`` placeholders and a
distribution of realistic slot values sampled from real records.  B4 inflates
these archetypes into full synthetic scenarios.

Extraction strategy (v1, simple enough to work without scikit-learn):
  1. Within a family, bucket records by their first dominant-signal line.
  2. Per bucket pick the most representative record (longest with most signal).
  3. Replace variable tokens with typed slots.
  4. Sample slot values across all records in the bucket.
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel

from ci_triage_env.data.datasets._base import FailureRecord

# ---------------------------------------------------------------------------
# Slot extraction regexes — order matters (longer/more-specific first)
# ---------------------------------------------------------------------------

_SLOT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ISO timestamp (with or without T separator)
    ("TIMESTAMP", re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")),
    # Durations: 12.34s, 120ms, 60 seconds, 1m30s
    ("DURATION", re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m\d+s)\b|\b\d+(?:\.\d+)? seconds?\b")),
    # Memory sizes
    ("MEMSIZE", re.compile(r"\b\d+(?:\.\d+)?[kKmMgG]i?[bB]\b|\b\d+[kK][bB]\b")),
    # PIDs / process IDs
    ("PID", re.compile(r"\bpid[=: ]+\d+\b|\bprocess \d+\b|\bpid \d+\b", re.IGNORECASE)),
    # Line numbers inside file references: file.py:123:
    ("LINENO", re.compile(r"(?<=:)\d+(?=:|\b)")),
    # Standalone numbers (2+ digits): goroutine IDs, ports, counts, addresses
    ("NUM", re.compile(r"\b\d{2,}\b")),
    # Test-function names  (word chars followed by parens or preceded by FAIL/ERROR)
    ("TEST_NAME", re.compile(r"(?:FAIL|ERROR|FAILED)\s+(\S+)")),
]

# Words that appear in meaningful "first error lines" for each family
_FAMILY_SIGNALS: dict[str, re.Pattern[str]] = {
    "infra_resource": re.compile(r"OOM|out of memory|ENOSPC|EMFILE|disk full|cannot allocate", re.IGNORECASE),
    "infra_network": re.compile(r"connection refused|getaddrinfo|DNS|TLS|EHOSTUNREACH|ENETUNREACH", re.IGNORECASE),
    "race_flake": re.compile(r"DATA RACE|race detected|concurrent map|deadlock", re.IGNORECASE),
    "timing_flake": re.compile(r"deadline exceeded|timeout exceeded|timed out|context canceled", re.IGNORECASE),
    "dependency_drift": re.compile(r"version mismatch|peer dep|Cargo\.lock|go\.sum|incompatible", re.IGNORECASE),
    "real_bug": re.compile(r"AssertionError|panic:|NullPointerException|AttributeError|assert .* ==", re.IGNORECASE),
    "ambiguous": re.compile(r"FAIL|ERROR|exception", re.IGNORECASE),
}

_INFORMATIVE_TOOLS: dict[str, list[str]] = {
    "infra_resource": ["read_logs:kernel", "cluster_metrics:5m"],
    "infra_network": ["read_logs:system", "cluster_metrics:5m"],
    "race_flake": ["read_logs:full", "inspect_test_code:concurrent"],
    "timing_flake": ["read_logs:full", "cluster_metrics:15m"],
    "dependency_drift": ["read_logs:build", "recent_commits:main"],
    "real_bug": ["read_logs:full", "inspect_test_code:failing"],
    "ambiguous": ["read_logs:full", "cluster_metrics:5m", "query_flake_history:test"],
}

_MINIMAL_EVIDENCE: dict[str, list[str]] = {
    "infra_resource": ["cluster_metrics:5m"],
    "infra_network": ["cluster_metrics:5m"],
    "race_flake": ["read_logs:full"],
    "timing_flake": ["cluster_metrics:15m"],
    "dependency_drift": ["recent_commits:main"],
    "real_bug": ["inspect_test_code:failing"],
    "ambiguous": ["read_logs:full"],
}


class Archetype(BaseModel):
    """Templated log pattern for one failure family."""

    archetype_id: str
    family: str
    pattern_summary: str
    log_template: str
    slot_distributions: dict[str, list[str]]
    informative_tools_hint: list[str]
    minimal_evidence_hint: list[str]


def _extract_slots(text: str) -> tuple[str, dict[str, list[str]]]:
    """Replace variable tokens with ``{SLOT}`` placeholders.

    Returns the templated text and a dict of slot_name → list of captured
    values (the values from *this* record only; the caller merges across the
    cluster).
    """
    result = text
    distributions: dict[str, list[str]] = defaultdict(list)

    for slot_name, pattern in _SLOT_PATTERNS:
        def _replace(m: re.Match[str], sn: str = slot_name) -> str:
            val = m.group(0)
            distributions[sn].append(val)
            return f"{{{sn}}}"

        result = pattern.sub(_replace, result)

    return result, dict(distributions)


def _dominant_signal_line(text: str, family: str) -> str:
    """Return the first line that matches the family's signal pattern."""
    signal_pat = _FAMILY_SIGNALS.get(family, re.compile(r"FAIL|ERROR", re.IGNORECASE))
    for line in text.splitlines():
        if signal_pat.search(line):
            return line.strip()[:120]
    return text.splitlines()[0].strip()[:120] if text.strip() else ""


def _bucket_records(records: list[FailureRecord], family: str, n: int) -> list[list[FailureRecord]]:
    """Group records into up to ``n`` buckets by first signal-line similarity."""
    buckets: dict[str, list[FailureRecord]] = defaultdict(list)
    for record in records:
        key = _dominant_signal_line(record.log_text, family)
        # Collapse to first 60 chars so minor variation doesn't over-split
        buckets[key[:60]].append(record)

    sorted_buckets = sorted(buckets.values(), key=len, reverse=True)
    if len(sorted_buckets) <= n:
        return sorted_buckets
    # Merge smallest buckets beyond limit into last retained bucket
    merged = sorted_buckets[:n]
    for extra in sorted_buckets[n:]:
        merged[-1].extend(extra)
    return merged


def _pick_representative(bucket: list[FailureRecord]) -> FailureRecord:
    """Pick the record with the most text (likely most informative)."""
    return max(bucket, key=lambda r: len(r.log_text))


def _merge_slot_distributions(
    per_record: list[dict[str, list[str]]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)
    for d in per_record:
        for slot, vals in d.items():
            merged[slot].extend(vals)
    # Deduplicate while preserving order; cap at 10 samples per slot
    seen: dict[str, set[str]] = defaultdict(set)
    result: dict[str, list[str]] = {}
    for slot, vals in merged.items():
        unique: list[str] = []
        for v in vals:
            if v not in seen[slot]:
                seen[slot].add(v)
                unique.append(v)
        result[slot] = unique[:10]
    return result


class ArchetypeExtractor:
    """Extract templated archetypes from a family's ``FailureRecord`` list."""

    def extract(
        self,
        records: list[FailureRecord],
        family: str,
        n_archetypes: int = 4,
    ) -> list[Archetype]:
        if not records:
            return []

        buckets = _bucket_records(records, family, n_archetypes)
        archetypes: list[Archetype] = []

        for idx, bucket in enumerate(buckets):
            representative = _pick_representative(bucket)
            log_preview = representative.log_text[:2000]

            all_distributions: list[dict[str, list[str]]] = []
            for rec in bucket:
                _, dist = _extract_slots(rec.log_text[:2000])
                all_distributions.append(dist)

            template, _own_dist = _extract_slots(log_preview)
            slot_dists = _merge_slot_distributions(all_distributions)

            signal_line = _dominant_signal_line(representative.log_text, family)
            summary = f"{family.replace('_', ' ').title()} — {signal_line[:80]}" if signal_line else family

            archetype_id = f"{family}_{idx + 1:03d}"
            archetypes.append(
                Archetype(
                    archetype_id=archetype_id,
                    family=family,
                    pattern_summary=summary,
                    log_template=template,
                    slot_distributions=slot_dists,
                    informative_tools_hint=_INFORMATIVE_TOOLS.get(family, ["read_logs:full"]),
                    minimal_evidence_hint=_MINIMAL_EVIDENCE.get(family, ["read_logs:full"]),
                )
            )

        return archetypes
