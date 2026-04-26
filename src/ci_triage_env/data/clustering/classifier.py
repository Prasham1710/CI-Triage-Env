"""Rule-based + optional LLM classifier for CI failure families.

Rule-based handles ~70%+ of cases; LLM fallback is called only for records
the rules cannot confidently classify, keeping OpenAI cost under $5.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from ci_triage_env.data.datasets._base import FailureRecord

FAMILIES: list[str] = [
    "real_bug",
    "race_flake",
    "timing_flake",
    "infra_network",
    "infra_resource",
    "dependency_drift",
    "ambiguous",
]

_RULES: dict[str, list[re.Pattern[str]]] = {
    "infra_resource": [
        re.compile(r"OOMKilled|out of memory|cannot allocate|killed by OS|137 SIGKILL", re.IGNORECASE),
        re.compile(r"no space left on device|disk full|ENOSPC", re.IGNORECASE),
        re.compile(r"too many open files|EMFILE"),
    ],
    "infra_network": [
        re.compile(r"DNS resolution|unable to resolve|getaddrinfo failed|connection refused"),
        re.compile(r"TLS handshake.*timeout|x509:.*certificate"),
        re.compile(r"socket: connection reset|EHOSTUNREACH|ENETUNREACH"),
    ],
    "race_flake": [
        re.compile(r"data race|race detected|WARNING: DATA RACE", re.IGNORECASE),
        re.compile(r"concurrent map writes|fatal error: concurrent"),
        re.compile(r"deadlock detected"),
    ],
    "timing_flake": [
        re.compile(r"deadline exceeded|context canceled|timeout exceeded"),
        re.compile(r"test timed out after \d+", re.IGNORECASE),
    ],
    "dependency_drift": [
        re.compile(r"Cargo\.lock|package-lock\.json|go\.sum.*conflict"),
        re.compile(r"npm ERR! peer dep|incompatible dependency"),
        re.compile(r"version (mismatch|conflict)"),
    ],
}

# Patterns that, if matched, strongly suggest a real bug (assertion, panic, etc.)
_REAL_BUG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"AssertionError|assert .* failed|FAILED assert", re.IGNORECASE),
    re.compile(r"panic: "),
    re.compile(r"NullPointerException|AttributeError|TypeError|ValueError"),
    re.compile(r"FAIL\b.*\(.*s\)"),
]

# Confidence threshold above which a family is counted as a clear "hit"
_HIT_THRESHOLD = 0.3


class RuleBasedClassifier:
    """Keyword + regex matching to classify obvious CI failure cases.

    Returns ``("unknown", 0.0)`` for records that match no rules; these
    become candidates for the LLM fallback in ``classify_all``.
    """

    def classify(self, record: FailureRecord) -> tuple[str, float]:
        """Return ``(family, confidence)`` where confidence is in ``[0, 1]``."""
        text = record.log_text

        scores: dict[str, float] = {}
        for family, patterns in _RULES.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches:
                scores[family] = matches / len(patterns)

        if not scores:
            # Try the real_bug heuristics as a last resort
            rb_matches = sum(1 for p in _REAL_BUG_PATTERNS if p.search(text))
            if rb_matches:
                return ("real_bug", rb_matches / len(_REAL_BUG_PATTERNS))
            return ("unknown", 0.0)

        hit_families = [f for f, s in scores.items() if s > _HIT_THRESHOLD]
        if len(hit_families) > 1:
            return ("ambiguous", min(scores[f] for f in hit_families))

        best_family = max(scores, key=lambda f: scores[f])
        return (best_family, scores[best_family])


class LLMClassifier:
    """Fallback for records the rule-based classifier marked ``'unknown'``.

    Calls ``openai`` (must be installed) and stops once ``budget_usd`` is spent.
    """

    SYSTEM_PROMPT = (
        "You are a CI failure classifier. Given a failure log, output exactly "
        "one label from: real_bug, race_flake, timing_flake, infra_network, "
        "infra_resource, dependency_drift, ambiguous.\n\n"
        "Choose ambiguous only if multiple causes are plausible and no single "
        "one dominates. Respond with the label only, no explanation."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        budget_usd: float = 5.0,
    ) -> None:
        from openai import OpenAI  # optional dependency — imported lazily

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.budget = budget_usd
        self.spent: float = 0.0

    def classify_batch(self, records: list[FailureRecord]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for record in records:
            if self.spent >= self.budget:
                results.append(("unknown", 0.0))
                continue
            log_excerpt = record.log_text[:3000]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": log_excerpt},
                ],
                max_tokens=20,
            )
            label = response.choices[0].message.content.strip().lower()
            if label not in FAMILIES:
                label = "unknown"
            self.spent += self._estimate_cost(response)
            results.append((label, 0.7))
        return results

    def _estimate_cost(self, response: object) -> float:
        usage = response.usage  # type: ignore[attr-defined]
        # gpt-4o-mini pricing ($/1M tokens): input $0.15, output $0.60
        return (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000


def classify_all(
    records: list[FailureRecord],
    openai_api_key: str | None = None,
) -> dict[str, list[FailureRecord]]:
    """Classify all records into family buckets.

    Rule-based first; LLM fallback for residuals if ``openai_api_key`` given.
    Unresolvable residuals land in ``"real_bug"`` as a safe default.
    """
    rule_clf = RuleBasedClassifier()
    by_family: dict[str, list[FailureRecord]] = {f: [] for f in FAMILIES}
    unknowns: list[FailureRecord] = []

    for record in records:
        family, _conf = rule_clf.classify(record)
        if family == "unknown":
            unknowns.append(record)
        else:
            by_family[family].append(record)

    if unknowns and openai_api_key:
        llm = LLMClassifier(openai_api_key)
        llm_results = llm.classify_batch(unknowns)
        for record, (family, _conf) in zip(unknowns, llm_results, strict=False):
            target = family if family in by_family else "real_bug"
            by_family[target].append(record)
        unknowns_after_llm = [
            r for r, (f, _) in zip(unknowns, llm_results, strict=False) if f == "unknown"
        ]
        unknowns = unknowns_after_llm
        print(f"LLM classified residuals; spent ~${llm.spent:.4f}")

    # Any remaining unknowns fall into real_bug
    by_family["real_bug"].extend(unknowns)

    return by_family
