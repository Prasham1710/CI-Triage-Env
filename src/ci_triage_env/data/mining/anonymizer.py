"""Best-effort anonymizer for mined GitHub Actions logs.

Replaces project-specific identifiers (usernames, emails, hostnames, paths,
IPv4s, SHAs) with neutral tokens. Long random IDs (full + short SHAs) are
hashed to a stable short token so distinct IDs remain distinguishable in the
output without leaking the original. This is *best-effort* — the README's
Limitations section warns that perfect anonymization is impossible.

Order matters: emails must be replaced before ``@USER`` matches the local
part, and full 40-char SHAs must be replaced before the more permissive
8-char hex pattern would clobber a substring.
"""

from __future__ import annotations

import hashlib
import re
from re import Match


def hash_short(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def _sub_full_sha(m: Match[str]) -> str:
    return f"sha-{hash_short(m.group())}"


def _sub_short_hex(m: Match[str]) -> str:
    return f"hex-{hash_short(m.group())}"


# Compiled in-order; each tuple is (pattern, replacement). Replacement may be a
# callable for hashing or a literal string for redaction.
_PATTERNS: list[tuple[re.Pattern[str], object]] = [
    # Emails first (before the @USER pattern eats the local part).
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL"),
    # Full SHAs (40 hex) before short hex (8 hex) so the longer match wins.
    (re.compile(r"\b[a-f0-9]{40}\b"), _sub_full_sha),
    # User home dirs.
    (re.compile(r"/(?:home|Users)/[^/\s]+/"), "/PATH/USER/"),
    # GitHub @-mentions.
    (re.compile(r"@[A-Za-z0-9][A-Za-z0-9_\-]{0,38}\b"), "@USER"),
    # IPv4. Run before short-hex so 192.168.1.1 isn't mistaken for hex tokens.
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "IP"),
    # Short hex tokens (8 chars) — order after the long-SHA pattern. The
    # negative lookbehinds keep this from re-matching the 8-char hash inside
    # an already-substituted ``sha-<8hex>`` or ``hex-<8hex>`` token.
    (re.compile(r"(?<!sha-)(?<!hex-)\b[a-f0-9]{8}\b"), _sub_short_hex),
    # Common internal-hostname prefixes.
    (re.compile(r"\b(?:corp|internal)\.[A-Za-z0-9.\-_]+"), "HOST"),
]


def anonymize(text: str) -> str:
    """Return ``text`` with project-specific identifiers replaced by tokens.

    Idempotent on the documented patterns: applying ``anonymize`` twice gives
    the same string (the replacement tokens themselves don't match any
    pattern).
    """
    out = text
    for pat, repl in _PATTERNS:
        if callable(repl):
            out = pat.sub(repl, out)  # type: ignore[arg-type]
        else:
            out = pat.sub(repl, out)
    return out
