"""Redaction utility (Section 3.4: "never persist secrets or raw sensitive
data into artifacts or logs").

Wordle itself carries no real PII/credentials, so this module has nothing
domain-specific to redact in this proxy target — it's exercised by a unit
test with synthetic secret-shaped strings. In the real banking environment
this is the seam that matters: every write to an artifact or a log passes
through `redact()` first, and the pattern list below would be extended with
account numbers, SSNs, auth tokens, and anything the target app's DOM might
leak (e.g. a session cookie visible in a debug panel).
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(r"\b(sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE)),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    out = text
    for _name, pattern in _PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


def redact_dict(data: dict) -> dict:
    """Recursively redact string values in a JSON-like dict/list structure."""

    def _walk(value):
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    return _walk(data)
