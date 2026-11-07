"""Remove credentials from text before it is embedded, stored, or logged.

Covers: credentials in URLs (`scheme://user:pass@host`), `key=value` secrets (password,
token, api_key, secret, ...), JSON-style `"password": "..."`, bearer tokens, and AWS access
key ids. Patterns are deliberately conservative so ordinary prose is left untouched (a test
checks that redacting the synthetic corpus changes only the files flagged as containing
fake secrets).
"""

import re

MASK = "[REDACTED]"

_SECRET_KEYS = (
    r"password|passwd|pwd|secret|client_secret|token|access_token|refresh_token|auth_token"
    r"|api[_-]?key|aws_secret_access_key"
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # postgresql://user:pass@host -> postgresql://user:[REDACTED]@host
    (re.compile(r"(?P<pre>[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:)[^\s@/]+@"), rf"\g<pre>{MASK}@"),
    # password=..., token=..., api_key=...  (value runs to whitespace or a separator)
    (
        re.compile(rf"(?i)\b(?P<key>{_SECRET_KEYS})(?P<sep>\s*=\s*)(?P<val>\"[^\"]*\"|'[^']*'|[^\s,;&]+)"),
        rf"\g<key>\g<sep>{MASK}",
    ),
    # "password": "..."  (JSON / YAML-ish with quoted value)
    (
        re.compile(rf"(?i)(?P<key>[\"']?\b(?:{_SECRET_KEYS})[\"']?\s*:\s*)(?P<val>\"[^\"]*\"|'[^']*')"),
        rf'\g<key>"{MASK}"',
    ),
    # Authorization: Bearer abc.def.ghi
    (re.compile(r"(?i)\b(?P<pre>bearer\s+)[A-Za-z0-9\-._~+/]+=*"), rf"\g<pre>{MASK}"),
    # AWS access key ids
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), MASK),
]


def redact(text: str) -> str:
    """Return `text` with every recognised credential replaced by [REDACTED]."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
