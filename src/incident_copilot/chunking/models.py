"""Data types shared by the chunkers. Pure data, no I/O."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


def normalize_text(text: str) -> str:
    """Canonical form used for hashing: LF line endings, no trailing spaces, trimmed."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit. `text` is exactly what gets embedded and stored."""

    index: int
    text: str
    section: str | None = None
    ts_start: datetime | None = None
    ts_end: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return sha256_text(self.text)

    def key(self, path: str) -> str:
        """Stable identity of a chunk: document path + position."""
        return f"{path}#{self.index}"


@dataclass(frozen=True)
class DocumentInfo:
    """Document-level facts extracted from a file (stored on the `documents` row)."""

    title: str | None
    pipeline_name: str | None
    # All pipelines the document is about (runbooks can apply to several).
    pipelines: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
