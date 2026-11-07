"""Read-only access to what is indexed, for the Knowledge Base explorer (GET /documents).

Nothing here writes to the database or the data directory, and nothing triggers ingestion.
Text leaves this module only through redact(): indexed chunks were already redacted when they
were stored, and an original file is redacted when it is read, so a value the indexer masked
is masked here too. Original files are read only from DATA_DIR, through the same path checks
as /ingest.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.ingestion.sources import UnsafePathError, normalize_root
from incident_copilot.redaction import redact

MAX_ORIGINAL_BYTES = 512 * 1024  # larger files are shown as indexed chunks only


class DocumentNotFoundError(LookupError):
    """No indexed document has this path."""


@dataclass(frozen=True)
class DocumentSummary:
    path: str
    source_type: str
    title: str | None
    pipeline: str | None
    sections: tuple[str, ...]
    chunk_count: int
    content_hash: str
    updated_at: str | None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IndexedChunk:
    chunk_index: int
    section: str | None
    text: str
    ts_start: str | None = None
    ts_end: str | None = None


@dataclass(frozen=True)
class DocumentDetail:
    summary: DocumentSummary
    chunks: tuple[IndexedChunk, ...]
    # The file under DATA_DIR, redacted; None when it is missing, too large or unreadable.
    original: str | None
    original_note: str


def _sections(meta: dict) -> tuple[str, ...]:
    raw = meta.get("sections")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [raw]
    return tuple(str(s) for s in raw) if isinstance(raw, list) else ()


def _public_metadata(meta: dict) -> dict:
    """Metadata as shown to users: strings redacted; the section list is reported separately."""
    out = {}
    for key, value in sorted(meta.items()):
        if key == "sections":
            continue
        out[key] = redact(value) if isinstance(value, str) else value
    return out


def _summary(row) -> DocumentSummary:
    path, source_type, title, pipeline, content_hash, updated, meta, n = row
    meta = meta or {}
    return DocumentSummary(path=path, source_type=source_type,
                           title=redact(title) if title else None, pipeline=pipeline,
                           sections=_sections(meta), chunk_count=int(n or 0),
                           content_hash=content_hash,
                           updated_at=updated.isoformat() if updated else None,
                           metadata=_public_metadata(meta))


_SUMMARY_SQL = """
    SELECT d.path, d.source_type, d.title, d.pipeline_name, d.content_hash, d.updated_at,
           d.metadata, count(c.id)
    FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
"""


def list_documents() -> list[DocumentSummary]:
    with connect() as conn:
        rows = conn.execute(_SUMMARY_SQL + " GROUP BY d.id ORDER BY d.source_type, d.path"
                            ).fetchall()
    return [_summary(r) for r in rows]


def read_original(data_dir: Path, path: str) -> tuple[str | None, str]:
    """(redacted file text or None, note saying which). Read-only, inside DATA_DIR only."""
    try:
        clean = normalize_root(data_dir, path)
    except UnsafePathError:
        return None, "The source path is not inside the data directory; showing indexed chunks."
    file = data_dir / clean
    if not file.is_file():
        return None, ("The original file is not available to the API (for example it was "
                      "removed after indexing); showing the indexed chunks.")
    if file.stat().st_size > MAX_ORIGINAL_BYTES:
        return None, "The original file is too large to show; showing the indexed chunks."
    try:
        text = file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "The original file could not be read; showing the indexed chunks."
    masked = redact(text)
    note = "Original source file from the data directory."
    if masked != text:
        note += " Sensitive values are masked as [REDACTED], as they were when indexed."
    return masked, note


def get_document(path: str, *, with_original: bool = True) -> DocumentDetail:
    with connect() as conn:
        row = conn.execute(_SUMMARY_SQL + " WHERE d.path = %s GROUP BY d.id", (path,)
                           ).fetchone()
        if row is None:
            raise DocumentNotFoundError(path)
        chunks = conn.execute(
            "SELECT c.chunk_index, c.section, c.content, c.ts_start, c.ts_end FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE d.path = %s ORDER BY c.chunk_index",
            (path,)).fetchall()
    original, note = (read_original(Path(get_settings().data_dir), path) if with_original
                      else (None, "Original file not requested."))
    return DocumentDetail(
        summary=_summary(row),
        chunks=tuple(IndexedChunk(i, s, redact(t or ""), a.isoformat() if a else None,
                                  b.isoformat() if b else None)
                     for i, s, t, a, b in chunks),
        original=original, original_note=note)
