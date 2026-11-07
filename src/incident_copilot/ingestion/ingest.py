"""Incremental, idempotent indexing of the data directory into documents + chunks.

Plain Python: no FastAPI, CLI, or Airflow imports, so a CLI, an API route, or an Airflow task
can all call `ingest_paths()`.

Per file:
  unchanged  same SHA-256 of the bytes (and same embedding model + chunker version): skipped,
             zero embedding calls
  new/changed  read -> redact -> chunk -> embed (outside any transaction) -> then ONE
             transaction upserts the document and replaces all of its chunks, so readers
             never see a mix of old and new chunks
  deleted    with prune=True, documents under the scanned roots whose file is gone are
             removed (chunks cascade). Nothing outside the scanned roots is touched.

A Postgres advisory lock allows one indexing run at a time; a second concurrent run raises
IndexingLockedError immediately instead of waiting.
"""

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from psycopg.types.json import Jsonb

from incident_copilot.chunking import CHUNKER_VERSION, CHUNKERS, DESCRIBERS, Chunk, DocumentInfo
from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.embeddings import Embedder
from incident_copilot.ingestion.sources import SourceFile, discover, is_under, normalize_root
from incident_copilot.redaction import redact

logger = logging.getLogger("incident_copilot.ingestion")

# Arbitrary constant identifying "an indexing run" in pg_advisory_lock's key space.
INGEST_LOCK_KEY = 72_010_020_011


class IndexingLockedError(RuntimeError):
    """Another indexing run holds the lock."""


@dataclass
class IngestSummary:
    dry_run: bool
    roots: list[str]
    run_id: str | None = None
    files_seen: int = 0
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: int = 0
    deleted: list[str] = field(default_factory=list)
    chunks_written: int = 0
    embed_calls: int = 0
    texts_embedded: int = 0
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Prepared:
    source: SourceFile
    file_hash: str
    info: DocumentInfo
    chunks: list[Chunk]


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prepare(source: SourceFile, data: bytes, file_hash: str) -> _Prepared:
    # Redact the whole file first, so no derived text (chunks, titles, metadata) can carry
    # a credential into the database or the embedding model.
    text = redact(data.decode("utf-8", errors="replace").replace("\r\n", "\n"))
    info = DESCRIBERS[source.source_type](text, source.path)
    chunks = CHUNKERS[source.source_type](text, source.path)
    return _Prepared(source, file_hash, info, chunks)


def _write_document(conn, prep: _Prepared, vectors, embedder_name: str, run_id) -> None:
    """Upsert one document and replace all its chunks, atomically."""
    src = prep.source
    doc_meta = {
        **prep.info.metadata,
        "pipelines": prep.info.pipelines,
        "embedding_model": embedder_name,
        "chunker_version": CHUNKER_VERSION,
    }
    with conn.transaction():
        doc_id = conn.execute(
            """
            INSERT INTO documents (path, source_type, title, pipeline_name, content_hash,
                                   metadata, last_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (path) DO UPDATE SET
                source_type = EXCLUDED.source_type, title = EXCLUDED.title,
                pipeline_name = EXCLUDED.pipeline_name, content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata, last_run_id = EXCLUDED.last_run_id,
                updated_at = now()
            RETURNING id
            """,
            (src.path, src.source_type, prep.info.title, prep.info.pipeline_name,
             prep.file_hash, Jsonb(doc_meta), run_id),
        ).fetchone()[0]
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, content, content_hash, embedding,
                                    ts_start, ts_end, section, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (doc_id, c.index, c.text, c.content_hash, vec, c.ts_start, c.ts_end,
                     c.section,
                     Jsonb({**c.metadata, "chunk_key": c.key(src.path),
                            "source_type": src.source_type}))
                    for c, vec in zip(prep.chunks, vectors, strict=True)
                ],
            )


def ingest_paths(
    paths: list[str] | None = None,
    *,
    embedder: Embedder,
    data_dir: str | Path | None = None,
    prune: bool = False,
    dry_run: bool = False,
    force: bool = False,
    trigger: str = "manual",
) -> IngestSummary:
    """Index `paths` (relative to the data directory; default: all of it).

    Raises UnsafePathError for paths outside the data directory and IndexingLockedError if
    another run is in progress.
    """
    started = time.perf_counter()
    base = Path(data_dir or get_settings().data_dir)
    roots = sorted({normalize_root(base, p) for p in (paths or ["."])})
    summary = IngestSummary(dry_run=dry_run, roots=roots)

    with connect(autocommit=True) as conn:
        locked = conn.execute("SELECT pg_try_advisory_lock(%s)", (INGEST_LOCK_KEY,)).fetchone()[0]
        if not locked:
            raise IndexingLockedError("another indexing run is in progress; try again later")
        try:
            run_id = None
            if not dry_run:
                run_id = conn.execute(
                    "INSERT INTO ingestion_runs (trigger) VALUES (%s) RETURNING id", (trigger,)
                ).fetchone()[0]
                summary.run_id = str(run_id)
            try:
                _run(conn, base, roots, embedder, summary, prune, dry_run, force, run_id)
            except Exception as exc:
                if run_id is not None:
                    conn.execute(
                        "UPDATE ingestion_runs SET status = 'failed', error = %s, "
                        "finished_at = now() WHERE id = %s",
                        (redact(f"{type(exc).__name__}: {exc}")[:1000], run_id),
                    )
                raise
            if run_id is not None:
                conn.execute(
                    "UPDATE ingestion_runs SET status = 'success', docs_seen = %s, "
                    "docs_changed = %s, finished_at = now() WHERE id = %s",
                    (summary.files_seen,
                     len(summary.added) + len(summary.updated) + len(summary.deleted), run_id),
                )
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (INGEST_LOCK_KEY,))

    summary.duration_s = round(time.perf_counter() - started, 3)
    return summary


def _run(conn, base, roots, embedder, summary, prune, dry_run, force, run_id) -> None:
    sources = discover(base, roots)
    summary.files_seen = len(sources)
    existing = {
        path: (content_hash, model, version)
        for path, content_hash, model, version in conn.execute(
            "SELECT path, content_hash, metadata->>'embedding_model', "
            "(metadata->>'chunker_version')::int FROM documents"
        ).fetchall()
    }

    for source in sources:
        data = source.abs_path.read_bytes()
        file_hash = file_sha256(data)
        known = existing.get(source.path)
        if (not force and known is not None
                and known == (file_hash, embedder.name, CHUNKER_VERSION)):
            summary.unchanged += 1
            continue

        prep = _prepare(source, data, file_hash)
        (summary.updated if known else summary.added).append(source.path)
        summary.chunks_written += len(prep.chunks)
        if dry_run:
            continue
        # Embed before opening the write transaction: a slow or failing model call never
        # leaves a document half-replaced.
        vectors = embedder.embed_documents([c.text for c in prep.chunks])
        summary.embed_calls += 1
        summary.texts_embedded += len(prep.chunks)
        _write_document(conn, prep, vectors, embedder.name, run_id)
        logger.info("%s %s (%d chunks)", "updated" if known else "added", source.path,
                    len(prep.chunks))

    if prune:
        found = {s.path for s in sources}
        gone = sorted(p for p in existing
                      if p not in found and any(is_under(p, r) for r in roots))
        summary.deleted = gone
        if gone and not dry_run:
            with conn.transaction():
                conn.execute("DELETE FROM documents WHERE path = ANY(%s)", (gone,))
            for p in gone:
                logger.info("deleted %s", p)
