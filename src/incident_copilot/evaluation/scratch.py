"""Throwaway indexes for evaluation, so experiments never touch the production database or
the default chunking configuration.

    with temporary_index(embedder=e, window=4) as idx:
        ... VectorRetriever(e).retrieve(...)   # reads idx.name, not the main database

Inside the block the app's settings point at a temporary database `<db>_<label>_w<window>`
indexed with ERROR_CONTEXT_ENTRIES=window. Afterwards (even on error) the window and the
DATABASE_URL setting are restored and the temporary database is dropped.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from incident_copilot.chunking import logs as log_chunker
from incident_copilot.config import get_settings
from incident_copilot.db_migrate import apply_migrations
from incident_copilot.embeddings import Embedder
from incident_copilot.ingestion import IngestSummary, ingest_paths


@dataclass(frozen=True)
class TemporaryIndex:
    name: str
    url: str
    window: int
    summary: IngestSummary


def _admin(base_url: str) -> psycopg.Connection:
    return psycopg.connect(make_conninfo(base_url, dbname="postgres"), autocommit=True)


def _set_database_url(value: str | None) -> None:
    # Evaluation-only: re-point this process's settings at another database.
    if value is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = value
    get_settings.cache_clear()


@contextmanager
def temporary_index(*, embedder: Embedder, window: int | None = None,
                    data_dir: str | None = None, label: str = "eval") -> Iterator[TemporaryIndex]:
    settings = get_settings()
    base = settings.database_url
    data_dir = data_dir or settings.data_dir
    original_window = log_chunker.ERROR_CONTEXT_ENTRIES
    window = original_window if window is None else window
    name = f"{conninfo_to_dict(base).get('dbname', 'incident_copilot')}_{label}_w{window}"
    url = make_conninfo(base, dbname=name)
    previous_env = os.environ.get("DATABASE_URL")

    with _admin(base) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{name}"')
    try:
        apply_migrations(url)
        log_chunker.ERROR_CONTEXT_ENTRIES = window
        _set_database_url(url)
        summary = ingest_paths(embedder=embedder, data_dir=data_dir)
        yield TemporaryIndex(name=name, url=url, window=window, summary=summary)
    finally:
        log_chunker.ERROR_CONTEXT_ENTRIES = original_window
        _set_database_url(previous_env)
        with _admin(base) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
