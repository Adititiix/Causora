"""Database access. One small module so connection handling lives in one place."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import psycopg
from pgvector.psycopg import register_vector

from incident_copilot.config import get_settings
from incident_copilot.redaction import redact

# Set inside shared_connection(): {"conn": psycopg.Connection | None}.
_shared: ContextVar[dict | None] = ContextVar("incident_copilot_shared_connection",
                                              default=None)


@contextmanager
def shared_connection() -> Iterator[None]:
    """Within this block, connect() reuses one connection instead of opening a new one each
    time (one /ask makes 5-7 short queries; a new connection costs ~28 ms in the API
    container, a reused one ~0.4 ms). Opened on first use, closed at the end. Each connect()
    block still commits on a clean exit and rolls back on an exception; a connection that
    broke is dropped and the next connect() opens a fresh one. autocommit=True callers always
    get their own connection."""
    state: dict = {"conn": None}
    token = _shared.set(state)
    try:
        yield
    finally:
        _shared.reset(token)
        if state["conn"] is not None:
            state["conn"].close()


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a connection with the pgvector type registered (or, inside shared_connection(),
    reuse that one).

    Commits on clean exit, rolls back on exception. With autocommit=True every statement
    commits on its own and `conn.transaction()` blocks are real transactions.
    """
    state = _shared.get()
    if state is None or autocommit:
        with psycopg.connect(get_settings().database_url, autocommit=autocommit) as conn:
            register_vector(conn)
            yield conn
        return
    conn = state["conn"]
    if conn is None or conn.closed:
        conn = psycopg.connect(get_settings().database_url)
        register_vector(conn)
        state["conn"] = conn
    try:
        yield conn
    except BaseException:
        try:
            conn.rollback()
        except psycopg.Error:  # the connection itself broke: start afresh next time
            conn.close()
            state["conn"] = None
        raise
    else:
        conn.commit()


def mask_dsn(dsn: str) -> str:
    """A connection string that is safe to print: the password is replaced."""
    return redact(dsn)


def db_status() -> dict:
    """Return facts that prove the schema is really there. Raises if DB is unreachable."""
    with connect() as conn:
        version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchone()
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()
        documents = conn.execute("SELECT count(*) FROM documents").fetchone()
        by_type = conn.execute(
            "SELECT d.source_type, count(DISTINCT d.id), count(c.id), count(c.embedding) "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.source_type ORDER BY d.source_type"
        ).fetchall()
    return {
        "pgvector_version": version[0] if version else None,
        "tables": tables[0],
        "chunks": chunks[0],
        "documents": documents[0],
        "by_source_type": {
            t: {"documents": d, "chunks": c, "embedded_chunks": e} for t, d, c, e in by_type
        },
    }
