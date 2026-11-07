"""Shared fixtures.

Database tests create a throwaway database (copilot_test_<random>), apply every
db/init/*.sql to it in name order, and drop it afterwards, so they never touch your dev data.
If Postgres is unreachable they are skipped, not failed; the skip reason says why.
"""

import uuid
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.conninfo import make_conninfo

from incident_copilot.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "db" / "init" / "001_schema.sql"
INIT_SQL = sorted((ROOT / "db" / "init").glob("*.sql"))


@pytest.fixture(scope="session")
def test_db_url():
    base = get_settings().database_url
    try:
        admin = psycopg.connect(
            make_conninfo(base, dbname="postgres"), autocommit=True, connect_timeout=3
        )
    except psycopg.OperationalError as exc:
        # The last line of a psycopg error names the real cause (e.g. password failed).
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        reason = lines[-1] if lines else ""
        pytest.skip(f"Postgres not reachable ({type(exc).__name__}: {reason}). "
                    "Start it with `docker compose up -d db` and check DATABASE_URL.")
    name = f"copilot_test_{uuid.uuid4().hex[:8]}"
    with admin:
        admin.execute(f'CREATE DATABASE "{name}"')
        url = make_conninfo(base, dbname=name)
        try:
            with psycopg.connect(url, autocommit=True) as conn:
                for sql in INIT_SQL:
                    conn.execute(sql.read_text(encoding="utf-8"))
            yield url
        finally:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def db_conn(test_db_url):
    """A connection whose work is rolled back after each test."""
    with psycopg.connect(test_db_url) as conn:
        register_vector(conn)
        yield conn
        conn.rollback()


@pytest.fixture
def use_database(monkeypatch):
    """Point the app's settings at a given database URL for the duration of one test."""

    def _use(url: str) -> None:
        monkeypatch.setenv("DATABASE_URL", url)
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _use
    get_settings.cache_clear()


@pytest.fixture
def ingest_db(test_db_url, use_database):
    """The test database, emptied before and after the test, with the app pointed at it.

    Ingestion commits for real (it cannot run inside a rolled-back test transaction), so
    tests that use it must leave the shared test database empty for the others.
    """

    def truncate():
        with psycopg.connect(test_db_url, autocommit=True) as conn:
            conn.execute("TRUNCATE ingestion_runs, documents, chunks, queries, retrievals "
                         "CASCADE")

    truncate()
    use_database(test_db_url)
    yield test_db_url
    truncate()


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """Freshly generated synthetic dataset, so tests never depend on `make data` having run."""
    import generate_synthetic_data as gen

    out = tmp_path_factory.mktemp("data")
    eval_out = out / "eval" / "questions.jsonl"
    result = gen.write_all(out, eval_out, gen.DEFAULT_SEED)
    return {"root": out, "eval": eval_out, "result": result, "gen": gen}
