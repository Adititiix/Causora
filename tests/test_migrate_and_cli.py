import shutil

import psycopg
import pytest

from incident_copilot.db_migrate import apply_migrations, migration_files
from incident_copilot.ingestion import cli
from incident_copilot.ingestion.ingest import INGEST_LOCK_KEY

pytestmark = pytest.mark.db


def test_migration_files_are_applied_in_order_and_idempotent(test_db_url):
    names = [f.name for f in migration_files()]
    assert names == sorted(names) and names[:2] == ["001_schema.sql", "002_chunk_content_hash.sql"]
    assert apply_migrations(test_db_url) == names
    assert apply_migrations(test_db_url) == names  # second time is a no-op
    with psycopg.connect(test_db_url) as conn:
        col = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'content_hash'"
        ).fetchone()
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchone()[0]
    assert col == ("text",)
    assert tables == 8  # migrations add columns, not tables


def test_migration_upgrades_a_database_that_only_has_001(test_db_url):
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        conn.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_hash")
    apply_migrations(test_db_url)
    with psycopg.connect(test_db_url) as conn:
        assert conn.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'content_hash'"
        ).fetchone()[0] == 1


@pytest.fixture
def corpus(dataset, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    return root


def run_cli(capsys, *args):
    code = cli.main(["index", "--embedder", "hashing", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_index_then_second_run_is_a_no_op(ingest_db, corpus, capsys):
    code, out, _ = run_cli(capsys, "--root", str(corpus))
    assert code == 0
    assert "added:          104" in out
    # The DSN is printed with its password masked (URL or key=value form).
    password = psycopg.conninfo.conninfo_to_dict(ingest_db)["password"]
    assert password not in out and "[REDACTED]" in out.splitlines()[0]
    code, out, _ = run_cli(capsys, "--root", str(corpus))
    assert code == 0
    assert "unchanged:      104" in out and "embed calls:    0 (0 texts)" in out


def test_cli_dry_run_lists_changes(ingest_db, corpus, capsys):
    code, out, _ = run_cli(capsys, "--root", str(corpus), "--path", "schemas", "--dry-run")
    assert code == 0
    assert "dry run" in out and "would be added: schemas/orders_raw.md" in out
    with psycopg.connect(ingest_db) as conn:
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_cli_reports_a_concurrent_run_clearly(ingest_db, corpus, capsys):
    with psycopg.connect(ingest_db, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (INGEST_LOCK_KEY,))
        code, _, err = run_cli(capsys, "--root", str(corpus))
        other.execute("SELECT pg_advisory_unlock(%s)", (INGEST_LOCK_KEY,))
    assert code == 2
    assert "another indexing run is in progress" in err


def test_cli_rejects_paths_outside_the_data_directory(ingest_db, corpus, capsys):
    code, _, err = run_cli(capsys, "--root", str(corpus), "--path", "../etc")
    assert code == 3
    assert "must not contain '..'" in err
