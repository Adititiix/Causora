"""db.shared_connection(): connect() reuses one connection inside the block, with the same
commit/rollback behaviour per connect() block as a fresh connection."""

import psycopg
import pytest

from incident_copilot.db import connect, shared_connection

pytestmark = pytest.mark.db


def _pid(conn) -> int:
    return conn.execute("SELECT pg_backend_pid()").fetchone()[0]


def test_outside_the_block_every_connect_is_a_new_connection(test_db_url, use_database):
    use_database(test_db_url)
    with connect() as a:
        first = _pid(a)
    with connect() as b:
        assert _pid(b) != first


def test_inside_the_block_connect_reuses_one_connection_and_closes_it(test_db_url,
                                                                      use_database):
    use_database(test_db_url)
    with shared_connection():
        with connect() as a:
            first = _pid(a)
        with connect() as b:
            assert _pid(b) == first
        with connect(autocommit=True) as c:  # autocommit callers keep their own
            assert _pid(c) != first
    assert a.closed


def test_each_block_commits_or_rolls_back_on_its_own(test_db_url, use_database):
    use_database(test_db_url)
    with psycopg.connect(test_db_url, autocommit=True) as admin:
        admin.execute("CREATE TABLE shared_conn_probe (v int)")
        try:
            with shared_connection():
                with connect() as conn:
                    conn.execute("INSERT INTO shared_conn_probe VALUES (1)")
                # Committed: another session sees it while the shared one is still open.
                assert admin.execute("SELECT count(*) FROM shared_conn_probe").fetchone()[0] == 1
                with pytest.raises(RuntimeError), connect() as conn:
                    conn.execute("INSERT INTO shared_conn_probe VALUES (2)")
                    raise RuntimeError("boom")
                with connect() as conn:  # still usable after the rollback
                    assert conn.execute("SELECT array_agg(v) FROM shared_conn_probe"
                                        ).fetchone()[0] == [1]
                with pytest.raises(psycopg.errors.DivisionByZero), connect() as conn:
                    conn.execute("SELECT 1 / 0")
                with connect() as conn:  # a failed statement does not poison later blocks
                    assert conn.execute("SELECT 2").fetchone()[0] == 2
        finally:
            admin.execute("DROP TABLE shared_conn_probe")


def test_a_broken_connection_is_replaced(test_db_url, use_database):
    use_database(test_db_url)
    with shared_connection(), psycopg.connect(test_db_url, autocommit=True) as admin:
        with connect() as conn:
            first = _pid(conn)
        admin.execute("SELECT pg_terminate_backend(%s)", (first,))
        with pytest.raises(psycopg.OperationalError), connect() as conn:
            conn.execute("SELECT 1")
        with connect() as conn:
            assert _pid(conn) != first
