from pathlib import Path

import psycopg
import pytest
from pgvector import Vector

pytestmark = pytest.mark.db

DIM = 384


def unit(i: int) -> Vector:
    """A 384-dim vector with a single 1.0 at position i."""
    v = [0.0] * DIM
    v[i] = 1.0
    return Vector(v)


def add_document(conn, path="runbooks/x.md", source_type="runbook") -> str:
    return conn.execute(
        "INSERT INTO documents (path, source_type, content_hash) VALUES (%s, %s, 'h') "
        "RETURNING id",
        (path, source_type),
    ).fetchone()[0]


def add_chunk(conn, doc_id, idx, content, embedding=None):
    conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
        "VALUES (%s, %s, %s, %s)",
        (doc_id, idx, content, embedding),
    )


def test_expected_tables_exist(db_conn):
    rows = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    assert {r[0] for r in rows} == {
        "ingestion_runs", "documents", "chunks", "queries", "retrievals",
        "eval_questions", "eval_runs", "eval_results",
    }


def test_schema_is_idempotent(db_conn):
    sql = (Path(__file__).resolve().parents[1] / "db" / "init" / "001_schema.sql").read_text()
    db_conn.execute(sql)  # applying a second time must not error


def test_vector_search_returns_nearest_chunk(db_conn):
    doc = add_document(db_conn)
    add_chunk(db_conn, doc, 0, "about disks", unit(0))
    add_chunk(db_conn, doc, 1, "about memory", unit(1))
    add_chunk(db_conn, doc, 2, "about network", unit(2))
    row = db_conn.execute(
        "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 1", (unit(1),)
    ).fetchone()
    assert row[0] == "about memory"


def test_full_text_search_matches_error_names_and_identifiers(db_conn):
    doc = add_document(db_conn)
    add_chunk(db_conn, doc, 0, "KeyError: 'customer_tier' raised in apply_transforms")
    add_chunk(db_conn, doc, 1, 'violates unique constraint "orders_fact_pkey"')
    add_chunk(db_conn, doc, 2, "Memory usage 3.8 GiB of 4.0 GiB limit")

    def search(term):
        return [
            r[0]
            for r in db_conn.execute(
                "SELECT content FROM chunks WHERE tsv @@ plainto_tsquery('simple', %s)", (term,)
            ).fetchall()
        ]

    assert len(search("KeyError")) == 1
    assert len(search("customer_tier")) == 1
    assert len(search("orders_fact_pkey")) == 1
    assert search("nonexistent_token") == []


def test_chunks_can_exist_before_embedding(db_conn):
    doc = add_document(db_conn)
    add_chunk(db_conn, doc, 0, "not embedded yet", None)
    assert db_conn.execute("SELECT embedding FROM chunks").fetchone()[0] is None


def test_wrong_embedding_dimension_is_rejected(db_conn):
    doc = add_document(db_conn)
    with pytest.raises(psycopg.Error):
        add_chunk(db_conn, doc, 0, "too short", Vector([0.1] * 10))


def test_invalid_source_type_is_rejected(db_conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        add_document(db_conn, source_type="memo")


def test_document_path_is_unique(db_conn):
    add_document(db_conn, "runbooks/dup.md")
    with pytest.raises(psycopg.errors.UniqueViolation):
        add_document(db_conn, "runbooks/dup.md")


def test_deleting_a_document_cascades_to_chunks(db_conn):
    doc = add_document(db_conn)
    add_chunk(db_conn, doc, 0, "will disappear", unit(0))
    db_conn.execute("DELETE FROM documents WHERE id = %s", (doc,))
    assert db_conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_retrieval_log_links_query_and_chunk(db_conn):
    doc = add_document(db_conn)
    add_chunk(db_conn, doc, 0, "evidence", unit(0))
    chunk_id = db_conn.execute("SELECT id FROM chunks").fetchone()[0]
    qid = db_conn.execute(
        "INSERT INTO queries (question, abstained) VALUES ('why?', false) RETURNING id"
    ).fetchone()[0]
    db_conn.execute(
        "INSERT INTO retrievals (query_id, chunk_id, rank, keyword_score, vector_score) "
        "VALUES (%s, %s, 1, 0.4, 0.9)",
        (qid, chunk_id),
    )
    assert db_conn.execute("SELECT count(*) FROM retrievals").fetchone()[0] == 1
