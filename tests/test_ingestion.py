"""Incremental indexing against a real (throwaway) Postgres database."""

import shutil

import psycopg
import pytest

from fakes import CountingEmbedder, WrongDimensionEmbedder
from incident_copilot.ingestion import IndexingLockedError, ingest_paths
from incident_copilot.ingestion.ingest import INGEST_LOCK_KEY

pytestmark = pytest.mark.db

RUNBOOK = "runbooks/column-renamed-upstream.md"
LOG = "logs/orders_etl/2026-09-04__transform_orders.log"


@pytest.fixture
def corpus(dataset, tmp_path):
    """A private, writable copy of the generated dataset."""
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    return root


def q(url, sql, params=None):
    with psycopg.connect(url) as conn:
        return conn.execute(sql, params).fetchall()


def chunk_ids(url):
    return dict(q(url, "SELECT c.id, d.path FROM chunks c "
                       "JOIN documents d ON d.id = c.document_id"))


def doc_ids(url):
    return dict(q(url, "SELECT path, id FROM documents"))


def test_first_run_indexes_every_source_type(ingest_db, corpus):
    emb = CountingEmbedder()
    s = ingest_paths(embedder=emb, data_dir=corpus)
    assert (len(s.added), len(s.updated), s.unchanged, len(s.deleted)) == (104, 0, 0, 0)
    assert s.embed_calls == emb.document_calls == 104
    assert s.texts_embedded == s.chunks_written == len(emb.texts)

    per_type = dict(q(ingest_db, "SELECT source_type, count(*) FROM documents GROUP BY 1"))
    assert per_type == {"runbook": 18, "log": 60, "schema": 12, "incident": 14}
    assert q(ingest_db, "SELECT count(*) FROM chunks")[0][0] == s.chunks_written
    missing = q(ingest_db, "SELECT count(*) FROM chunks WHERE embedding IS NULL "
                           "OR content_hash IS NULL OR metadata->>'chunk_key' IS NULL")
    assert missing[0][0] == 0
    run = q(ingest_db, "SELECT status, docs_seen, docs_changed FROM ingestion_runs")
    assert run == [("success", 104, 104)]


def test_stored_metadata_for_logs_and_runbooks(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    rows = q(ingest_db, """
        SELECT c.chunk_index, c.section, c.ts_start, c.ts_end, c.metadata, d.pipeline_name,
               d.title
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.path = %s ORDER BY c.chunk_index""", (LOG,))
    assert [r[1] for r in rows] == ["run summary", "error context"]
    for _, _, ts_start, ts_end, meta, pipeline, title in rows:
        assert ts_start is not None and ts_start <= ts_end
        assert meta["task"] == "transform_orders" and meta["status"] == "FAILED"
        assert meta["chunk_key"].startswith(LOG + "#")
        assert pipeline == "orders_etl"
        assert title == "orders_etl / transform_orders / 2026-09-04"
    doc_meta = q(ingest_db, "SELECT metadata FROM documents WHERE path = %s", (RUNBOOK,))[0][0]
    assert doc_meta["pipelines"] == ["orders_etl", "customer_360"]
    assert doc_meta["embedding_model"] == "hashing-bow-384"


def test_second_run_changes_nothing_and_embeds_nothing(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    docs_before, chunks_before = doc_ids(ingest_db), chunk_ids(ingest_db)

    emb = CountingEmbedder()
    s = ingest_paths(embedder=emb, data_dir=corpus)
    assert (len(s.added), len(s.updated), s.unchanged, len(s.deleted)) == (0, 0, 104, 0)
    assert s.embed_calls == 0 and emb.document_calls == 0 and s.chunks_written == 0
    assert doc_ids(ingest_db) == docs_before
    assert chunk_ids(ingest_db) == chunks_before


def test_modified_file_replaces_only_that_documents_chunks(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    docs_before, chunks_before = doc_ids(ingest_db), chunk_ids(ingest_db)

    path = corpus / RUNBOOK
    path.write_text(path.read_text(encoding="utf-8") + "- Page the upstream owner.\n",
                    encoding="utf-8")
    emb = CountingEmbedder()
    s = ingest_paths(embedder=emb, data_dir=corpus)
    assert s.updated == [RUNBOOK] and not s.added and s.unchanged == 103
    assert emb.document_calls == 1

    chunks_after = chunk_ids(ingest_db)
    assert doc_ids(ingest_db) == docs_before  # document id is stable
    others_before = {cid for cid, p in chunks_before.items() if p != RUNBOOK}
    others_after = {cid for cid, p in chunks_after.items() if p != RUNBOOK}
    assert others_before == others_after
    old = {cid for cid, p in chunks_before.items() if p == RUNBOOK}
    new = {cid for cid, p in chunks_after.items() if p == RUNBOOK}
    assert new and not (old & new)
    text = q(ingest_db, "SELECT string_agg(content, ' ') FROM chunks c JOIN documents d "
                        "ON d.id = c.document_id WHERE d.path = %s", (RUNBOOK,))[0][0]
    assert "Page the upstream owner." in text


def test_failed_write_leaves_the_old_chunks_in_place(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    before = q(ingest_db, "SELECT d.content_hash, array_agg(c.id ORDER BY c.id) FROM documents d "
                          "JOIN chunks c ON c.document_id = d.id WHERE d.path = %s "
                          "GROUP BY 1", (RUNBOOK,))
    path = corpus / RUNBOOK
    path.write_text(path.read_text(encoding="utf-8") + "- new line\n", encoding="utf-8")

    with pytest.raises(psycopg.Error):  # vector(384) rejects the 10-dim vectors
        ingest_paths(embedder=WrongDimensionEmbedder(), data_dir=corpus)
    after = q(ingest_db, "SELECT d.content_hash, array_agg(c.id ORDER BY c.id) FROM documents d "
                         "JOIN chunks c ON c.document_id = d.id WHERE d.path = %s "
                         "GROUP BY 1", (RUNBOOK,))
    assert after == before
    statuses = [r[0] for r in q(ingest_db, "SELECT status FROM ingestion_runs ORDER BY started_at")]
    assert statuses == ["success", "failed"]


def test_new_file_under_extra_is_indexed_as_its_type(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    extra = corpus / "extra" / "runbooks"
    extra.mkdir(parents=True, exist_ok=True)
    (extra / "my-runbook.md").write_text("# Runbook: Mine\n\n## Symptoms\n\n- It broke.\n",
                                         encoding="utf-8")
    s = ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    assert s.added == ["extra/runbooks/my-runbook.md"]
    assert q(ingest_db, "SELECT source_type FROM documents WHERE path = %s",
             ("extra/runbooks/my-runbook.md",)) == [("runbook",)]


def test_deleted_file_needs_prune_and_dry_run_only_reports(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    (corpus / RUNBOOK).unlink()

    s = ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    assert s.deleted == [] and s.unchanged == 103
    assert len(doc_ids(ingest_db)) == 104

    s = ingest_paths(embedder=CountingEmbedder(), data_dir=corpus, prune=True, dry_run=True)
    assert s.deleted == [RUNBOOK]
    assert RUNBOOK in doc_ids(ingest_db)

    chunks_before = q(ingest_db, "SELECT count(*) FROM chunks")[0][0]
    s = ingest_paths(embedder=CountingEmbedder(), data_dir=corpus, prune=True)
    assert s.deleted == [RUNBOOK]
    assert RUNBOOK not in doc_ids(ingest_db)
    assert q(ingest_db, "SELECT count(*) FROM chunks")[0][0] == chunks_before - 6  # cascaded


def test_prune_only_touches_the_scanned_roots(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    (corpus / RUNBOOK).unlink()
    (corpus / LOG).unlink()
    s = ingest_paths(["runbooks"], embedder=CountingEmbedder(), data_dir=corpus, prune=True)
    assert s.deleted == [RUNBOOK]
    assert s.files_seen == 17
    docs = doc_ids(ingest_db)
    assert LOG in docs and RUNBOOK not in docs


def test_dry_run_writes_nothing_and_embeds_nothing(ingest_db, corpus):
    emb = CountingEmbedder()
    s = ingest_paths(embedder=emb, data_dir=corpus, dry_run=True)
    assert len(s.added) == 104 and s.chunks_written > 0
    assert s.embed_calls == 0 and emb.document_calls == 0 and s.run_id is None
    assert q(ingest_db, "SELECT count(*) FROM documents")[0][0] == 0
    assert q(ingest_db, "SELECT count(*) FROM ingestion_runs")[0][0] == 0


def test_no_fake_secret_reaches_the_database(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    assert q(ingest_db, "SELECT count(*) FROM chunks WHERE content ILIKE '%%FAKE-%%' "
                        "OR metadata::text ILIKE '%%FAKE-%%'")[0][0] == 0
    assert q(ingest_db, "SELECT count(*) FROM documents WHERE metadata::text ILIKE '%%FAKE-%%' "
                        "OR coalesce(title, '') ILIKE '%%FAKE-%%'")[0][0] == 0
    redacted_docs = q(ingest_db, "SELECT count(DISTINCT document_id) FROM chunks "
                                 "WHERE content LIKE '%%[REDACTED]%%'")[0][0]
    assert redacted_docs == 8


def test_embedded_text_is_redacted(ingest_db, corpus):
    emb = CountingEmbedder()
    ingest_paths(embedder=emb, data_dir=corpus)
    assert emb.texts and not any("FAKE-" in t for t in emb.texts)


def test_a_concurrent_run_is_rejected(ingest_db, corpus):
    with psycopg.connect(ingest_db, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (INGEST_LOCK_KEY,))
        emb = CountingEmbedder()
        with pytest.raises(IndexingLockedError, match="in progress"):
            ingest_paths(embedder=emb, data_dir=corpus)
        assert emb.document_calls == 0
        other.execute("SELECT pg_advisory_unlock(%s)", (INGEST_LOCK_KEY,))
    assert len(ingest_paths(embedder=CountingEmbedder(), data_dir=corpus).added) == 104


def test_changing_the_embedding_model_re_embeds_everything(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    s = ingest_paths(embedder=CountingEmbedder(name="another-model"), data_dir=corpus)
    assert len(s.updated) == 104 and s.embed_calls == 104


def test_force_re_embeds_unchanged_files(ingest_db, corpus):
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    s = ingest_paths(["schemas"], embedder=CountingEmbedder(), data_dir=corpus, force=True)
    assert len(s.updated) == 12 and s.embed_calls == 12
