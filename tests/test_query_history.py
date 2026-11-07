"""Query logging survives re-indexing and pruning, and never points at replacement content."""

import shutil

import psycopg
import pytest

from fakes import CountingEmbedder
from incident_copilot.embeddings import HashingEmbedder
from incident_copilot.generation import FakeLLMProvider, answer_question
from incident_copilot.ingestion import ingest_paths
from incident_copilot.observability import log_query, query_history
from incident_copilot.retrieval import Filters, VectorRetriever

pytestmark = pytest.mark.db

RUNBOOK = "runbooks/column-renamed-upstream.md"


@pytest.fixture
def corpus(ingest_db, dataset, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    ingest_paths(embedder=CountingEmbedder(), data_dir=root)
    return root


def ask_and_log(question="KeyError customer_tier column renamed upstream transform", k=4):
    outcome = answer_question(
        question, retriever=VectorRetriever(HashingEmbedder()),
        llm_factory=lambda: FakeLLMProvider([{
            "insufficient_evidence": False, "summary": "s",
            "evidence": [{"claim": "c", "evidence_ids": ["E1"]}], "inference": [],
            "next_steps": []}]),
        k=k, min_similarity=0.0, filters=Filters(source_types=("runbook", "schema")))
    return outcome, log_query(outcome, retriever_name="vector", request={"top_k": k})


def q(url, sql, params=None):
    with psycopg.connect(url) as conn:
        return conn.execute(sql, params).fetchall()


def test_a_query_row_and_snapshot_rows_are_written(corpus, ingest_db):
    outcome, qid = ask_and_log()
    (question, answer, abstained, config, model, latency, meta) = q(
        ingest_db, "SELECT question, answer, abstained, retrieval_config, model, latency_ms, "
                   "metadata FROM queries WHERE id = %s", (qid,))[0]
    assert answer == "s" and abstained is False and config == "vector"
    assert model.startswith("fake-llm") and latency is not None
    assert meta["status"] == "answered" and meta["request"] == {"top_k": 4}
    history = query_history(qid)
    assert [h["evidence_id"] for h in history] == ["E1", "E2", "E3", "E4"]
    assert [h["used_in_answer"] for h in history] == [True, False, False, False]
    for h, hit in zip(history, outcome.hits, strict=True):
        assert (h["path"], h["content"], h["chunk_id"]) == (hit.path, hit.text, hit.chunk_id)
        assert h["is_current"] is True


def test_reindexing_a_document_keeps_history_and_unlinks_replaced_chunks(corpus, ingest_db):
    outcome, qid = ask_and_log()
    assert RUNBOOK in {h.path for h in outcome.hits}
    before = {h["evidence_id"]: h for h in query_history(qid)}

    path = corpus / RUNBOOK
    path.write_text(path.read_text(encoding="utf-8").replace("Update the transform",
                                                             "Patch the transform"),
                    encoding="utf-8")
    assert ingest_paths(embedder=CountingEmbedder(), data_dir=corpus).updated == [RUNBOOK]

    after = {h["evidence_id"]: h for h in query_history(qid)}
    assert after.keys() == before.keys()  # nothing deleted
    for eid, h in after.items():
        assert h["content"] == before[eid]["content"]  # snapshot is immutable
        if h["path"] == RUNBOOK:
            assert h["chunk_id"] is None and h["is_current"] is False
        else:
            assert h["chunk_id"] == before[eid]["chunk_id"] and h["is_current"] is True
    # The query itself is untouched.
    assert q(ingest_db, "SELECT answer FROM queries WHERE id = %s", (qid,)) == [("s",)]


def test_no_history_row_ever_links_to_different_content(corpus, ingest_db):
    ask_and_log()
    path = corpus / RUNBOOK
    path.write_text(path.read_text(encoding="utf-8") + "- extra\n", encoding="utf-8")
    ingest_paths(embedder=CountingEmbedder(), data_dir=corpus)
    mismatched = q(ingest_db, """
        SELECT count(*) FROM retrievals r JOIN chunks c ON c.id = r.chunk_id
        WHERE c.content_hash IS DISTINCT FROM r.chunk_content_hash
           OR c.content IS DISTINCT FROM r.content""")
    assert mismatched == [(0,)]


def test_pruning_a_document_keeps_history(corpus, ingest_db):
    _, qid = ask_and_log()
    n = len(query_history(qid))
    (corpus / RUNBOOK).unlink()
    assert ingest_paths(embedder=CountingEmbedder(), data_dir=corpus, prune=True).deleted == [
        RUNBOOK]
    history = query_history(qid)
    assert len(history) == n
    gone = [h for h in history if h["path"] == RUNBOOK]
    assert gone and all(h["chunk_id"] is None and h["content"] for h in gone)


def test_deleting_a_query_still_removes_its_retrievals(corpus, ingest_db):
    _, qid = ask_and_log()
    with psycopg.connect(ingest_db) as conn:
        conn.execute("DELETE FROM queries WHERE id = %s", (qid,))
    assert q(ingest_db, "SELECT count(*) FROM retrievals WHERE query_id = %s", (qid,)) == [(0,)]


def test_retrieval_fk_is_set_null(test_db_url):
    rule = q(test_db_url, "SELECT confdeltype FROM pg_constraint "
                          "WHERE conname = 'retrievals_chunk_id_fkey'")
    assert rule == [("n",)]
