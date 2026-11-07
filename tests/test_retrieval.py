"""Vector retrieval and the error-link retriever against a real (throwaway) Postgres."""

import shutil

import numpy as np
import pytest

from fakes import CountingEmbedder
from incident_copilot.embeddings import HashingEmbedder
from incident_copilot.ingestion import ingest_paths
from incident_copilot.retrieval import (
    ErrorLinkRetriever,
    Filters,
    Retriever,
    VectorRetriever,
    get_retriever,
)

pytestmark = pytest.mark.db

KEYERROR_LOG = "logs/orders_etl/2026-09-04__transform_orders.log"


@pytest.fixture
def indexed(ingest_db, dataset, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    ingest_paths(embedder=CountingEmbedder(), data_dir=root)
    return root


def test_retrievers_implement_the_protocol():
    e = HashingEmbedder()
    assert isinstance(VectorRetriever(e), Retriever)
    assert isinstance(ErrorLinkRetriever(e), Retriever)
    assert isinstance(get_retriever("vector", e), VectorRetriever)
    with pytest.raises(ValueError):
        get_retriever("bm25", e)


def test_vector_ranking_puts_the_matching_chunk_first(indexed):
    hits = VectorRetriever(HashingEmbedder()).retrieve(
        "KeyError customer_tier transform_orders orders_etl normalize_loyalty", 5)
    # The matching log is in the top 3 (alongside documents about the same rename, such as
    # INC-2041 and the column-renamed runbook) and ahead of unrelated documents.
    assert KEYERROR_LOG in [h.path for h in hits[:3]]
    unrelated = VectorRetriever(HashingEmbedder()).retrieve(
        "KeyError customer_tier transform_orders orders_etl normalize_loyalty", 300)
    rank = {h.path: h.rank for h in reversed(unrelated)}
    assert rank[KEYERROR_LOG] < rank["runbooks/disk-full.md"]


def test_scores_are_cosine_similarity_sorted_with_ranks(indexed):
    e = HashingEmbedder()
    question = "disk full no space left on device"
    hits = VectorRetriever(e).retrieve(question, 7)
    assert [h.rank for h in hits] == list(range(1, 8))
    sims = [h.scores["vector_similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)
    q = e.embed_query(question)
    for h in hits:
        assert h.scores["vector_similarity"] == pytest.approx(1 - h.scores["vector_distance"])
        assert h.score == h.scores["vector_similarity"]
        expected = float(np.dot(q, e.embed_documents([h.text])[0]))
        assert h.scores["vector_similarity"] == pytest.approx(expected, abs=1e-4)


def test_hits_carry_chunk_metadata(indexed):
    hits = VectorRetriever(HashingEmbedder()).retrieve("transform_orders failed", 10,
                                                       Filters(source_types=("log",)))
    assert hits and all(h.source_type == "log" for h in hits)
    for h in hits:
        assert h.ts_start is not None and h.ts_start <= h.ts_end
        assert h.metadata["task"] and h.metadata["chunk_key"] == f"{h.path}#{h.chunk_index}"
        assert len(h.content_hash) == 64 and h.chunk_id and h.document_id


def test_source_type_filter(indexed):
    hits = VectorRetriever(HashingEmbedder()).retrieve(
        "KeyError customer_tier", 10, Filters(source_types=("runbook", "schema")))
    assert len(hits) == 10
    assert {h.source_type for h in hits} <= {"runbook", "schema"}


def test_pipeline_filter_includes_runbooks_that_apply_to_it(indexed):
    hits = VectorRetriever(HashingEmbedder()).retrieve(
        "KeyError column renamed", 40, Filters(pipeline="customer_360"))
    assert hits
    for h in hits:
        assert (h.pipeline_name == "customer_360"
                or "customer_360" in h.metadata.get("pipelines", [])), h.path
    paths = {h.path for h in hits}
    assert "runbooks/column-renamed-upstream.md" in paths  # applies to customer_360
    assert not any(p.startswith("logs/orders_etl/") for p in paths)


def test_filters_that_match_nothing_return_no_hits(indexed):
    assert VectorRetriever(HashingEmbedder()).retrieve(
        "anything", 5, Filters(pipeline="no_such_pipeline")) == []


# ---- error link ---------------------------------------------------------------------------


@pytest.fixture
def tiny(ingest_db, tmp_path):
    """A corpus where the question shares words with the log but not with the runbook."""
    root = tmp_path / "data"
    (root / "logs" / "p1").mkdir(parents=True)
    (root / "runbooks").mkdir()
    (root / "logs" / "p1" / "2026-01-01__t1.log").write_text(
        "[2026-01-01T02:00:00.000+0000] {x.py:1} INFO - Starting attempt 1 of 1\n"
        "[2026-01-01T02:00:05.000+0000] {x.py:1} ERROR - Task failed with exception\n"
        "Traceback (most recent call last):\n"
        '  File "/dags/t1.py", line 3, in t1\n'
        "    ratio = a / b\n"
        "ZeroDivisionError: division by zero\n"
        "[2026-01-01T02:00:06.000+0000] {x.py:1} INFO - Marking task as FAILED.\n",
        encoding="utf-8")
    (root / "runbooks" / "division.md").write_text(
        "# Runbook: Arithmetic\n\n## Symptoms\n\n- ZeroDivisionError: division by zero\n",
        encoding="utf-8")
    (root / "runbooks" / "other.md").write_text(
        "# Runbook: Network\n\n## Symptoms\n\n- Connection reset by peer\n", encoding="utf-8")
    ingest_paths(embedder=HashingEmbedder(), data_dir=root)
    return root


QUESTION = "Why did t1 in p1 fail on 2026-01-01 attempt"


def test_error_link_adds_the_runbook_named_by_the_logs_exception(tiny):
    e = HashingEmbedder()
    plain = VectorRetriever(e).retrieve(QUESTION, 2, Filters(source_types=("log",)))
    assert plain[0].path == "logs/p1/2026-01-01__t1.log"

    hits = ErrorLinkRetriever(e).retrieve(QUESTION, 3)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1)) and len(hits) <= 3
    linked = [h for h in hits if h.origin == "error_link"]
    assert [h.path for h in linked] == ["runbooks/division.md"]
    link = linked[0]
    assert link.linked_from == "logs/p1/2026-01-01__t1.log"
    log_pos = next(i for i, h in enumerate(hits) if h.path == link.linked_from)
    assert hits[log_pos + 1] is link  # placed right after its log
    # vector_similarity stays similarity to the QUESTION; the link score is separate.
    q = e.embed_query(QUESTION)
    expected = float(np.dot(q, e.embed_documents([link.text])[0]))
    assert link.scores["vector_similarity"] == pytest.approx(expected, abs=1e-4)
    assert link.scores["link_similarity"] > link.scores["vector_similarity"]


def test_error_link_respects_a_source_type_filter_without_runbooks(tiny):
    hits = ErrorLinkRetriever(HashingEmbedder()).retrieve(QUESTION, 3,
                                                          Filters(source_types=("log",)))
    assert all(h.origin == "vector" and h.source_type == "log" for h in hits)


def test_error_link_never_exceeds_k_or_duplicates(indexed):
    hits = ErrorLinkRetriever(HashingEmbedder()).retrieve(
        "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-04?", 5)
    assert len(hits) == 5
    assert len({h.chunk_id for h in hits}) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
