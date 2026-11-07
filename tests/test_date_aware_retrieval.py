"""DateAwareRetriever against a real Postgres: wrong-date logs are dropped only when safe."""

import pytest

from incident_copilot.embeddings import HashingEmbedder
from incident_copilot.ingestion import ingest_paths
from incident_copilot.retrieval import (
    DateAwareRetriever,
    ErrorLinkRetriever,
    Filters,
    VectorRetriever,
    get_retriever,
)
from incident_copilot.retrieval.date_aware import clear_log_catalog_cache, load_log_catalog

pytestmark = pytest.mark.db

D1, D2 = "2026-09-04", "2026-09-21"
LOG_D1 = f"logs/orders_etl/{D1}__transform_orders.log"
LOG_D2 = f"logs/orders_etl/{D2}__transform_orders.log"
LOG_OTHER = f"logs/payments_sync/{D1}__pull_payments_api.log"


def log_text(day, error_lines):
    return "\n".join([
        f"[{day}T02:00:00.000+0000] {{x.py:1}} INFO - Starting attempt 1 of 1",
        f"[{day}T02:00:05.000+0000] {{x.py:1}} ERROR - Task failed with exception",
        "Traceback (most recent call last):",
        '  File "/dags/t.py", line 3, in run',
        "    step()",
        *error_lines,
        f"[{day}T02:00:06.000+0000] {{x.py:1}} INFO - Marking task as FAILED.",
    ]) + "\n"


@pytest.fixture
def corpus(ingest_db, tmp_path):
    """Two transform_orders logs that differ only in date and error, plus two runbooks."""
    root = tmp_path / "data"
    (root / "logs" / "orders_etl").mkdir(parents=True)
    (root / "logs" / "payments_sync").mkdir(parents=True)
    (root / "runbooks").mkdir()
    (root / LOG_D1).write_text(log_text(D1, ["ZeroDivisionError: division by zero"]), "utf-8")
    (root / LOG_D2).write_text(log_text(D2, ["KeyError: 'customer_tier'"]), "utf-8")
    (root / LOG_OTHER).write_text(log_text(D1, ["TimeoutError: upstream timed out"]), "utf-8")
    (root / "runbooks" / "division.md").write_text(
        "# Runbook: Division\n\n## Symptoms\n\n- ZeroDivisionError: division by zero\n", "utf-8")
    (root / "runbooks" / "rename.md").write_text(
        "# Runbook: Rename\n\n## Symptoms\n\n- KeyError: 'customer_tier' after a rename\n",
        "utf-8")
    ingest_paths(embedder=HashingEmbedder(), data_dir=root)
    clear_log_catalog_cache()
    yield root
    clear_log_catalog_cache()


def paths(hits):
    return [h.path for h in hits]


def log_paths(hits):
    return {h.path for h in hits if h.source_type == "log"}


def ids(hits):
    return [h.chunk_id for h in hits]


Q = "Why did the transform_orders task in the orders_etl pipeline fail on {}?"


def test_catalog_lists_every_indexed_log(corpus):
    catalog = load_log_catalog()
    assert catalog.entries == {("orders_etl", "transform_orders", D1),
                               ("orders_etl", "transform_orders", D2),
                               ("payments_sync", "pull_payments_api", D1)}


def test_wrong_date_log_is_retrieved_without_the_filter(corpus):
    """The failure mode being fixed: plain vector search returns both dates."""
    hits = VectorRetriever(HashingEmbedder()).retrieve(Q.format(D2), 10)
    assert {LOG_D1, LOG_D2} <= log_paths(hits)


def test_explicit_date_keeps_only_that_log_and_all_non_log_evidence(corpus):
    e = HashingEmbedder()
    r = DateAwareRetriever(e, VectorRetriever)
    hits = r.retrieve(Q.format(D2), 10)
    assert log_paths(hits) == {LOG_D2}
    assert {"runbooks/division.md", "runbooks/rename.md"} <= set(paths(hits))
    info = r.last_info["date_filter"]
    assert info["applied"] is True and info["date"] == D2
    assert (info["pipeline"], info["task"]) == ("orders_etl", "transform_orders")
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_date_only_question_keeps_every_log_of_that_date(corpus):
    hits = DateAwareRetriever(HashingEmbedder(), VectorRetriever).retrieve(
        f"What failed on {D1}?", 10)
    assert log_paths(hits) == {LOG_D1, LOG_OTHER}


@pytest.mark.parametrize("question", [
    "Why did the transform_orders task in the orders_etl pipeline fail on 09/04/2026?",
    "Why did the transform_orders task in the orders_etl pipeline fail on Sep 21?",
    "Why did the transform_orders task in the orders_etl pipeline fail yesterday?",
    "Why does the transform_orders task in the orders_etl pipeline keep failing?",
    "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-05?",
    "Why did the ml_features pipeline fail on 2026-09-21?",
])
def test_uncertain_or_missing_dates_return_exactly_the_unfiltered_results(corpus, question):
    e = HashingEmbedder()
    r = DateAwareRetriever(e, VectorRetriever)
    filtered = r.retrieve(question, 10)
    assert ids(filtered) == ids(VectorRetriever(e).retrieve(question, 10))
    assert r.last_info["date_filter"]["applied"] is False
    assert r.last_info["date_filter"]["reason"]


def test_request_without_logs_or_with_a_conflicting_pipeline_is_not_narrowed(corpus):
    e = HashingEmbedder()
    r = DateAwareRetriever(e, VectorRetriever)
    hits = r.retrieve(Q.format(D2), 10, Filters(source_types=("runbook",)))
    assert r.last_info["date_filter"]["reason"] == "request excludes logs"
    assert {h.source_type for h in hits} == {"runbook"}
    r.retrieve(Q.format(D2), 10, Filters(pipeline="payments_sync"))
    assert r.last_info["date_filter"]["applied"] is False


def test_error_link_follows_the_log_of_the_requested_date(corpus):
    e = HashingEmbedder()
    question = Q.format(D2)
    plain = ErrorLinkRetriever(e).retrieve(question, 4)
    dated = DateAwareRetriever(e, ErrorLinkRetriever).retrieve(question, 4)
    linked = [h for h in dated if h.origin == "error_link"]
    assert all(h.linked_from == LOG_D2 for h in linked)
    assert "runbooks/rename.md" in paths(dated)
    assert LOG_D1 not in paths(dated)
    assert LOG_D2 in paths(plain) or LOG_D1 in paths(plain)  # plain search sees both dates


def test_registry_names(corpus):
    e = HashingEmbedder()
    assert get_retriever("vector+date", e).name == "vector+date"
    assert get_retriever("vector+error_link+date", e).name == "vector+error_link+date"
