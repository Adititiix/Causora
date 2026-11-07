"""HTTP API: /ask, /ingest, /health/db. The embedder is the offline HashingEmbedder and the
LLM is FakeLLMProvider (scripted); nothing here calls a network or a real model."""

import shutil

import psycopg
import pytest
from fastapi.testclient import TestClient

from incident_copilot.api.main import app, get_embedder_dep, get_llm_factory
from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.embeddings import HashingEmbedder
from incident_copilot.generation import (
    INSUFFICIENT_EVIDENCE_TEXT,
    FakeLLMProvider,
    LLMError,
    get_llm_provider,
)
from incident_copilot.ingestion import ingest_paths
from incident_copilot.ingestion.ingest import INGEST_LOCK_KEY

DOWN = "postgresql://nobody:nothing@127.0.0.1:1/none?connect_timeout=2"
QUESTION = "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-04?"


def good_answer(ids=("E1", "E2")):
    return {"insufficient_evidence": False, "summary": "It failed with KeyError.",
            "evidence": [{"claim": f"claim from {i}", "evidence_ids": [i]} for i in ids],
            "inference": [{"claim": "column renamed", "based_on": [ids[0]],
                           "confidence": "medium"}],
            "next_steps": ["Update the transform."]}


@pytest.fixture
def llm():
    return FakeLLMProvider()


@pytest.fixture
def client(llm, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    app.dependency_overrides[get_embedder_dep] = HashingEmbedder
    app.dependency_overrides[get_llm_factory] = lambda: (lambda: llm)
    yield TestClient(app)
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def set_env(monkeypatch, **values):
    for k, v in values.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()


@pytest.fixture
def indexed(ingest_db, dataset, tmp_path, monkeypatch):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    ingest_paths(embedder=HashingEmbedder(), data_dir=root)
    # The hashing fake's similarities are not on bge's scale; open the gate for API tests.
    set_env(monkeypatch, DATA_DIR=root, MIN_SIMILARITY=0.0, RETRIEVER="vector")
    return root


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    assert body["error"]["code"] == code and body["error"]["message"]
    return body["error"]


# ---- /ask validation (no database needed: validation runs first) ------------------------


@pytest.mark.parametrize("payload", [
    {},
    {"question": ""},
    {"question": "    "},
    {"question": "x" * 1001},
    {"question": "q", "top_k": 0},
    {"question": "q", "top_k": 21},
    {"question": "q", "top_k": "five"},
    {"question": "q", "source_types": ["memo"]},
    {"question": "q", "source_types": []},
    {"question": "q", "pipeline": "orders etl; drop"},
    {"question": "q", "unexpected": True},
    # Conversation history: bounded and strict.
    {"question": "q", "history": [{"question": f"q{i}"} for i in range(7)]},
    {"question": "q", "history": [{"question": ""}]},
    {"question": "q", "history": [{"question": "x" * 1001}]},
    {"question": "q", "history": [{"question": "q", "answer": "x" * 2001}]},
    {"question": "q", "history": [{"question": "q", "cited_paths": ["p"] * 21}]},
    {"question": "q", "history": [{"question": "q", "role": "system"}]},
    {"question": "q", "history": "not a list"},
])
def test_ask_validation_errors_use_the_error_body(client, payload):
    err = assert_error(client.post("/ask", json=payload), 422, "validation_error")
    assert isinstance(err["details"], list) and err["details"]


def test_ask_rejects_a_non_json_body(client):
    resp = client.post("/ask", content=b"not json", headers={"content-type": "application/json"})
    assert_error(resp, 422, "validation_error")


def test_ask_returns_503_when_the_database_is_down(client, use_database):
    use_database(DOWN)
    err = assert_error(client.post("/ask", json={"question": "q", "retrieval_only": True}),
                       503, "db_unavailable")
    assert "nothing" not in err["message"]  # no credentials in errors


# ---- /ask behaviour ---------------------------------------------------------------------


@pytest.mark.db
def test_retrieval_only_returns_evidence_without_the_llm(client, indexed, llm, ingest_db):
    resp = client.post("/ask", json={"question": f"  {QUESTION}  ", "retrieval_only": True,
                                     "top_k": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "retrieval_only" and body["answer"] is None
    assert body["insufficient_evidence"] is None and body["citations"] == []
    assert [r["evidence_id"] for r in body["retrieved"]] == ["E1", "E2", "E3", "E4", "E5"]
    first = body["retrieved"][0]
    assert {"vector_similarity", "vector_distance"} <= set(first["scores"])
    assert first["path"] and first["text"] and first["origin"] == "vector"
    assert llm.call_count == 0
    with psycopg.connect(ingest_db) as conn:
        question, n = conn.execute(
            "SELECT q.question, count(r.id) FROM queries q JOIN retrievals r "
            "ON r.query_id = q.id WHERE q.id = %s GROUP BY 1", (body["query_id"],)).fetchone()
    assert question == QUESTION and n == 5  # stripped question; one row per retrieved chunk


@pytest.mark.db
def test_answer_with_verified_citations(client, indexed, llm, ingest_db):
    llm._responses.append(good_answer())
    body = client.post("/ask", json={"question": QUESTION, "top_k": 4}).json()
    assert body["status"] == "answered" and body["insufficient_evidence"] is False
    assert body["answer"] == "It failed with KeyError."
    assert body["model"].startswith("fake-llm")
    by_id = {r["evidence_id"]: r for r in body["retrieved"]}
    assert [c["evidence_id"] for c in body["citations"]] == ["E1", "E2"]
    for c in body["citations"]:
        r = by_id[c["evidence_id"]]
        assert (c["path"], c["section"], c["chunk_id"]) == (r["path"], r["section"],
                                                            r["chunk_id"])
        assert r["used_in_answer"] is True
    assert body["evidence"][0]["evidence_ids"] == ["E1"]
    assert body["inference"][0]["confidence"] == "medium"
    assert llm.call_count == 1
    with psycopg.connect(ingest_db) as conn:
        used = conn.execute("SELECT evidence_id FROM retrievals WHERE query_id = %s AND "
                            "used_in_answer ORDER BY rank", (body["query_id"],)).fetchall()
    assert used == [("E1",), ("E2",)]


@pytest.mark.db
def test_invalid_citation_ids_are_dropped(client, indexed, llm):
    bad = good_answer()
    bad["evidence"].append({"claim": "invented", "evidence_ids": ["E77"]})
    llm._responses.append(bad)
    body = client.post("/ask", json={"question": QUESTION, "top_k": 3}).json()
    assert [e["claim"] for e in body["evidence"]] == ["claim from E1", "claim from E2"]
    assert body["validation"]["invalid_ids"] == ["E77"]


@pytest.mark.db
def test_experimental_retriever_keeps_the_response_shape(client, indexed, monkeypatch):
    from incident_copilot.api.schemas import RetrievedChunk
    from incident_copilot.retrieval.evidence import clear_resolution_cache

    clear_resolution_cache()  # the test database is re-indexed between tests
    set_env(monkeypatch, RETRIEVER="vector+error_link+date")  # the default it extends
    baseline = client.post("/ask", json={"question": QUESTION, "retrieval_only": True}).json()
    set_env(monkeypatch, RETRIEVER="vector+error_link+evidence+date")
    resp = client.post("/ask", json={"question": QUESTION, "retrieval_only": True})
    body = resp.json()
    clear_resolution_cache()
    assert resp.status_code == 200 and body["retriever"] == "vector+error_link+evidence+date"
    assert set(body) == set(baseline)
    assert body["retrieved"] and len(body["retrieved"]) <= 8
    for item in body["retrieved"]:
        assert set(item) == set(RetrievedChunk.model_fields)
    assert "provenance" not in resp.text and "vector_rank" not in resp.text
    assert body["gate"]["best_similarity"] == baseline["gate"]["best_similarity"]


def _an_incident(ingest_db):
    with psycopg.connect(ingest_db) as conn:
        return conn.execute("SELECT path, title FROM documents WHERE source_type = 'incident' "
                            "ORDER BY path LIMIT 1").fetchone()


@pytest.mark.db
def test_follow_up_with_history_is_resolved_logged_and_keeps_the_schema(client, indexed, llm,
                                                                        ingest_db):
    from incident_copilot.observability.query_log import safe_question
    from incident_copilot.retrieval.date_aware import clear_log_catalog_cache

    clear_log_catalog_cache()
    path, title = _an_incident(ingest_db)
    plain = client.post("/ask", json={"question": QUESTION, "retrieval_only": True}).json()
    history = [{"question": "Which past incident hit us last month?", "answer": "That one.",
                "cited_paths": [path, "incidents/does-not-exist.md"]}]
    resp = client.post("/ask", json={"question": "How do we prevent it from happening again?",
                                     "retrieval_only": True, "history": history})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == set(plain)  # same response fields with or without history
    conv = body["retrieval_info"]["conversation"]
    assert conv["status"] == "contextualized" and title in conv["retrieval_query"]
    assert conv["subjects"] == [title]
    assert {"embedding_ms", "retrieval_ms", "llm_called"} <= set(body["timings"])
    assert body["timings"]["llm_called"] is False and llm.call_count == 0
    with psycopg.connect(ingest_db) as conn:
        question, meta = conn.execute("SELECT question, metadata FROM queries WHERE id = %s",
                                      (body["query_id"],)).fetchone()
    assert question == "How do we prevent it from happening again?"  # the user's own words
    assert meta["request"]["history_turns"] == 1 and "history" not in meta["request"]
    assert meta["retrieval_info"]["conversation"]["retrieval_query"] == \
        safe_question(conv["retrieval_query"])
    assert meta["timings"]["llm_called"] is False


@pytest.mark.db
def test_requests_do_not_share_conversation_state(client, indexed, ingest_db):
    from incident_copilot.retrieval.date_aware import clear_log_catalog_cache

    clear_log_catalog_cache()
    path, title = _an_incident(ingest_db)
    q = "How do we prevent it from happening again?"
    with_history = client.post("/ask", json={
        "question": q, "retrieval_only": True,
        "history": [{"question": "What happened?", "cited_paths": [path]}]}).json()
    without = client.post("/ask", json={"question": q, "retrieval_only": True}).json()
    other_user = client.post("/ask", json={
        "question": q, "retrieval_only": True,
        "history": [{"question": "what is my name"}]}).json()
    assert title in with_history["retrieval_info"]["conversation"]["retrieval_query"]
    assert "conversation" not in without["retrieval_info"]
    assert other_user["retrieval_info"]["conversation"]["status"] == "no_context"
    assert other_user["retrieval_info"]["conversation"]["retrieval_query"] == q


@pytest.mark.db
def test_retry_exhaustion_is_a_502_with_attempts_logged(client, indexed, llm, ingest_db):
    from incident_copilot.generation import LLMError

    llm._responses.append(LLMError("Gemini API error 504: Deadline expired (after 4 attempts)",
                                   attempts=4, status=504, transient=True))
    resp = client.post("/ask", json={"question": QUESTION})
    err = assert_error(resp, 502, "llm_upstream_error")
    assert "504" in err["message"] and "after 4 attempts" in err["message"]
    with psycopg.connect(ingest_db) as conn:
        meta = conn.execute("SELECT metadata FROM queries WHERE id = %s",
                            (err["query_id"],)).fetchone()[0]
    assert meta["timings"]["llm_called"] is True and meta["timings"]["llm_attempts"] == 4


@pytest.mark.db
def test_ask_reports_the_effective_top_k(client, indexed):
    default = client.post("/ask", json={"question": QUESTION, "retrieval_only": True}).json()
    chosen = client.post("/ask", json={"question": QUESTION, "retrieval_only": True,
                                       "top_k": 3}).json()
    assert default["top_k"] == 8 and chosen["top_k"] == 3 and len(chosen["retrieved"]) <= 3


# ---- /documents (read-only Knowledge Base) ------------------------------------------------


def _snapshot(ingest_db):
    with psycopg.connect(ingest_db) as conn:
        return conn.execute("SELECT count(*), max(updated_at), (SELECT count(*) FROM chunks), "
                            "(SELECT count(*) FROM queries) FROM documents").fetchone()


@pytest.mark.db
def test_documents_lists_the_index_without_changing_it(client, indexed, ingest_db):
    before = _snapshot(ingest_db)
    resp = client.get("/documents")
    assert resp.status_code == 200, resp.text
    docs = resp.json()
    with psycopg.connect(ingest_db) as conn:
        n_docs, n_chunks = conn.execute(
            "SELECT count(*), (SELECT count(*) FROM chunks) FROM documents").fetchone()
    assert len(docs) == n_docs and sum(d["chunk_count"] for d in docs) == n_chunks
    assert {d["source_type"] for d in docs} <= {"runbook", "log", "schema", "incident"}
    runbook = next(d for d in docs if d["source_type"] == "runbook")
    assert runbook["title"] and runbook["sections"] and "sections" not in runbook["metadata"]
    assert _snapshot(ingest_db) == before  # read-only: nothing written, nothing logged


@pytest.mark.db
def test_document_detail_has_chunks_and_the_masked_original(client, indexed, ingest_db):
    path = next(d["path"] for d in client.get("/documents").json()
                if d["source_type"] == "runbook")
    # A secret written into the source file after indexing must never be shown.
    with open(indexed / path, "a", encoding="utf-8") as f:
        f.write("\npassword=hunter2-very-secret\n")
    body = client.get(f"/documents/{path}").json()
    assert body["document"]["path"] == path
    assert [c["chunk_index"] for c in body["chunks"]] == \
        sorted(c["chunk_index"] for c in body["chunks"])
    assert body["original"] and "hunter2-very-secret" not in body["original"]
    assert "[REDACTED]" in body["original"] and "masked" in body["original_note"]


@pytest.mark.db
def test_document_detail_without_the_original_file_shows_chunks_only(client, indexed):
    path = next(d["path"] for d in client.get("/documents").json()
                if d["source_type"] == "schema")
    (indexed / path).unlink()
    body = client.get(f"/documents/{path}").json()
    assert body["original"] is None and body["chunks"]
    assert "not available" in body["original_note"]


@pytest.mark.db
@pytest.mark.parametrize("path", ["nope.md", "..%2F.env", "%2E%2E/%2E%2E/etc/passwd"])
def test_unknown_or_unsafe_document_paths_are_404(client, indexed, path):
    assert_error(client.get(f"/documents/{path}"), 404, "document_not_found")


@pytest.mark.db
def test_explain_uses_only_the_selected_document_and_validates_sections(client, indexed, llm):
    path = next(d["path"] for d in client.get("/documents").json()
                if d["source_type"] == "runbook")
    llm._responses.append({"summary": "A runbook.", "not_documented": [],
                           "points": [{"topic": "overview", "text": "It covers a failure.",
                                       "basis": "documented", "section_ids": ["S1"]},
                                      {"topic": "remediation", "text": "Invented step.",
                                       "basis": "documented", "section_ids": ["S99"]}]})
    resp = client.post("/documents/explain", json={"path": path})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == path and [p["section_ids"] for p in body["points"]] == [["S1"]]
    assert body["dropped_points"] == 1 and body["sections"][0]["id"] == "S1"
    assert llm.call_count == 1 and path in llm.calls[0]["user"]
    assert "Valid ids: S1" in llm.calls[0]["user"]


@pytest.mark.db
def test_explain_errors_are_classified_like_ask(client, indexed, llm):
    from incident_copilot.generation import LLMError
    from incident_copilot.generation.explain import clear_explain_cache

    clear_explain_cache()
    assert_error(client.post("/documents/explain", json={"path": "nope.md"}), 404,
                 "document_not_found")
    assert_error(client.post("/documents/explain", json={"path": "x", "extra": 1}), 422,
                 "validation_error")
    path = next(d["path"] for d in client.get("/documents").json()
                if d["source_type"] == "incident")
    llm._responses.append(LLMError("Gemini API error 504: Deadline expired", attempts=4,
                                   status=504, transient=True))
    err = assert_error(client.post("/documents/explain", json={"path": path}), 502,
                       "llm_upstream_error")
    assert "504" in err["message"]
    llm._responses.append("not json")
    assert_error(client.post("/documents/explain", json={"path": path}), 502,
                 "llm_bad_response")


@pytest.mark.db
def test_refusal_note_and_dropped_claims_are_logged_but_not_returned(client, indexed, llm):
    llm._responses.append({"insufficient_evidence": True,
                           "summary": "The evidence shows symptoms but no fix.",
                           "evidence": [{"claim": "an invented claim", "evidence_ids": ["C3"]}],
                           "inference": [], "next_steps": []})
    resp = client.post("/ask", json={"question": QUESTION, "top_k": 3})
    body = resp.json()
    assert body["answer"] == INSUFFICIENT_EVIDENCE_TEXT
    assert "refusal_note" not in body and "dropped_claims" not in body
    assert "symptoms but no fix" not in resp.text and "an invented claim" not in resp.text
    with connect() as conn:
        meta = conn.execute("SELECT metadata FROM queries WHERE id = %s",
                            (body["query_id"],)).fetchone()[0]
    assert meta["refusal_note"] == "The evidence shows symptoms but no fix."
    assert meta["dropped_claims"] == [{"field": "evidence", "claim": "an invented claim",
                                       "ids": ["C3"], "invalid_ids": ["C3"]}]


@pytest.mark.db
def test_all_citations_invalid_means_insufficient_evidence(client, indexed, llm):
    llm._responses.append(good_answer(ids=("E50", "E51")))
    body = client.post("/ask", json={"question": QUESTION, "top_k": 3}).json()
    assert body["answer"] == INSUFFICIENT_EVIDENCE_TEXT and body["insufficient_evidence"]
    assert body["citations"] == [] and body["evidence"] == []


@pytest.mark.db
def test_below_min_similarity_refuses_without_calling_the_llm(client, indexed, llm,
                                                               monkeypatch):
    set_env(monkeypatch, MIN_SIMILARITY=0.99)
    body = client.post("/ask", json={"question": "What is the on-call phone number for the "
                                                 "payments team?"}).json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] == "Not enough evidence to determine the cause"
    assert body["insufficient_reason"] == "below_min_similarity"
    assert body["gate"]["passed"] is False and body["retrieved"]
    assert llm.call_count == 0


@pytest.mark.db
def test_llm_not_configured_is_503_with_setup_instructions(indexed, monkeypatch):
    set_env(monkeypatch, GEMINI_API_KEY="")
    app.dependency_overrides[get_embedder_dep] = HashingEmbedder
    app.dependency_overrides[get_llm_factory] = lambda: get_llm_provider  # the real factory
    try:
        resp = TestClient(app).post("/ask", json={"question": QUESTION})
    finally:
        app.dependency_overrides.clear()
    err = assert_error(resp, 503, "llm_not_configured")
    assert "GEMINI_API_KEY" in err["message"] and "retrieval_only" in err["message"]


@pytest.mark.db
def test_llm_upstream_failure_is_502(client, indexed, llm):
    llm._responses.append(LLMError("Gemini API error 500: internal"))
    assert_error(client.post("/ask", json={"question": QUESTION}), 502, "llm_upstream_error")


@pytest.mark.db
def test_llm_malformed_output_is_502(client, indexed, llm):
    llm._responses.append("I think it was a KeyError")
    assert_error(client.post("/ask", json={"question": QUESTION}), 502, "llm_bad_response")


@pytest.mark.db
def test_filters_are_applied(client, indexed):
    body = client.post("/ask", json={"question": QUESTION, "retrieval_only": True, "top_k": 10,
                                     "source_types": ["runbook"],
                                     "pipeline": "customer_360"}).json()
    assert body["retrieved"]
    assert {r["source_type"] for r in body["retrieved"]} == {"runbook"}


@pytest.mark.db
def test_default_top_k_comes_from_settings(client, indexed, monkeypatch):
    set_env(monkeypatch, ASK_TOP_K=3)
    body = client.post("/ask", json={"question": QUESTION, "retrieval_only": True}).json()
    assert len(body["retrieved"]) == 3


# ---- /ingest ------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../secrets", "runbooks/../../etc", "/etc/passwd",
                                 "C:\\Windows", "\\\\server\\share"])
def test_ingest_rejects_path_traversal(client, bad):
    err = assert_error(client.post("/ingest", json={"paths": [bad]}), 422, "invalid_path")
    assert "secrets" not in err["message"] or ".." in err["message"]


@pytest.mark.parametrize("payload", [{"paths": "runbooks"}, {"paths": [""]},
                                     {"prune": "sometimes"}, {"force": True}])
def test_ingest_validation_errors(client, payload):
    assert_error(client.post("/ingest", json=payload), 422, "validation_error")


@pytest.mark.db
def test_ingest_dry_run_then_real_run(client, ingest_db, dataset, tmp_path, monkeypatch):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    set_env(monkeypatch, DATA_DIR=root)
    dry = client.post("/ingest", json={"paths": ["schemas"], "dry_run": True}).json()
    assert dry["dry_run"] is True and len(dry["added"]) == 12 and dry["embed_calls"] == 0
    real = client.post("/ingest", json={"paths": ["schemas"]}).json()
    assert len(real["added"]) == 12 and real["embed_calls"] == 12 and real["run_id"]
    again = client.post("/ingest", json={}).json()
    assert again["unchanged"] == 12 and len(again["added"]) == 92


@pytest.mark.db
def test_ingest_returns_409_while_another_run_holds_the_lock(client, ingest_db, dataset,
                                                             monkeypatch):
    set_env(monkeypatch, DATA_DIR=dataset["root"])
    with psycopg.connect(ingest_db, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (INGEST_LOCK_KEY,))
        resp = client.post("/ingest", json={})
        other.execute("SELECT pg_advisory_unlock(%s)", (INGEST_LOCK_KEY,))
    assert_error(resp, 409, "indexing_in_progress")


def test_ingest_returns_503_when_the_database_is_down(client, use_database):
    use_database(DOWN)
    assert_error(client.post("/ingest", json={"dry_run": True}), 503, "db_unavailable")


# ---- /health/db ---------------------------------------------------------------------------


@pytest.mark.db
def test_health_db_reports_document_and_chunk_counts(client, indexed):
    body = client.get("/health/db").json()
    assert body["status"] == "ok" and body["documents"] == 104 and body["chunks"] > 0
    assert body["by_source_type"]["incident"]["documents"] == 14
    assert body["by_source_type"]["log"]["chunks"] == body["by_source_type"]["log"][
        "embedded_chunks"]


def test_health_db_503_keeps_detail_and_adds_error(client, use_database):
    use_database(DOWN)
    resp = client.get("/health/db")
    assert resp.status_code == 503
    body = resp.json()
    assert "database unavailable" in body["detail"]
    assert body["error"]["code"] == "db_unavailable"


# ---- clarification for vague questions (generation/clarify.py) ----------------------------

PAYMENT = "why did the payment fail?"


@pytest.mark.parametrize("subject", [
    {"incident_id": "INC-2043", "unexpected": True},                   # extra field
    {"pipeline": "payments_sync"},                                     # partial run
    {"pipeline": "payments_sync", "task": "pull_payments_api"},        # no date
    {"incident_id": "INC2043"},                                        # malformed id
    {"pipeline": "payments sync; drop", "task": "t", "date": "2026-09-16"},
    {"pipeline": "p", "task": "t", "date": "16/09/2026"},              # not ISO
    {},
    "INC-2043",
])
def test_ask_rejects_malformed_subjects(client, subject):
    assert_error(client.post("/ask", json={"question": PAYMENT, "subject": subject}),
                 422, "validation_error")


@pytest.mark.db
def test_an_unknown_subject_is_rejected_and_recorded(client, indexed, llm, ingest_db):
    resp = client.post("/ask", json={"question": PAYMENT, "subject": {"incident_id": "INC-9999"}})
    err = assert_error(resp, 422, "validation_error")
    assert "does not match any indexed" in err["message"] and llm.call_count == 0
    with psycopg.connect(ingest_db) as conn:
        category, meta = conn.execute("SELECT error_category, metadata FROM queries "
                                      "WHERE id = %s", (err["query_id"],)).fetchone()
    assert category == "validation_error"
    assert meta["request"]["subject"] == {"incident_id": "INC-9999"}


@pytest.mark.db
def test_a_picked_subject_is_context_for_retrieval_and_the_gate_still_decides(
        client, indexed, llm, ingest_db, monkeypatch):
    from incident_copilot.api import main

    plain = client.post("/ask", json={"question": PAYMENT, "retrieval_only": True}).json()
    monkeypatch.setattr(main, "make_clarifier",
                        lambda *a, **k: pytest.fail("a picked subject is never clarified"))
    body = client.post("/ask", json={"question": PAYMENT, "retrieval_only": True,
                                     "subject": {"incident_id": "INC-2043"}}).json()
    assert set(body) == set(plain) and body["clarification"] is None
    conv = body["retrieval_info"]["conversation"]
    assert conv["status"] == "contextualized"
    assert conv["reason"] == "subject selected from a clarification"
    assert conv["retrieval_query"].startswith(PAYMENT + " (Selected: ")
    assert "INC-2043" in conv["retrieval_query"] and conv["subjects"]
    # The gate is not relaxed for a picked subject.
    set_env(monkeypatch, MIN_SIMILARITY=0.99)
    refused = client.post("/ask", json={"question": PAYMENT,
                                        "subject": {"incident_id": "INC-2043"}}).json()
    assert refused["insufficient_reason"] == "below_min_similarity"
    assert refused["clarification"] is None and llm.call_count == 0
    with psycopg.connect(ingest_db) as conn:
        meta = conn.execute("SELECT metadata FROM queries WHERE id = %s",
                            (refused["query_id"],)).fetchone()[0]
    assert meta["request"]["subject"] == {"incident_id": "INC-2043"}


@pytest.mark.db
def test_a_vague_gate_refusal_returns_candidates_without_calling_the_llm(
        client, indexed, llm, ingest_db, monkeypatch):
    from incident_copilot.api import main
    from incident_copilot.generation.clarify import Candidate

    offered = [Candidate(id="run:payments_sync/pull_payments_api/2026-09-16",
                         label="payments_sync / pull_payments_api on 2026-09-16 (INC-2043)",
                         description="INC-2043: Upstream API rate limit", incident_id="INC-2043",
                         pipeline="payments_sync", task="pull_payments_api", date="2026-09-16",
                         paths=("incidents/INC-2043.md",), best_similarity=0.69),
               Candidate(id="doc:incidents/INC-2044.md", label="INC-2044: Expired credentials",
                         description="", incident_id="INC-2044", pipeline=None, task=None,
                         date=None, paths=("incidents/INC-2044.md",), best_similarity=0.68)]
    monkeypatch.setattr(main, "make_clarifier", lambda *a, **k: lambda q, hits: offered)
    set_env(monkeypatch, MIN_SIMILARITY=0.99)
    resp = client.post("/ask", json={"question": PAYMENT})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "insufficient_evidence" and body["insufficient_evidence"] is True
    assert body["insufficient_reason"] == "ambiguous_query"
    assert body["citations"] == [] and body["evidence"] == [] and llm.call_count == 0
    cands = body["clarification"]["candidates"]
    assert [c["id"] for c in cands] == [c.id for c in offered]
    assert cands[0]["subject"] == {"incident_id": "INC-2043", "pipeline": "payments_sync",
                                   "task": "pull_payments_api", "date": "2026-09-16"}
    assert cands[1]["subject"] == {"incident_id": "INC-2044", "pipeline": None, "task": None,
                                   "date": None}
    with psycopg.connect(ingest_db) as conn:
        meta = conn.execute("SELECT metadata FROM queries WHERE id = %s",
                            (body["query_id"],)).fetchone()[0]
    assert meta["insufficient_reason"] == "ambiguous_query"
    assert [c["id"] for c in meta["clarification"]["candidates"]] == [c.id for c in offered]


@pytest.mark.db
def test_other_responses_carry_no_clarification(client, indexed, llm):
    llm._responses.append(good_answer())
    assert client.post("/ask", json={"question": QUESTION}).json()["clarification"] is None
    body = client.post("/ask", json={"question": QUESTION, "retrieval_only": True}).json()
    assert body["clarification"] is None


# ---- the picked failure is kept for follow-ups ----------------------------------------------

PICKED = {"incident_id": "INC-2043"}


def _turn(question=PAYMENT, subject=PICKED):
    return {"question": question, "answer": "INC-2043 was a rate limit.",
            "cited_paths": ["incidents/INC-2043.md"], "subject": subject}


@pytest.mark.parametrize("subject", [
    {"incident_id": "INC-2043", "extra": 1}, {"pipeline": "p"}, {"incident_id": "bad"}])
def test_a_history_turn_subject_is_validated_like_a_picked_one(client, subject):
    resp = client.post("/ask", json={"question": "What is the solution for it?",
                                     "history": [_turn(subject=subject)]})
    assert_error(resp, 422, "validation_error")


@pytest.mark.db
@pytest.mark.parametrize("follow_up", ["What is the solution for it?", "How do we prevent it?"])
def test_a_follow_up_keeps_the_previous_turns_subject(client, indexed, llm, monkeypatch,
                                                      follow_up):
    from incident_copilot.api import main

    monkeypatch.setattr(main, "make_clarifier",
                        lambda *a, **k: pytest.fail("a kept subject is never clarified"))
    body = client.post("/ask", json={"question": follow_up, "retrieval_only": True,
                                     "history": [_turn()]}).json()
    conv = body["retrieval_info"]["conversation"]
    assert conv["status"] == "contextualized"
    assert conv["reason"] == "subject kept from the previous turn"
    assert conv["subject"]["incident_id"] == "INC-2043"
    assert conv["retrieval_query"].startswith(follow_up + " (Selected: ")
    assert llm.call_count == 0


@pytest.mark.db
def test_a_standalone_question_does_not_inherit_the_previous_subject(client, indexed):
    body = client.post("/ask", json={"question": QUESTION, "retrieval_only": True,
                                     "history": [_turn()]}).json()
    assert body["retrieval_info"].get("conversation", {}).get("subject") is None


@pytest.mark.db
def test_a_kept_subject_does_not_bypass_the_gate(client, indexed, llm, monkeypatch):
    set_env(monkeypatch, MIN_SIMILARITY=0.99)
    body = client.post("/ask", json={"question": "What is the solution for it?",
                                     "history": [_turn()]}).json()
    assert body["insufficient_reason"] == "below_min_similarity" and llm.call_count == 0
    assert body["retrieval_info"]["conversation"]["subject"]["incident_id"] == "INC-2043"


@pytest.mark.db
def test_a_kept_subject_that_is_no_longer_indexed_is_ignored(client, indexed):
    resp = client.post("/ask", json={"question": "What is the solution for it?",
                                     "retrieval_only": True,
                                     "history": [_turn(subject={"incident_id": "INC-9999"})]})
    assert resp.status_code == 200  # only a picked subject must exist; a kept one may lapse
    assert resp.json()["retrieval_info"].get("conversation", {}).get("subject") is None


@pytest.mark.db
def test_a_picked_subject_is_reported_back_for_the_next_turn(client, indexed):
    body = client.post("/ask", json={"question": PAYMENT, "retrieval_only": True,
                                     "subject": PICKED}).json()
    subject = body["retrieval_info"]["conversation"]["subject"]
    assert subject["incident_id"] == "INC-2043" and subject["pipeline"]


@pytest.mark.db
def test_next_step_evidence_is_returned_aligned_with_next_steps(client, indexed, llm):
    llm._responses.append({**good_answer(), "next_steps": [
        {"step": "Update the transform.", "evidence_ids": ["E1"]},
        {"step": "Invented step.", "evidence_ids": ["E77"]},
        "Plain string step."]})
    body = client.post("/ask", json={"question": QUESTION, "top_k": 3}).json()
    assert body["next_steps"] == ["Update the transform.", "Plain string step."]
    assert body["next_step_evidence"] == [["E1"], []]


@pytest.mark.db
def test_next_step_labels_and_unknowns_are_returned_aligned(client, indexed, llm):
    llm._responses.append({**good_answer(), "unknowns": ["The provider quota is unknown."],
                           "next_steps": [
                               {"step": "Update the transform.", "evidence_ids": ["E1"],
                                "kind": "prevention"},
                               "Plain string step."]})
    body = client.post("/ask", json={"question": QUESTION, "top_k": 3}).json()
    assert len(body["next_step_kinds"]) == len(body["next_step_support"]) == 2
    assert body["next_step_support"][1] == "unsupported"
    assert body["unknowns"] == ["The provider quota is unknown."]


@pytest.mark.db
def test_a_refusal_carries_no_step_labels_or_unknowns(client, indexed, monkeypatch):
    set_env(monkeypatch, MIN_SIMILARITY=0.99)
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert body["status"] == "insufficient_evidence"
    assert body["next_step_kinds"] == body["next_step_support"] == body["unknowns"] == []


@pytest.mark.db
def test_a_picked_incident_gets_its_own_remediation_after_the_gate(client, indexed, llm):
    llm._responses.append(good_answer())
    body = client.post("/ask", json={"question": PAYMENT, "top_k": 2,
                                     "subject": PICKED}).json()
    assert body["status"] == "answered", body
    added = [h for h in body["retrieved"] if h["origin"] == "subject_remediation"]
    assert added and {h["path"] for h in added} == {"incidents/INC-2043.md"}
    assert {h["section"] for h in added} <= {"Resolution", "Follow-ups"}
    ids = [h["chunk_id"] for h in body["retrieved"]]
    assert len(ids) == len(set(ids))  # nothing twice
    # The gate was decided before they were added.
    found = [h["scores"]["vector_similarity"] for h in body["retrieved"]
             if h["origin"] != "subject_remediation"]
    assert body["gate"]["best_similarity"] == max(found)
    # The model saw them: they are in its prompt as evidence blocks.
    assert all(f'id="{h["evidence_id"]}"' in llm.calls[-1]["user"] for h in added)


@pytest.mark.db
def test_an_unrelated_question_gets_no_remediation(client, indexed, llm):
    llm._responses.append(good_answer())
    body = client.post("/ask", json={"question": QUESTION, "top_k": 3}).json()
    assert not [h for h in body["retrieved"] if h["origin"] == "subject_remediation"]
