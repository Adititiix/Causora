"""Failed /ask requests are recorded with status, latency and a safe error category."""

import shutil

import psycopg
import pytest
from fastapi.testclient import TestClient

from incident_copilot.api.main import app, get_embedder_dep, get_llm_factory
from incident_copilot.config import get_settings
from incident_copilot.embeddings import HashingEmbedder
from incident_copilot.generation import FakeLLMProvider, LLMError, get_llm_provider
from incident_copilot.ingestion import ingest_paths
from incident_copilot.observability import ERROR_CATEGORIES, safe_question
from incident_copilot.retrieval.date_aware import clear_log_catalog_cache

QUESTION = "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-04?"
DOWN = "postgresql://nobody:nothing@127.0.0.1:1/none?connect_timeout=2"


def set_env(monkeypatch, **values):
    for k, v in values.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()


@pytest.fixture
def indexed(ingest_db, dataset, tmp_path, monkeypatch):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    ingest_paths(embedder=HashingEmbedder(), data_dir=root)
    set_env(monkeypatch, DATA_DIR=root, MIN_SIMILARITY=0.0, GEMINI_API_KEY="",
            RETRIEVER="vector+error_link+date")
    clear_log_catalog_cache()
    yield ingest_db
    clear_log_catalog_cache()


def client_with(llm_factory):
    app.dependency_overrides[get_embedder_dep] = HashingEmbedder
    app.dependency_overrides[get_llm_factory] = lambda: llm_factory
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def row(url, query_id):
    with psycopg.connect(url) as conn:
        r = conn.execute(
            "SELECT question, answer, status_code, error_category, latency_ms, model, "
            "metadata, (SELECT count(*) FROM retrievals WHERE query_id = q.id) "
            "FROM queries q WHERE id = %s", (query_id,)).fetchone()
    keys = ["question", "answer", "status_code", "error_category", "latency_ms", "model",
            "metadata", "retrievals"]
    return dict(zip(keys, r, strict=True))


def all_rows(url):
    with psycopg.connect(url) as conn:
        return conn.execute("SELECT status_code, error_category, question FROM queries "
                            "ORDER BY created_at").fetchall()


@pytest.mark.db
def test_llm_not_configured_is_recorded_with_its_evidence(indexed):
    resp = client_with(get_llm_provider).post("/ask", json={"question": QUESTION, "top_k": 5})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["code"] == "llm_not_configured" and err["query_id"]
    r = row(indexed, err["query_id"])
    assert (r["status_code"], r["error_category"], r["answer"]) == (503, "llm_not_configured",
                                                                    None)
    assert r["latency_ms"] >= 0 and r["retrievals"] == 5
    assert r["metadata"]["error"]["category"] == "llm_not_configured"
    assert r["metadata"]["retrieval_info"]["date_filter"]["applied"] is True


@pytest.mark.db
def test_llm_upstream_and_bad_response_are_recorded(indexed):
    fake = FakeLLMProvider([LLMError("upstream said api_key=SECRET123"), "not json"])
    c = client_with(lambda: fake)
    first = c.post("/ask", json={"question": QUESTION}).json()["error"]
    second = c.post("/ask", json={"question": QUESTION}).json()["error"]
    assert (first["code"], second["code"]) == ("llm_upstream_error", "llm_bad_response")
    r1, r2 = row(indexed, first["query_id"]), row(indexed, second["query_id"])
    assert (r1["status_code"], r2["status_code"]) == (502, 502)
    assert r1["model"].startswith("fake-llm") and r1["retrievals"] == 8
    assert "SECRET123" not in str(r1["metadata"]) and "[REDACTED]" in str(r1["metadata"])
    assert "SECRET123" not in first["message"]


@pytest.mark.db
def test_validation_errors_are_recorded_with_a_redacted_question(indexed):
    c = client_with(lambda: FakeLLMProvider())
    resp = c.post("/ask", json={"question": "token=abc123 why did it fail?", "top_k": 99})
    assert resp.status_code == 422
    err = resp.json()["error"]
    r = row(indexed, err["query_id"])
    assert (r["status_code"], r["error_category"]) == (422, "validation_error")
    assert r["question"] == "token=[REDACTED] why did it fail?"
    assert r["retrievals"] == 0 and r["latency_ms"] >= 0
    bad_json = c.post("/ask", content=b"{not json", headers={"content-type": "application/json"})
    assert bad_json.status_code == 422
    assert row(indexed, bad_json.json()["error"]["query_id"])["question"] == ""


@pytest.mark.db
def test_validation_errors_on_other_routes_are_not_recorded_as_queries(indexed):
    c = client_with(lambda: FakeLLMProvider())
    assert c.post("/ingest", json={"paths": "runbooks"}).status_code == 422
    assert all_rows(indexed) == []


@pytest.mark.db
def test_unexpected_errors_are_500_recorded_without_leaking_details(indexed):
    class BrokenEmbedder(HashingEmbedder):
        def embed_query(self, text):
            raise RuntimeError("boom password=hunter2")

    app.dependency_overrides[get_embedder_dep] = BrokenEmbedder
    app.dependency_overrides[get_llm_factory] = lambda: (lambda: FakeLLMProvider())
    resp = TestClient(app, raise_server_exceptions=False).post("/ask", json={"question": "q"})
    assert resp.status_code == 500
    err = resp.json()["error"]
    assert err["code"] == "internal_error" and "hunter2" not in resp.text
    r = row(indexed, err["query_id"])
    assert (r["status_code"], r["error_category"]) == (500, "internal_error")
    assert "hunter2" not in str(r["metadata"])


@pytest.mark.db
def test_successful_requests_are_status_200_and_store_redacted_questions(indexed):
    c = client_with(lambda: FakeLLMProvider())
    body = c.post("/ask", json={"question": f"{QUESTION} password=hunter2",
                                "retrieval_only": True}).json()
    r = row(indexed, body["query_id"])
    assert (r["status_code"], r["error_category"]) == (200, None)
    assert "hunter2" not in r["question"] and r["question"].endswith("password=[REDACTED]")
    assert body["retrieval_info"]["date_filter"]["applied"] is True
    assert body["retrieval_info"]["date_filter"]["date"] == "2026-09-04"


def test_database_outage_is_reported_but_cannot_be_recorded(use_database, monkeypatch):
    use_database(DOWN)
    c = client_with(lambda: FakeLLMProvider())
    resp = c.post("/ask", json={"question": "q", "retrieval_only": True})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["code"] == "db_unavailable" and "query_id" not in err
    # A 422 while the database is down is still answered (the failed write is swallowed).
    assert c.post("/ask", json={"question": ""}).status_code == 422


def test_safe_question_and_categories():
    assert safe_question("  token=abc rest  ") == "token=[REDACTED] rest"
    assert safe_question(None) == "" and safe_question(42) == ""
    assert len(safe_question("x" * 5000)) == 1000
    assert {"validation_error", "internal_error", "llm_not_configured"} <= ERROR_CATEGORIES
