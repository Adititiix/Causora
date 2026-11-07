from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from incident_copilot.api.main import app
from incident_copilot.config import Settings


def test_settings_defaults_match_schema_dimension():
    # chunks.embedding is vector(384); changing the model means a schema migration.
    assert Settings().embedding_dim == 384


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    assert Settings().retrieval_top_k == 7


def test_env_example_stays_in_sync_with_settings():
    """Every key in .env.example must be a real setting (or a docker-compose-only variable)."""
    compose_only = {"postgres_user", "postgres_password", "postgres_db", "postgres_port",
                    "api_port"}
    path = Path(__file__).resolve().parents[1] / ".env.example"
    keys = {
        line.split("=", 1)[0].strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    unknown = keys - set(Settings.model_fields) - compose_only
    assert not unknown, f".env.example has keys the app does not read: {sorted(unknown)}"
    # And the other direction: every setting is documented for the user.
    undocumented = set(Settings.model_fields) - keys
    assert not undocumented, f"settings missing from .env.example: {sorted(undocumented)}"


def test_health_does_not_need_database():
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_db_returns_503_when_database_is_down(use_database):
    # Port 1 is never a Postgres server.
    use_database("postgresql://nobody:nothing@127.0.0.1:1/none?connect_timeout=2")
    resp = TestClient(app).get("/health/db")
    assert resp.status_code == 503
    assert "database unavailable" in resp.json()["detail"]


@pytest.mark.db
def test_health_db_reports_pgvector_and_tables(test_db_url, use_database):
    use_database(test_db_url)
    resp = TestClient(app).get("/health/db")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["pgvector_version"]
    assert body["tables"] == 8
    assert body["chunks"] == 0
