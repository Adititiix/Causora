"""Evaluation harness, temporary indexes and logged-answer loading against Postgres."""

import json
import os
import shutil

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from incident_copilot.chunking import logs as log_chunker
from incident_copilot.config import get_settings
from incident_copilot.embeddings import HashingEmbedder
from incident_copilot.evaluation.answers import load_logged_answers, split_by_provider
from incident_copilot.evaluation.harness import (
    RetrieverConfig,
    deep_ranks,
    default_configs,
    index_fingerprint,
    load_questions,
    run_retrieval_eval,
)
from incident_copilot.evaluation.scratch import temporary_index
from incident_copilot.generation import FakeLLMProvider, answer_question
from incident_copilot.ingestion import ingest_paths
from incident_copilot.observability import log_query
from incident_copilot.retrieval import VectorRetriever

SEED = "eval/questions.seed.jsonl"


def test_load_questions_reads_the_seed_file_without_changing_it(dataset):
    before = (dataset["eval"]).read_bytes()
    records, problems = load_questions(dataset["eval"])
    assert len(records) == 50 and problems == []
    assert dataset["eval"].read_bytes() == before


def test_load_questions_reports_malformed_records(tmp_path):
    p = tmp_path / "q.jsonl"
    good = {"external_id": "Q1", "question": "q", "ground_truth": "g", "category": "c",
            "should_abstain": False, "expected_paths": []}
    p.write_text("\n".join([json.dumps(good), "{not json", json.dumps({"external_id": "Q2"})]),
                 encoding="utf-8")
    records, problems = load_questions(p)
    assert [r["external_id"] for r in records] == ["Q1"]
    assert problems[0]["line"] == 2 and "invalid JSON" in problems[0]["problem"]
    assert problems[1]["id"] == "Q2" and "missing fields" in problems[1]["problem"]


def test_default_configs_cover_every_registered_retriever():
    names = [c.name for c in default_configs()]
    assert names == ["vector", "vector+error_link", "vector+date", "vector+error_link+date"]
    assert all(c.description for c in default_configs())


@pytest.fixture
def indexed(ingest_db, dataset, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    ingest_paths(embedder=HashingEmbedder(), data_dir=root)
    return root


@pytest.mark.db
def test_run_retrieval_eval_rows(indexed, dataset):
    questions, _ = load_questions(dataset["eval"])
    config = RetrieverConfig("vector", VectorRetriever)
    rows = run_retrieval_eval(questions[:5] + questions[-1:], config, HashingEmbedder(),
                              max_k=8, repeats=2)
    assert len(rows) == 6
    first = rows[0]
    assert first["config"] == "vector" and len(first["retrieved_paths"]) == 8
    assert len(first["latency_runs_ms"]) == 2 and first["latency_ms"] > 0
    assert {"hit@1", "recall@5", "all@8", "rr"} <= set(first)
    assert first["expected_source_types"] == ["log", "runbook"]
    unanswerable = rows[-1]
    assert unanswerable["should_abstain"] and "hit@1" not in unanswerable
    ranks = deep_ranks(questions[:2], VectorRetriever(HashingEmbedder()), depth=400)
    assert set(ranks) == {questions[0]["external_id"], questions[1]["external_id"]}
    assert all(set(v) == set(q["expected_paths"]) for v, q in zip(ranks.values(), questions,
                                                                  strict=False))


@pytest.mark.db
def test_index_fingerprint(indexed):
    fp = index_fingerprint()
    assert fp["documents"] == 104 and fp["chunks"] == fp["embedded_chunks"] > 0
    assert fp["embedding_models"] == ["hashing-bow-384"] and len(fp["corpus_sha256"]) == 64


@pytest.mark.db
def test_temporary_index_restores_window_settings_and_drops_the_database(ingest_db, dataset,
                                                                          tmp_path):
    root = tmp_path / "data"
    shutil.copytree(dataset["root"], root)
    env_before = os.environ.get("DATABASE_URL")
    window_before = log_chunker.ERROR_CONTEXT_ENTRIES
    with temporary_index(embedder=HashingEmbedder(), window=4, data_dir=str(root),
                         label="pytest") as idx:
        assert log_chunker.ERROR_CONTEXT_ENTRIES == 4
        assert conninfo_to_dict(get_settings().database_url)["dbname"] == idx.name
        assert idx.summary.chunks_written > 0 and idx.window == 4
        hits = VectorRetriever(HashingEmbedder()).retrieve("KeyError customer_tier", 3)
        assert hits
    assert log_chunker.ERROR_CONTEXT_ENTRIES == window_before
    assert os.environ.get("DATABASE_URL") == env_before
    with psycopg.connect(make_conninfo(ingest_db, dbname="postgres")) as admin:
        left = admin.execute("SELECT count(*) FROM pg_database WHERE datname = %s",
                             (idx.name,)).fetchone()[0]
    assert left == 0
    # The test database itself was never written by the temporary index.
    with psycopg.connect(ingest_db) as conn:
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


@pytest.mark.db
def test_logged_answers_are_split_by_provider(indexed):
    fake = FakeLLMProvider([{"insufficient_evidence": False, "summary": "s",
                             "evidence": [{"claim": "c", "evidence_ids": ["E1"]}],
                             "inference": [], "next_steps": []}])
    for retrieval_only in (False, True):
        out = answer_question("KeyError customer_tier", retriever=VectorRetriever(
            HashingEmbedder()), llm_factory=lambda: fake, k=3, min_similarity=0.0,
            retrieval_only=retrieval_only)
        log_query(out, retriever_name="vector", request={})
    split = split_by_provider(load_logged_answers())
    assert len(split["fake_scripted"]) == 1 and len(split["no_llm_call"]) == 1
    assert split["real_llm"] == []
    rec = split["fake_scripted"][0]
    assert rec["evidence_ids"] == ["E1", "E2", "E3"] and len(rec["evidence_paths"]) == 3


@pytest.mark.db
def test_interleaved_runs_time_every_config_for_every_question(indexed, dataset):
    from incident_copilot.evaluation.harness import run_retrieval_eval_many

    questions, _ = load_questions(dataset["eval"])
    configs = default_configs()[:2]  # vector, vector+error_link
    rows = run_retrieval_eval_many(questions[:3], configs, HashingEmbedder(), max_k=5, repeats=2,
                                   labels={"vector": "v", "vector+error_link": "v+e"})
    assert [r["config"] for r in rows] == ["v"] * 3 + ["v+e"] * 3
    assert all(len(r["latency_runs_ms"]) == 2 for r in rows)
    assert [r["id"] for r in rows[:3]] == [q["external_id"] for q in questions[:3]]
