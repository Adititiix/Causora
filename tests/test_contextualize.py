"""Follow-up resolution (retrieval/contextualize.py) and how answer_question uses it.
Offline: a fixed log catalogue and document lookup stand in for the database."""

import json
from pathlib import Path

import pytest

from incident_copilot.generation import FakeLLMProvider, answer_question
from incident_copilot.generation.prompt import build_user_prompt
from incident_copilot.retrieval import Hit
from incident_copilot.retrieval.contextualize import (
    AMBIGUOUS,
    CONTEXTUALIZED,
    MAX_QUERY_CHARS,
    MAX_TURNS,
    NO_CONTEXT,
    STANDALONE,
    DocInfo,
    Turn,
    contextualize,
)
from incident_copilot.retrieval.query_scope import LogCatalog

CATALOG = LogCatalog(frozenset({
    ("payments_sync", "pull_payments_api", "2026-08-22"),
    ("payments_sync", "pull_payments_api", "2026-09-16"),
    ("orders_etl", "transform_orders", "2026-09-04"),
    ("orders_etl", "transform_orders", "2026-09-16"),
    ("orders_etl", "wait_for_orders_export", "2026-09-01"),
}))

DOCS = {
    "incidents/INC-2043.md": DocInfo(
        "incidents/INC-2043.md", "incident",
        "INC-2043: Upstream API rate limit (HTTP 429) in payments_sync", "INC-2043",
        "payments_sync", "pull_payments_api", "2026-09-16"),
    "incidents/INC-2032.md": DocInfo(
        "incidents/INC-2032.md", "incident",
        "INC-2032: Task killed for running out of memory in orders_etl", "INC-2032",
        "orders_etl", "transform_orders", "2026-08-12"),
    "incidents/INC-2035.md": DocInfo(
        "incidents/INC-2035.md", "incident",
        "INC-2035: Dependency version conflict after image rebuild in orders_etl", "INC-2035",
        "orders_etl", "transform_orders", "2026-08-18"),
    "logs/orders_etl/2026-09-16__transform_orders.log": DocInfo(
        "logs/orders_etl/2026-09-16__transform_orders.log", "log",
        "orders_etl / transform_orders / 2026-09-16", None, "orders_etl", "transform_orders",
        "2026-09-16", "ImportError: cannot import name 'soft_unicode' from 'markupsafe'"),
    "runbooks/dependency-version-conflict.md": DocInfo(
        "runbooks/dependency-version-conflict.md", "runbook",
        "Runbook: Dependency version conflict after image rebuild"),
    "runbooks/api-rate-limited.md": DocInfo(
        "runbooks/api-rate-limited.md", "runbook", "Runbook: Upstream API rate limit (HTTP 429)"),
}


def lookup(paths):
    lookup.calls.append(list(paths))
    return {p: DOCS[p] for p in paths if p in DOCS}


lookup.calls = []

RATE_LIMIT_TURN = Turn(
    "Which past incident was caused by the payments provider API rate limiting our requests, "
    "and how long did it last?", "INC-2043 lasted 205 minutes.", ("incidents/INC-2043.md",))
IMPORT_ERROR_TURN = Turn(
    "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-16?",
    "ImportError ...", ("logs/orders_etl/2026-09-16__transform_orders.log",
                        "runbooks/dependency-version-conflict.md"))


def ctx(question, history):
    return contextualize(question, history, CATALOG, lookup)


# ---- resolution ---------------------------------------------------------------------------


def test_follow_up_after_a_known_incident_resolves_to_that_incident():
    c = ctx("What should we check first to prevent this happening again?", [RATE_LIMIT_TURN])
    assert c.status == CONTEXTUALIZED and c.source_turn == 1
    assert c.subjects == ("INC-2043: Upstream API rate limit (HTTP 429) in payments_sync",)
    assert c.retrieval_query.startswith("What should we check first to prevent this")
    assert "INC-2043: Upstream API rate limit (HTTP 429)" in c.retrieval_query
    assert c.question == "What should we check first to prevent this happening again?"
    assert "not evidence" in c.context_note


def test_follow_up_after_a_log_diagnosis_keeps_run_error_and_runbook_topic():
    c = ctx("How do we prevent it from happening again?", [IMPORT_ERROR_TURN])
    assert c.status == CONTEXTUALIZED
    assert "transform_orders on 2026-09-16" in c.retrieval_query
    assert "Dependency version conflict" in c.retrieval_query


def test_second_follow_up_keeps_the_context_from_the_first():
    first = Turn("What should we check first to prevent this happening again?",
                 "Use a shared pool ...", ("incidents/INC-2043.md",
                                           "runbooks/api-rate-limited.md"))
    c = ctx("And what was the root cause?", [RATE_LIMIT_TURN, first])
    assert c.status == CONTEXTUALIZED and c.source_turn == 1
    assert "INC-2043" in c.retrieval_query and "API rate limit" in c.retrieval_query


def test_a_refused_turn_is_skipped_and_the_earlier_subject_is_used():
    refused = Turn("How do we prevent it?")  # no citations, no subject of its own
    c = ctx("Why did it happen?", [RATE_LIMIT_TURN, refused])
    assert c.status == CONTEXTUALIZED and c.source_turn == 2 and "INC-2043" in c.retrieval_query


@pytest.mark.parametrize("question", [
    "Why did the wait_for_orders_export task in the orders_etl pipeline fail on 2026-09-01?",
    "A metric task fails with ZeroDivisionError. How should I fix it?",
    "Which past incident was caused by an unpinned dependency, and how was it resolved?",
    "Who owns the orders_fact table and what is its freshness SLA?",
    "What happened in INC-2032?",
    "What is the retention period for the clickstream Kafka topic in production?",
    "what is my name",
])
def test_a_new_standalone_question_does_not_inherit_the_previous_incident(question):
    c = ctx(question, [RATE_LIMIT_TURN])
    assert c.status == STANDALONE and c.retrieval_query == question and c.subjects == ()
    assert c.context_note is None


def test_no_seed_question_is_treated_as_a_follow_up():
    """All 50 seed questions are standalone, so none may pick up an unrelated history."""
    path = Path(__file__).resolve().parents[1] / "eval" / "questions.seed.jsonl"
    seeds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert len(seeds) == 50
    for history in ([RATE_LIMIT_TURN], [IMPORT_ERROR_TURN]):
        wrong = [s["external_id"] for s in seeds
                 if ctx(s["question"], history).status != STANDALONE]
        assert wrong == []


def test_an_answer_about_several_incidents_asks_which_one():
    turn = Turn("Which orders_etl incidents involved transform_orders?", "Two ...",
                ("incidents/INC-2032.md", "incidents/INC-2035.md"))
    # The earlier question names a pipeline/task, but both cited incidents match it.
    c = ctx("How long did it last?", [turn])
    assert c.status == AMBIGUOUS and c.retrieval_query == "How long did it last?"
    assert any("INC-2032" in x for x in c.candidates) and any("INC-2035" in x
                                                              for x in c.candidates)
    assert "Which one do you mean" in c.clarification() and c.context_note is None


def test_log_and_incident_about_the_same_run_are_not_ambiguous():
    turn = Turn("What failed in payments last month?", "...",
                ("incidents/INC-2043.md", "logs/orders_etl/2026-09-16__transform_orders.log"))
    assert ctx("Why did it fail?", [turn]).status == AMBIGUOUS  # different runs
    same_run = DocInfo("logs/payments_sync/2026-09-16__pull_payments_api.log", "log",
                       None, None, "payments_sync", "pull_payments_api", "2026-09-16",
                       "HTTPError: 429")
    turn = Turn("What failed in payments?", "...",
                ("incidents/INC-2043.md", same_run.path))
    c = contextualize("Why did it fail?", [turn], CATALOG,
                      lambda paths: {p: v for p, v in {**DOCS, same_run.path: same_run}.items()
                                     if p in paths})
    assert c.status == CONTEXTUALIZED and "pull_payments_api on 2026-09-16" in c.retrieval_query


def test_a_new_date_replaces_the_old_run_but_keeps_pipeline_and_task():
    c = ctx("What about the run on 2026-09-04?", [IMPORT_ERROR_TURN])
    assert c.status == CONTEXTUALIZED
    assert "orders_etl / transform_orders" in c.retrieval_query
    assert "2026-09-16" not in c.retrieval_query and "ImportError" not in c.retrieval_query
    # Nor the old run's runbook: the 2026-09-04 run failed for a different reason.
    assert "Dependency version conflict" not in c.retrieval_query


def test_follow_up_without_any_usable_earlier_subject_is_no_context():
    c = ctx("How do we prevent it?", [Turn("what is my name")])
    assert c.status == NO_CONTEXT and c.retrieval_query == "How do we prevent it?"


@pytest.mark.parametrize("history", [None, []])
def test_no_history_is_standalone(history):
    c = ctx("How do we prevent it from happening again?", history)
    assert c.status == STANDALONE and c.reason == "no earlier turns"


def test_unknown_cited_paths_from_the_client_are_ignored():
    turn = Turn("Tell me more", None, ("incidents/INC-9999.md", "../../etc/passwd"))
    assert ctx("Why did it fail?", [turn]).status == NO_CONTEXT


def test_history_and_query_are_bounded():
    turns = [RATE_LIMIT_TURN] + [Turn(f"filler question {i}") for i in range(MAX_TURNS)]
    # The rate-limit turn is older than MAX_TURNS, so it is no longer considered.
    assert ctx("Why did it happen?", turns).status == NO_CONTEXT
    long_q = "Why did it happen again " + "x " * 600
    c = ctx(long_q, [RATE_LIMIT_TURN])
    assert c.status == CONTEXTUALIZED and len(c.retrieval_query) <= MAX_QUERY_CHARS
    assert "INC-2043" in c.retrieval_query


def test_resolution_is_a_pure_function_of_its_inputs():
    """No state survives between calls, so one user's history cannot reach another."""
    a = ctx("How do we prevent it?", [RATE_LIMIT_TURN])
    b = ctx("How do we prevent it?", [IMPORT_ERROR_TURN])
    c = ctx("How do we prevent it?", None)
    again = ctx("How do we prevent it?", [RATE_LIMIT_TURN])
    assert "INC-2043" in a.retrieval_query and "INC-2043" not in b.retrieval_query
    assert c.retrieval_query == "How do we prevent it?" and again == a


# ---- answer_question with a contextualization ---------------------------------------------


def _hit(rank, path, score, source_type="incident", section="Follow-ups", text="Use a pool"):
    return Hit(chunk_id=f"c{rank}", document_id=f"d{rank}", path=path, source_type=source_type,
               section=section, ts_start=None, ts_end=None, text=text, chunk_index=rank - 1,
               content_hash=None, pipeline_name=None, rank=rank,
               scores={"vector_similarity": score, "vector_distance": 1 - score})


class RecordingRetriever:
    name = "recording"

    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def retrieve(self, question, k, filters=None):
        self.queries.append(question)
        return self.hits[:k]


ANSWER = {"insufficient_evidence": False, "summary": "Use a shared pool.",
          "evidence": [{"claim": "INC-2043 follow-ups: use a shared pool", "evidence_ids": ["E1"]}],
          "inference": [], "next_steps": ["Use a shared pool for each external API."]}


def test_retrieval_uses_the_resolved_query_and_the_model_gets_the_real_question():
    c = ctx("How do we prevent it from happening again?", [RATE_LIMIT_TURN])
    retriever = RecordingRetriever([_hit(1, "incidents/INC-2043.md", 0.9)])
    llm = FakeLLMProvider([ANSWER])
    out = answer_question(c.question, retriever=retriever, llm_factory=lambda: llm, k=8,
                          min_similarity=0.72, contextualization=c)
    assert retriever.queries == [c.retrieval_query]
    prompt = llm.calls[0]["user"]
    assert "<question>\nHow do we prevent it from happening again?\n</question>" in prompt
    assert "<conversation_context>" in prompt and "It is not evidence" in prompt
    assert out.status == "answered" and out.question == c.question
    assert out.retrieval_info["conversation"]["status"] == CONTEXTUALIZED
    assert [x.path for x in out.citations] == ["incidents/INC-2043.md"]


def test_conversation_context_cannot_bypass_the_similarity_gate():
    c = ctx("Who was the on-call engineer for it?", [RATE_LIMIT_TURN])
    assert c.status == CONTEXTUALIZED
    llm = FakeLLMProvider([])
    out = answer_question(c.question, retriever=RecordingRetriever(
        [_hit(1, "incidents/INC-2043.md", 0.70)]), llm_factory=lambda: llm, k=8,
        min_similarity=0.72, contextualization=c)
    assert out.insufficient_reason == "below_min_similarity" and llm.call_count == 0


def test_missing_evidence_still_abstains_after_contextualization():
    c = ctx("How much did it cost us?", [RATE_LIMIT_TURN])
    llm = FakeLLMProvider([{"insufficient_evidence": True, "summary": "No cost is stated.",
                            "evidence": [], "inference": [], "next_steps": []}])
    out = answer_question(c.question, retriever=RecordingRetriever(
        [_hit(1, "incidents/INC-2043.md", 0.8)]), llm_factory=lambda: llm, k=8,
        min_similarity=0.72, contextualization=c)
    assert out.status == "insufficient_evidence"
    assert out.insufficient_reason == "model_insufficient_evidence" and out.citations == []


def test_invalid_citations_are_still_dropped_for_follow_ups():
    c = ctx("How do we prevent it?", [RATE_LIMIT_TURN])
    bad = {**ANSWER, "evidence": [{"claim": "made up", "evidence_ids": ["E7"]}]}
    out = answer_question(c.question, retriever=RecordingRetriever(
        [_hit(1, "incidents/INC-2043.md", 0.9)]), llm_factory=lambda: FakeLLMProvider([bad]),
        k=8, min_similarity=0.72, contextualization=c)
    assert out.insufficient_reason == "no_valid_citations" and out.validation["invalid_ids"]


def test_ambiguous_reference_asks_for_clarification_without_retrieval_or_llm():
    turn = Turn("Which orders_etl incidents involved transform_orders?", "Two ...",
                ("incidents/INC-2032.md", "incidents/INC-2035.md"))
    c = ctx("How long did it last?", [turn])
    retriever, llm = RecordingRetriever([]), FakeLLMProvider([])
    out = answer_question(c.question, retriever=retriever, llm_factory=lambda: llm, k=8,
                          min_similarity=0.72, contextualization=c)
    assert retriever.queries == [] and llm.call_count == 0
    assert out.status == "insufficient_evidence"
    assert out.insufficient_reason == "ambiguous_reference"
    assert "Which one do you mean" in out.answer and out.timings["llm_called"] is False


def test_without_history_the_prompt_is_byte_identical_to_before():
    hits = [_hit(1, "incidents/INC-2043.md", 0.9)]
    assert build_user_prompt("q", hits) == build_user_prompt("q", hits, None)
    assert "<conversation_context>" not in build_user_prompt("q", hits)
    standalone = ctx("What happened in INC-2032?", [RATE_LIMIT_TURN])
    assert standalone.context_note is None
