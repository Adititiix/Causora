"""Streamlit UI helpers: the HTTP client (with a mock transport) and response parsing.
Offline: no API, no Streamlit server, no LLM."""

import json

import httpx
import pytest

from incident_copilot.config import Settings
from incident_copilot.ui.client import CausoraClient
from incident_copilot.ui.view import (
    EXAMPLE_QUESTIONS,
    FOLLOWUP_HINT,
    INSUFFICIENT_REASONS,
    MAX_HISTORY_TURNS,
    build_view,
    escape_md,
    failure_hint,
    followup_hint,
    history_for_request,
    next_step_sources,
    parse_chunks,
)

RUNBOOK_RESOLUTION = ("# Runbook: Dependency version conflict after image rebuild\n\n"
                      "## Resolution\n1. Roll back to the previous worker image.\n"
                      "2. Pin versions in a constraints file and rebuild.\n"
                      "3. Clear the failed tasks.")


def _chunk(eid, path, source_type, section, text, used=False, sim=0.8, origin="vector"):
    return {"evidence_id": eid, "rank": int(eid[1:]), "chunk_id": f"c{eid}", "path": path,
            "source_type": source_type, "section": section, "chunk_index": 0,
            "ts_start": None, "ts_end": None,
            "scores": {"vector_similarity": sim, "vector_distance": 1 - sim},
            "origin": origin, "linked_from": None, "used_in_answer": used, "text": text}


def answered_response() -> dict:
    """Shaped like a real /ask response (Q016 in eval/results/20261028T114348Z_answers)."""
    return {
        "query_id": "q-1", "status": "answered",
        "answer": "It failed with ImportError: cannot import name 'soft_unicode'.",
        "insufficient_evidence": False, "insufficient_reason": None,
        "evidence": [{"claim": "The task failed with an ImportError.",
                      "evidence_ids": ["E1"]}],
        "inference": [{"claim": "A dependency version conflict.", "based_on": ["E1", "E2"],
                       "confidence": "high"}],
        "next_steps": ["Roll back to the previous worker image.",
                       "Pin versions in a constraints file and rebuild.",
                       "Ask the platform team."],
        "citations": [{"evidence_id": "E1", "chunk_id": "cE1",
                       "path": "logs/orders_etl/2026-09-16__transform_orders.log",
                       "source_type": "log", "section": "error context", "chunk_index": 1,
                       "ts_start": None, "ts_end": None}],
        "retrieved": [
            _chunk("E1", "logs/orders_etl/2026-09-16__transform_orders.log", "log",
                   "error context", "ImportError: cannot import name 'soft_unicode'",
                   used=True),
            _chunk("E2", "runbooks/dependency-version-conflict.md", "runbook", "Resolution",
                   RUNBOOK_RESOLUTION, sim=0.55, origin="resolution_companion"),
        ],
        "model": "gemini-3.5-flash-lite", "retriever": "vector+error_link+date",
        "gate": {"best_similarity": 0.85, "min_similarity": 0.72, "passed": True},
        "validation": {}, "tokens_in": 10, "tokens_out": 5, "latency_ms": 2200,
        "retrieval_ms": 300, "generation_ms": 1800, "retrieval_info": {},
    }


# ---- view -------------------------------------------------------------------------------


def test_build_view_separates_facts_inferences_and_steps():
    view = build_view(answered_response())
    assert view.status == "answered" and not view.insufficient
    assert [c.text for c in view.facts] == ["The task failed with an ImportError."]
    assert view.facts[0].ids == ("E1",)
    assert view.inferences[0].confidence == "high"
    assert view.inferences[0].ids == ("E1", "E2")
    assert view.cited_ids == ["E1"]
    assert view.chunk("E2").label == ("Troubleshooting guide · "
                                      "runbooks/dependency-version-conflict.md · "
                                      "Resolution")
    assert view.chunk("E1").similarity == pytest.approx(0.8)
    assert view.chunk("E9") is None


def test_next_steps_are_matched_only_on_exact_wording():
    view = build_view(answered_response())
    by_text = {s.text: s.matched_ids for s in view.next_steps}
    assert by_text["Roll back to the previous worker image."] == ("E2",)
    assert by_text["Pin versions in a constraints file and rebuild."] == ("E2",)
    # Not in any chunk: no source is claimed.
    assert by_text["Ask the platform team."] == ()


def test_next_step_match_ignores_case_numbering_and_punctuation_but_not_paraphrase():
    chunks = parse_chunks([_chunk("E3", "runbooks/x.md", "runbook", "Resolution",
                                  "1. Wait for the rate window to reset, then clear the task.")])
    assert next_step_sources("2) wait for the rate window to reset then clear the task",
                             chunks) == ("E3",)
    assert next_step_sources("Wait until the rate limit resets and retry.", chunks) == ()
    # Very short steps could match almost anything, so they never do.
    assert next_step_sources("Clear the", chunks) == ()
    # Whole words only: "task" must not match inside "tasks".
    assert next_step_sources("then clear the tas", chunks) == ()


def test_insufficient_response_is_explained_and_has_no_answer_parts():
    data = {"status": "insufficient_evidence", "insufficient_evidence": True,
            "insufficient_reason": "below_min_similarity",
            "answer": "Not enough evidence to determine the cause.",
            "evidence": [], "inference": [], "next_steps": [], "citations": [],
            "retrieved": [], "gate": {"best_similarity": 0.63, "min_similarity": 0.72,
                                      "passed": False}}
    view = build_view(data)
    assert view.insufficient
    assert view.insufficient_explanation == INSUFFICIENT_REASONS["below_min_similarity"]
    assert view.best_similarity == pytest.approx(0.63)
    assert view.facts == view.inferences == view.next_steps == []


def test_unknown_insufficient_reason_is_shown_verbatim():
    view = build_view({"status": "insufficient_evidence", "insufficient_reason": "new_reason"})
    assert view.insufficient_explanation == "new_reason"


@pytest.mark.parametrize("data", [
    {}, {"status": "answered"},
    {"status": "answered", "evidence": None, "inference": "x", "next_steps": [None, " ", 3],
     "retrieved": [None, {"no_id": 1}], "citations": [{}], "gate": None, "latency_ms": "x"},
])
def test_missing_or_malformed_fields_render_as_nothing(data):
    view = build_view(data)
    assert view.facts == view.inferences == view.next_steps == view.chunks == []
    assert view.cited_ids == []
    assert view.latency_ms is None and view.best_similarity is None


def test_escape_md_shows_text_literally():
    assert escape_md("**bold** [x](http://e) <b>") == \
        r"\*\*bold\*\* \[x\]\(http://e\) \<b\>"
    assert escape_md(None) == ""


REFUSED = {"status": "insufficient_evidence", "insufficient_evidence": True,
           "insufficient_reason": "below_min_similarity",
           "answer": "Not enough evidence to determine the cause."}


@pytest.mark.parametrize("question", [
    "What should we check first to prevent this happening again?",  # logged UI refusal
    "Why did it fail?",
    "Is that the same issue as last week?",
    "How do we prevent it?",
    "Did THIS happen before?",
])
def test_followup_hint_for_refused_context_dependent_questions(question):
    assert followup_hint(question, build_view(REFUSED)) == FOLLOWUP_HINT


@pytest.mark.parametrize("question", [
    "What is the retention period for the clickstream Kafka topic in production?",
    "How do we prevent the payments_sync API rate limit (HTTP 429) from recurring?",
    # Whole words only: "item", "within" and "thistle" contain "it"/"this".
    "Which item within the thistle pipeline failed?",
    "",
    None,
])
def test_no_followup_hint_for_self_contained_questions(question):
    assert followup_hint(question, build_view(REFUSED)) is None


def test_followup_hint_only_on_refusals():
    question = "What should we check first to prevent this happening again?"
    assert followup_hint(question, build_view(answered_response())) is None
    retrieval_only = {"status": "retrieval_only", "retrieved": []}
    assert followup_hint(question, build_view(retrieval_only)) is None
    model_refusal = {**REFUSED, "insufficient_reason": "model_insufficient_evidence"}
    assert followup_hint(question, build_view(model_refusal)) == FOLLOWUP_HINT


def test_no_hint_when_the_api_resolved_the_follow_up_or_asked_which_one():
    question = "How do we prevent it?"
    for status in ("contextualized", "ambiguous"):
        data = {**REFUSED, "retrieval_info": {"conversation": {"status": status}}}
        assert followup_hint(question, build_view(data)) is None
    unresolved = {**REFUSED, "retrieval_info": {"conversation": {"status": "no_context"}}}
    assert followup_hint(question, build_view(unresolved)) == FOLLOWUP_HINT


def test_view_reads_how_a_follow_up_was_interpreted():
    data = answered_response()
    data["retrieval_info"] = {"conversation": {
        "status": "contextualized", "subjects": ["INC-2043: Upstream API rate limit"],
        "retrieval_query": "How do we prevent it? (Earlier ...)"}}
    data["timings"] = {"llm_attempts": 2}
    view = build_view(data)
    assert view.interpreted_as == ("INC-2043: Upstream API rate limit",)
    assert view.timings["llm_attempts"] == 2
    assert build_view(answered_response()).interpreted_as == ()
    standalone = {**answered_response(), "retrieval_info": {"conversation": {
        "status": "standalone", "subjects": ["ignored"]}}}
    assert build_view(standalone).interpreted_as == ()


def _messages(*pairs):
    out = []
    for question, result in pairs:
        out += [{"role": "user", "content": question},
                {"role": "assistant", "result": result, "question": question}]
    return out


def test_history_for_request_carries_question_answer_and_cited_paths():
    from incident_copilot.ui.client import ApiFailure, AskResult

    refused = {**REFUSED, "citations": []}
    msgs = _messages(("Why did transform_orders fail on 2026-09-16?",
                      AskResult(data=answered_response())),
                     ("what is my name", AskResult(data=refused)),
                     ("How do we prevent it?", AskResult(failure=ApiFailure("timeout", "slow"))))
    turns = history_for_request(msgs)
    assert turns[0] == {"question": "Why did transform_orders fail on 2026-09-16?",
                        "answer": "It failed with ImportError: cannot import name "
                                  "'soft_unicode'.",
                        "cited_paths": ["logs/orders_etl/2026-09-16__transform_orders.log"]}
    # A refusal or an error keeps only its question; the conversation stays usable.
    assert turns[1] == {"question": "what is my name"}
    assert turns[2] == {"question": "How do we prevent it?"}


def test_history_for_request_is_bounded_and_matches_the_api_schema():
    from incident_copilot.api.schemas import AskRequest
    from incident_copilot.ui.client import AskResult

    long_answer = {**answered_response(), "answer": "x" * 5000}
    msgs = _messages(*[(f"question {i} " + "y" * 1500, AskResult(data=long_answer))
                       for i in range(10)])
    turns = history_for_request(msgs)
    assert len(turns) == MAX_HISTORY_TURNS and turns[-1]["question"].startswith("question 9")
    assert all(len(t["question"]) <= 1000 and len(t["answer"]) <= 2000 for t in turns)
    AskRequest(question="How do we prevent it?", history=turns)  # extra="forbid" everywhere


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "result": None}],
                                      [{"role": "user", "content": "   "}], [None, "x"]])
def test_history_for_request_handles_empty_or_malformed_messages(messages):
    assert history_for_request(messages) == []


def test_ask_sends_history_only_when_there_is_one():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=answered_response())

    client = _client(handler)
    client.ask("How do we prevent it?", history=[])
    client.ask("How do we prevent it?", history=[{"question": "Why did INC-2043 happen?"}])
    assert "history" not in seen[0]
    assert seen[1]["history"] == [{"question": "Why did INC-2043 happen?"}]


def test_top_k_limits_match_the_api_schema_and_invalid_values_are_not_sent():
    from incident_copilot.api.schemas import AskRequest
    from incident_copilot.ui.client import TOP_K_MAX, TOP_K_MIN

    bounds = {type(m).__name__: m for m in AskRequest.model_fields["top_k"].metadata}
    assert (bounds["Ge"].ge, bounds["Le"].le) == (TOP_K_MIN, TOP_K_MAX)
    sent = []
    client = _client(lambda r: sent.append(json.loads(r.content)) or httpx.Response(
        200, json=answered_response()))
    for bad in (0, 21, -3, 2.5, "8", True):
        result = client.ask("q", top_k=bad)
        assert not result.ok and result.failure.code == "invalid_top_k"
    assert sent == []
    assert client.ask("q", top_k=TOP_K_MAX).ok and sent[-1]["top_k"] == TOP_K_MAX


def test_outcome_kinds_and_evidence_labels():
    from incident_copilot.ui.view import evidence_label, outcome_kind

    assert outcome_kind(build_view(answered_response())) == "answered"
    assert evidence_label(build_view(answered_response())) == \
        "Retrieved evidence: 2 chunks, 1 cited in the answer"
    cases = {"below_min_similarity": ("gate", "below the similarity threshold"),
             "no_hits": ("gate", "below the similarity threshold"),
             "model_insufficient_evidence": ("model", "sent to the model; no answer"),
             "no_valid_citations": ("withheld", "sent to the model; no answer"),
             "ambiguous_reference": ("clarify", "sent to the model; no answer")}
    for reason, (kind, label) in cases.items():
        view = build_view({**REFUSED, "insufficient_reason": reason,
                           "retrieved": answered_response()["retrieved"]})
        assert outcome_kind(view) == kind
        assert label in evidence_label(view) and "0 cited" not in evidence_label(view)
    ro = build_view({"status": "retrieval_only", "retrieved": answered_response()["retrieved"]})
    assert outcome_kind(ro) == "retrieval_only" and "no answer requested" in evidence_label(ro)


def test_effective_top_k_is_read_from_the_response():
    assert build_view({**answered_response(), "top_k": 5}).top_k == 5
    assert build_view({**answered_response(), "top_k": "5"}).top_k is None


DOCS = [
    {"path": "runbooks/api-rate-limited.md", "source_type": "runbook",
     "title": "Runbook: Upstream API rate limit (HTTP 429)", "pipeline": None,
     "sections": ["Symptoms", "Prevention"], "metadata": {"runbook_id": "RB-007"}},
    {"path": "runbooks/deadlock-detected.md", "source_type": "runbook",
     "title": "Runbook: Database deadlock during load", "metadata": {}},
    {"path": "incidents/INC-2043.md", "source_type": "incident", "title": "INC-2043",
     "pipeline": "payments_sync", "metadata": {"failed_task": "pull_payments_api"}},
    {"no_path": True}, "not a dict",
]


def test_filter_documents_by_type_and_words_in_title_path_pipeline_metadata():
    from incident_copilot.ui.view import filter_documents

    def paths(types, q):
        return [d["path"] for d in filter_documents(DOCS, types, q)]

    assert paths(["runbook"], "") == ["runbooks/api-rate-limited.md",
                                      "runbooks/deadlock-detected.md"]
    assert paths(None, "rate limit") == ["runbooks/api-rate-limited.md"]
    assert paths([], "DEADLOCK") == ["runbooks/deadlock-detected.md"]
    assert paths(None, "pull_payments_api") == ["incidents/INC-2043.md"]  # metadata value
    assert paths(None, "rb-007") == ["runbooks/api-rate-limited.md"]
    assert paths(None, "prevention") == ["runbooks/api-rate-limited.md"]  # section name
    assert paths(["schema"], "") == [] and paths(None, "nothing-matches") == []


def test_explanation_points_are_grouped_in_reading_order():
    from incident_copilot.ui.view import group_points

    points = [{"topic": "prevention", "text": "b"}, {"topic": "overview", "text": "a"},
              {"topic": "mystery", "text": "c"}, {"topic": "overview", "text": ""}, None]
    assert [(t, [p["text"] for p in ps]) for t, ps in group_points(points)] == \
        [("overview", ["a"]), ("prevention", ["b"]), ("mystery", ["c"])]


def test_document_client_calls_the_read_only_routes():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.raw_path.decode()))
        if request.url.path == "/documents":
            return httpx.Response(200, json=DOCS[:1])
        if request.url.path == "/documents/explain":
            return httpx.Response(200, json={"summary": "s", "points": []})
        return httpx.Response(200, json={"document": DOCS[0], "chunks": []})

    client = _client(handler)
    assert client.list_documents().ok and client.get_document("runbooks/a b.md").ok
    assert client.explain("runbooks/a b.md").ok
    assert seen == [("GET", "/documents"), ("GET", "/documents/runbooks/a%20b.md"),
                    ("POST", "/documents/explain")]
    bad = _client(lambda r: httpx.Response(200, json={"unexpected": 1}))
    assert bad.list_documents().failure.code == "bad_response"
    assert bad.get_document("x").failure.code == "bad_response"
    assert bad.explain("x").failure.code == "bad_response"
    missing = _client(lambda r: httpx.Response(404, json={"error": {
        "code": "document_not_found", "message": "no indexed document has this path"}}))
    assert missing.get_document("x").failure.code == "document_not_found"


def test_examples_and_hints():
    assert len(EXAMPLE_QUESTIONS) >= 3 and all(q.strip() for q in EXAMPLE_QUESTIONS)
    assert failure_hint("timeout") and failure_hint("llm_not_configured")
    assert failure_hint("never-heard-of-it") is None


class _ListRetriever:
    name = "list"

    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k, filters=None):
        return self.hits[:k]


def _hit(rank, path, source_type, section, text, sim):
    from incident_copilot.retrieval import Hit

    return Hit(chunk_id=f"c{rank}", document_id=f"d{rank}", path=path,
               source_type=source_type, section=section, ts_start=None, ts_end=None,
               text=text, chunk_index=rank - 1, content_hash=None, pipeline_name=None,
               rank=rank, scores={"vector_similarity": sim, "vector_distance": 1 - sim})


def _api_json(model_answer, sim=0.85) -> dict:
    """An /ask body produced by the real API code (answer_question -> _ask_response ->
    AskResponse JSON), with a fake retriever and FakeLLMProvider instead of DB and Gemini."""
    from incident_copilot.api.main import _ask_response
    from incident_copilot.api.schemas import AskResponse
    from incident_copilot.generation import FakeLLMProvider, answer_question

    hits = [_hit(1, "logs/orders_etl/2026-09-16__transform_orders.log", "log",
                 "error context", "ImportError: cannot import name 'soft_unicode'", sim),
            _hit(2, "runbooks/dependency-version-conflict.md", "runbook", "Resolution",
                 RUNBOOK_RESOLUTION, sim - 0.3)]
    llm = FakeLLMProvider([model_answer])
    outcome = answer_question("why?", retriever=_ListRetriever(hits),
                              llm_factory=lambda: llm, k=8, min_similarity=0.72)
    return json.loads(AskResponse.model_validate(
        _ask_response(outcome, "q-1", "list")).model_dump_json())


def test_view_reads_a_real_api_answer():
    data = _api_json({"insufficient_evidence": False, "summary": "ImportError at import.",
                      "evidence": [{"claim": "It raised ImportError.", "evidence_ids": ["E1"]}],
                      "inference": [{"claim": "Dependency conflict.", "based_on": ["E1", "E2"],
                                     "confidence": "medium"}],
                      "next_steps": ["Roll back to the previous worker image."]})
    view = build_view(data)
    assert view.status == "answered" and view.summary == "ImportError at import."
    assert view.facts[0].ids == ("E1",) and view.inferences[0].confidence == "medium"
    assert view.cited_ids == ["E1", "E2"]  # citations come from evidence + inference ids
    assert [c.used_in_answer for c in view.chunks] == [True, True]
    assert view.chunk("E1").similarity == pytest.approx(0.85)
    assert view.next_steps[0].matched_ids == ("E2",)
    assert view.model and view.query_id == "q-1"


def test_view_reads_real_api_refusals():
    gate = build_view(_api_json({}, sim=0.5))  # below the gate: the fake LLM is never called
    assert gate.insufficient and gate.insufficient_reason == "below_min_similarity"
    assert gate.summary and gate.best_similarity == pytest.approx(0.5)
    assert gate.facts == gate.next_steps == [] and len(gate.chunks) == 2

    model = build_view(_api_json({"insufficient_evidence": True, "summary": "x",
                                  "evidence": [], "inference": [], "next_steps": []}))
    assert model.insufficient_reason == "model_insufficient_evidence"
    assert model.insufficient_explanation == INSUFFICIENT_REASONS["model_insufficient_evidence"]
    assert model.cited_ids == [] and not any(c.used_in_answer for c in model.chunks)


# ---- client -----------------------------------------------------------------------------


def _client(handler) -> CausoraClient:
    return CausoraClient("http://api.test/", timeout=3, transport=httpx.MockTransport(handler))


def test_ask_sends_only_defined_request_fields():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=answered_response())

    client = _client(handler)
    assert client.ask("why?").ok
    assert client.ask("why?", retrieval_only=True, top_k=5).ok
    assert seen[0] == ("POST", "/ask", {"question": "why?"})
    assert seen[1][2] == {"question": "why?", "retrieval_only": True, "top_k": 5}


def test_ask_request_matches_the_api_schema():
    from incident_copilot.api.schemas import AskRequest

    sent = []
    _client(lambda r: sent.append(json.loads(r.content)) or httpx.Response(
        200, json=answered_response())).ask("q", retrieval_only=True, top_k=3)
    AskRequest(**sent[0])  # extra="forbid": an invented field would raise


def test_ask_api_error_body_becomes_a_failure():
    body = {"error": {"code": "llm_not_configured", "message": "no key", "query_id": "q-9"}}
    result = _client(lambda r: httpx.Response(503, json=body)).ask("q")
    assert not result.ok and result.data is None
    assert (result.failure.code, result.failure.status, result.failure.query_id) == \
        ("llm_not_configured", 503, "q-9")


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="<html>oops</html>"),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json=["not", "an", "answer"]),
])
def test_ask_unexpected_responses_are_bad_response(response):
    result = _client(lambda r: response).ask("q")
    assert not result.ok and result.failure.code == "bad_response"


@pytest.mark.parametrize("exc, code", [
    (httpx.ReadTimeout("slow"), "timeout"),
    (httpx.ConnectError("refused"), "unreachable"),
])
def test_ask_transport_errors_become_failures(exc, code):
    def handler(request):
        raise exc

    result = _client(handler).ask("q")
    assert not result.ok and result.failure.code == code


def test_health_reports_api_and_database_state():
    ok = _client(lambda r: httpx.Response(200, json={"status": "ok", "documents": 104,
                                                     "chunks": 324})).health()
    assert ok.api_reachable and ok.db_ok and (ok.documents, ok.chunks) == (104, 324)

    db_down = _client(lambda r: httpx.Response(
        503, json={"detail": "x", "error": {"code": "db_unavailable",
                                            "message": "database unavailable"}})).health()
    assert db_down.api_reachable and not db_down.db_ok
    assert db_down.message == "database unavailable"

    def refuse(request):
        raise httpx.ConnectError("refused")

    down = _client(refuse).health()
    assert not down.api_reachable and not down.db_ok


def test_ui_settings_have_safe_defaults():
    s = Settings(_env_file=None)
    assert s.ui_api_url.startswith("http://127.0.0.1")
    assert s.ui_request_timeout_seconds > s.llm_timeout_seconds


# ---- clarification for vague questions ----------------------------------------------------

CLARIFIED = {**REFUSED, "insufficient_reason": "ambiguous_query",
             "answer": "Several different failures match this question.",
             "clarification": {"reason": "several_failures_match", "candidates": [
                 {"id": "run:p/t/2026-09-16", "label": "p / t on 2026-09-16 (INC-2043)",
                  "description": "INC-2043: rate limit",
                  "subject": {"incident_id": "INC-2043", "pipeline": "p", "task": "t",
                              "date": "2026-09-16"}},
                 {"id": "doc:x", "label": "INC-2044", "description": None,
                  "subject": {"incident_id": "INC-2044", "pipeline": None, "task": None,
                              "date": None}}]}}


def test_a_clarification_is_its_own_kind_of_refusal():
    from incident_copilot.ui.view import evidence_label, outcome_kind

    view = build_view({**CLARIFIED, "retrieved": answered_response()["retrieved"]})
    assert outcome_kind(view) == "clarify" and view.insufficient
    assert "not sent to the model" in evidence_label(view)
    assert "asking which one you mean" in view.insufficient_explanation
    assert [c.id for c in view.choices] == ["run:p/t/2026-09-16", "doc:x"]
    assert view.choices[0].subject == {"incident_id": "INC-2043", "pipeline": "p", "task": "t",
                                       "date": "2026-09-16"}
    assert view.choices[1].subject == {"incident_id": "INC-2044"}  # nulls dropped
    assert view.choices[1].description == ""


@pytest.mark.parametrize("clarification", [
    None, "x", {"candidates": "x"}, {"candidates": [None, 5, {}]},
    {"candidates": [{"id": "a", "label": "A"}]},                      # no subject
    {"candidates": [{"id": "a", "label": "A", "subject": {"other": "x"}}]},
    {"candidates": [{"label": "A", "subject": {"incident_id": "INC-1"}}]},  # no id
])
def test_malformed_clarifications_offer_no_choices(clarification):
    assert build_view({**CLARIFIED, "clarification": clarification}).choices == []


def test_responses_without_a_clarification_have_no_choices():
    assert build_view(answered_response()).choices == []
    assert build_view(REFUSED).choices == []


def test_ask_sends_a_subject_only_when_one_was_picked():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=answered_response())

    client = _client(handler)
    assert client.ask("why?", subject=None).ok
    assert client.ask("why?", subject={}).ok
    assert client.ask("why?", subject={"incident_id": "INC-2043", "pipeline": None}).ok
    assert seen == [{"question": "why?"}, {"question": "why?"},
                    {"question": "why?", "subject": {"incident_id": "INC-2043"}}]


def test_the_subject_the_client_sends_matches_the_api_schema():
    from incident_copilot.api.schemas import AskRequest

    req = AskRequest(question="why?", subject=CLARIFIED["clarification"]["candidates"][0]
                     ["subject"])
    assert req.subject.incident_id == "INC-2043" and req.subject.date == "2026-09-16"


# ---- next-step citations, kept subjects, typed replies ---------------------------------------


def test_next_steps_carry_the_ids_the_api_validated():
    data = {**answered_response(), "next_steps": ["Update the transform.", "Restart."],
            "next_step_evidence": [["E1"], []]}
    steps = build_view(data).next_steps
    assert [s.evidence_ids for s in steps] == [("E1",), ()]
    older = {**answered_response(), "next_steps": ["Update the transform."]}
    assert build_view(older).next_steps[0].evidence_ids == ()  # no field: unsupported


def test_history_carries_the_subject_the_api_reported():
    from incident_copilot.ui.client import AskResult
    from incident_copilot.ui.view import history_for_request

    data = {**answered_response(), "retrieval_info": {"conversation": {
        "status": "contextualized", "subject": {"incident_id": "INC-2045", "pipeline": None,
                                                "junk": "x"}}}}
    turns = history_for_request([{"role": "user", "content": "why did it fail?"},
                                 {"role": "assistant", "result": AskResult(data=data)}])
    assert turns[0]["subject"] == {"incident_id": "INC-2045"}


def test_local_messages_are_not_turns():
    from incident_copilot.ui.view import history_for_request

    msgs = [{"role": "user", "content": "i mean about the shipment"},
            {"role": "assistant", "local": True, "question": "q", "choices": []}]
    assert history_for_request(msgs) == []


def _choices():
    from incident_copilot.ui.view import Choice

    return [Choice("a", "shipments_sync / transform_shipments on 2026-10-06 (INC-2045)",
                   "INC-2045: Upstream column rename", {"incident_id": "INC-2045"}),
            Choice("b", "shipments_sync / load_shipments_fact on 2026-10-15",
                   "Logged error: UniqueViolation", {"pipeline": "shipments_sync",
                                                     "task": "load_shipments_fact",
                                                     "date": "2026-10-15"})]


@pytest.mark.parametrize("text, expected", [
    ("i mean about the shipment", ["a", "b"]),       # both: ask again among them
    ("the column rename one", ["a"]),                # one: that pick
    ("inc-2045", ["a"]),
    ("the 2026-10-15 one", ["b"]),
    ("why did the orders_etl load fail on 2026-09-04?", []),  # names its own subject
    ("how do we prevent it from happening again in production please?", []),  # > 8 words
    ("", []),
])
def test_typed_replies_are_matched_only_when_they_point_at_a_choice(text, expected):
    from incident_copilot.ui.view import match_choices

    assert [c.id for c in match_choices(text, _choices())] == expected
