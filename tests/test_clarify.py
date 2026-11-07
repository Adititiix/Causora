"""Clarification for vague questions (generation/clarify.py and its use in answer_question).
Offline: hand-made hits and index metadata, a scripted retriever and a fake LLM that must
never be called, so no database, model or network.

The shapes mirror what was measured on the real index: "why did the payment fail?" is
refused by the gate (best 0.691) while its top 8 hold three different failures; Q047, Q050,
M014 and M015 are gate refusals that must stay refusals.
"""

from dataclasses import replace

import pytest

from incident_copilot.generation import answer_question
from incident_copilot.generation.clarify import (
    CLARIFICATION_TEXT,
    Candidate,
    describe_selection,
    find_candidates,
)
from incident_copilot.retrieval import Hit
from incident_copilot.retrieval.contextualize import (
    AMBIGUOUS,
    CONTEXTUALIZED,
    NO_CONTEXT,
    Anchors,
    Contextualization,
    DocInfo,
    extract_anchors,
)
from incident_copilot.retrieval.query_scope import LogCatalog

FLOOR, MAX = 0.67, 5
NONE = Anchors()

SENSOR_LOG = "extra/logs/finance_reporting/2026-10-12__wait_for_payments_sync.log"
SENSOR_INC = "extra/incidents/INC-2046.md"
PULL_INC = "incidents/INC-2043.md"
LOAD_LOG = "logs/payments_sync/2026-09-17__load_payments_fact.log"
RUNBOOK = "runbooks/api-rate-limited.md"
SCHEMA = "schemas/payments_fact.md"

DOCS = {
    SENSOR_LOG: DocInfo(SENSOR_LOG, "log", "finance_reporting / wait_for_payments_sync",
                        pipeline="finance_reporting", task="wait_for_payments_sync",
                        date="2026-10-12", error="AirflowSensorTimeout: Sensor has timed out"),
    SENSOR_INC: DocInfo(SENSOR_INC, "incident", "INC-2046: Rate-limited payments pull",
                        incident_id="INC-2046", pipeline="finance_reporting",
                        task="wait_for_payments_sync", date="2026-10-12"),
    PULL_INC: DocInfo(PULL_INC, "incident", "INC-2043: Upstream API rate limit (HTTP 429)",
                      incident_id="INC-2043", pipeline="payments_sync",
                      task="pull_payments_api", date="2026-09-16"),
    LOAD_LOG: DocInfo(LOAD_LOG, "log", "payments_sync / load_payments_fact",
                      pipeline="payments_sync", task="load_payments_fact", date="2026-09-17",
                      error="psycopg2.errors.UniqueViolation: duplicate key"),
    RUNBOOK: DocInfo(RUNBOOK, "runbook", "Runbook: Upstream API rate limit"),
    SCHEMA: DocInfo(SCHEMA, "schema", "Table: payments_fact"),
}


def hit(path, score, section="sec", rank=1):
    return Hit(chunk_id=f"{path}#{section}", document_id=f"doc:{path}", path=path,
               source_type=DOCS[path].source_type if path in DOCS else "incident",
               section=section, ts_start=None, ts_end=None, text=f"{path} {section}",
               chunk_index=0, content_hash="h", pipeline_name=None, rank=rank,
               scores={"vector_similarity": score, "vector_distance": round(1 - score, 6)})


PAYMENT_HITS = [hit(SENSOR_LOG, 0.691, "error context"), hit(PULL_INC, 0.689, "Overview"),
                hit(SENSOR_INC, 0.684, "Summary"), hit(RUNBOOK, 0.70, "Symptoms"),
                hit(LOAD_LOG, 0.676, "run summary"), hit(PULL_INC, 0.66, "Timeline"),
                hit(SCHEMA, 0.65)]


def candidates(hits=PAYMENT_HITS, docs=DOCS, **kw):
    options = {"own_anchors": NONE, "min_similarity": FLOOR, "max_candidates": MAX} | kw
    return find_candidates(hits, docs, **options)


# ---- detection ----------------------------------------------------------------------------


def test_distinct_failures_become_candidates_in_a_deterministic_order():
    out = candidates()
    assert [c.id for c in out] == ["run:finance_reporting/wait_for_payments_sync/2026-10-12",
                                   "run:payments_sync/pull_payments_api/2026-09-16",
                                   "run:payments_sync/load_payments_fact/2026-09-17"]
    assert [c.best_similarity for c in out] == [0.691, 0.689, 0.676]
    assert candidates(list(reversed(PAYMENT_HITS))) == out  # input order does not matter


def test_a_log_and_the_incident_of_the_same_run_are_one_candidate():
    sensor = candidates()[0]
    assert sensor.paths == (SENSOR_INC, SENSOR_LOG)  # several chunks, two documents, one run
    assert sensor.incident_id == "INC-2046"
    assert sensor.label == "finance_reporting / wait_for_payments_sync on 2026-10-12 (INC-2046)"


def test_runbooks_and_schemas_are_never_candidates():
    assert all(c.pipeline for c in candidates())
    assert not candidates([hit(RUNBOOK, 0.71), hit(SCHEMA, 0.70)])


def test_one_failure_does_not_trigger_a_clarification():
    one = [hit(SENSOR_LOG, 0.69), hit(SENSOR_INC, 0.68), hit(RUNBOOK, 0.70)]
    assert candidates(one) == []


def test_the_similarity_floor_is_enforced():
    # Floor 0.68 drops the load log (0.676): two failures remain.
    assert [c.best_similarity for c in candidates(min_similarity=0.68)] == [0.691, 0.689]
    # Floor 0.69 keeps only the sensor run (0.691): one failure, so no clarification.
    assert candidates(min_similarity=0.69) == []
    weak = [hit(PULL_INC, 0.669), hit(LOAD_LOG, 0.60), hit(SENSOR_LOG, 0.691)]
    assert candidates(weak) == []  # 0.669 < 0.67: only one failure qualifies


def test_more_failures_than_the_cap_means_no_clarification():
    assert len(candidates(max_candidates=3)) == 3
    assert candidates(max_candidates=2) == []  # three qualify: too vague for a short list
    assert candidates(max_candidates=1) == []


@pytest.mark.parametrize("anchors", [
    Anchors(incidents=("INC-2045",)), Anchors(pipelines=("payments_sync",)),
    Anchors(tasks=("pull_payments_api",)), Anchors(errors=("HTTP 429",)),
    Anchors(identifiers=("ml_features",)),
])
def test_a_question_with_a_subject_of_its_own_stays_a_refusal(anchors):
    assert candidates(own_anchors=anchors) == []


def test_unknown_identifiers_keep_the_refusal():
    assert candidates(unknown_identifiers=("ml_features",)) == []


def test_labels_and_descriptions_come_only_from_index_metadata():
    out = {c.id: c for c in candidates()}
    pull = out["run:payments_sync/pull_payments_api/2026-09-16"]
    assert pull.description == "INC-2043: Upstream API rate limit (HTTP 429)"
    load = out["run:payments_sync/load_payments_fact/2026-09-17"]
    assert load.incident_id is None and "(INC" not in load.label  # no incident invented
    assert load.description == "Logged error: psycopg2.errors.UniqueViolation: duplicate key"
    assert pull.subject == {"pipeline": "payments_sync", "task": "pull_payments_api",
                            "date": "2026-09-16", "incident_id": "INC-2043"}
    assert load.subject == {"pipeline": "payments_sync", "task": "load_payments_fact",
                            "date": "2026-09-17"}


def test_missing_or_partial_metadata_is_never_filled_in():
    docs = dict(DOCS)
    docs[PULL_INC] = DocInfo(PULL_INC, "incident", None, incident_id="INC-2043")  # no run info
    out = candidates(docs=docs)
    doc = next(c for c in out if c.id == f"doc:{'INC-2043'}")
    assert (doc.pipeline, doc.task, doc.date) == (None, None, None)
    assert doc.label == PULL_INC and doc.subject == {"incident_id": "INC-2043"}
    assert candidates(docs={}) == []  # hits that are not indexed are ignored


# ---- the protected benchmark refusals (measured: gate-refused, must stay so) --------------

CATALOG = LogCatalog(frozenset({("payments_sync", "pull_payments_api", "2026-09-10"),
                                ("orders_etl", "transform_orders", "2026-09-04"),
                                ("shipments_sync", "transform_shipments", "2026-10-06")}))


@pytest.mark.parametrize("qid, question", [
    ("Q047", "Why did the ml_features pipeline fail on 2026-09-09?"),
    ("Q050", "Who was the on-call engineer during incident INC-9999?"),
    ("M014", "How many customer orders were delivered late by the carrier because of "
             "INC-2045?"),
])
def test_protected_questions_name_a_subject_so_they_are_never_clarified(qid, question):
    own = extract_anchors(question, CATALOG)
    assert own.has_subject, qid
    assert candidates(own_anchors=own) == []  # even with several failures in the evidence


def test_m015_has_no_subject_but_its_evidence_is_below_the_floor():
    # M015 ("Who at logistics-eng approved migration 0311?") is vague, but nothing it
    # retrieves reaches the floor (best 0.618 on the real index).
    question = "Who at logistics-eng approved migration 0311?"
    assert not extract_anchors(question, CATALOG).has_subject
    weak = [hit(SENSOR_LOG, 0.618), hit(PULL_INC, 0.60), hit(LOAD_LOG, 0.59)]
    assert candidates(weak) == []


# ---- a picked candidate -------------------------------------------------------------------


def test_a_picked_run_is_described_from_the_index_and_searched_with_its_label():
    sel = describe_selection("why did the payment fail?",
                             {"pipeline": "finance_reporting", "task": "wait_for_payments_sync",
                              "date": "2026-10-12"}, [DOCS[SENSOR_LOG], DOCS[SENSOR_INC]])
    assert sel.label == "finance_reporting / wait_for_payments_sync on 2026-10-12 (INC-2046)"
    assert sel.retrieval_query == ("why did the payment fail? (Selected: finance_reporting / "
                                   "wait_for_payments_sync on 2026-10-12 (INC-2046))")


@pytest.mark.parametrize("subject", [
    {"incident_id": "INC-9999"},                                         # not indexed
    {"pipeline": "payments_sync", "task": "pull_payments_api", "date": "2026-01-01"},
    {"incident_id": "INC-2043", "pipeline": "finance_reporting",          # fields disagree
     "task": "wait_for_payments_sync", "date": "2026-10-12"},
    {},
])
def test_a_subject_that_matches_no_indexed_failure_is_rejected(subject):
    assert describe_selection("q", subject, DOCS.values()) is None


# ---- answer_question ----------------------------------------------------------------------


class ScriptedRetriever:
    name = "scripted"

    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def retrieve(self, question, k, filters=None):
        self.queries.append(question)
        return list(self.hits[:k])


def never_called():
    pytest.fail("the LLM must not be called")


def clarifier_for(docs=DOCS, anchors=NONE):
    calls = []

    def clarify(question, hits):
        calls.append(question)
        return find_candidates(hits, docs, own_anchors=anchors, min_similarity=FLOOR,
                               max_candidates=MAX)
    clarify.calls = calls
    return clarify


def ask(hits, **kw):
    return answer_question("why did the payment fail?", retriever=ScriptedRetriever(hits),
                           llm_factory=never_called, k=8, min_similarity=0.72, **kw)


def test_a_vague_gate_refusal_becomes_a_clarification_without_an_llm_call():
    out = ask(PAYMENT_HITS, clarifier=clarifier_for())
    assert out.status == "insufficient_evidence" and out.insufficient_evidence is True
    assert out.insufficient_reason == "ambiguous_query"
    assert out.answer == CLARIFICATION_TEXT
    assert out.gate["passed"] is False and out.timings["llm_called"] is False
    assert out.citations == [] and out.evidence == [] and out.next_steps == []
    assert [c["id"] for c in out.clarification["candidates"]] == [c.id for c in candidates()]


def test_without_a_clarifier_or_candidates_the_refusal_is_unchanged():
    assert ask(PAYMENT_HITS).insufficient_reason == "below_min_similarity"
    out = ask(PAYMENT_HITS, clarifier=clarifier_for(anchors=Anchors(pipelines=("x",))))
    assert out.insufficient_reason == "below_min_similarity" and out.clarification is None


def test_no_hits_is_never_clarified():
    clarify = clarifier_for()
    assert ask([], clarifier=clarify).insufficient_reason == "no_hits"
    assert clarify.calls == []


def test_a_question_that_passes_the_gate_is_never_clarified():
    from incident_copilot.generation import FakeLLMProvider
    clarify = clarifier_for()
    strong = [hit(SENSOR_LOG, 0.80), hit(PULL_INC, 0.79)]
    llm = FakeLLMProvider([{"insufficient_evidence": True, "summary": "missing",
                            "evidence": [], "inference": [], "next_steps": []}])
    out = answer_question("q", retriever=ScriptedRetriever(strong), llm_factory=lambda: llm,
                          k=8, min_similarity=0.72, clarifier=clarify)
    assert out.insufficient_reason == "model_insufficient_evidence" and clarify.calls == []


@pytest.mark.parametrize("status", [CONTEXTUALIZED, AMBIGUOUS])
def test_a_follow_up_resolved_from_history_or_a_picked_subject_is_not_clarified(status):
    clarify = clarifier_for()
    ctx = Contextualization(status, "r", "why did the payment fail?",
                            "why did the payment fail? (Selected: x)", subjects=("x",),
                            candidates=("a", "b"))
    out = ask(PAYMENT_HITS, clarifier=clarify, contextualization=ctx)
    assert out.insufficient_reason != "ambiguous_query" and clarify.calls == []


def test_a_follow_up_without_context_can_still_be_clarified():
    ctx = Contextualization(NO_CONTEXT, "r", "why did the payment fail?",
                            "why did the payment fail?")
    assert ask(PAYMENT_HITS, clarifier=clarifier_for(),
               contextualization=ctx).insufficient_reason == "ambiguous_query"


def test_after_a_pick_the_normal_gate_still_decides():
    # The picked subject narrows retrieval; weak evidence for it is still refused.
    ctx = Contextualization(CONTEXTUALIZED, "subject selected from a clarification",
                            "why did the payment fail?",
                            "why did the payment fail? (Selected: payments_sync / "
                            "pull_payments_api on 2026-09-16 (INC-2043))",
                            subjects=("payments_sync / pull_payments_api on 2026-09-16",))
    r = ScriptedRetriever([hit(PULL_INC, 0.70)])
    out = answer_question("why did the payment fail?", retriever=r, llm_factory=never_called,
                          k=8, min_similarity=0.72, contextualization=ctx)
    assert out.insufficient_reason == "below_min_similarity"
    assert r.queries == [ctx.retrieval_query]  # retrieval used the picked subject


def test_candidate_dicts_carry_everything_the_client_needs():
    c = candidates()[1]
    d = c.as_dict()
    assert set(d) == {"id", "label", "description", "incident_id", "pipeline", "task", "date",
                      "paths", "best_similarity", "subject"}
    assert d["subject"] == c.subject and isinstance(c, Candidate)


# ---- topic: unrelated failures are not offered --------------------------------------------

SHIP_RENAME_INC = "extra/incidents/INC-2045.md"
SHIP_RENAME_LOG = "extra/logs/shipments_sync/2026-10-06__transform_shipments.log"
SHIP_DUP_LOG = "extra/logs/shipments_sync/2026-10-15__load_shipments_fact.log"
SENSOR_INC_ORDERS = "incidents/INC-2037.md"
DUP_RUNBOOK = "runbooks/duplicate-key-violation.md"
RENAME_RUNBOOK = "runbooks/column-renamed-upstream.md"

SHIP_DOCS = {
    SHIP_RENAME_INC: DocInfo(SHIP_RENAME_INC, "incident",
                             "INC-2045: Upstream column rename breaks transform_shipments",
                             incident_id="INC-2045", pipeline="shipments_sync",
                             task="transform_shipments", date="2026-10-06"),
    SHIP_RENAME_LOG: DocInfo(SHIP_RENAME_LOG, "log", "t", pipeline="shipments_sync",
                             task="transform_shipments", date="2026-10-06",
                             error="KeyError: 'carrier_code'"),
    SHIP_DUP_LOG: DocInfo(SHIP_DUP_LOG, "log", "t", pipeline="shipments_sync",
                          task="load_shipments_fact", date="2026-10-15",
                          error="psycopg2.errors.UniqueViolation: duplicate key"),
    SENSOR_INC_ORDERS: DocInfo(SENSOR_INC_ORDERS, "incident",
                               "INC-2037: Sensor timed out waiting for upstream in orders_etl",
                               incident_id="INC-2037", pipeline="orders_etl",
                               task="wait_for_orders_export", date="2026-08-21"),
    DUP_RUNBOOK: DocInfo(DUP_RUNBOOK, "runbook", "Runbook: Duplicate key violation on load"),
    RENAME_RUNBOOK: DocInfo(RENAME_RUNBOOK, "runbook", "Runbook: Column renamed upstream"),
}


def ship_hit(path, score, section="sec", origin="vector", linked_from=None, companion_of=None):
    meta = {"provenance": {"companion_of": companion_of}} if companion_of else {}
    return Hit(chunk_id=f"{path}#{section}", document_id=f"doc:{path}", path=path,
               source_type=SHIP_DOCS[path].source_type, section=section, ts_start=None,
               ts_end=None, text=f"{path} {section}", chunk_index=0, content_hash="h",
               pipeline_name=None, rank=1, origin=origin, linked_from=linked_from,
               metadata=meta,
               scores={"vector_similarity": score, "vector_distance": round(1 - score, 6)})


SHIP_HITS = [ship_hit(SHIP_RENAME_INC, 0.717, "Summary"),
             ship_hit(SENSOR_INC_ORDERS, 0.691, "Timeline"),
             ship_hit(SHIP_DUP_LOG, 0.686, "middle"),
             ship_hit(DUP_RUNBOOK, 0.515, "Symptoms", "error_link", linked_from=SHIP_DUP_LOG),
             ship_hit(DUP_RUNBOOK, 0.499, "Resolution", "resolution_companion",
                      companion_of=DUP_RUNBOOK),
             ship_hit(SHIP_RENAME_INC, 0.642, "Inferred cause"),
             ship_hit(RENAME_RUNBOOK, 0.60, "Resolution")]


def test_failures_sharing_no_topic_word_with_the_question_are_left_out():
    out = find_candidates(SHIP_HITS, SHIP_DOCS, own_anchors=NONE, min_similarity=FLOOR,
                          max_candidates=MAX, question="Why did the shipment fail?")
    assert [c.label for c in out] == [
        "shipments_sync / transform_shipments on 2026-10-06 (INC-2045)",
        "shipments_sync / load_shipments_fact on 2026-10-15"]  # orders_etl INC-2037 dropped


def test_a_question_without_topic_words_keeps_every_failure():
    out = find_candidates(SHIP_HITS, SHIP_DOCS, own_anchors=NONE, min_similarity=FLOOR,
                          max_candidates=MAX, question="Why did it fail?")
    assert len(out) == 3


def test_topic_terms_ignore_generic_failure_words():
    from incident_copilot.generation.clarify import topic_terms

    assert topic_terms("why did the task fail? what is the solution for it") == set()
    assert topic_terms("i mean about the shipment") == {"shipm"}
    assert "shipm" in topic_terms("shipments_sync / load_shipments_fact")


@pytest.mark.parametrize("question, expected", [
    ("why did the shipment task fail?", True),
    ("what caused the failure of the shipment?", True),
    ("A task disappears with return code -9 and OOMKilled. What should I do?", False),  # Q022
    ("Which past incident was caused by the payments provider rate limiting?", False),  # Q037
    ("What is the retention period for the clickstream Kafka topic?", False),           # Q049
])
def test_only_questions_about_one_particular_failure_are_clarified_past_the_gate(question,
                                                                                  expected):
    from incident_copilot.generation.clarify import asks_about_one_failure

    assert asks_about_one_failure(question) is expected


def test_a_vague_question_that_passes_the_gate_is_clarified_not_mixed():
    strong = [ship_hit(SHIP_RENAME_INC, 0.749, "Summary"), ship_hit(SHIP_DUP_LOG, 0.718),
              ship_hit(DUP_RUNBOOK, 0.70, "Resolution")]

    def clarify(question, hits):
        return find_candidates(hits, SHIP_DOCS, own_anchors=NONE, min_similarity=FLOOR,
                               max_candidates=MAX, question=question)

    out = answer_question("why did the shipment task fail?", retriever=ScriptedRetriever(strong),
                          llm_factory=never_called, k=8, min_similarity=0.72, clarifier=clarify)
    assert out.insufficient_reason == "ambiguous_query" and out.gate["passed"] is True
    assert len(out.clarification["candidates"]) == 2 and out.citations == []


def test_a_how_to_question_that_passes_the_gate_is_answered_as_before():
    from incident_copilot.generation import FakeLLMProvider

    strong = [ship_hit(SHIP_RENAME_INC, 0.80), ship_hit(SHIP_DUP_LOG, 0.79)]
    llm = FakeLLMProvider([{"insufficient_evidence": True, "summary": "x", "evidence": [],
                            "inference": [], "next_steps": []}])
    calls = []
    out = answer_question("A load task fails with a duplicate key. How do I fix it?",
                          retriever=ScriptedRetriever(strong), llm_factory=lambda: llm, k=8,
                          min_similarity=0.72,
                          clarifier=lambda q, h: calls.append(q) or candidates())
    assert calls == [] and out.insufficient_reason == "model_insufficient_evidence"


# ---- choices for "what is the solution for it?" after an answer about several failures -----


def test_an_ambiguous_follow_up_offers_the_earlier_failures_as_choices():
    from incident_copilot.generation.clarify import candidates_from_docs

    docs = [SHIP_DOCS[SHIP_RENAME_INC], SHIP_DOCS[SHIP_RENAME_LOG], SHIP_DOCS[SHIP_DUP_LOG],
            SHIP_DOCS[DUP_RUNBOOK]]
    out = candidates_from_docs(docs)
    assert [c.id for c in out] == ["run:shipments_sync/transform_shipments/2026-10-06",
                                   "run:shipments_sync/load_shipments_fact/2026-10-15"]
    assert out[0].paths == (SHIP_RENAME_INC, SHIP_RENAME_LOG) and out[0].best_similarity is None
    ctx = Contextualization(AMBIGUOUS, "r", "what is the solution for it?",
                            "what is the solution for it?", candidates=("a", "b"),
                            candidate_docs=tuple(docs))
    outcome = answer_question("what is the solution for it?", retriever=ScriptedRetriever([]),
                              llm_factory=never_called, k=8, min_similarity=0.72,
                              contextualization=ctx)
    assert outcome.insufficient_reason == "ambiguous_reference"
    assert [c["id"] for c in outcome.clarification["candidates"]] == [c.id for c in out]


# ---- a picked failure: evidence about other failures is left out --------------------------

RENAME_RUN = ("run", "shipments_sync", "transform_shipments", "2026-10-06")
DUP_RUN = ("run", "shipments_sync", "load_shipments_fact", "2026-10-15")


def test_scoping_to_the_rename_drops_the_duplicate_key_failure_and_its_remediation():
    from incident_copilot.generation.clarify import scope_hits

    kept = scope_hits(SHIP_HITS, SHIP_DOCS, RENAME_RUN)
    assert [(h.path, h.section) for h in kept] == [
        (SHIP_RENAME_INC, "Summary"), (SHIP_RENAME_INC, "Inferred cause"),
        (RENAME_RUNBOOK, "Resolution")]
    # The duplicate-key runbook came only via the other run's error line, so it and its
    # Resolution companion are gone; the orders_etl incident is another failure too.
    assert not any(h.path in (DUP_RUNBOOK, SHIP_DUP_LOG, SENSOR_INC_ORDERS) for h in kept)


def test_scoping_to_the_duplicate_key_failure_drops_the_rename():
    from incident_copilot.generation.clarify import scope_hits

    kept = scope_hits(SHIP_HITS, SHIP_DOCS, DUP_RUN)
    assert SHIP_RENAME_INC not in {h.path for h in kept}  # no rename evidence to attribute
    assert {h.path for h in kept} == {SHIP_DUP_LOG, DUP_RUNBOOK, RENAME_RUNBOOK}


def test_a_runbook_found_by_the_question_itself_is_kept():
    from incident_copilot.generation.clarify import scope_hits

    hits = [ship_hit(DUP_RUNBOOK, 0.80, "Symptoms")]  # plain vector hit, not a link
    assert scope_hits(hits, SHIP_DOCS, RENAME_RUN) == hits


def test_scoped_evidence_still_goes_through_the_gate():
    from incident_copilot.generation.clarify import scope_hits

    # Only the other failure was strong: after scoping, the best similarity is too low.
    hits = [ship_hit(SHIP_DUP_LOG, 0.80), ship_hit(SHIP_RENAME_INC, 0.65)]
    out = answer_question("what is the solution for it?", retriever=ScriptedRetriever(hits),
                          llm_factory=never_called, k=8, min_similarity=0.72,
                          scope=lambda h: scope_hits(h, SHIP_DOCS, RENAME_RUN))
    assert out.insufficient_reason == "below_min_similarity"
    assert [h.path for h in out.hits] == [SHIP_RENAME_INC] and out.hits[0].rank == 1


def test_a_selection_reports_its_run_and_validated_subject():
    sel = describe_selection("q", {"incident_id": "INC-2045"},
                             [SHIP_DOCS[SHIP_RENAME_INC], SHIP_DOCS[SHIP_RENAME_LOG]])
    assert sel.run_key == RENAME_RUN
    assert sel.subject == {"pipeline": "shipments_sync", "task": "transform_shipments",
                           "date": "2026-10-06", "incident_id": "INC-2045"}


# ---- next steps cite real evidence ----------------------------------------------------------


def _answer_with_steps(steps):
    from incident_copilot.generation import FakeLLMProvider

    llm = FakeLLMProvider([{"insufficient_evidence": False, "summary": "Renamed column.",
                            "evidence": [{"claim": "KeyError", "evidence_ids": ["E1"]}],
                            "inference": [], "next_steps": steps}])
    hits = [ship_hit(SHIP_RENAME_INC, 0.80), ship_hit(RENAME_RUNBOOK, 0.78, "Resolution")]
    hits = [replace(h, rank=i) for i, h in enumerate(hits, start=1)]  # E1, E2
    return answer_question("q", retriever=ScriptedRetriever(hits), llm_factory=lambda: llm,
                           k=8, min_similarity=0.72)


def test_next_steps_keep_their_validated_evidence_ids():
    out = _answer_with_steps([{"step": "Map carrier_id to carrier_code", "evidence_ids": ["E2"]},
                              {"step": "Clear the task", "evidence_ids": ["[E1]", "E2"]}])
    assert out.next_steps == ["Map carrier_id to carrier_code", "Clear the task"]
    assert out.next_step_evidence == [["E2"], ["E1", "E2"]]


def test_a_step_citing_evidence_that_was_not_retrieved_is_dropped():
    out = _answer_with_steps([{"step": "Upsert the load", "evidence_ids": ["E9"]},
                              {"step": "Map carrier_id", "evidence_ids": ["E2"]}])
    assert out.next_steps == ["Map carrier_id"] and out.next_step_evidence == [["E2"]]
    assert {"field": "next_steps", "claim": "Upsert the load", "ids": ["E9"],
            "invalid_ids": ["E9"]} in out.dropped_claims


def test_a_step_without_evidence_is_kept_but_marked_unsupported():
    out = _answer_with_steps(["Restart everything", {"step": "Check", "evidence_ids": []}])
    assert out.next_steps == ["Restart everything", "Check"]
    assert out.next_step_evidence == [[], []]  # the page shows these as unsupported


def test_claim_validation_is_unchanged_by_next_step_citations():
    out = _answer_with_steps([{"step": "x", "evidence_ids": ["E2"]}])
    assert out.evidence == [{"claim": "KeyError", "evidence_ids": ["E1"]}]
    assert [c.evidence_id for c in out.citations] == ["E1"]  # steps do not add citations
    assert out.validation == {"dropped_evidence_claims": 0, "dropped_inference_claims": 0,
                              "invalid_ids": []}
