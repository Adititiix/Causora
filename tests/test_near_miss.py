"""Near misses, the selected failure's own remediation, and next-step labels
(generation/clarify.py resolve_near_miss / remediation_hits, generation/answer.py).

Offline: hand-made hits and index metadata shaped like the real index (measured 2026-11-04):
  * "why did the shipments fail?" best 0.711 (< 0.72); the top 8 hold INC-2045 only, the
    index has two shipments failures (INC-2045, INC-2047)  -> clarify, never refuse or answer;
  * "why did the loyalty join fail?" best 0.693; the index has ONE join_loyalty failure, and
    another failure's error text names the loyalty table  -> answer, scoped to that failure;
  * "why did the kubernetes upgrade fail?" / "what is the on-call phone number for the
    payments team?"  -> stay refusals;
  * INC-2046 picked: its own "Resolution and verification" / "Prevention" were missing from
    the evidence, so the only remediation was the generic sensor-timeout runbook.
"""

import json
from dataclasses import replace

import pytest

from incident_copilot.generation import FakeLLMProvider, answer_question
from incident_copilot.generation.answer import classify_next_steps, parse_model_answer
from incident_copilot.generation.clarify import (
    CLARIFICATION_TEXT,
    NearMiss,
    pick_remediation_rows,
    resolve_near_miss,
)
from incident_copilot.retrieval import Hit
from incident_copilot.retrieval.contextualize import Anchors, DocInfo

FLOOR, MAX = 0.67, 5
NONE = Anchors()

SHIP_INC = "extra/incidents/INC-2045.md"
SHIP_LOG = "extra/logs/shipments_sync/2026-10-06__transform_shipments.log"
SHIP2_INC = "extra/incidents/INC-2047.md"
SHIP2_LOG = "extra/logs/shipments_sync/2026-10-15__load_shipments_fact.log"
ORDERS_INC = "incidents/INC-2037.md"
LOYALTY_LOG = "logs/customer_360/2026-09-22__join_loyalty.log"
PROFILE_LOG = "logs/customer_360/2026-08-07__build_customer_profile.log"
INVENTORY_LOG = "logs/inventory_snapshot/2026-08-16__extract_inventory.log"
PAY_SCHEMA = "extra/schemas/pipeline_payments_sync.md"
DUP_RUNBOOK = "runbooks/duplicate-key-violation.md"
RENAME_RUNBOOK = "runbooks/column-renamed-upstream.md"
SENSOR_INC = "extra/incidents/INC-2046.md"
SENSOR_LOG = "extra/logs/finance_reporting/2026-10-12__wait_for_payments_sync.log"
SENSOR_RUNBOOK = "runbooks/sensor-timeout.md"


def _run(path, kind, pipeline, task, day, title=None, incident=None, error=None):
    return DocInfo(path, kind, title, incident_id=incident, pipeline=pipeline, task=task,
                   date=day, error=error)


CATALOG = (
    _run(SHIP_INC, "incident", "shipments_sync", "transform_shipments", "2026-10-06",
         "INC-2045: carrier_code renamed upstream", "INC-2045"),
    _run(SHIP_LOG, "log", "shipments_sync", "transform_shipments", "2026-10-06",
         error="KeyError: 'carrier_code'"),
    _run(SHIP2_INC, "incident", "shipments_sync", "load_shipments_fact", "2026-10-15",
         "INC-2047: duplicate shipments load", "INC-2047"),
    _run(SHIP2_LOG, "log", "shipments_sync", "load_shipments_fact", "2026-10-15",
         error="psycopg2.errors.UniqueViolation: duplicate key value"),
    _run(ORDERS_INC, "incident", "orders_etl", "wait_for_orders_export", "2026-08-21",
         "INC-2037: orders export sensor timeout", "INC-2037"),
    _run(LOYALTY_LOG, "log", "customer_360", "join_loyalty", "2026-09-22",
         error="KeyError: 'customer_tier'"),
    # Its error text names the loyalty table: it must not match "loyalty join".
    _run(PROFILE_LOG, "log", "customer_360", "build_customer_profile", "2026-08-07",
         error="DataQualityError: freshness check failed for loyalty_members"),
    _run(INVENTORY_LOG, "log", "inventory_snapshot", "extract_inventory", "2026-08-16",
         error="psycopg2.OperationalError: connection to server"),
    _run(SENSOR_INC, "incident", "finance_reporting", "wait_for_payments_sync", "2026-10-12",
         "INC-2046: Rate-limited payments pull times out the finance_reporting sensor",
         "INC-2046"),
)
DOCS = {d.path: d for d in CATALOG} | {
    PAY_SCHEMA: DocInfo(PAY_SCHEMA, "schema", "pipeline_payments_sync",
                        pipeline="payments_sync"),
    DUP_RUNBOOK: DocInfo(DUP_RUNBOOK, "runbook", "Runbook: duplicate key violation"),
    RENAME_RUNBOOK: DocInfo(RENAME_RUNBOOK, "runbook", "Runbook: column renamed upstream"),
    SENSOR_RUNBOOK: DocInfo(SENSOR_RUNBOOK, "runbook", "Runbook: sensor timed out"),
    SENSOR_LOG: _run(SENSOR_LOG, "log", "finance_reporting", "wait_for_payments_sync",
                     "2026-10-12", error="AirflowSensorTimeout: Sensor has timed out"),
}


def hit(path, score, section="sec", rank=1, origin="vector", linked_from=None):
    source_type = DOCS[path].source_type if path in DOCS else "incident"
    return Hit(chunk_id=f"{path}#{section}", document_id=f"doc:{path}", path=path,
               source_type=source_type, section=section, ts_start=None, ts_end=None,
               text=f"{path} {section}", chunk_index=0, content_hash="h",
               pipeline_name=DOCS[path].pipeline if path in DOCS else None, rank=rank,
               scores={"vector_similarity": score, "vector_distance": round(1 - score, 6)},
               origin=origin, linked_from=linked_from)


def ranked(hits):
    return [replace(h, rank=i) for i, h in enumerate(hits, start=1)]


SHIPMENTS_HITS = ranked([
    hit(SHIP_INC, 0.711, "Summary"), hit(SHIP_LOG, 0.679, "error context"),
    hit(DUP_RUNBOOK, 0.545, "Overview", origin="error_link", linked_from=SHIP_LOG),
    hit(ORDERS_INC, 0.675, "Timeline (UTC)"), hit(SHIP_INC, 0.627, "Inferred cause")])
LOYALTY_HITS = ranked([
    hit(LOYALTY_LOG, 0.693, "error context"),
    hit(RENAME_RUNBOOK, 0.549, "Overview", origin="error_link", linked_from=LOYALTY_LOG),
    hit(PROFILE_LOG, 0.654, "error context")])
KUBERNETES_HITS = ranked([hit(INVENTORY_LOG, 0.656, "run summary")])
PHONE_HITS = ranked([hit(PAY_SCHEMA, 0.657, "pipeline_payments_sync"),
                     hit(SENSOR_INC, 0.644, "Overview")])


def near_miss(question, hits, **kw):
    options = {"own_anchors": NONE, "min_similarity": FLOOR, "max_candidates": MAX} | kw
    return resolve_near_miss(question, hits, DOCS, CATALOG, **options)


# ---- the decision ---------------------------------------------------------------------------


def test_several_indexed_failures_on_the_topic_are_asked_about_even_if_one_was_retrieved():
    out = near_miss("Why did the shipments fail?", SHIPMENTS_HITS)
    assert out.resolved is None
    assert [c.id for c in out.candidates] == [
        "run:shipments_sync/transform_shipments/2026-10-06",  # retrieved: most similar first
        "run:shipments_sync/load_shipments_fact/2026-10-15"]  # only in the index
    assert [c.best_similarity for c in out.candidates] == [0.711, None]
    assert out.candidates[1].incident_id == "INC-2047"  # labelled from index metadata


def test_the_one_indexed_failure_on_the_topic_is_resolved_and_the_evidence_scoped_to_it():
    out = near_miss("Why did the loyalty join fail?", LOYALTY_HITS)
    assert out.candidates == ()
    assert out.resolved.id == "run:customer_360/join_loyalty/2026-09-22"
    assert out.resolved.subject == {"pipeline": "customer_360", "task": "join_loyalty",
                                    "date": "2026-09-22"}
    # The other failure's log is left out; the runbook linked from this failure's log stays.
    assert [h.path for h in out.hits] == [LOYALTY_LOG, RENAME_RUNBOOK]


def test_a_failure_matches_by_its_identity_not_by_tables_its_error_names():
    out = near_miss("Why did the loyalty join fail?", LOYALTY_HITS)
    assert PROFILE_LOG not in out.resolved.paths


@pytest.mark.parametrize("question, hits", [
    ("Why did the kubernetes upgrade fail?", KUBERNETES_HITS),  # nothing indexed matches
    ("What is the on-call phone number for the payments team?", PHONE_HITS),  # not a failure
])
def test_genuinely_unsupported_questions_stay_refused(question, hits):
    assert near_miss(question, hits) == NearMiss()


def test_the_matched_failure_must_itself_be_retrieved_at_the_floor():
    weak = ranked([replace(h, scores={"vector_similarity": 0.66, "vector_distance": 0.34})
                   for h in LOYALTY_HITS])
    assert near_miss("Why did the loyalty join fail?", weak) == NearMiss()


def test_evidence_off_the_topic_never_starts_a_near_miss():
    # "shipments" words, but the only strong hit is about orders: no shipments evidence.
    hits = ranked([hit(ORDERS_INC, 0.70, "Summary")])
    assert near_miss("Why did the shipments fail?", hits) == NearMiss()


def test_a_question_with_its_own_subject_or_an_unknown_identifier_is_not_a_near_miss():
    assert near_miss("Why did the shipments fail?", SHIPMENTS_HITS,
                     own_anchors=Anchors(incidents=("INC-9999",))) == NearMiss()
    assert near_miss("Why did the shipments fail?", SHIPMENTS_HITS,
                     unknown_identifiers=("ml_features",)) == NearMiss()


def test_too_many_matching_failures_is_a_refusal_not_a_long_list():
    assert near_miss("Why did the shipments fail?", SHIPMENTS_HITS,
                     max_candidates=1) == NearMiss()


# ---- answer_question with a near-miss resolver ----------------------------------------------


class ScriptedRetriever:
    name = "scripted"

    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k, filters=None):
        return list(self.hits)


def never_called():
    raise AssertionError("the LLM must not be called")


ANSWER = {"insufficient_evidence": False, "summary": "The join failed on customer_tier.",
          "evidence": [{"claim": "KeyError: 'customer_tier'", "evidence_ids": ["E1"]}],
          "inference": [], "unknowns": ["Why the column was missing is not documented."],
          "next_steps": [{"step": "Follow the column-rename runbook", "evidence_ids": ["E2"],
                          "kind": "recovery"}]}


def resolver(question, hits):
    return near_miss(question, hits)


def test_a_near_miss_clarification_calls_no_llm_and_keeps_the_real_gate():
    out = answer_question("Why did the shipments fail?",
                          retriever=ScriptedRetriever(SHIPMENTS_HITS),
                          llm_factory=never_called, k=8, min_similarity=0.72,
                          near_miss=resolver)
    assert out.insufficient_reason == "ambiguous_query" and out.answer == CLARIFICATION_TEXT
    assert out.clarification["reason"] == "several_failures_in_index"
    assert out.gate == {"best_similarity": 0.711, "min_similarity": 0.72, "passed": False}
    assert out.timings["llm_called"] is False


def test_a_resolved_near_miss_is_answered_about_that_failure_only():
    prompts = []

    class Recording(FakeLLMProvider):
        def generate(self, system, user, schema):
            prompts.append(user)
            return super().generate(system, user, schema)

    out = answer_question("Why did the loyalty join fail?",
                          retriever=ScriptedRetriever(LOYALTY_HITS),
                          llm_factory=lambda: Recording([json.dumps(ANSWER)]), k=8,
                          min_similarity=0.72, near_miss=resolver)
    assert out.status == "answered"
    assert out.gate["passed"] is False and out.gate["best_similarity"] == 0.693
    assert out.gate["accepted_by"] == {"rule": "sole_matching_failure",
                                       "candidate": "run:customer_360/join_loyalty/2026-09-22"}
    assert [h.path for h in out.hits] == [LOYALTY_LOG, RENAME_RUNBOOK]
    assert PROFILE_LOG not in prompts[0]  # the other failure never reaches the model
    assert "matches exactly one failure in the index" in prompts[0]
    conversation = out.retrieval_info["conversation"]
    assert conversation["status"] == "matched_in_index"
    assert conversation["subject"]["task"] == "join_loyalty"  # a follow-up keeps it
    assert out.unknowns == ["Why the column was missing is not documented."]


def test_without_a_resolver_a_near_miss_is_refused_as_before():
    out = answer_question("Why did the loyalty join fail?",
                          retriever=ScriptedRetriever(LOYALTY_HITS),
                          llm_factory=never_called, k=8, min_similarity=0.72)
    assert out.insufficient_reason == "below_min_similarity"


def test_a_picked_or_kept_subject_is_never_re_resolved():
    from incident_copilot.retrieval.contextualize import CONTEXTUALIZED, Contextualization

    def must_not_run(question, hits):
        raise AssertionError("resolver called for a decided subject")

    ctx = Contextualization(CONTEXTUALIZED, "subject kept from the previous turn",
                            "How do we prevent it?", "How do we prevent it? (Selected: x)",
                            subjects=("x",))
    out = answer_question("How do we prevent it?", retriever=ScriptedRetriever(LOYALTY_HITS),
                          llm_factory=never_called, k=8, min_similarity=0.72,
                          contextualization=ctx, near_miss=must_not_run)
    assert out.insufficient_reason == "below_min_similarity"


def test_a_passing_question_never_consults_the_near_miss_resolver():
    def must_not_run(question, hits):
        raise AssertionError("resolver called for a passing question")

    hits = ranked([hit(SHIP_INC, 0.80, "Summary")])
    out = answer_question("How do I fix a renamed column?", retriever=ScriptedRetriever(hits),
                          llm_factory=lambda: FakeLLMProvider([json.dumps(ANSWER)]), k=8,
                          min_similarity=0.72, near_miss=must_not_run)
    assert out.status == "answered"


# ---- the selected failure's own remediation ------------------------------------------------


def _row(chunk_id, section, source_type="incident"):
    # (chunk id, document id, path, source type, section, ...) as load_document_chunks gives.
    return (chunk_id, "d", SENSOR_INC, source_type, section)


INC_2046_ROWS = [_row("c0", "Overview"), _row("c1", "Summary"), _row("c2", "Inferred cause"),
                 _row("c3", "Resolution and verification"), _row("c4", "Prevention")]


def test_the_incidents_documented_resolution_and_prevention_are_picked():
    assert [r[0] for r in pick_remediation_rows(INC_2046_ROWS, set())] == ["c3", "c4"]


def test_sections_already_in_the_evidence_are_not_added_twice():
    assert [r[0] for r in pick_remediation_rows(INC_2046_ROWS, {"c3"})] == ["c4"]


def test_generated_reports_use_resolution_and_follow_ups():
    rows = [_row("a", "Root cause"), _row("b", "Resolution"), _row("c", "Follow-ups")]
    assert [r[0] for r in pick_remediation_rows(rows, set())] == ["b", "c"]


def test_remediation_is_added_after_the_gate_and_never_lifts_it():
    sensor = ranked([hit(SENSOR_INC, 0.70, "Overview"), hit(SENSOR_LOG, 0.69, "error context")])
    added = replace(hit(SENSOR_INC, 0.95, "Resolution and verification"),
                    origin="subject_remediation")
    out = answer_question("Why did the payments fail?", retriever=ScriptedRetriever(sensor),
                          llm_factory=never_called, k=8, min_similarity=0.72,
                          remediation=lambda paths, hits: [added], subject_paths=(SENSOR_INC,))
    assert out.insufficient_reason == "below_min_similarity"  # 0.95 did not count
    assert [h.section for h in out.hits] == ["Overview", "error context"]


def test_remediation_is_appended_with_the_next_evidence_id():
    sensor = ranked([hit(SENSOR_INC, 0.81, "Overview"), hit(SENSOR_RUNBOOK, 0.55, "Resolution")])
    added = replace(hit(SENSOR_INC, 0.80, "Resolution and verification"),
                    origin="subject_remediation")
    llm = FakeLLMProvider([json.dumps(ANSWER | {"next_steps": [
        {"step": "Cancel the backfill, then clear the sensor", "evidence_ids": ["E3"],
         "kind": "recovery"},
        {"step": "Fix or rerun the export", "evidence_ids": ["E2"], "kind": "recovery"}]})])
    out = answer_question("Why did the payments fail?", retriever=ScriptedRetriever(sensor),
                          llm_factory=lambda: llm, k=8, min_similarity=0.72,
                          remediation=lambda paths, hits: [added], subject_paths=(SENSOR_INC,))
    assert [(h.rank, h.origin) for h in out.hits][-1] == (3, "subject_remediation")
    assert out.next_step_evidence == [["E3"], ["E2"]]
    # The incident's own recovery is documented; the runbook's is only general guidance.
    assert out.next_step_support == ["incident_report", "general_guidance"]


# ---- next-step labels -----------------------------------------------------------------------


def _classify(steps, hits, subject_paths=()):
    answer = parse_model_answer(json.dumps({"insufficient_evidence": False, "summary": "s",
                                            "evidence": [], "inference": [],
                                            "next_steps": steps}))
    by_id = {f"E{h.rank}": h for h in ranked(hits)}
    texts = [s["step"] if isinstance(s, dict) else s for s in steps]
    ids = [s.get("evidence_ids", []) if isinstance(s, dict) else [] for s in steps]
    return classify_next_steps(answer, texts, ids, by_id, subject_paths)


def test_the_kind_follows_the_cited_sections_over_the_models_label():
    hits = [hit(SENSOR_INC, 0.8, "Prevention"), hit(SENSOR_RUNBOOK, 0.6, "Resolution")]
    kinds, _ = _classify([{"step": "Add a pool", "evidence_ids": ["E1"], "kind": "recovery"},
                          {"step": "Clear it", "evidence_ids": ["E2"], "kind": "prevention"}],
                         hits)
    assert kinds == ["prevention", "recovery"]


def test_mixed_or_other_sections_keep_the_models_kind():
    hits = [hit(SENSOR_INC, 0.8, "Summary"), hit(SENSOR_RUNBOOK, 0.6, "Prevention")]
    kinds, _ = _classify([{"step": "x", "evidence_ids": ["E1", "E2"], "kind": "prevention"}],
                         hits)
    assert kinds == ["prevention"]


def test_runbook_only_steps_are_general_guidance_never_the_incidents_fix():
    hits = [hit(SENSOR_RUNBOOK, 0.6, "Resolution"), hit(SENSOR_LOG, 0.7, "error context")]
    _, support = _classify([{"step": "Rerun the export", "evidence_ids": ["E1", "E2"]},
                            "Restart everything"], hits)
    assert support == ["general_guidance", "unsupported"]


def test_with_a_subject_only_its_own_incident_report_counts_as_documented():
    hits = [hit(SENSOR_INC, 0.8, "Resolution and verification"), hit(SHIP_INC, 0.7, "Summary")]
    _, support = _classify([{"step": "a", "evidence_ids": ["E1"]},
                            {"step": "b", "evidence_ids": ["E2"]}], hits, (SENSOR_INC,))
    assert support == ["incident_report", "general_guidance"]


def test_an_answer_without_kinds_or_unknowns_still_parses():
    answer = parse_model_answer(json.dumps({"insufficient_evidence": False, "summary": "s",
                                            "evidence": [], "inference": [],
                                            "next_steps": [{"step": "x",
                                                            "evidence_ids": ["E1"]}]}))
    assert answer.next_steps[0].kind == "recovery" and answer.unknowns == []
