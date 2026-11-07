"""Related documents (retrieval/relations.py and the evidence step that uses them). Offline: a
hand-made catalogue and a scripted vector backend, so no database, model or network.

The catalogue mirrors the shapes measured on eval/questions.multidoc.jsonl: an incident whose
facts name a failed task and an upstream table, the log of that task on the same date, a
table schema, a pipeline definition that reads and writes tables, and an incident that names
an earlier one.
"""

from dataclasses import replace

import numpy as np
import pytest

from incident_copilot.generation import answer_question
from incident_copilot.retrieval import EvidenceRetriever, Filters, Hit, evidence
from incident_copilot.retrieval.evidence import RELATED_ORIGIN, select_evidence
from incident_copilot.retrieval.relations import (
    Candidate,
    build_catalog,
    document_references,
    question_references,
    rank_candidates,
    related_candidates,
)

INC_A = "extra/incidents/INC-2045.md"
INC_B = "extra/incidents/INC-2046.md"
INC_OLD = "incidents/INC-2043.md"
LOG_A = "extra/logs/shipments_sync/2026-10-06__transform_shipments.log"
LOG_OTHER_DAY = "extra/logs/shipments_sync/2026-10-15__transform_shipments.log"
LOG_DOWNSTREAM = "extra/logs/finance_reporting/2026-10-06__check_source_freshness.log"
RAW = "extra/schemas/shipments_raw.md"
FACT = "extra/schemas/shipments_fact.md"
PIPE = "extra/schemas/pipeline_shipments_sync.md"
RUNBOOK = "runbooks/column-renamed-upstream.md"
UNRELATED = "incidents/INC-2031.md"

ROWS = [
    (INC_A, "incident", "INC-2045: Upstream rename breaks transform_shipments", "shipments_sync",
     {"date": "2026-10-06", "failed_task": "transform_shipments"},
     "- **Pipeline:** shipments_sync\n- **Failed task:** transform_shipments\n"
     "The rename in shipments_raw broke the transform; see runbook RB-001."),
    (INC_B, "incident", "INC-2046: Rate-limited pull", "finance_reporting",
     {"date": "2026-10-12"},
     "Neither follow-up of INC-2043 was in place; the run of payments_sync was slow."),
    (INC_OLD, "incident", "INC-2043: Upstream API rate limit", "payments_sync",
     {"date": "2026-09-16"}, "The provider API limit was exceeded."),
    (UNRELATED, "incident", "INC-2031: Row count dropped", "orders_etl",
     {"date": "2026-08-09"}, "The extract was partial. Shipments and transforms were fine."),
    (LOG_A, "log", "shipments_sync / transform_shipments / 2026-10-06", "shipments_sync",
     {"date": "2026-10-06", "task": "transform_shipments"},
     "Loaded 48,213 rows from shipments_raw\nKeyError: 'carrier_code'"),
    (LOG_OTHER_DAY, "log", "shipments_sync / transform_shipments / 2026-10-15", "shipments_sync",
     {"date": "2026-10-15", "task": "transform_shipments"}, "Loaded rows from shipments_raw"),
    (LOG_DOWNSTREAM, "log", "finance_reporting / check_source_freshness / 2026-10-06",
     "finance_reporting", {"date": "2026-10-06", "task": "check_source_freshness"},
     "freshness check failed for table shipments_fact"),
    (RAW, "schema", "Table: shipments_raw", None, {"table_name": "shipments_raw"},
     "# Table: shipments_raw\nRenamed carrier_code to carrier_id."),
    (FACT, "schema", "Table: shipments_fact", "shipments_sync",
     {"table_name": "shipments_fact", "pipelines": ["shipments_sync"]},
     "# Table: shipments_fact\nPopulated by shipments_sync."),
    (PIPE, "schema", "Pipeline: shipments_sync", None, {"dag_id": "shipments_sync"},
     "# Pipeline: shipments_sync\n- **Reads:** shipments_raw\n- **Writes:** shipments_fact"),
    (RUNBOOK, "runbook", "Runbook: Column renamed upstream", None, {"runbook_id": "RB-001"},
     "KeyError after a rename."),
]


@pytest.fixture
def cat():
    return build_catalog(ROWS)


# ---- references -------------------------------------------------------------------------


def test_incident_links_to_the_log_of_its_failed_task_on_the_same_date_only(cat):
    refs = document_references(INC_A, cat)
    assert LOG_A in refs
    assert LOG_OTHER_DAY not in refs  # same pipeline and task, another date
    assert LOG_DOWNSTREAM not in refs  # same date, but the report names neither its task


def test_a_log_links_back_to_the_incident_of_its_date_and_to_the_table_it_read(cat):
    refs = document_references(LOG_A, cat)
    assert INC_A in refs and RAW in refs
    assert INC_A not in document_references(LOG_OTHER_DAY, cat)


def test_incident_links_to_the_schema_of_a_table_it_names(cat):
    assert RAW in document_references(INC_A, cat)
    assert FACT not in document_references(INC_A, cat)  # not named


def test_pipeline_definition_links_to_the_tables_it_reads_and_writes(cat):
    assert document_references(PIPE, cat) == [FACT, RAW]  # tables in name order


def test_explicit_incident_and_runbook_ids_are_followed(cat):
    assert INC_OLD in document_references(INC_B, cat)
    assert RUNBOOK in document_references(INC_A, cat)
    assert INC_A not in document_references(INC_A, cat)  # never itself


def test_question_references_ids_tables_and_pipeline_definitions(cat):
    q = "How do INC-2043 and RB-001 relate to shipments_raw in the shipments_sync pipeline?"
    assert question_references(q, cat) == [INC_OLD, RUNBOOK, RAW, PIPE]


# ---- what is never followed -------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "Why did shipments fail after the transform?",          # plain words, no identifiers
    "Was shipments_raw_v2 or my_shipments_raw affected?",   # identifiers that only contain one
    "What happened in INC-9999 and RB-999?",                # ids nothing in the index carries
    "",
])
def test_generic_words_and_unknown_references_promote_nothing(cat, question):
    assert question_references(question, cat) == []


def test_a_document_sharing_words_but_no_identifier_is_not_related(cat):
    # INC-2031 mentions "Shipments and transforms" but no table, task, pipeline or id.
    for path in (INC_A, LOG_A, PIPE, RAW):
        assert UNRELATED not in document_references(path, cat)
    assert document_references(UNRELATED, cat) == []


def test_missing_and_malformed_rows_are_skipped():
    cat = build_catalog([
        None, (), ("only", "two"), ("", "incident", "INC-1: x", None, {}, "text"),
        ("incidents/INC-0001.md", "incident", None, None, None, None),  # no title or text
        ("logs/p/2026-01-01__t.log", "log", "t", "p", {"date": 5}, "x"),  # bad metadata
        ROWS[0],
    ])
    assert set(cat.documents) == {"incidents/INC-0001.md", "logs/p/2026-01-01__t.log", INC_A}
    assert cat.incidents == {"INC-2045": INC_A}  # an incident without a title has no id
    assert cat.logs == []
    assert document_references("not/indexed.md", cat) == []
    # INC-2045 names RB-001 and shipments_raw, but neither is in this catalogue.
    assert document_references(INC_A, cat) == []


# ---- candidates: filters, de-duplication, ranking ----------------------------------------


def test_candidates_are_unique_and_skip_what_the_evidence_already_shows(cat):
    q = "Is shipments_raw the cause of INC-2045?"
    cands = related_candidates(q, [LOG_A, LOG_A, INC_A], cat, Filters(), max_seed_documents=3)
    paths = [c.path for c in cands]
    assert len(paths) == len(set(paths))
    assert INC_A not in paths and LOG_A not in paths  # already in the evidence
    assert paths[0] == RAW and cands[0].source is None  # named by the question first


def test_only_the_first_seed_documents_are_expanded(cat):
    cands = related_candidates("x", [LOG_DOWNSTREAM, INC_B, INC_A], cat, Filters(),
                               max_seed_documents=2)
    assert {c.source for c in cands} <= {LOG_DOWNSTREAM, INC_B}
    assert INC_OLD in [c.path for c in cands]
    assert RUNBOOK not in [c.path for c in cands]  # only INC_A (third) names it


@pytest.mark.parametrize("filters, excluded", [
    (Filters(source_types=("incident", "log")), RAW),            # a type the request excludes
    (Filters(pipeline="payments_sync"), RAW),                     # another pipeline
    (Filters(log_date="2026-10-15"), LOG_A),                      # dated scope: other date
    (Filters(log_task="check_source_freshness"), LOG_A),          # dated scope: other task
    (Filters(log_pipeline="finance_reporting"), LOG_A),           # dated scope: other pipeline
])
def test_request_filters_are_respected(cat, filters, excluded):
    cands = related_candidates("x", [INC_A], cat, filters, max_seed_documents=3)
    assert excluded not in [c.path for c in cands]


def test_ranking_is_question_first_then_evidence_order_then_similarity():
    cands = [Candidate("b", "s2", 2), Candidate("a", "s1", 1), Candidate("q", None, 0),
             Candidate("c", "s1", 1), Candidate("none", "s1", 1)]
    sims = {"a": 0.70, "b": 0.99, "c": 0.75, "q": 0.10, "none": None}
    assert [c.path for c in rank_candidates(cands, sims.get, 10)] == ["q", "c", "a", "b"]
    assert [c.path for c in rank_candidates(cands, sims.get, 2)] == ["q", "c"]
    assert rank_candidates(cands, sims.get, 0) == []


# ---- placement and top_k ----------------------------------------------------------------


def hit(path, score, section="sec", chunk=None, origin="vector"):
    return Hit(chunk_id=chunk or f"{path}#{section}", document_id=f"doc:{path}", path=path,
               source_type=path.split("/")[-2].rstrip("s") if "/" in path else "incident",
               section=section, ts_start=None, ts_end=None, text=f"{path} {section}",
               chunk_index=0, content_hash="h", pipeline_name=None, rank=0,
               scores={"vector_similarity": score, "vector_distance": round(1 - score, 6)},
               origin=origin)


def test_related_chunks_never_remove_a_document_and_respect_top_k():
    primary = [hit("a", 0.9, "1"), hit("b", 0.8, "1"), hit("a", 0.7, "2"), hit("c", 0.6, "1")]
    related = [hit("r1", 0.5, origin=RELATED_ORIGIN), hit("r2", 0.4, origin=RELATED_ORIGIN)]
    out = select_evidence(primary, {}, {}, 4, True, related=related)
    assert len(out) == 4 and [h.rank for h in out] == [1, 2, 3, 4]
    # Only the extra chunk of "a" could make room: one related chunk gets in, in order.
    assert [(h.path, h.section) for h in out] == [("a", "1"), ("b", "1"), ("c", "1"),
                                                  ("r1", "sec")]


def test_related_chunks_fill_free_slots_and_never_duplicate_a_chunk():
    primary = [hit("a", 0.9), hit("b", 0.8)]
    related = [hit("a", 0.9), hit("r1", 0.5), hit("r1", 0.5)]
    out = select_evidence(primary, {}, {}, 8, True, related=related)
    assert [h.path for h in out] == ["a", "b", "r1"]


def test_a_runbook_resolution_is_never_given_up_for_a_related_chunk():
    primary = [hit("runbooks/x.md", 0.9, "Symptoms"), hit("runbooks/x.md", 0.8, "Resolution")]
    out = select_evidence(primary, {}, {}, 2, True,
                          related=[hit("r1", 0.5, origin=RELATED_ORIGIN)])
    assert [h.section for h in out] == ["Symptoms", "Resolution"]


# ---- the retriever step -----------------------------------------------------------------


class Vec:
    def __init__(self, text):
        self.text = text
        self.array = np.ones(4, dtype=np.float32)

    def to_numpy(self):
        return self.array


class FakeEmbedder:
    dim = 384
    name = "fake"

    def embed_query(self, text):
        return Vec(text)


class FakeVector:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query_vec, k, filters=None, score_vec=None):
        hits = [h for h in self.hits
                if not (filters and filters.source_types)
                or h.source_type in filters.source_types]
        return [replace(h, rank=i) for i, h in enumerate(hits[:k], start=1)]


def chunk_row(path, source_type, section, embedding):
    """A row shaped like evidence._DOCUMENT_CHUNKS_SQL (cosine with the all-ones question
    vector: (1,0,0,0) -> 0.5, (1,1,0,0) -> 0.707, (1,1,1,1) -> 1.0)."""
    return (f"{path}#{section}", f"doc:{path}", path, source_type, section, None, None,
            f"{path} {section}", 0, "h", None, {}, np.array(embedding, dtype=np.float32))


def retriever(monkeypatch, primary, rows, **options):
    monkeypatch.setattr(evidence, "load_relation_catalog", lambda: build_catalog(ROWS))
    monkeypatch.setattr(evidence, "load_document_chunks",
                        lambda paths: {p: rows[p] for p in paths if p in rows})
    monkeypatch.setattr(evidence, "load_resolution_sections", lambda: {})
    r = EvidenceRetriever(FakeEmbedder(), related_documents=True, **options)
    r.vector = FakeVector(primary)
    return r


def test_the_step_adds_the_named_table_and_keeps_the_best_similarity(monkeypatch):
    primary = [replace(hit(LOG_A, 0.83, "error context"), source_type="log"),
               replace(hit(INC_A, 0.80, "Summary"), source_type="incident"),
               replace(hit(INC_A, 0.79, "Overview"), source_type="incident")]
    rows = {RAW: [chunk_row(RAW, "schema", "shipments_raw", (1, 1, 0, 0))]}
    out = retriever(monkeypatch, primary, rows).retrieve("Why did it fail?", 8)
    added = [h for h in out if h.origin == RELATED_ORIGIN]
    assert [h.path for h in added] == [RAW] and added[0].linked_from == LOG_A
    assert max(h.score for h in out) == 0.83  # the refusal gate sees the same best hit


def test_a_chunk_more_similar_than_the_top_hit_is_never_added(monkeypatch):
    primary = [replace(hit(LOG_A, 0.6, "error context"), source_type="log")]
    rows = {RAW: [chunk_row(RAW, "schema", "shipments_raw", (1, 1, 1, 1))]}  # cosine 1.0
    out = retriever(monkeypatch, primary, rows).retrieve("Why?", 8)
    assert [h.path for h in out] == [LOG_A]


def test_the_step_is_off_by_default_and_honours_its_limit(monkeypatch):
    primary = [replace(hit(PIPE, 0.83), source_type="schema")]
    rows = {RAW: [chunk_row(RAW, "schema", "shipments_raw", (1, 0, 0, 0))],
            FACT: [chunk_row(FACT, "schema", "shipments_fact", (1, 1, 0, 0))]}
    r = retriever(monkeypatch, primary, rows, max_related=1)
    assert [h.path for h in r.retrieve("x", 8)] == [PIPE, FACT]  # the more similar one
    r.related_documents = False
    assert [h.path for h in r.retrieve("x", 8)] == [PIPE]


def test_an_unanswerable_question_stays_refused_by_the_gate(monkeypatch):
    # Weak evidence that names a document: the related chunk is added, but it can never
    # raise the best similarity, so the gate still refuses without calling a model.
    primary = [replace(hit(INC_B, 0.62, "Overview"), source_type="incident")]
    rows = {INC_OLD: [chunk_row(INC_OLD, "incident", "Overview", (1, 0, 0, 0))]}
    r = retriever(monkeypatch, primary, rows)
    out = answer_question("What was the exact rate limit during INC-2046?", retriever=r,
                          llm_factory=lambda: pytest.fail("the model must not be called"),
                          k=8, min_similarity=0.72)
    assert out.insufficient_evidence and out.gate["passed"] is False
    assert INC_OLD in [h.path for h in out.hits]



# ---- the cause section of the top incident report -----------------------------------------


def incident_hit(section, score):
    return replace(hit(INC_B, score, section), source_type="incident")


def cause_rows(embedding=(1, 0, 0, 0)):
    return {INC_B: [chunk_row(INC_B, "incident", "Observed evidence", (1, 1, 0, 0)),
                    chunk_row(INC_B, "incident", "Inferred cause", embedding)]}


def test_a_cause_question_gets_the_top_incidents_cause_section(monkeypatch):
    primary = [incident_hit("Observed evidence", 0.79), incident_hit("Timeline", 0.75)]
    out = retriever(monkeypatch, primary, cause_rows()).retrieve(
        "What caused the HTTP 429 responses, and how certain is that?", 8)
    cause = [h for h in out if h.origin == evidence.CAUSE_ORIGIN]
    assert [(h.path, h.section) for h in cause] == [(INC_B, "Inferred cause")]
    assert max(h.score for h in out) == 0.79


@pytest.mark.parametrize("question, top", [
    ("List the timeline of the payments pull.", incident_hit("Observed evidence", 0.79)),
    ("What caused it?", replace(hit(LOG_A, 0.79, "run summary"), source_type="log")),
])
def test_no_cause_section_without_a_cause_question_or_an_incident_on_top(monkeypatch,
                                                                         question, top):
    out = retriever(monkeypatch, [top], cause_rows()).retrieve(question, 8)
    assert not [h for h in out if h.origin == evidence.CAUSE_ORIGIN]


def test_the_cause_section_is_not_repeated_or_allowed_above_the_top_hit(monkeypatch):
    shown = [incident_hit("Inferred cause", 0.79)]
    out = retriever(monkeypatch, shown, cause_rows()).retrieve("Why did it happen?", 8)
    assert [h.section for h in out if h.path == INC_B] == ["Inferred cause"]
    above = retriever(monkeypatch, [incident_hit("Observed evidence", 0.6)],
                      cause_rows(embedding=(1, 1, 1, 1))).retrieve("Why?", 8)
    assert not [h for h in above if h.origin == evidence.CAUSE_ORIGIN]
