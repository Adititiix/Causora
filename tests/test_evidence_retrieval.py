"""Evidence selection (retrieval/evidence.py). Offline: a scripted vector backend stands in for
pgvector, so these run without a database, a model or a network.

The fixtures reproduce the shapes measured on the seed set: Q015 (the target log is the 8th
vector hit and anchors the runbook link), Q016 (three incident reports fill the top 8),
Q022 (only the runbook's Symptoms section matches), Q049 (an unanswerable fact question
whose evidence still contains a failed log and a linked runbook).
"""

import random
from dataclasses import replace

import numpy as np
import pytest

from incident_copilot.config import get_settings
from incident_copilot.generation import FakeLLMProvider, answer_question
from incident_copilot.retrieval import (
    EXPERIMENTAL_RETRIEVERS,
    RETRIEVERS,
    DateAwareRetriever,
    ErrorLinkRetriever,
    EvidenceRetriever,
    Hit,
    date_aware,
    evidence,
    get_retriever,
)
from incident_copilot.retrieval.evidence import (
    asks_about_failure,
    companion_targets,
    diversify,
    select_evidence,
)
from incident_copilot.retrieval.query_scope import LogCatalog

LOG = "logs/orders_etl/2026-08-31__load_orders_fact.log"
RUNBOOK = "runbooks/not-null-violation.md"
ERROR = "psycopg2.errors.NotNullViolation: null value in column customer_id"
Q015 = "Why did the load_orders_fact task in the orders_etl pipeline fail on 2026-08-31?"


class Vec:
    """What the fake embedder returns: the text (to look results up) plus a vector."""

    def __init__(self, text: str, array=None):
        self.text = text
        self.array = np.ones(4, dtype=np.float32) if array is None else array

    def to_numpy(self):
        return self.array


class FakeEmbedder:
    dim = 384
    name = "fake"

    def embed_query(self, text):
        return Vec(text)


class FakeVector:
    """search() returns scripted hits for the query text; filters are recorded, and a
    source-type filter is applied the way the SQL applies it."""

    def __init__(self, results: dict[str, list[Hit]]):
        self.results = results
        self.calls: list[dict] = []

    def search(self, query_vec, k, filters=None, score_vec=None):
        self.calls.append({"query": query_vec.text, "k": k, "filters": filters})
        hits = list(self.results.get(query_vec.text, []))
        if filters and filters.source_types:
            hits = [h for h in hits if h.source_type in filters.source_types]
        out = []
        for i, h in enumerate(hits[:k], start=1):
            scores = dict(h.scores)
            if score_vec is not None:
                scores["search_similarity"] = scores["vector_similarity"]
            out.append(replace(h, rank=i, scores=scores))
        return out


def hit(path, score, section="sec", source_type=None, chunk=None, error=None):
    source_type = source_type or path.split("/")[0].rstrip("s")
    meta = {"error": error} if error else {}
    return Hit(chunk_id=chunk or f"{path}#{section}", document_id=f"doc:{path}", path=path,
               source_type=source_type, section=section, ts_start=None, ts_end=None,
               text=f"{path} {section}", chunk_index=0, content_hash="h", pipeline_name=None,
               rank=0, scores={"vector_similarity": score, "vector_distance": round(1 - score, 6)},
               metadata=meta)


def resolution_row(path, chunk=None, embedding=(1.0, 0.0, 0.0, 0.0)):
    """A row shaped like _RESOLUTION_SQL's (embedding last). The default embedding has cosine
    0.5 with the fake question vector (all ones), below every fixture's top hit, as a
    Resolution section is in the real index."""
    return (chunk or f"{path}#Resolution", f"doc:{path}", path, "runbook", "Resolution", None,
            None, "1. Default or filter the nulls. 2. Re-run.", 4, "h", None, {},
            np.array(embedding, dtype=np.float32))


def q015_backend():
    incidents = [hit("incidents/INC-2036.md", 0.857, "Summary"),
                 hit("incidents/INC-2036.md", 0.839, "Timeline"),
                 hit("incidents/INC-2034.md", 0.838, "Summary"),
                 hit("incidents/INC-2036.md", 0.832, "Overview"),
                 hit("incidents/INC-2034.md", 0.827, "Timeline"),
                 hit("incidents/INC-2040.md", 0.821, "Timeline"),
                 hit("incidents/INC-2039.md", 0.819, "Overview")]
    log = hit(LOG, 0.8187, "run summary", error=ERROR)
    more = [hit(LOG, 0.812, "error context", error=ERROR),
            hit("incidents/INC-2040.md", 0.81, "Overview")]
    return FakeVector({Q015: incidents + [log] + more,
                       ERROR: [hit(RUNBOOK, 0.518, "Symptoms")]})


@pytest.fixture
def sections(monkeypatch):
    rows = {RUNBOOK: resolution_row(RUNBOOK),
            "runbooks/oom-killed.md": resolution_row("runbooks/oom-killed.md")}
    monkeypatch.setattr(evidence, "load_resolution_sections", lambda: rows)
    return rows


def build(cls=EvidenceRetriever, backend=None, **options):
    r = cls(FakeEmbedder(), **options)
    r.vector = backend or q015_backend()
    return r


def paths(hits):
    return [(h.path, h.section, h.origin) for h in hits]


# ---- anchors (Q015) -----------------------------------------------------------------------


def test_baseline_error_link_drops_the_log_it_links_from():
    """Documents the defect being fixed: the link takes the last slot, which was the log."""
    hits = build(ErrorLinkRetriever).retrieve(Q015, 8)
    assert len(hits) == 8 and LOG not in {h.path for h in hits}
    assert hits[-1].origin == "error_link" and hits[-1].linked_from == LOG


def test_a_linked_runbook_does_not_displace_its_anchor_log(sections):
    hits = build(protect_anchors=True, resolution_companions=False,
                 max_chunks_per_document=None).retrieve(Q015, 8)
    assert len(hits) == 8
    got = paths(hits)
    log_at = got.index((LOG, "run summary", "vector"))
    assert got[log_at + 1] == (RUNBOOK, "Symptoms", "error_link")   # the link follows its log
    assert hits[0].path == "incidents/INC-2036.md"                  # the top hit is untouched
    assert hits[log_at].metadata["provenance"] == {"vector_rank": 8}
    assert hits[log_at + 1].metadata["provenance"]["anchor"] == LOG


# ---- resolution companions ----------------------------------------------------------------


def test_the_resolution_of_a_linked_runbook_is_added_for_a_failure_question(sections):
    hits = build().retrieve(Q015, 8)
    got = paths(hits)
    i = got.index((RUNBOOK, "Symptoms", "error_link"))
    assert got[i + 1] == (RUNBOOK, "Resolution", "resolution_companion")
    assert LOG in {h.path for h in hits} and len(hits) == 8
    companion = hits[i + 1]
    assert companion.text.startswith("1. Default or filter")
    assert companion.metadata["provenance"] == {"companion_of": RUNBOOK}
    assert companion.score <= hits[0].score


def test_no_resolution_is_added_to_a_question_that_is_not_about_a_failure(sections):
    question = "What is the retention period for the clickstream Kafka topic?"  # Q049 shape
    backend = FakeVector({question: [hit("incidents/INC-2038.md", 0.764, "Overview"),
                                     hit(LOG, 0.714, "error context", error=ERROR)],
                          ERROR: [hit(RUNBOOK, 0.64, "Symptoms")]})
    hits = build(backend=backend).retrieve(question, 8)
    assert (RUNBOOK, "Symptoms", "error_link") in paths(hits)
    assert all(h.origin != "resolution_companion" for h in hits)


def test_a_weakly_matched_unlinked_runbook_gets_no_resolution(sections):
    question = "A task fails with return code -9. What should I do?"
    backend = FakeVector({question: [hit("logs/a/2026-01-01__t.log", 0.80, "run summary"),
                                     hit("runbooks/oom-killed.md", 0.60, "Symptoms")]})
    hits = build(backend=backend, companion_min_similarity=0.72).retrieve(question, 8)
    assert all(h.origin != "resolution_companion" for h in hits)
    strong = FakeVector({question: [hit("runbooks/oom-killed.md", 0.82, "Symptoms")]})
    hits = build(backend=strong, companion_min_similarity=0.72).retrieve(question, 8)
    assert paths(hits)[1] == ("runbooks/oom-killed.md", "Resolution", "resolution_companion")


def test_a_resolution_already_retrieved_is_not_added_twice(sections):
    question = "A load task fails with NotNullViolation. How do I fix it?"
    backend = FakeVector({question: [hit(RUNBOOK, 0.80, "Symptoms"),
                                     hit(RUNBOOK, 0.75, "Resolution",
                                         chunk=f"{RUNBOOK}#Resolution")]})
    hits = build(backend=backend, max_chunks_per_document=None).retrieve(question, 8)
    assert [h.chunk_id for h in hits].count(f"{RUNBOOK}#Resolution") == 1
    assert all(h.origin != "resolution_companion" for h in hits)


def test_failure_intent():
    for q in ("Why did the transform_orders task fail on 2026-09-04?",
              "A task disappears with OOMKilled exit code 137. What should I do?",
              "How do I resolve a replica timeout?", "Steps to recover from OffsetOutOfRange"):
        assert asks_about_failure(q), q
    for q in ("What is the on-call phone number for the payments team?",
              "What Kubernetes version do the Airflow workers run?",
              "Who was the on-call engineer during incident INC-9999?"):
        assert not asks_about_failure(q), q


def test_companion_targets_are_bounded_and_ordered():
    hits = [hit(f"runbooks/r{i}.md", 0.9, "Symptoms") for i in range(4)]
    assert companion_targets(hits, "why did it fail", 0.72, 2) == ["runbooks/r0.md",
                                                                  "runbooks/r1.md"]
    assert companion_targets(hits, "why did it fail", 0.72, 0) == []


# ---- diversity (Q016) ---------------------------------------------------------------------


def test_document_cap_lets_the_target_log_into_the_top_k():
    crowd = [hit("incidents/A.md", 0.86, "s1"), hit("incidents/B.md", 0.83, "s1"),
             hit("incidents/A.md", 0.82, "s2"), hit("incidents/C.md", 0.82, "s1"),
             hit("incidents/B.md", 0.81, "s2"), hit("incidents/C.md", 0.81, "s2"),
             hit("incidents/A.md", 0.81, "s3"), hit("incidents/B.md", 0.80, "s3"),
             hit(LOG, 0.806, "run summary")]
    assert LOG not in {h.path for h in crowd[:8]}
    chosen = diversify(crowd, 8, 2)
    assert LOG in {h.path for h in chosen} and len(chosen) == 8
    # The cap allows 7 here; the 8th slot is back-filled with the best skipped chunk.
    assert [h.section for h in chosen if h.path == "incidents/A.md"] == ["s1", "s2", "s3"]
    assert [h.section for h in chosen if h.path == "incidents/B.md"] == ["s1", "s2"]


def test_document_cap_backfills_when_one_document_is_all_there_is():
    only = [hit(RUNBOOK, 0.9 - i / 100, f"s{i}") for i in range(6)]
    chosen = diversify(only, 5, 2)
    assert [h.section for h in chosen] == ["s0", "s1", "s2", "s3", "s4"]  # rank order kept
    assert diversify(only, 5, None) == only[:5]
    assert diversify([], 5, 2) == []


# ---- general contract ---------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_top_k_is_respected_and_chunks_are_unique(sections, k):
    hits = build().retrieve(Q015, k)
    assert len(hits) == k
    assert len({h.chunk_id for h in hits}) == k
    assert [h.rank for h in hits] == list(range(1, k + 1))
    assert hits[0].path == "incidents/INC-2036.md"  # the top question hit is always kept


def test_select_evidence_removes_duplicates_and_keeps_order():
    a, b = hit("logs/x/2026-01-01__t.log", 0.9, "run summary", error="E"), hit("x.md", 0.8)
    link = hit(RUNBOOK, 0.5, "Symptoms")
    out = select_evidence([a, b, a], {a.path: link}, {}, 8, protect_anchors=True)
    assert [h.chunk_id for h in out] == [a.chunk_id, link.chunk_id, b.chunk_id]


def test_ties_are_broken_deterministically(sections):
    tied = [hit(f"incidents/T{i}.md", 0.8, "Summary", chunk=f"c{i}") for i in range(6)]
    results = set()
    for seed in range(5):
        order = tied[:]
        random.Random(seed).shuffle(order)
        r = build(backend=FakeVector({"q": order}))
        results.add(tuple(h.chunk_id for h in r.retrieve("q", 4)))
    assert results == {("c0", "c1", "c2", "c3")}


def test_no_candidates_means_no_hits(sections):
    assert build(backend=FakeVector({})).retrieve(Q015, 8) == []


def test_best_similarity_and_the_gate_are_unchanged(sections):
    base = build(ErrorLinkRetriever).retrieve(Q015, 8)
    new = build().retrieve(Q015, 8)
    assert max(h.score for h in new) == max(h.score for h in base)
    for threshold in (0.80, 0.86):  # below and above the best hit
        outs = [answer_question(Q015, retriever=build(cls), llm_factory=lambda: FakeLLMProvider(),
                                k=8, min_similarity=threshold, retrieval_only=True)
                for cls in (ErrorLinkRetriever, EvidenceRetriever)]
        assert outs[0].gate == outs[1].gate


def test_a_companion_more_similar_than_the_top_hit_is_left_out(monkeypatch):
    # Only possible if the approximate index missed the chunk; it must not raise the gate.
    rows = {RUNBOOK: resolution_row(RUNBOOK, embedding=(1.0, 1.0, 1.0, 1.0))}  # cosine 1.0
    monkeypatch.setattr(evidence, "load_resolution_sections", lambda: rows)
    hits = build().retrieve(Q015, 8)
    assert all(h.origin != "resolution_companion" for h in hits)
    assert max(h.score for h in hits) == 0.857


def test_a_request_without_runbooks_gets_no_links_or_companions(sections):
    from incident_copilot.retrieval import Filters

    hits = build().retrieve(Q015, 8, Filters(source_types=("incident", "log")))
    assert {h.source_type for h in hits} <= {"incident", "log"}
    assert all(h.origin == "vector" for h in hits)


# ---- dates --------------------------------------------------------------------------------


@pytest.fixture
def catalog(monkeypatch):
    cat = LogCatalog(frozenset({("orders_etl", "load_orders_fact", "2026-08-31")}))
    monkeypatch.setattr(date_aware, "load_log_catalog", lambda: cat)


def test_a_reliable_date_narrows_log_candidates(sections, catalog):
    r = DateAwareRetriever(FakeEmbedder(), inner_cls=EvidenceRetriever)
    r.inner.vector = q015_backend()
    r.retrieve(Q015, 8)
    f = r.inner.vector.calls[0]["filters"]
    assert (f.log_date, f.log_pipeline, f.log_task) == ("2026-08-31", "orders_etl",
                                                        "load_orders_fact")
    assert r.last_info["date_filter"]["applied"] is True


def test_without_a_date_nothing_is_filtered_and_undated_evidence_stays(sections, catalog):
    question = "A load task fails with NotNullViolation. How do I fix it?"
    undated = [hit(RUNBOOK, 0.80, "Symptoms"), hit("schemas/orders_fact.md", 0.75)]
    r = DateAwareRetriever(FakeEmbedder(), inner_cls=EvidenceRetriever)
    r.inner.vector = FakeVector({question: undated})
    hits = r.retrieve(question, 8)
    assert r.inner.vector.calls[0]["filters"].log_date is None
    assert {h.path for h in hits} >= {RUNBOOK, "schemas/orders_fact.md"}


# ---- backward compatibility ---------------------------------------------------------------


def test_established_retrievers_and_the_default_are_unchanged():
    assert list(RETRIEVERS) == ["vector", "vector+error_link", "vector+date",
                                "vector+error_link+date"]
    assert not set(EXPERIMENTAL_RETRIEVERS) & set(RETRIEVERS)
    known = set(RETRIEVERS) | set(EXPERIMENTAL_RETRIEVERS)
    assert get_settings().retriever in known  # .env may pick any registered retriever
    from incident_copilot.config import Settings

    # The default was switched deliberately (owner-approved 2026-10-29) to the evaluated
    # evidence retriever; it must stay registered and must not change by accident.
    assert Settings.model_fields["retriever"].default == \
        "vector+error_link+evidence+context+scoped+date"
    assert Settings.model_fields["retriever"].default in EXPERIMENTAL_RETRIEVERS
    for name in EXPERIMENTAL_RETRIEVERS:
        assert get_retriever(name, FakeEmbedder()).name == name
    with pytest.raises(ValueError, match="unknown retriever"):
        get_retriever("vector+nonsense", FakeEmbedder())


# ---- error-context companions (Q016) ------------------------------------------------------

Q016 = "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-16?"
LOG16 = "logs/orders_etl/2026-09-16__transform_orders.log"
RB16 = "runbooks/dependency-version-conflict.md"
ERR16 = "ImportError: cannot import name 'soft_unicode' from 'markupsafe'"


def context_row(path, chunk=None):
    """An "error context" row (cosine 0.5 with the fake question vector)."""
    return (chunk or f"{path}#error context", f"doc:{path}", path, "log", "error context", None,
            None, "Worker image: ...2026.09.16-2\nTraceback ... from markupsafe import "
            "soft_unicode", 1, "h", "orders_etl", {"error": ERR16},
            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32))


def q016_backend():
    """The measured Q016 shape: incident reports of the same task on other dates outrank the
    one log of the asked date, and only its run-summary chunk is among the candidates."""
    incidents = [hit("incidents/INC-2032.md", 0.856, "Summary"),
                 hit("incidents/INC-2035.md", 0.831, "Summary"),
                 hit("incidents/INC-2032.md", 0.826, "Timeline"),
                 hit("incidents/INC-2041.md", 0.821, "Summary"),
                 hit("incidents/INC-2035.md", 0.819, "Timeline"),
                 hit("incidents/INC-2041.md", 0.814, "Overview"),
                 hit("incidents/INC-2032.md", 0.812, "Overview"),
                 hit("incidents/INC-2035.md", 0.810, "Overview")]
    return FakeVector({Q016: incidents + [hit(LOG16, 0.806, "run summary", error=ERR16)],
                       ERR16: [hit(RB16, 0.586, "Symptoms")]})


@pytest.fixture
def contexts(monkeypatch):
    rows = {RB16: resolution_row(RB16)}
    monkeypatch.setattr(evidence, "load_resolution_sections", lambda: rows)
    ctx = {LOG16: context_row(LOG16), LOG: context_row(LOG)}
    monkeypatch.setattr(evidence, "load_log_error_contexts", lambda: ctx)
    return ctx


def dated(day="2026-09-16"):
    from incident_copilot.retrieval import Filters

    return Filters(log_date=day, log_pipeline="orders_etl", log_task="transform_orders")


def test_a_dated_failed_log_gets_its_traceback_next_to_it(contexts):
    hits = build(backend=q016_backend(), log_context_companions=True).retrieve(Q016, 8,
                                                                                dated())
    got = paths(hits)
    i = got.index((LOG16, "run summary", "vector"))
    assert got[i + 1] == (LOG16, "error context", "error_context_companion")
    assert got[i + 2] == (RB16, "Symptoms", "error_link")
    assert got[i + 3] == (RB16, "Resolution", "resolution_companion")
    assert len(hits) == 8 and len({h.chunk_id for h in hits}) == 8
    assert hits[0].path == "incidents/INC-2032.md" and hits[0].score == 0.856  # gate input
    assert hits[i + 1].metadata["provenance"] == {"companion_of": LOG16}


def test_the_traceback_is_not_added_without_a_date_scope(contexts):
    # Without log_date the shown logs may be examples of the error, not the incident asked
    # about (runbook questions), so no extra log chunk is spent on them.
    hits = build(backend=q016_backend(), log_context_companions=True).retrieve(Q016, 8)
    assert all(h.origin != "error_context_companion" for h in hits)


def test_the_traceback_rule_is_off_unless_enabled(contexts):
    hits = build(backend=q016_backend()).retrieve(Q016, 8, dated())
    assert all(h.origin != "error_context_companion" for h in hits)


def test_no_traceback_companion_when_it_is_already_shown_or_the_log_did_not_fail(contexts):
    shown = FakeVector({Q016: [hit(LOG16, 0.81, "run summary", error=ERR16),
                               hit(LOG16, 0.77, "error context", error=ERR16,
                                   chunk=f"{LOG16}#error context")]})
    hits = build(backend=shown, log_context_companions=True).retrieve(Q016, 8, dated())
    assert [h.chunk_id for h in hits].count(f"{LOG16}#error context") == 1
    assert all(h.origin != "error_context_companion" for h in hits)
    ok_log = FakeVector({Q016: [hit(LOG16, 0.81, "run summary")]})   # no error metadata
    hits = build(backend=ok_log, log_context_companions=True).retrieve(Q016, 8, dated())
    assert all(h.origin != "error_context_companion" for h in hits)


def test_traceback_companions_are_bounded_by_max_links(contexts):
    logs = [f"logs/orders_etl/2026-09-16__t{i}.log" for i in range(4)]
    contexts.update({p: context_row(p) for p in logs})
    backend = FakeVector({Q016: [hit(p, 0.8 - i / 100, "run summary", error=ERR16)
                                 for i, p in enumerate(logs)]})
    hits = build(backend=backend, log_context_companions=True, max_links=2).retrieve(
        Q016, 8, dated())
    added = [h.path for h in hits if h.origin == "error_context_companion"]
    assert added == logs[:2]


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_traceback_companions_respect_top_k(contexts, k):
    hits = build(backend=q016_backend(), log_context_companions=True).retrieve(Q016, k,
                                                                                dated())
    assert len(hits) == k and hits[0].score == 0.856
    assert len({h.chunk_id for h in hits}) == k


def test_the_existing_evidence_variant_is_unchanged_by_the_new_rule():
    assert EXPERIMENTAL_RETRIEVERS["vector+error_link+evidence+date"](FakeEmbedder()).inner \
        .log_context_companions is False
    assert EXPERIMENTAL_RETRIEVERS["vector+error_link+evidence+context+date"](
        FakeEmbedder()).inner.log_context_companions is True


# ---- cap scoped to dated-log questions (Q037) ---------------------------------------------


def q037_backend():
    """One incident report answers the question; its Overview (duration) ranks 3rd."""
    q = "Which past incident was caused by the payments API rate limiting us, how long?"
    inc = "incidents/INC-2043.md"
    others = [hit(f"logs/p/2026-09-{10 + i}__pull.log", 0.80 - i / 100, "error context")
              for i in range(8)]  # enough other candidates for the cap to bind
    return q, FakeVector({q: [hit(inc, 0.86, "Root cause"), hit(inc, 0.85, "Timeline"),
                              hit(inc, 0.84, "Overview"), hit(inc, 0.83, "Summary"),
                              *others]})


def test_scoped_cap_keeps_every_section_of_an_undated_single_document_answer(contexts):
    q, backend = q037_backend()
    capped = build(backend=backend).retrieve(q, 8)
    scoped = build(backend=q037_backend()[1], cap_only_when_dated=True).retrieve(q, 8)
    assert "Overview" not in [h.section for h in capped if h.path == "incidents/INC-2043.md"]
    assert [h.section for h in scoped if h.path == "incidents/INC-2043.md"] == [
        "Root cause", "Timeline", "Overview", "Summary"]


def test_scoped_cap_still_caps_a_dated_question(contexts):
    hits = build(backend=q016_backend(), log_context_companions=True,
                 cap_only_when_dated=True).retrieve(Q016, 8, dated())
    assert LOG16 in {h.path for h in hits}
    assert max(sum(h.path == p for h in hits) for p in {h.path for h in hits}) <= 2 + 1


def test_a_retrieved_traceback_is_not_trimmed_away_for_protected_chunks(contexts, monkeypatch):
    """Q008 shape: the dated log's error-context chunk is a low-ranked vector hit; the link
    and Resolution companion take two slots, which used to push it out."""
    q = "Why did the validate_orders task in the orders_etl pipeline fail on 2026-08-19?"
    log = "logs/orders_etl/2026-08-19__validate_orders.log"
    rb = "runbooks/type-mismatch-na-values.md"
    err = "pyarrow.lib.ArrowInvalid: Could not convert 'N/A'"
    monkeypatch.setattr(evidence, "load_resolution_sections", lambda: {rb: resolution_row(rb)})
    incidents = [hit(f"incidents/INC-{2030 + i}.md", 0.835 - i / 100, "Summary")
                 for i in range(5)]
    backend = FakeVector({q: [*incidents[:2], hit(log, 0.819, "run summary", error=err),
                              # lowest-ranked candidate, as in the real Q008 run
                              *incidents[2:], hit(log, 0.79, "error context", error=err,
                                                  chunk=f"{log}#error context")],
                          err: [hit(rb, 0.6, "Symptoms")]})
    hits = build(backend=backend, log_context_companions=True,
                 cap_only_when_dated=True).retrieve(q, 8, dated("2026-08-19"))
    got = paths(hits)
    assert (log, "error context", "vector") in got          # kept, not re-added
    assert (rb, "Resolution", "resolution_companion") in got and len(hits) == 8
    off = build(backend=FakeVector(backend.results)).retrieve(q, 8, dated("2026-08-19"))
    assert (log, "error context", "vector") not in paths(off)  # rule off: trimmed as before
