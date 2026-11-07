"""Day 3-5: vector search, then keyword (full-text) search + RRF fusion, then reranking.

Each retriever implements the same interface, `retrieve(question, k, filters) -> list[Hit]`,
so the eval harness can swap retrievers by name. So far:
  vector                  cosine similarity on the question only
  vector+error_link       vector, plus runbooks found via the exception line of retrieved logs
  vector+date             vector, with log chunks narrowed to an unambiguous date in the question
  vector+error_link+date  both of the above

EXPERIMENTAL_RETRIEVERS (opt-in, never a default; see retrieval/evidence.py) add evidence
selection to vector+error_link+date, one step at a time and all together:
  vector+error_link+anchors+date     a linked runbook never pushes out the log it came from
  vector+error_link+resolution+date  + the Resolution section of a matched runbook
  vector+error_link+diverse+date     at most 2 chunks per document
  vector+error_link+evidence+date    all three
  vector+error_link+evidence+context+date
                                     + a dated failed log's error-context chunk (traceback)
  vector+error_link+evidence+context+scoped+date
                                     + the per-document cap only for dated-log questions
                                     + documents the question or the evidence names
                                       explicitly (retrieval/relations.py; since 2026-11-01)
  vector+error_link+evidence+context+scoped+norelations+date
                                     the line above without related documents
"""

from functools import partial

from incident_copilot.embeddings import Embedder
from incident_copilot.retrieval.base import SOURCE_TYPES, Filters, Hit, Retriever
from incident_copilot.retrieval.date_aware import DateAwareRetriever
from incident_copilot.retrieval.error_link import ErrorLinkRetriever
from incident_copilot.retrieval.evidence import EvidenceRetriever
from incident_copilot.retrieval.vector import VectorRetriever

RETRIEVERS = {
    "vector": VectorRetriever,
    "vector+error_link": ErrorLinkRetriever,
    "vector+date": partial(DateAwareRetriever, inner_cls=VectorRetriever),
    "vector+error_link+date": partial(DateAwareRetriever, inner_cls=ErrorLinkRetriever),
}


def _evidence(name: str, **options):
    return partial(DateAwareRetriever,
                   inner_cls=partial(EvidenceRetriever, name=name, **options))


EXPERIMENTAL_RETRIEVERS = {
    "vector+error_link+anchors+date": _evidence(
        "vector+error_link+anchors", resolution_companions=False, max_chunks_per_document=None),
    "vector+error_link+resolution+date": _evidence(
        "vector+error_link+resolution", protect_anchors=False, max_chunks_per_document=None),
    "vector+error_link+diverse+date": _evidence(
        "vector+error_link+diverse", protect_anchors=False, resolution_companions=False),
    "vector+error_link+evidence+date": _evidence("vector+error_link+evidence"),
    "vector+error_link+evidence+context+date": _evidence(
        "vector+error_link+evidence+context", log_context_companions=True),
    # The default (config.py). Since 2026-11-01 it also adds related documents
    # (retrieval/relations.py); result folders from before that date measured it without.
    "vector+error_link+evidence+context+scoped+date": _evidence(
        "vector+error_link+evidence+context+scoped", log_context_companions=True,
        cap_only_when_dated=True, related_documents=True),
    # The default as it was before related documents, kept for comparisons.
    "vector+error_link+evidence+context+scoped+norelations+date": _evidence(
        "vector+error_link+evidence+context+scoped+norelations", log_context_companions=True,
        cap_only_when_dated=True),
}


def get_retriever(name: str, embedder: Embedder) -> Retriever:
    factory = RETRIEVERS.get(name) or EXPERIMENTAL_RETRIEVERS.get(name)
    if factory is None:
        raise ValueError(f"unknown retriever {name!r}; choose one of "
                         f"{sorted(RETRIEVERS) + sorted(EXPERIMENTAL_RETRIEVERS)}")
    return factory(embedder)


__all__ = ["EXPERIMENTAL_RETRIEVERS", "RETRIEVERS", "SOURCE_TYPES", "DateAwareRetriever",
           "ErrorLinkRetriever", "EvidenceRetriever", "Filters", "Hit", "Retriever",
           "VectorRetriever", "get_retriever"]
