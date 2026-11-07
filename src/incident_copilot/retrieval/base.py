"""The retrieval interface shared by every retriever (vector now; hybrid and rerank later).

A later HybridRetriever or RerankingRetriever implements the same `retrieve()` and adds its
own entries to `Hit.scores` (e.g. "keyword", "rrf", "rerank"), so callers, the API and the
eval harness need no changes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

SOURCE_TYPES = ("runbook", "log", "schema", "incident")


@dataclass(frozen=True)
class Filters:
    source_types: tuple[str, ...] | None = None
    # Matches documents.pipeline_name, or any pipeline a runbook "applies to".
    pipeline: str | None = None
    # Constraints on LOG chunks only (other source types are never excluded by these).
    # Set by DateAwareRetriever when the question names an unambiguous date.
    log_date: str | None = None
    log_pipeline: str | None = None
    log_task: str | None = None

    def as_dict(self) -> dict:
        out = {"source_types": list(self.source_types) if self.source_types else None,
               "pipeline": self.pipeline}
        if self.log_date or self.log_pipeline or self.log_task:
            out |= {"log_date": self.log_date, "log_pipeline": self.log_pipeline,
                    "log_task": self.log_task}
        return out


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    path: str
    source_type: str
    section: str | None
    ts_start: datetime | None
    ts_end: datetime | None
    text: str
    chunk_index: int
    content_hash: str | None
    pipeline_name: str | None
    rank: int  # 1-based, in final order
    # Named scores, e.g. {"vector_similarity": 0.81, "vector_distance": 0.19}.
    # "vector_similarity" is always the similarity to the QUESTION.
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    # How the hit was found: "vector" (question search) or "error_link" (see error_link.py).
    origin: str = "vector"
    linked_from: str | None = None  # path of the log whose error line found this hit

    @property
    def score(self) -> float:
        """The score the ranking is based on (vector similarity for the vector retriever)."""
        return self.scores.get("vector_similarity", 0.0)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int, filters: Filters | None = None) -> list[Hit]: ...
