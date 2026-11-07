"""Date-aware retrieval: wrap any retriever and, when the question names one unambiguous
date (plus, optionally, a known pipeline/task), drop LOG chunks from other dates, pipelines
or tasks. Runbooks, schemas and incidents are never filtered. Every decision, applied or not,
is kept in `last_info` so the API response and the query log show what happened.

Why: embeddings barely separate `2026-09-04` from `2026-09-21`; in the baseline evaluation
the dominant failure was a log of the right task on the wrong date.
"""

import time
from dataclasses import replace

from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.embeddings import Embedder
from incident_copilot.retrieval.base import Filters, Hit, Retriever
from incident_copilot.retrieval.query_scope import LogCatalog, QueryScope, resolve_scope

CATALOG_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, LogCatalog]] = {}


def load_log_catalog() -> LogCatalog:
    """(pipeline, task, date) of every indexed log, cached briefly per database."""
    url = get_settings().database_url
    hit = _cache.get(url)
    if hit and time.monotonic() - hit[0] < CATALOG_TTL_SECONDS:
        return hit[1]
    with connect() as conn:
        rows = conn.execute(
            "SELECT pipeline_name, metadata->>'task', metadata->>'date' FROM documents "
            "WHERE source_type = 'log' AND pipeline_name IS NOT NULL "
            "AND metadata->>'task' IS NOT NULL AND metadata->>'date' IS NOT NULL"
        ).fetchall()
    catalog = LogCatalog(frozenset((p, t, d) for p, t, d in rows))
    _cache[url] = (time.monotonic(), catalog)
    return catalog


def clear_log_catalog_cache() -> None:
    _cache.clear()


class DateAwareRetriever:
    def __init__(self, embedder: Embedder, inner_cls: type, suffix: str = "+date"):
        self.inner: Retriever = inner_cls(embedder)
        self.name = f"{self.inner.name}{suffix}"
        self.last_scope: QueryScope | None = None

    @property
    def last_question_vec(self):
        """The inner retriever's embedding of the last question, if it keeps one."""
        return getattr(self.inner, "last_question_vec", None)

    @property
    def last_info(self) -> dict:
        return {"date_filter": self.last_scope.as_dict()} if self.last_scope else {}

    def retrieve(self, question: str, k: int, filters: Filters | None = None) -> list[Hit]:
        filters = filters or Filters()
        if filters.source_types and "log" not in filters.source_types:
            self.last_scope = QueryScope(False, "request excludes logs")
            return self.inner.retrieve(question, k, filters)
        scope = resolve_scope(question, load_log_catalog())
        if scope.applied and filters.pipeline and scope.pipeline \
                and filters.pipeline != scope.pipeline:
            scope = replace(scope, applied=False,
                            reason="question pipeline differs from the request's pipeline")
        self.last_scope = scope
        if scope.applied:
            filters = replace(filters, log_date=scope.date, log_pipeline=scope.pipeline,
                              log_task=scope.task)
        return self.inner.retrieve(question, k, filters)
