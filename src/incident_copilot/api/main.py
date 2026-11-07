"""FastAPI entrypoint.

Routes are added layer by layer:
  Day 1  /health, /health/db
  Day 3  POST /ask, POST /ingest
  Day 9  POST /ingest/failure
  Day 11 GET  /metrics/summary

Authentication is out of scope so far: do not expose this API beyond your machine.
"""

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from incident_copilot import __version__
from incident_copilot.api.errors import ApiError, error_body, install_error_handlers
from incident_copilot.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentDetailOut,
    DocumentSummaryOut,
    ErrorBody,
    ExplainRequest,
    ExplainResponse,
    IngestRequest,
    IngestResponse,
)
from incident_copilot.config import get_settings
from incident_copilot.db import db_status, shared_connection
from incident_copilot.embeddings import (
    Embedder,
    EmbeddingDimensionError,
    EmbeddingUnavailableError,
    check_schema_dim,
    get_embedder,
)
from incident_copilot.generation import (
    SETUP_INSTRUCTIONS,
    AskOutcome,
    LLMBadResponseError,
    LLMError,
    LLMNotConfiguredError,
    LLMProvider,
    answer_question,
    get_llm_provider,
)
from incident_copilot.generation.clarify import (
    describe_selection,
    load_subject_documents,
    make_clarifier,
    make_near_miss_resolver,
    remediation_hits,
    scope_hits,
)
from incident_copilot.generation.explain import explain_document
from incident_copilot.generation.prompt import evidence_id
from incident_copilot.ingestion import IndexingLockedError, UnsafePathError, ingest_paths
from incident_copilot.observability import log_failed_query, log_query
from incident_copilot.redaction import redact
from incident_copilot.retrieval import Filters, get_retriever
from incident_copilot.retrieval.contextualize import (
    CONTEXTUALIZED,
    Contextualization,
    Turn,
    contextualize,
    extract_anchors,
    is_follow_up,
    load_document_info,
)
from incident_copilot.retrieval.date_aware import load_log_catalog
from incident_copilot.retrieval.documents import (
    DocumentNotFoundError,
    get_document,
    list_documents,
)

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("incident_copilot")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast on a dimension the schema cannot store. The model itself loads lazily.
    check_schema_dim(get_settings().embedding_dim)
    yield


app = FastAPI(title="Incident Copilot", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def _start_timer(request: Request, call_next):
    # Lets the validation handler report latency for rejected /ask requests.
    request.state.t0 = time.perf_counter()
    return await call_next(request)


def _elapsed_ms(t0: float | None) -> int:
    return int(round((time.perf_counter() - t0) * 1000)) if t0 is not None else 0


async def _record_rejected_ask(request: Request, exc: RequestValidationError,
                               message: str) -> str | None:
    """422 on /ask: record it (redacted question, fixed category); never raises."""
    if request.url.path != "/ask" or request.method != "POST":
        return None
    body = exc.body if isinstance(exc.body, dict) else {}
    return await run_in_threadpool(
        log_failed_query, question=body.get("question"), status_code=422,
        error_category="validation_error",
        latency_ms=_elapsed_ms(getattr(request.state, "t0", None)), message=message)


install_error_handlers(app, on_validation_error=_record_rejected_ask)

ERROR_RESPONSES = {code: {"model": ErrorBody} for code in (409, 422, 500, 502, 503)}


# ---- dependencies (overridden in tests) ---------------------------------------------------


@lru_cache(maxsize=1)
def _shared_embedder() -> Embedder:
    return get_embedder("fastembed")


def get_embedder_dep() -> Embedder:
    return _shared_embedder()


class _TimedEmbedder:
    """Per-request wrapper that adds up time spent embedding the question."""

    def __init__(self, inner: Embedder):
        self._inner, self.ms = inner, 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def embed_query(self, text):
        t0 = time.perf_counter()
        try:
            return self._inner.embed_query(text)
        finally:
            self.ms += (time.perf_counter() - t0) * 1000


def get_llm_factory() -> Callable[[], LLMProvider]:
    """A factory, so the provider is only built (and the key only required) when needed."""
    return get_llm_provider


def _database_error(exc: psycopg.Error) -> ApiError:
    if isinstance(exc, psycopg.OperationalError):
        return ApiError(503, "db_unavailable",
                        f"database unavailable: {type(exc).__name__}")
    return ApiError(503, "database_error",
                    f"database error: {type(exc).__name__}. If the schema is out of date run "
                    "`python -m incident_copilot.db_migrate`.")


# ---- health -------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Does not touch the database."""
    return {"status": "ok", "version": __version__}


@app.get("/health/db")
def health_db():
    """Readiness: the database is reachable and pgvector is installed; corpus counts."""
    try:
        return {"status": "ok", **db_status()}
    except Exception as exc:  # noqa: BLE001 - surface any connection/schema problem as 503
        logger.exception("database health check failed")
        detail = f"database unavailable: {type(exc).__name__}"
        # `detail` is kept for backward compatibility; `error` is the API-wide format.
        return JSONResponse(status_code=503,
                            content={"detail": detail, **error_body("db_unavailable", detail)})


# ---- ask ----------------------------------------------------------------------------------


def _ask_response(outcome: AskOutcome, query_id: str, retriever: str,
                  top_k: int | None = None) -> AskResponse:
    used = outcome.used_chunk_ids
    return AskResponse(
        query_id=query_id,
        status=outcome.status,
        answer=outcome.answer,
        insufficient_evidence=outcome.insufficient_evidence,
        insufficient_reason=outcome.insufficient_reason,
        evidence=outcome.evidence,
        inference=outcome.inference,
        next_steps=outcome.next_steps,
        citations=[c.__dict__ for c in outcome.citations],
        retrieved=[
            {"evidence_id": evidence_id(h), "rank": h.rank, "chunk_id": h.chunk_id,
             "path": h.path, "source_type": h.source_type, "section": h.section,
             "chunk_index": h.chunk_index, "ts_start": h.ts_start, "ts_end": h.ts_end,
             "scores": h.scores, "origin": h.origin, "linked_from": h.linked_from,
             "used_in_answer": h.chunk_id in used, "text": h.text}
            for h in outcome.hits
        ],
        model=outcome.model,
        retriever=retriever,
        gate=outcome.gate,
        validation=outcome.validation,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
        latency_ms=outcome.latency_ms,
        retrieval_ms=outcome.retrieval_ms,
        generation_ms=outcome.generation_ms,
        retrieval_info=outcome.retrieval_info,
        timings=outcome.timings,
        top_k=top_k,
        clarification=outcome.clarification,
        next_step_evidence=outcome.next_step_evidence,
        next_step_kinds=outcome.next_step_kinds,
        next_step_support=outcome.next_step_support,
        unknowns=outcome.unknowns,
    )


def _kept_subject(req: AskRequest) -> dict | None:
    """The previous turn's subject, for a follow-up ("what is the solution for it?") that
    names no subject of its own. None for a standalone question, so a new question never
    inherits an old failure."""
    last = req.history[-1] if req.history else None
    if req.subject is not None or last is None or last.subject is None:
        return None
    own = extract_anchors(req.question, load_log_catalog())
    follow_up, _ = is_follow_up(req.question, own)
    return last.subject.model_dump(exclude_none=True) if follow_up else None


def _subject_context(question: str, subject: dict, reason: str, *, must_exist: bool):
    """(Contextualization, evidence scope, the subject's indexed paths) for a picked or kept
    subject; (None, None, ()) for a kept subject that no longer matches the index. A picked
    one that matches nothing is a 422."""
    selection = describe_selection(question, subject, load_subject_documents(subject))
    if selection is None:
        if must_exist:
            raise ApiError(422, "validation_error",
                           "subject does not match any indexed incident or failed run")
        return None, None, ()
    ctx = Contextualization(CONTEXTUALIZED, reason, question, selection.retrieval_query,
                            subjects=(selection.label,), subject=selection.subject)

    def scope(hits):
        return scope_hits(hits, load_document_info(sorted({h.path for h in hits})),
                          selection.run_key)
    return ctx, scope, selection.paths


def _classify_ask_error(exc: Exception) -> ApiError:
    """Map an exception to the API error the client sees: a fixed code, and a message that
    is our own text or the exception text passed through redact()."""
    text = redact(str(exc))[:500]
    if isinstance(exc, ApiError):  # already classified by the handler (an unknown subject)
        return exc
    if isinstance(exc, LLMNotConfiguredError):
        return ApiError(503, "llm_not_configured", SETUP_INSTRUCTIONS)
    if isinstance(exc, LLMBadResponseError):
        return ApiError(502, "llm_bad_response", text)
    if isinstance(exc, LLMError):
        return ApiError(502, "llm_upstream_error", text)
    if isinstance(exc, psycopg.Error):
        return _database_error(exc)
    if isinstance(exc, EmbeddingUnavailableError):
        return ApiError(503, "embedding_unavailable", text)
    if isinstance(exc, EmbeddingDimensionError):
        return ApiError(503, "embedding_misconfigured", text)
    return ApiError(500, "internal_error", "internal error; details are in the server log")


@app.post("/ask", response_model=AskResponse, responses=ERROR_RESPONSES)
def ask(
    req: AskRequest,
    embedder: Annotated[Embedder, Depends(get_embedder_dep)],
    llm_factory: Annotated[Callable[[], LLMProvider], Depends(get_llm_factory)],
) -> AskResponse:
    """Retrieve evidence and answer with verified citations, or refuse when evidence is weak.
    Every request that reaches this handler is recorded in `queries`, failures included."""
    # One database connection for the whole request instead of one per query
    # (db.shared_connection); behaviour is otherwise unchanged.
    with shared_connection():
        return _ask(req, embedder, llm_factory)


def _ask(req: AskRequest, embedder: Embedder,
         llm_factory: Callable[[], LLMProvider]) -> AskResponse:
    t0 = time.perf_counter()
    settings = get_settings()
    k = req.top_k or settings.ask_top_k
    filters = Filters(tuple(req.source_types) if req.source_types else None, req.pipeline)
    # Only the number of history turns is recorded; each turn was logged when it was asked.
    request_meta = {"top_k": k, "retrieval_only": req.retrieval_only, **filters.as_dict(),
                    "history_turns": len(req.history or [])}
    if req.subject is not None:
        request_meta["subject"] = req.subject.model_dump(exclude_none=True)
    timed = _TimedEmbedder(embedder)
    retriever = get_retriever(settings.retriever, timed)
    try:
        # Stateless: the conversation exists only in this request, so nothing can carry over
        # between users or sessions.
        ctx, context_ms, scope, subject_paths = None, 0, None, ()
        kept = _kept_subject(req)
        if req.subject is not None or kept is not None:
            # A failure picked from a clarification, or kept from the previous turn for a
            # follow-up that names nothing of its own: searched for and shown to the model as
            # context, never as evidence; evidence about other failures is left out, and the
            # gate and citation checks still decide.
            t_ctx = time.perf_counter()
            ctx, scope, subject_paths = _subject_context(
                req.question, req.subject.model_dump(exclude_none=True)
                if req.subject is not None else kept,
                "subject selected from a clarification" if req.subject is not None
                else "subject kept from the previous turn",
                must_exist=req.subject is not None)
            context_ms = _elapsed_ms(t_ctx)
        if ctx is None and req.history:
            t_ctx = time.perf_counter()
            ctx = contextualize(
                req.question,
                [Turn(t.question, t.answer, tuple(t.cited_paths)) for t in req.history],
                load_log_catalog(), load_document_info)
            context_ms = _elapsed_ms(t_ctx)
        outcome = answer_question(
            req.question, retriever=retriever, llm_factory=llm_factory, k=k,
            min_similarity=settings.min_similarity, filters=filters,
            retrieval_only=req.retrieval_only, contextualization=ctx,
            # Never for a picked or kept subject: that question is already narrowed down.
            clarifier=None if scope is not None else make_clarifier(
                retriever, min_similarity=settings.clarify_min_similarity,
                max_candidates=settings.clarify_max_candidates),
            scope=scope,
            # A refused question may still match one or several indexed failures.
            near_miss=None if scope is not None else make_near_miss_resolver(
                retriever, min_similarity=settings.clarify_min_similarity,
                max_candidates=settings.clarify_max_candidates),
            # The selected failure's own documented resolution and prevention.
            remediation=lambda paths, hits: remediation_hits(
                paths, hits, getattr(retriever, "last_question_vec", None)),
            subject_paths=subject_paths,
        )
        outcome.timings |= {"embedding_ms": round(timed.ms), "contextualization_ms": context_ms}
        query_id = log_query(outcome, retriever_name=retriever.name, request=request_meta)
    except Exception as exc:  # noqa: BLE001 - every failure is classified, recorded, re-raised
        error = _classify_ask_error(exc)
        if error.code == "internal_error":
            logger.exception("ask failed with an unexpected error")
        partial = getattr(exc, "partial_outcome", None)
        if partial is not None:
            partial.timings["embedding_ms"] = round(timed.ms)
        # A database outage cannot be recorded in the database; the warning below is all.
        if error.code != "db_unavailable":
            error.query_id = log_failed_query(
                question=req.question, status_code=error.status, error_category=error.code,
                latency_ms=_elapsed_ms(t0), message=error.message, request=request_meta,
                retriever_name=retriever.name, outcome=partial)
        logger.warning("ask failed status=%s category=%s latency_ms=%d query_id=%s",
                       error.status, error.code, _elapsed_ms(t0), error.query_id)
        raise error from None
    logger.info("ask query_id=%s status=%s reason=%s k=%d best=%s conversation=%s "
                "latency_ms=%d timings=%s model=%s", query_id, outcome.status,
                outcome.insufficient_reason, k, outcome.gate.get("best_similarity"),
                ctx.status if ctx else None, outcome.latency_ms, outcome.timings,
                outcome.model)
    return _ask_response(outcome, query_id, retriever.name, k)


# ---- documents (read-only Knowledge Base) -------------------------------------------------


@app.get("/documents", response_model=list[DocumentSummaryOut], responses=ERROR_RESPONSES)
def documents() -> list[DocumentSummaryOut]:
    """Every indexed document with its type, title, sections, chunk count and metadata.
    Read-only: nothing is written and no ingestion is triggered."""
    try:
        return [DocumentSummaryOut(**d.__dict__) for d in list_documents()]
    except psycopg.Error as exc:
        raise _database_error(exc) from None


@app.get("/documents/{path:path}", response_model=DocumentDetailOut, responses=ERROR_RESPONSES)
def document(path: str) -> DocumentDetailOut:
    """One document: its indexed chunks, and the original file (sensitive values masked) when
    it is available under DATA_DIR."""
    try:
        detail = get_document(path)
    except DocumentNotFoundError:
        raise ApiError(404, "document_not_found", "no indexed document has this path") from None
    except psycopg.Error as exc:
        raise _database_error(exc) from None
    return DocumentDetailOut(document=DocumentSummaryOut(**detail.summary.__dict__),
                             chunks=[c.__dict__ for c in detail.chunks],
                             original=detail.original, original_note=detail.original_note)


@app.post("/documents/explain", response_model=ExplainResponse, responses=ERROR_RESPONSES)
def explain(
    req: ExplainRequest,
    llm_factory: Annotated[Callable[[], LLMProvider], Depends(get_llm_factory)],
) -> ExplainResponse:
    """Explain one indexed document with the LLM, citing its own sections. Only on request:
    opening a document never calls the model. Uses the indexed (redacted) chunks only."""
    t0 = time.perf_counter()
    try:
        detail = get_document(req.path, with_original=False)
        s = detail.summary
        result = explain_document(path=s.path, source_type=s.source_type, title=s.title,
                                  content_hash=s.content_hash, chunks=detail.chunks,
                                  llm_factory=llm_factory)
    except DocumentNotFoundError:
        raise ApiError(404, "document_not_found", "no indexed document has this path") from None
    except Exception as exc:  # noqa: BLE001 - classified exactly like /ask
        error = _classify_ask_error(exc)
        if error.code == "internal_error":
            logger.exception("explain failed with an unexpected error")
        logger.warning("explain failed status=%s category=%s latency_ms=%d", error.status,
                       error.code, _elapsed_ms(t0))
        raise error from None
    logger.info("explain path=%s cached=%s points=%d dropped=%d truncated=%s latency_ms=%d "
                "attempts=%s", result.path, result.cached, len(result.points),
                result.dropped_points, result.truncated, _elapsed_ms(t0), result.llm_attempts)
    return ExplainResponse(**result.as_dict())


# ---- ingest -------------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse, responses=ERROR_RESPONSES)
def ingest(
    req: IngestRequest, embedder: Annotated[Embedder, Depends(get_embedder_dep)]
) -> IngestResponse:
    """Index new and changed files (synchronous). Paths are relative to the data directory."""
    try:
        summary = ingest_paths(req.paths, embedder=embedder, prune=req.prune,
                               dry_run=req.dry_run, trigger="manual")
    except UnsafePathError as exc:
        raise ApiError(422, "invalid_path", str(exc)) from None
    except IndexingLockedError as exc:
        raise ApiError(409, "indexing_in_progress", str(exc)) from None
    except psycopg.Error as exc:
        raise _database_error(exc) from None
    except EmbeddingUnavailableError as exc:
        raise ApiError(503, "embedding_unavailable", str(exc)) from None
    return IngestResponse(**summary.as_dict())
