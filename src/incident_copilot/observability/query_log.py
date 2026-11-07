"""Write one `queries` row and its `retrievals` rows per /ask, and read history back.

Each retrieval row keeps a snapshot of the evidence exactly as it was shown (path, section,
timestamps, content hash, redacted content). `chunk_id` is a live link that becomes NULL when
the chunk is replaced or pruned (ON DELETE SET NULL, db/init/003). Re-indexing therefore never
deletes history, and an old row can never point at replacement content.
"""

import logging

from psycopg.types.json import Jsonb

from incident_copilot.db import connect
from incident_copilot.generation.answer import AskOutcome
from incident_copilot.generation.prompt import evidence_id
from incident_copilot.redaction import redact

logger = logging.getLogger("incident_copilot.observability")

MAX_LOGGED_QUESTION_CHARS = 1000

# The only values ever written to queries.error_category.
ERROR_CATEGORIES = frozenset({
    "validation_error", "llm_not_configured", "llm_upstream_error", "llm_bad_response",
    "db_unavailable", "database_error", "embedding_unavailable", "embedding_misconfigured",
    "internal_error",
})


def safe_question(question: object) -> str:
    """What may be stored for a question: redacted, truncated, never a non-string."""
    if not isinstance(question, str):
        return ""
    return redact(question.strip())[:MAX_LOGGED_QUESTION_CHARS]


def _insert(*, question: str, answer, abstained: bool, retriever_name, latency_ms,
            retrieval_ms, generation_ms, tokens_in, tokens_out, model, metadata: dict,
            hits, used: set, status_code: int, error_category: str | None) -> str:
    with connect() as conn:
        query_id = conn.execute(
            """
            INSERT INTO queries (question, answer, abstained, retrieval_config, latency_ms,
                                 retrieval_ms, generation_ms, tokens_in, tokens_out, model,
                                 metadata, status_code, error_category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (safe_question(question), answer, abstained, retriever_name, latency_ms,
             retrieval_ms, generation_ms, tokens_in, tokens_out, model, Jsonb(metadata),
             status_code, error_category),
        ).fetchone()[0]
        if hits:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO retrievals (query_id, chunk_id, rank, vector_score,
                                            used_in_answer, evidence_id, document_path,
                                            source_type, chunk_index, section, ts_start,
                                            ts_end, chunk_content_hash, content)
                    -- The subselect stores NULL instead of failing if the chunk was
                    -- replaced between retrieval and logging.
                    VALUES (%s, (SELECT id FROM chunks WHERE id = %s::uuid), %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (query_id, h.chunk_id, h.rank, h.scores.get("vector_similarity"),
                         h.chunk_id in used, evidence_id(h), h.path, h.source_type,
                         h.chunk_index, h.section, h.ts_start, h.ts_end, h.content_hash,
                         h.text)
                        for h in hits
                    ],
                )
    return str(query_id)


def _retrieval_info(outcome: AskOutcome) -> dict:
    """retrieval_info as stored: a follow-up's resolved retrieval query contains the user's
    question, so it is redacted and truncated like the question itself."""
    info = dict(outcome.retrieval_info or {})
    conv = info.get("conversation")
    if isinstance(conv, dict) and "retrieval_query" in conv:
        info["conversation"] = {**conv, "retrieval_query": safe_question(conv["retrieval_query"])}
    return info


def _retrieved_meta(outcome: AskOutcome) -> list[dict]:
    return [{"evidence_id": evidence_id(h), "origin": h.origin, "linked_from": h.linked_from,
             "scores": h.scores} for h in outcome.hits]


def log_query(outcome: AskOutcome, *, retriever_name: str, request: dict) -> str:
    """Persist a successful /ask (HTTP 200); returns the new query id."""
    metadata = {
        "request": request,
        "status": outcome.status,
        "insufficient_reason": outcome.insufficient_reason,
        "gate": outcome.gate,
        "validation": outcome.validation,
        "retrieval_info": _retrieval_info(outcome),
        # Stage latencies, whether the LLM was called and how many attempts it took.
        "timings": outcome.timings,
        # Diagnosis only: what the model said was missing, and claims validation removed.
        "refusal_note": outcome.refusal_note,
        "dropped_claims": outcome.dropped_claims,
        "response": {
            "evidence": outcome.evidence,
            "inference": outcome.inference,
            "next_steps": outcome.next_steps,
            "next_step_evidence": outcome.next_step_evidence,
            "next_step_kinds": outcome.next_step_kinds,
            "next_step_support": outcome.next_step_support,
            "unknowns": outcome.unknowns,
            "citations": [c.evidence_id for c in outcome.citations],
        },
        "retrieved": _retrieved_meta(outcome),
        # The failures offered for a vague question (insufficient_reason "ambiguous_query").
        "clarification": outcome.clarification,
    }
    return _insert(question=outcome.question, answer=outcome.answer,
                   abstained=bool(outcome.insufficient_evidence), retriever_name=retriever_name,
                   latency_ms=outcome.latency_ms, retrieval_ms=outcome.retrieval_ms,
                   generation_ms=outcome.generation_ms, tokens_in=outcome.tokens_in,
                   tokens_out=outcome.tokens_out, model=outcome.model, metadata=metadata,
                   hits=outcome.hits, used=outcome.used_chunk_ids, status_code=200,
                   error_category=None)


def log_failed_query(*, question: object, status_code: int, error_category: str,
                     latency_ms: int, message: str = "", request: dict | None = None,
                     retriever_name: str | None = None,
                     outcome: AskOutcome | None = None) -> str | None:
    """Persist a failed /ask. Returns the query id, or None if it could not be stored (for
    example because the database itself is down); it never raises, so logging can never
    replace the original error. Only a fixed category, our own (redacted, truncated) error
    message, and the redacted question are stored."""
    if error_category not in ERROR_CATEGORIES:
        error_category = "internal_error"
    metadata = {"request": request or {}, "status": "error",
                "error": {"category": error_category, "message": redact(message)[:300]}}
    if outcome is not None:
        metadata |= {"gate": outcome.gate, "retrieval_info": _retrieval_info(outcome),
                     "timings": outcome.timings, "retrieved": _retrieved_meta(outcome)}
    try:
        return _insert(question=question, answer=None, abstained=False,
                       retriever_name=retriever_name, latency_ms=latency_ms,
                       retrieval_ms=outcome.retrieval_ms if outcome else None,
                       generation_ms=outcome.generation_ms if outcome else None,
                       tokens_in=outcome.tokens_in if outcome else None,
                       tokens_out=outcome.tokens_out if outcome else None,
                       model=outcome.model if outcome else None, metadata=metadata,
                       hits=outcome.hits if outcome else [], used=set(),
                       status_code=status_code, error_category=error_category)
    except Exception as exc:  # noqa: BLE001 - observability must not mask the real error
        logger.warning("could not record failed /ask (status=%s category=%s latency_ms=%s): "
                       "%s", status_code, error_category, latency_ms, type(exc).__name__)
        return None


def query_history(query_id: str) -> list[dict]:
    """Retrievals of a past query. `is_current` is true only if the chunk still exists with
    identical content; otherwise the snapshot is the only faithful record of the evidence."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT r.evidence_id, r.rank, r.document_path, r.section, r.chunk_index,
                   r.chunk_content_hash, r.content, r.used_in_answer, r.vector_score,
                   r.chunk_id::text,
                   (c.id IS NOT NULL AND c.content_hash = r.chunk_content_hash) AS is_current
            FROM retrievals r LEFT JOIN chunks c ON c.id = r.chunk_id
            WHERE r.query_id = %s ORDER BY r.rank
            """,
            (query_id,),
        ).fetchall()
    keys = ["evidence_id", "rank", "path", "section", "chunk_index", "content_hash", "content",
            "used_in_answer", "vector_score", "chunk_id", "is_current"]
    return [dict(zip(keys, r, strict=True)) for r in rows]
