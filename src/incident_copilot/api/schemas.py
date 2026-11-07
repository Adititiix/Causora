"""Request and response models for the HTTP API."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SourceType = Literal["runbook", "log", "schema", "incident"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorBody(BaseModel):
    error: ErrorDetail


# ---- /ask ---------------------------------------------------------------------------------

MAX_HISTORY_TURNS = 6  # matches retrieval.contextualize.MAX_TURNS


_IDENT = r"^[A-Za-z0-9_.\-]+$"


class SubjectSelection(BaseModel):
    """The failure the user picked from a clarification (AskResponse.clarification): an
    incident id, a failed run (pipeline, task and date), or both. It must match an indexed
    incident or task log; it is context for retrieval and the model, never evidence."""

    model_config = ConfigDict(extra="forbid")

    incident_id: Annotated[str, StringConstraints(pattern=r"^INC-\d{3,6}$")] | None = None
    pipeline: Annotated[str, StringConstraints(min_length=1, max_length=100,
                                               pattern=_IDENT)] | None = None
    task: Annotated[str, StringConstraints(min_length=1, max_length=100,
                                           pattern=_IDENT)] | None = None
    date: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None

    @model_validator(mode="after")
    def _names_one_failure(self):
        run = (self.pipeline, self.task, self.date)
        if any(run) and not all(run):
            raise ValueError("pipeline, task and date must be given together")
        if not self.incident_id and not all(run):
            raise ValueError("give an incident_id, or a pipeline, task and date")
        return self


class ConversationTurn(BaseModel):
    """One earlier exchange, sent back by the client. Untrusted: it only helps resolve what a
    follow-up refers to (retrieval/contextualize.py) and is never shown to the model as
    evidence. Cited paths that are not in the index are ignored."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                               max_length=1000)]
    answer: Annotated[str, StringConstraints(max_length=2000)] | None = None
    cited_paths: list[Annotated[str, StringConstraints(min_length=1, max_length=300)]] = \
        Field(default_factory=list, max_length=20)
    # The failure this turn was about, as the response reported it (retrieval_info
    # .conversation.subject). A follow-up keeps it as context; it is validated against the
    # index and never used as evidence.
    subject: SubjectSelection | None = None


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                               max_length=1000)]
    top_k: int | None = Field(default=None, ge=1, le=20,
                              description="Chunks to retrieve (default ASK_TOP_K)")
    retrieval_only: bool = Field(default=False,
                                 description="Return retrieved evidence without calling the LLM")
    source_types: list[SourceType] | None = Field(default=None, min_length=1, max_length=4)
    pipeline: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                               max_length=100,
                                               pattern=r"^[A-Za-z0-9_.\-]+$")] | None = None
    history: list[ConversationTurn] | None = Field(
        default=None, max_length=MAX_HISTORY_TURNS,
        description="Earlier turns of this conversation, oldest first (optional). Used only "
                    "to resolve what a follow-up refers to; omit it for a standalone question.")
    subject: SubjectSelection | None = Field(
        default=None,
        description="A failure picked from a clarification (optional). Retrieval and the "
                    "model use it as context; the similarity gate and citation checks still "
                    "decide whether to answer.")


class RetrievedChunk(BaseModel):
    evidence_id: str
    rank: int
    chunk_id: str
    path: str
    source_type: str
    section: str | None
    chunk_index: int
    ts_start: datetime | None
    ts_end: datetime | None
    scores: dict[str, float]
    origin: str
    linked_from: str | None
    used_in_answer: bool
    text: str


class EvidenceItem(BaseModel):
    claim: str
    evidence_ids: list[str]


class InferenceItem(BaseModel):
    claim: str
    based_on: list[str]
    confidence: Literal["low", "medium", "high"]


class CitationOut(BaseModel):
    evidence_id: str
    chunk_id: str
    path: str
    source_type: str
    section: str | None
    chunk_index: int
    ts_start: datetime | None
    ts_end: datetime | None


class ClarificationSubjectOut(BaseModel):
    incident_id: str | None = None
    pipeline: str | None = None
    task: str | None = None
    date: str | None = None


class ClarificationCandidateOut(BaseModel):
    """One failure the vague question may mean. Every field comes from indexed metadata."""

    id: str
    label: str
    description: str
    incident_id: str | None
    pipeline: str | None
    task: str | None
    date: str | None
    paths: list[str]
    # None for failures offered from an earlier answer (ambiguous_reference), not a search.
    best_similarity: float | None
    # Send this back as AskRequest.subject to pick the candidate.
    subject: ClarificationSubjectOut


class ClarificationOut(BaseModel):
    reason: str
    candidates: list[ClarificationCandidateOut]


class AskResponse(BaseModel):
    query_id: str
    status: Literal["answered", "insufficient_evidence", "retrieval_only"]
    answer: str | None
    insufficient_evidence: bool | None
    insufficient_reason: str | None
    evidence: list[EvidenceItem]
    inference: list[InferenceItem]
    next_steps: list[str]
    citations: list[CitationOut]
    retrieved: list[RetrievedChunk]
    model: str | None
    retriever: str
    gate: dict
    validation: dict
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int
    retrieval_ms: int
    generation_ms: int | None
    # Retriever decisions beyond ranking, e.g. whether a date filter was applied and why, and
    # for a request with history, how the question was resolved (`conversation`).
    retrieval_info: dict = {}
    # Stage latencies (contextualization_ms, embedding_ms, retrieval_ms, generation_ms,
    # validation_ms), prompt_chars, and llm_called / llm_attempts.
    timings: dict = {}
    # The number of chunks requested: the request's top_k, or ASK_TOP_K when it had none.
    top_k: int | None = None
    # Only for insufficient_reason "ambiguous_query": the failures a vague question may mean
    # (generation/clarify.py). Pick one by sending its `subject` with the same question.
    clarification: ClarificationOut | None = None
    # The evidence ids each next step cites, aligned with next_steps ([] = no supporting
    # evidence id: shown as unsupported, never as a citation).
    next_step_evidence: list[list[str]] = []
    # Also aligned with next_steps, decided from the cited chunks (generation/answer.py):
    # "recovery" | "prevention", and "incident_report" (an incident report about this failure
    # states it) | "general_guidance" (only runbooks, schemas or logs) | "unsupported".
    next_step_kinds: list[Literal["recovery", "prevention"]] = []
    next_step_support: list[Literal["incident_report", "general_guidance", "unsupported"]] = []
    # What the evidence leaves open (an alternative not ruled out, a missing measurement).
    unknowns: list[str] = []


# ---- /documents (read-only Knowledge Base) ------------------------------------------------


class DocumentSummaryOut(BaseModel):
    path: str
    source_type: SourceType
    title: str | None
    pipeline: str | None
    sections: list[str]
    chunk_count: int
    content_hash: str
    updated_at: str | None
    metadata: dict


class IndexedChunkOut(BaseModel):
    chunk_index: int
    section: str | None
    text: str
    ts_start: str | None
    ts_end: str | None


class DocumentDetailOut(BaseModel):
    document: DocumentSummaryOut
    chunks: list[IndexedChunkOut]
    original: str | None = Field(description="The source file (sensitive values masked), or "
                                             "null when only indexed chunks are available")
    original_note: str


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                           max_length=300)]


class ExplainPointOut(BaseModel):
    topic: str
    text: str
    basis: Literal["documented", "inferred"]
    section_ids: list[str]


class ExplainSectionOut(BaseModel):
    id: str
    chunk_index: int
    section: str | None


class ExplainResponse(BaseModel):
    path: str
    title: str | None
    content_hash: str
    model: str | None
    summary: str
    points: list[ExplainPointOut]
    not_documented: list[str]
    sections: list[ExplainSectionOut]
    dropped_points: int
    truncated: bool
    cached: bool
    generation_ms: int | None
    llm_attempts: int | None
    tokens_in: int | None
    tokens_out: int | None


# ---- /ingest ------------------------------------------------------------------------------


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[Annotated[str, StringConstraints(min_length=1, max_length=500)]] | None = Field(
        default=None, max_length=100,
        description="Files or folders relative to the data directory (default: everything)",
    )
    prune: bool = False
    dry_run: bool = False


class IngestResponse(BaseModel):
    dry_run: bool
    roots: list[str]
    run_id: str | None
    files_seen: int
    added: list[str]
    updated: list[str]
    unchanged: int
    deleted: list[str]
    chunks_written: int
    embed_calls: int
    texts_embedded: int
    duration_s: float
