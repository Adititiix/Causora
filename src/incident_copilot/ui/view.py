"""Turn an /ask response (a plain dict) into what the chat page shows.

Pure functions, no Streamlit, so they are unit-tested. Every field is read defensively: a
missing or null field renders as nothing, never as made-up content. Labels shown to the
user come only from the response (paths, sections, ids).

Next steps carry the evidence ids the model cited for them (`next_step_evidence`, validated
by the API like claims); a step without one is shown as unsupported. `next_step_sources` (a
text match of the step's wording in the chunks) is kept for older responses and diagnosis
only: the page never presents a text match as a citation.
"""

import re
from dataclasses import dataclass, field

EXAMPLE_QUESTIONS = (
    "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-16?",
    "Why did the wait_for_orders_export task in the orders_etl pipeline fail on 2026-09-01?",
    "Which past incident was caused by the payments provider API rate limiting our "
    "requests, and how long did it last?",
    "A metric task fails with ZeroDivisionError. How should I fix it?",
    "What is the retention period for the clickstream Kafka topic in production?",
)

# insufficient_reason values set by generation/answer.py.
INSUFFICIENT_REASONS = {
    "no_hits": "No indexed document matched the question.",
    "below_min_similarity": "The closest evidence was below the similarity threshold, so "
                            "the model was not asked.",
    "model_insufficient_evidence": "The model read the retrieved evidence and reported that "
                                   "it does not answer the question.",
    "no_valid_citations": "The model's answer cited no valid evidence, so it was not shown.",
    "ambiguous_reference": "The question refers back to an answer that covered more than one "
                           "incident or run, so Causora is asking which one you mean instead "
                           "of guessing. Nothing was searched and the model was not asked.",
    "ambiguous_query": "Several different failures match this question, but none closely "
                       "enough to answer it, so Causora is asking which one you mean. The "
                       "model was not asked. The failure you pick is searched for and answered "
                       "only if its evidence passes the usual similarity threshold.",
}

# Mirrors api/schemas.py (MAX_HISTORY_TURNS and the ConversationTurn field limits).
MAX_HISTORY_TURNS = 6
_MAX_TURN_QUESTION = 1000
_MAX_TURN_ANSWER = 2000
_MAX_TURN_PATHS = 20

# Our own codes from client.py plus the API's error codes (api/main.py).
FAILURE_HINTS = {
    "timeout": "The API took too long. Gemini may be slow or rate limited; try again.",
    "unreachable": "Start the API (see the README) and check UI_API_URL.",
    "bad_response": "The API returned something the UI does not understand.",
    "llm_not_configured": "The API has no Gemini key configured. Retrieval-only mode still "
                          "works.",
    "llm_upstream_error": "Gemini timed out, was overloaded or rate limited, and the API's "
                          "retries did not get through. Your conversation is kept; ask again "
                          "in a moment.",
    "llm_bad_response": "Gemini returned a malformed answer; it was not shown.",
    "db_unavailable": "The API cannot reach the database. Start it with "
                      "`docker compose up -d db`.",
    "database_error": "The database schema may be out of date; run "
                      "`python -m incident_copilot.db_migrate`.",
    "embedding_unavailable": "The embedding model is not available on the API host.",
    "validation_error": "The request was rejected (the question may be empty or too long).",
}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$])")
_STEP_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def escape_md(text: str | None) -> str:
    """Model and document text is shown literally, never interpreted as Markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def _norm(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class Chunk:
    evidence_id: str
    path: str
    source_type: str
    section: str | None
    text: str
    similarity: float | None
    origin: str | None
    linked_from: str | None
    used_in_answer: bool

    @property
    def label(self) -> str:
        parts = [DOC_TYPE_NAMES.get(self.source_type, self.source_type or "?"), self.path or "?"]
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)


@dataclass(frozen=True)
class Choice:
    """A failure offered by a clarification (insufficient_reason "ambiguous_query"); every
    field comes from the API, which takes it from indexed metadata."""

    id: str
    label: str
    description: str
    subject: dict


@dataclass(frozen=True)
class Claim:
    text: str
    ids: tuple[str, ...]
    confidence: str | None = None  # inferences only


@dataclass(frozen=True)
class NextStep:
    text: str
    matched_ids: tuple[str, ...]  # chunks containing this exact wording (UI text match)
    evidence_ids: tuple[str, ...] = ()  # what the model cited for it, validated by the API
    # From the API (next_step_kinds / next_step_support), None when it sent none (older API).
    kind: str | None = None  # "recovery" | "prevention"
    support: str | None = None  # "incident_report" | "general_guidance" | "unsupported"


@dataclass
class AnswerView:
    status: str
    summary: str
    insufficient: bool
    insufficient_reason: str | None
    insufficient_explanation: str | None
    facts: list[Claim] = field(default_factory=list)
    inferences: list[Claim] = field(default_factory=list)
    next_steps: list[NextStep] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    best_similarity: float | None = None
    min_similarity: float | None = None
    model: str | None = None
    retriever: str | None = None
    latency_ms: int | None = None
    query_id: str | None = None
    # retrieval_info["conversation"] for a request sent with history (else empty).
    conversation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    top_k: int | None = None  # chunks requested for this answer (server default if unset)
    choices: list[Choice] = field(default_factory=list)  # a clarification's candidates
    unknowns: list[str] = field(default_factory=list)  # what the evidence leaves open
    # gate["accepted_by"]: a near miss answered about the one indexed failure it matches.
    accepted_by: dict = field(default_factory=dict)

    @property
    def matched_failure(self) -> str | None:
        """The failure a near miss was answered about (its label), else None."""
        if self.conversation.get("status") != "matched_in_index" or not self.accepted_by:
            return None
        subjects = [str(s) for s in _list(self.conversation.get("subjects")) if s]
        return subjects[0] if subjects else None

    @property
    def interpreted_as(self) -> tuple[str, ...]:
        """Subjects a follow-up was resolved to (empty unless it was contextualized)."""
        if self.conversation.get("status") != "contextualized":
            return ()
        return tuple(str(s) for s in _list(self.conversation.get("subjects")) if s)

    def chunk(self, evidence_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.evidence_id == evidence_id), None)


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _ids(value) -> tuple[str, ...]:
    return tuple(str(i) for i in _list(value) if i)


def _float(value) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def parse_chunks(retrieved) -> list[Chunk]:
    chunks = []
    for item in _list(retrieved):
        if not isinstance(item, dict) or not item.get("evidence_id"):
            continue
        scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        chunks.append(Chunk(
            evidence_id=str(item["evidence_id"]),
            path=str(item.get("path") or ""),
            source_type=str(item.get("source_type") or ""),
            section=item.get("section"),
            text=str(item.get("text") or ""),
            # Cosine similarity to the question (retrieval/vector.py), what the gate uses.
            similarity=_float(scores.get("vector_similarity")),
            origin=item.get("origin"),
            linked_from=item.get("linked_from"),
            used_in_answer=bool(item.get("used_in_answer")),
        ))
    return chunks


def next_step_sources(step: str, chunks: list[Chunk]) -> tuple[str, ...]:
    """Ids of chunks whose text contains the step's wording (ignoring case, punctuation and a
    leading list marker). Exact phrase only: a paraphrase is never claimed as a match."""
    needle = _norm(_STEP_PREFIX.sub("", step))
    if len(needle.split()) < 3:  # too short to mean anything
        return ()
    return tuple(c.evidence_id for c in chunks if f" {needle} " in f" {_norm(c.text)} ")


_SUBJECT_KEYS = ("incident_id", "pipeline", "task", "date")


def parse_choices(clarification) -> list[Choice]:
    """The candidates of a clarification, in the API's order; malformed entries (no id, label
    or subject) are skipped, never repaired."""
    raw = clarification.get("candidates") if isinstance(clarification, dict) else None
    out = []
    for c in _list(raw):
        if not isinstance(c, dict) or not c.get("id") or not c.get("label"):
            continue
        subject = c.get("subject") if isinstance(c.get("subject"), dict) else {}
        subject = {k: str(subject[k]) for k in _SUBJECT_KEYS if subject.get(k)}
        if subject:
            out.append(Choice(str(c["id"]), str(c["label"]), str(c.get("description") or ""),
                              subject))
    return out


def build_view(data: dict) -> AnswerView:
    status = str(data.get("status") or "unknown")
    chunks = parse_chunks(data.get("retrieved"))
    reason = data.get("insufficient_reason")
    insufficient = status == "insufficient_evidence" or bool(data.get("insufficient_evidence"))
    gate = data.get("gate") if isinstance(data.get("gate"), dict) else {}
    facts = [Claim(str(e.get("claim") or ""), _ids(e.get("evidence_ids")))
             for e in _list(data.get("evidence")) if isinstance(e, dict) and e.get("claim")]
    inferences = [Claim(str(i.get("claim") or ""), _ids(i.get("based_on")),
                        i.get("confidence"))
                  for i in _list(data.get("inference"))
                  if isinstance(i, dict) and i.get("claim")]
    step_ids = _list(data.get("next_step_evidence"))
    kinds, support = _list(data.get("next_step_kinds")), _list(data.get("next_step_support"))

    def at(values: list, n: int) -> str | None:
        return str(values[n]) if n < len(values) and isinstance(values[n], str) else None

    steps = [NextStep(s.strip(), next_step_sources(s, chunks),
                      _ids(step_ids[n]) if n < len(step_ids) else (),
                      at(kinds, n), at(support, n))
             for n, s in enumerate(_list(data.get("next_steps")))
             if isinstance(s, str) and s.strip()]
    cited = [str(c["evidence_id"]) for c in _list(data.get("citations"))
             if isinstance(c, dict) and c.get("evidence_id")]
    info = data.get("retrieval_info") if isinstance(data.get("retrieval_info"), dict) else {}
    conversation = info.get("conversation") if isinstance(info.get("conversation"), dict) else {}
    return AnswerView(
        status=status,
        summary=str(data.get("answer") or ""),
        insufficient=insufficient,
        insufficient_reason=reason,
        insufficient_explanation=(INSUFFICIENT_REASONS.get(reason, reason)
                                  if insufficient else None),
        facts=facts, inferences=inferences, next_steps=steps, cited_ids=cited,
        chunks=chunks,
        best_similarity=_float(gate.get("best_similarity")),
        min_similarity=_float(gate.get("min_similarity")),
        model=data.get("model"),
        retriever=data.get("retriever"),
        latency_ms=data.get("latency_ms") if isinstance(data.get("latency_ms"), int) else None,
        query_id=data.get("query_id"),
        conversation=conversation,
        timings=data.get("timings") if isinstance(data.get("timings"), dict) else {},
        top_k=data.get("top_k") if isinstance(data.get("top_k"), int) else None,
        choices=parse_choices(data.get("clarification")),
        unknowns=[str(u).strip() for u in _list(data.get("unknowns"))
                  if isinstance(u, str) and u.strip()],
        accepted_by=gate.get("accepted_by") if isinstance(gate.get("accepted_by"), dict)
        else {},
    )


def history_for_request(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """This session's earlier turns in the /ask `history` format, oldest first, bounded.

    `messages` alternate {"role": "user", "content"} and {"role": "assistant", "result"}.
    A turn carries the question, the answer summary and the paths the answer cited; a
    refused or failed turn carries only its question (which may still name an incident).
    A turn the API answered about a picked or kept failure also carries that `subject`, so a
    follow-up keeps it. Local messages (choices narrowed by the page, no API call) are not
    turns."""
    turns: list[dict] = []
    pending: str | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content")
            pending = content.strip() if isinstance(content, str) and content.strip() else None
            continue
        if msg.get("role") != "assistant" or pending is None:
            continue
        if msg.get("local"):
            pending = None
            continue
        turn: dict = {"question": pending[:_MAX_TURN_QUESTION]}
        result = msg.get("result")
        data = getattr(result, "data", None)
        subject = conversation_subject(data)
        if subject:
            turn["subject"] = subject
        if isinstance(data, dict) and data.get("status") == "answered":
            if isinstance(data.get("answer"), str) and data["answer"].strip():
                turn["answer"] = data["answer"].strip()[:_MAX_TURN_ANSWER]
            paths = [c.get("path") for c in _list(data.get("citations")) if isinstance(c, dict)]
            paths = list(dict.fromkeys(p for p in paths if isinstance(p, str) and p))
            if paths:
                turn["cited_paths"] = paths[:_MAX_TURN_PATHS]
        turns.append(turn)
        pending = None
    return turns[-max_turns:]


def conversation_subject(data) -> dict | None:
    """The failure an answer was about, as the API reported it (retrieval_info.conversation
    .subject): only its incident_id / pipeline / task / date strings, or None."""
    info = data.get("retrieval_info") if isinstance(data, dict) else None
    conv = info.get("conversation") if isinstance(info, dict) else None
    subject = conv.get("subject") if isinstance(conv, dict) else None
    if not isinstance(subject, dict):
        return None
    out = {k: str(subject[k]) for k in _SUBJECT_KEYS if subject.get(k)}
    return out or None


# ---- a typed reply to a clarification ("i mean about the shipment") -----------------------

MAX_REPLY_WORDS = 8  # longer text is a new question, not a pick
# snake_case names (shipments_sync, load_shipments_fact) and ISO dates.
_OWN_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|\d{4}-\d{2}-\d{2}")
_INCIDENT_ID = re.compile(r"\bINC-\d{3,6}\b", re.IGNORECASE)


def match_choices(text: str | None, choices: list[Choice]) -> list[Choice]:
    """The choices a short typed reply points at, in their order: the one whose incident id it
    names, else those sharing a topic word with it (clarify.topic_terms: "shipment" matches
    shipments_sync). [] for a longer text, or one that matches nothing."""
    from incident_copilot.generation.clarify import matching_by_topic, topic_terms

    text = (text or "").strip()
    if not text or len(text.split()) > MAX_REPLY_WORDS:
        return []
    named = {m.upper() for m in _INCIDENT_ID.findall(text)}
    if named:
        return [c for c in choices if (c.subject.get("incident_id") or "").upper() in named]
    # A pipeline/task/table name or a date of its own that no choice mentions: a new
    # question ("why did the orders_etl load fail on 2026-09-04?"), never a pick.
    own = {w.lower() for w in _OWN_IDENTIFIER.findall(text)}
    if own:
        choices = [c for c in choices
                   if all(w in f"{c.label} {c.description}".lower() for w in own)]
    return matching_by_topic(text, choices, lambda c: topic_terms(f"{c.label} {c.description}"))


def open_clarification(messages: list[dict]) -> tuple[str, list[Choice]] | None:
    """(question, choices) of the last message if it is an unanswered clarification, else
    None (a later user message means it was already answered)."""
    if not messages or not isinstance(messages[-1], dict):
        return None
    last = messages[-1]
    if last.get("role") != "assistant" or not last.get("question"):
        return None
    if last.get("local"):
        choices = last.get("choices") or []
    else:
        data = getattr(last.get("result"), "data", None)
        choices = build_view(data).choices if isinstance(data, dict) else []
    return (last["question"], list(choices)) if choices else None


def failure_hint(code: str) -> str | None:
    return FAILURE_HINTS.get(code)


# ---- what kind of result this is ----------------------------------------------------------

# (badge label, badge colour, one-line explanation) per kind of result.
OUTCOMES = {
    "answered": ("Answered from the evidence", "green",
                 "Grounded answer: every listed fact cites a retrieved chunk."),
    "gate": ("Refused: evidence too weak", "orange",
             "No retrieved chunk was similar enough to the question (similarity gate), so "
             "the model was not asked."),
    "model": ("Refused by the model", "orange",
              "The evidence was similar enough to send to the model, but the model reported "
              "that it does not answer the question. A high similarity score does not mean "
              "the answer is in the evidence."),
    "withheld": ("Answer withheld", "orange",
                 "The model's answer cited no valid evidence, so it was not shown."),
    "clarify": ("Clarification needed", "blue",
                "The question refers back to more than one incident or run."),
    "retrieval_only": ("Retrieval only", "gray", "No LLM was called."),
}


def outcome_kind(view: "AnswerView") -> str:
    if view.status == "retrieval_only":
        return "retrieval_only"
    if not view.insufficient:
        return "answered"
    return {"below_min_similarity": "gate", "no_hits": "gate",
            "model_insufficient_evidence": "model", "no_valid_citations": "withheld",
            "ambiguous_reference": "clarify",
            "ambiguous_query": "clarify"}.get(view.insufficient_reason or "", "model")


def evidence_label(view: "AnswerView") -> str:
    """The evidence expander's label. A refusal cites nothing; '0 cited' would wrongly suggest
    that nothing was retrieved, so the label says what happened to the chunks instead."""
    n = len(view.chunks)
    kind = outcome_kind(view)
    if kind == "answered":
        used = sum(c.used_in_answer for c in view.chunks)
        return f"Retrieved evidence: {n} chunks, {used} cited in the answer"
    if kind == "gate":
        return f"Retrieved evidence: {n} chunks (below the similarity threshold; not sent to " \
               "the model)"
    if kind == "retrieval_only":
        return f"Retrieved evidence: {n} chunks (no answer requested)"
    if view.insufficient_reason == "ambiguous_query":
        return (f"Retrieved evidence: {n} chunks (not sent to the model; Causora asked which "
                "failure you mean)")
    return f"Retrieved evidence: {n} chunks (sent to the model; no answer, so none cited)"


# ---- Knowledge Base -----------------------------------------------------------------------

# Labels for the source types that exist in the index (documents.source_type). Display only:
# the API, the database, retrieval and the evaluation sets keep the values themselves
# ("runbook" documents are shown as troubleshooting guides). Incident timelines are sections
# of incident reports, not separate documents. Dict order is the order types are listed in.
DOC_TYPES = {"runbook": "Troubleshooting guides", "log": "Task logs",
             "incident": "Incident reports", "schema": "Table schemas"}
DOC_TYPE_NAMES = {"runbook": "Troubleshooting guide", "log": "Task log",
                  "incident": "Incident report", "schema": "Table schema"}


def filter_documents(docs: list[dict], types: list[str] | None, query: str) -> list[dict]:
    """Documents of the given types whose title, path, pipeline or metadata values contain
    every word of `query` (case-insensitive). Malformed entries are skipped."""
    words = [w for w in (query or "").lower().split() if w]
    out = []
    for d in docs:
        if not isinstance(d, dict) or not d.get("path"):
            continue
        if types and d.get("source_type") not in types:
            continue
        meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        haystack = " ".join(str(v) for v in (d.get("title"), d.get("path"), d.get("pipeline"),
                                             *meta.values(), *(d.get("sections") or []))
                            if v).lower()
        if all(w in haystack for w in words):
            out.append(d)
    return out


TOPIC_LABELS = {"overview": "What it describes", "pipeline_or_task": "Pipeline or task",
                "stages": "Sequence of operations", "dependencies": "Dependencies",
                "failure_modes": "Failure modes and errors", "diagnosis": "Diagnosis",
                "remediation": "Remediation", "prevention": "Prevention",
                "terms": "Technical terms", "other": "Other"}


def group_points(points) -> list[tuple[str, list[dict]]]:
    """Explanation points grouped by topic in TOPIC_LABELS order (unknown topics last)."""
    groups: dict[str, list[dict]] = {}
    for p in _list(points):
        if isinstance(p, dict) and p.get("text"):
            groups.setdefault(str(p.get("topic") or "other"), []).append(p)
    order = list(TOPIC_LABELS)
    return sorted(groups.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order
                  else len(order))


# Words that point back at an earlier turn. The API resolves a follow-up from the history the
# page sends; this hint is for a refusal where that did not happen (first question, or no
# earlier turn named an incident, pipeline or task).
_CONTEXT_REF = re.compile(
    r"\b(?:this|that|these|those|it|again|same (?:issue|problem|error|failure|incident)|"
    r"prevent it)\b", re.IGNORECASE)

FOLLOWUP_HINT = ("This question seems to refer to something earlier, but Causora could not "
                 "link it to an incident, pipeline or task from this conversation. Ask again "
                 "and name the incident ID, pipeline, task or error message (for example "
                 "INC-2043, payments_sync, pull_payments_api or HTTP 429).")


def followup_hint(question: str | None, view: AnswerView) -> str | None:
    """A hint for a refused question that looks like an unresolved follow-up. Only shown,
    never acted on: the question is not rewritten and nothing is resent. Not shown when the
    API did resolve the follow-up (then the refusal is about the evidence) or asked which
    incident was meant."""
    if not view.insufficient or not question or not _CONTEXT_REF.search(question):
        return None
    if view.conversation.get("status") in ("contextualized", "ambiguous"):
        return None
    return FOLLOWUP_HINT
