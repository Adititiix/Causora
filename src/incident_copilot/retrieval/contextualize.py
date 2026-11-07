"""Resolve a follow-up question against earlier turns into a standalone RETRIEVAL query.

Deterministic, no LLM call. The user's question is never replaced: retrieval searches with
`retrieval_query`, the model still answers `question`, and the conversation is passed to the
model only as a note saying what the question refers to, never as evidence. The similarity
gate runs on the retrieval query exactly as it does for any other question.

A question is treated as a FOLLOW-UP only when all of these hold:
  * the request carries earlier turns;
  * it names no subject of its own: no incident id (INC-2043), known pipeline or task, or
    error name (KeyError, HTTP 429, ...);
  * it does not introduce a new one ("which past incident ...", "a load task fails ...");
  * it points back at something: it, this, that, again, the same issue, the incident, the
    root cause, what about ...
Anything else is STANDALONE and uses no history, so a new question never inherits an old
incident.

For a follow-up, the subject comes from the most recent earlier turn that has one: the
pipeline/task/date/incident named in that turn's question, otherwise the incidents and logs
its answer cited (looked up in the index, so an unknown path from the client is ignored).
If that turn is about more than one incident or failed run, the status is AMBIGUOUS and the
caller asks the user which one they mean instead of guessing. If no earlier turn has a
subject, the status is NO_CONTEXT and the question is searched as it is.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from incident_copilot.retrieval.query_scope import IDENT_RE, LogCatalog, extract_dates

MAX_TURNS = 6           # earlier turns considered (the request schema enforces the same cap)
MAX_SUBJECTS = 3        # subject labels added to one retrieval query
MAX_LABEL_CHARS = 160
MAX_QUERY_CHARS = 1000  # the /ask question limit

REFERENCE_RE = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|again|"
    r"same\s+(?:issue|problem|error|failure|incident|thing|task|pipeline)|"
    r"the\s+(?:incident|failure|error|issue|problem|outage|task|job|run|pipeline|fix|"
    r"root\s+cause|cause|runbook|resolution)|what\s+about|how\s+about|"
    r"what\s+happened\s+next|after\s+that)\b",
    re.IGNORECASE)
# "Which past incident ...", "What incident ...", "A load task fails ..." bring their own
# subject even though they may also say "it".
NEW_REFERENT_RE = re.compile(
    r"\b(?:which|what|a|an|any|another)\s+(?:[a-z_]+\s+)?"
    r"(?:incident|failure|error|outage|task|job|pipeline|dag)s?\b", re.IGNORECASE)
INCIDENT_RE = re.compile(r"\bINC-\d{3,6}\b", re.IGNORECASE)
ERROR_RE = re.compile(
    r"\b[A-Z][A-Za-z]*(?:Error|Exception|Timeout|Violation|Invalid|Detected)\b"
    r"|\bHTTP\s*\d{3}\b|\bSIGTERM\b|\bOOM\b")

STANDALONE, CONTEXTUALIZED, AMBIGUOUS, NO_CONTEXT = (
    "standalone", "contextualized", "ambiguous", "no_context")


@dataclass(frozen=True)
class Turn:
    """One earlier exchange, as the client reports it (untrusted)."""

    question: str
    answer: str | None = None
    cited_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocInfo:
    """What the index says about a cited document (documents.title and metadata)."""

    path: str
    source_type: str
    title: str | None = None
    incident_id: str | None = None
    pipeline: str | None = None
    task: str | None = None
    date: str | None = None
    error: str | None = None


DocLookup = Callable[[Sequence[str]], dict[str, DocInfo]]


@dataclass(frozen=True)
class Anchors:
    incidents: tuple[str, ...] = ()
    pipelines: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    # Other snake_case names (tables such as orders_fact, columns such as customer_tier).
    identifiers: tuple[str, ...] = ()

    @property
    def has_subject(self) -> bool:
        return bool(self.incidents or self.pipelines or self.tasks or self.errors
                    or self.identifiers)


@dataclass(frozen=True)
class Contextualization:
    status: str
    reason: str
    question: str
    retrieval_query: str
    subjects: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    source_turn: int | None = None  # 1 = the previous turn, 2 = the one before, ...
    own_anchors: dict = field(default_factory=dict)
    # AMBIGUOUS: the failures the earlier answer covered (index metadata), for a structured
    # "which one?" (generation/clarify.py). Not part of as_dict().
    candidate_docs: tuple = ()
    # A failure the user picked, or kept from the previous turn ({incident_id | pipeline,
    # task, date}); context for retrieval and the model, never evidence.
    subject: dict | None = None

    @property
    def context_note(self) -> str | None:
        """What the model is told about the conversation (only for a follow-up)."""
        if self.status != CONTEXTUALIZED:
            return None
        return ("This question follows earlier messages about: " + "; ".join(self.subjects)
                + ". Use this only to understand what the question refers to. It is not "
                "evidence: state facts only from the evidence blocks, and if they are about "
                "something else, say the evidence is insufficient.")

    def clarification(self) -> str:
        options = ", ".join(self.candidates)
        return ("Your question refers to an earlier answer that covered more than one "
                f"incident or failed run ({options}). Which one do you mean? Please ask again "
                "naming the incident ID, or the pipeline, task and date.")

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason,
                "retrieval_query": self.retrieval_query, "subjects": list(self.subjects),
                "candidates": list(self.candidates), "source_turn": self.source_turn,
                "own_anchors": self.own_anchors} | ({"subject": dict(self.subject)}
                                                    if self.subject else {})


def extract_anchors(text: str, catalog: LogCatalog) -> Anchors:
    words = set(IDENT_RE.findall(text.lower()))
    dates, _ = extract_dates(text)
    known = catalog.pipelines | catalog.tasks
    return Anchors(
        incidents=tuple(sorted({m.upper() for m in INCIDENT_RE.findall(text)})),
        pipelines=tuple(sorted(words & catalog.pipelines)),
        tasks=tuple(sorted(words & catalog.tasks)),
        dates=tuple(dates),
        errors=tuple(dict.fromkeys(ERROR_RE.findall(text))),
        identifiers=tuple(sorted(w for w in words if "_" in w.strip("_") and w not in known)),
    )


def is_follow_up(question: str, own: Anchors) -> tuple[bool, str]:
    if own.has_subject:
        return False, "the question names its own incident, pipeline, task or error"
    if NEW_REFERENT_RE.search(question):
        return False, "the question introduces its own subject"
    if not REFERENCE_RE.search(question):
        return False, "the question does not refer to earlier messages"
    return True, "the question refers to earlier messages"


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LABEL_CHARS else text[:MAX_LABEL_CHARS - 1] + "…"


def _event_key(d: DocInfo) -> tuple:
    """Logs and incident reports about the same failed run share a key."""
    if d.pipeline and d.task and d.date:
        return ("run", d.pipeline, d.task, d.date)
    return ("doc", d.incident_id or d.path)


def _label(d: DocInfo, new_run: bool) -> str:
    """A log's label names its run. When the follow-up names another date (`new_run`), only
    pipeline and task carry over: that run's date and error belong to the old run."""
    if d.source_type == "log":
        if new_run or not d.date:
            return _short(f"{d.pipeline} / {d.task}")
        return _short(f"{d.pipeline} / {d.task} on {d.date}" + (f": {d.error}" if d.error
                                                                else ""))
    return _short(d.title or d.path)


def _turn_subjects(turn: Turn, catalog: LogCatalog, docs: dict[str, DocInfo],
                   new_dates: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(subject labels, ambiguity candidates) for one earlier turn; both empty if the turn
    has no subject."""
    asked = extract_anchors(turn.question, catalog)
    cited = [docs[p] for p in dict.fromkeys(turn.cited_paths) if p in docs]
    events = [d for d in cited if d.source_type in ("incident", "log")]
    # Runbooks and schema docs the answer used add the topic in plain words, unless the
    # follow-up names another date: that is another run, whose failure may be different.
    topics = [] if new_dates else [_short(d.title) for d in cited
                                   if d.source_type in ("runbook", "schema") and d.title]
    if asked.has_subject:
        # What the user asked about wins; keep only cited documents that match it.
        def matches(d: DocInfo) -> bool:
            return ((not asked.incidents or d.incident_id in asked.incidents)
                    and (not asked.tasks or d.task in asked.tasks)
                    and (not asked.pipelines or d.pipeline in asked.pipelines)
                    and (not asked.dates or d.date in asked.dates))
        events = [d for d in events if matches(d)]
        if not events:
            named = [*asked.incidents, *asked.pipelines, *asked.tasks, *asked.dates,
                     *asked.errors, *asked.identifiers]
            if new_dates:
                named = [n for n in named if n not in asked.dates]
            return ([_short(" ".join(named))] + topics)[:MAX_SUBJECTS], []
    by_event: dict[tuple, DocInfo] = {}
    for d in events:
        # Prefer the log for a run (it carries the error line), then the incident report.
        by_event.setdefault(_event_key(d), d)
        if d.source_type == "log":
            by_event[_event_key(d)] = d
    if len(by_event) > 1:
        return [], list(by_event.values())[:MAX_SUBJECTS + 2]
    labels = [_label(d, bool(new_dates)) for d in by_event.values()] + topics
    return labels[:MAX_SUBJECTS], []


def contextualize(question: str, history: Iterable[Turn] | None, catalog: LogCatalog,
                  lookup: DocLookup) -> Contextualization:
    question = question.strip()
    turns = list(history or [])[-MAX_TURNS:]
    own = extract_anchors(question, catalog)
    own_dict = {k: list(v) for k, v in own.__dict__.items() if v}
    if not turns:
        return Contextualization(STANDALONE, "no earlier turns", question, question,
                                 own_anchors=own_dict)
    follow_up, reason = is_follow_up(question, own)
    if not follow_up:
        return Contextualization(STANDALONE, reason, question, question, own_anchors=own_dict)

    paths = [p for t in turns for p in t.cited_paths]
    docs = lookup(list(dict.fromkeys(paths))) if paths else {}
    for back, turn in enumerate(reversed(turns), 1):
        labels, candidates = _turn_subjects(turn, catalog, docs, own.dates)
        if candidates:
            # All cited documents of those failures (a log and its incident report together).
            keys = {_event_key(d) for d in candidates}
            group = tuple(d for p in dict.fromkeys(turn.cited_paths)
                          if (d := docs.get(p)) and d.source_type in ("incident", "log")
                          and _event_key(d) in keys)
            return Contextualization(AMBIGUOUS, f"turn -{back} covered several incidents or "
                                     "runs", question, question,
                                     candidates=tuple(_label(d, False) for d in candidates),
                                     source_turn=back, own_anchors=own_dict,
                                     candidate_docs=group)
        if labels:
            suffix = " (Earlier in this conversation: " + "; ".join(labels) + ")"
            query = (question[:MAX_QUERY_CHARS - len(suffix)] + suffix)[:MAX_QUERY_CHARS]
            return Contextualization(CONTEXTUALIZED, f"subject taken from turn -{back}",
                                     question, query, subjects=tuple(labels),
                                     source_turn=back, own_anchors=own_dict)
    return Contextualization(NO_CONTEXT, "no earlier turn names an incident, pipeline, task or "
                             "cites a document", question, question, own_anchors=own_dict)


# ---- index lookup (the only database access in this module) -----------------------------


def load_document_info(paths: Sequence[str]) -> dict[str, DocInfo]:
    """DocInfo for the given paths that exist in the index; unknown paths are dropped."""
    from incident_copilot.db import connect

    if not paths:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT path, source_type, title, pipeline_name, metadata->>'incident_id', "
            "coalesce(metadata->>'task', metadata->>'failed_task'), metadata->>'date', "
            "metadata->>'error' FROM documents WHERE path = ANY(%s)", (list(paths),)
        ).fetchall()
    return {r[0]: DocInfo(path=r[0], source_type=r[1], title=r[2], pipeline=r[3],
                          incident_id=r[4], task=r[5], date=r[6],
                          error=(r[7] or None) and _short(r[7]))
            for r in rows}
