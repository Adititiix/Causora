"""Clarification for vague questions: "which failure do you mean?" instead of a plain refusal.

A question such as "why did the payment fail?" can be refused by the similarity gate (no
chunk is similar enough) while its evidence points at several different failures. Then /ask
returns the candidates instead (insufficient_reason "ambiguous_query"; never an answer, no
LLM call), and the user picks one. The pick is sent back as a structured `subject`: retrieval
searches with the question plus that subject, the model is told what was picked (as context,
never as evidence), and the normal gate and citation checks decide again.

Candidates (`find_candidates`, pure and deterministic). All of these must hold, else none:
  * the question names no subject of its own (incident id, pipeline, task, error name or
    other snake_case identifier; contextualize.extract_anchors), and the date scope found no
    unknown identifier. A question about something specific that the evidence does not
    support stays a refusal (Q047 "ml_features", Q050 "INC-9999", M014 "INC-2045");
  * the evidence holds at least 2 distinct failures with a chunk at or above
    `min_similarity` (CLARIFY_MIN_SIMILARITY). A failure is a failed run (pipeline, task,
    date): its task log and its incident report are ONE candidate (contextualize's run
    grouping). Runbooks and schema docs describe no particular failure and are not
    candidates;
  * there are at most `max_candidates` of them (CLARIFY_MAX_CANDIDATES); more means the
    question is too vague to offer a short list.
They are ordered by their best similarity (ties by id). Every label and description comes
from indexed metadata (incident id, title, pipeline, task, date, logged error), never from
the model and never invented.

The caller decides when to ask (generation/answer.py: only after a below_min_similarity
refusal, and never for a follow-up resolved from history or a request that carries a subject).

Near misses (`resolve_near_miss`, only for a question the gate refused): the top-k hits can
hold just one of several failures the question's words match ("why did the shipments fail?"
found INC-2045 but not INC-2047), so the hit-based candidates alone would be one and the
question a plain refusal. Then the index's own list of failures (`load_failure_catalog`:
every incident report and failed task log, metadata only) decides, by the same topic words:
  * 2..max_candidates failures match: ask which one (catalog-only ones have no similarity);
  * exactly 1 matches and its own incident report or log is a hit at or above
    `min_similarity` (CLARIFY_MIN_SIMILARITY): answer about it, with the evidence scoped to
    it exactly as if the user had picked it (`scope_hits`); the 0.72 gate itself is not
    changed and the response says which rule accepted the question;
  * otherwise the refusal stands.
Both need a question about one particular failure (asks_about_one_failure), no subject or
unknown identifier of its own, and evidence on the topic at or above `min_similarity`.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from incident_copilot.retrieval.base import Hit
from incident_copilot.retrieval.contextualize import NEW_REFERENT_RE, Anchors, DocInfo, _event_key

CLARIFICATION_TEXT = ("Several different failures match this question. Which one do you mean?")
EVENT_TYPES = ("incident", "log")

# ---- topic: which candidates the question's own words point at ----------------------------

# Words that say nothing about WHICH failure is meant.
_GENERIC = frozenset("""
why what which when where how who did does do is was were are be been the a an of for to in on
at by with about and or it its this that these those i me my we our you your please can could
fail fails failed failing failure failures error errors cause caused causes wrong went go issue
issues problem problems task tasks pipeline pipelines job jobs run runs mean meant one ones
solution solutions fix fixed prevent happen happened next explain tell more again same just
""".split())
_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    return word[:5]


def topic_terms(text: str | None) -> set[str]:
    """Stems of the words that can identify a failure (shipment -> shipm, matching
    shipments_sync). Generic words (why, fail, task, ...) and words under 4 letters are
    ignored."""
    return {_stem(w) for w in _WORD.findall((text or "").lower().replace("_", " "))
            if len(w) >= 4 and w not in _GENERIC}


def _candidate_terms(parts: Iterable[str | None]) -> set[str]:
    return topic_terms(" ".join(p for p in parts if p))


def matching_by_topic(text: str | None, candidates: Sequence, terms_of) -> list:
    """The candidates sharing a topic term with `text`, in their order; [] if none does or
    `text` has no topic term."""
    wanted = topic_terms(text)
    return [c for c in candidates if wanted & terms_of(c)] if wanted else []


# A question about one particular failure ("why did the shipment fail?"), not a how-to or a
# search across failures ("a task fails with ...", "which incident was ...").
_FAILURE_QUESTION = re.compile(
    r"(?i)\b(?:why|fail\w*|error\w*|caus\w*|wrong|broke\w*|crash\w*)\b")


def asks_about_one_failure(question: str) -> bool:
    return bool(_FAILURE_QUESTION.search(question or "")) \
        and not NEW_REFERENT_RE.search(question or "")


@dataclass(frozen=True)
class Candidate:
    id: str  # "run:<pipeline>/<task>/<date>" or "doc:<path>"
    label: str
    description: str
    incident_id: str | None
    pipeline: str | None
    task: str | None
    date: str | None
    paths: tuple[str, ...]
    best_similarity: float | None  # None: offered from an earlier answer, not this search

    @property
    def subject(self) -> dict:
        """What the client sends back to pick this candidate (AskRequest.subject)."""
        if self.pipeline and self.task and self.date:
            out = {"pipeline": self.pipeline, "task": self.task, "date": self.date}
            return out | ({"incident_id": self.incident_id} if self.incident_id else {})
        return {"incident_id": self.incident_id} if self.incident_id else {}

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "incident_id": self.incident_id, "pipeline": self.pipeline, "task": self.task,
                "date": self.date, "paths": list(self.paths),
                "best_similarity": self.best_similarity, "subject": self.subject}


@dataclass
class _Group:
    key: tuple
    docs: dict[str, DocInfo] = field(default_factory=dict)
    best: float = 0.0


def _describe(key: tuple, docs: Iterable[DocInfo]) -> tuple[str, str, str | None]:
    """(label, description, incident id) from the indexed metadata of one failure."""
    docs = sorted(docs, key=lambda d: (d.source_type != "incident", d.path))
    incident = next((d for d in docs if d.source_type == "incident"), None)
    incident_id = incident.incident_id if incident else None
    if key[0] == "run":
        _, pipeline, task, day = key
        label = f"{pipeline} / {task} on {day}" + (f" ({incident_id})" if incident_id else "")
    else:
        label = (incident.title if incident and incident.title else docs[0].path)
    parts = []
    if incident and incident.title:
        parts.append(incident.title)
    error = next((d.error for d in docs if d.source_type == "log" and d.error), None)
    if error:
        parts.append(f"Logged error: {error}")
    return label, " · ".join(parts), incident_id


def _candidate(key: tuple, docs: Iterable[DocInfo], best: float | None) -> Candidate:
    docs = list(docs)
    label, description, incident_id = _describe(key, docs)
    run = key[0] == "run"
    return Candidate(
        id=(f"run:{key[1]}/{key[2]}/{key[3]}" if run else f"doc:{key[1]}"),
        label=label, description=description, incident_id=incident_id,
        pipeline=key[1] if run else None, task=key[2] if run else None,
        date=key[3] if run else None, paths=tuple(sorted(d.path for d in docs)),
        best_similarity=None if best is None else round(best, 6))


def candidate_terms(c: Candidate) -> set[str]:
    return _candidate_terms((c.pipeline, c.task, c.label, c.description))


def find_candidates(hits: Sequence[Hit], docs: Mapping[str, DocInfo], *, own_anchors: Anchors,
                    unknown_identifiers: Sequence[str] = (), min_similarity: float,
                    max_candidates: int, question: str = "") -> list[Candidate]:
    """The failures to offer (module docstring), or [] when the question must stay a refusal.
    `docs` is the index metadata of the hit paths (contextualize.load_document_info). With a
    `question`, failures that share none of its topic words are left out when others do
    ("shipment" keeps the shipments_sync runs and drops an orders_etl one)."""
    if own_anchors.has_subject or unknown_identifiers or max_candidates < 2:
        return []
    groups: dict[tuple, _Group] = {}
    for h in hits:
        d = docs.get(h.path)
        if d is None or d.source_type not in EVENT_TYPES:
            continue
        g = groups.setdefault(_event_key(d), _Group(_event_key(d)))
        g.docs[d.path] = d
        g.best = max(g.best, h.score)
    chosen = [_candidate(g.key, g.docs.values(), g.best) for g in groups.values()
              if g.best >= min_similarity]
    chosen = matching_by_topic(question, chosen, candidate_terms) or chosen
    if not 2 <= len(chosen) <= max_candidates:
        return []
    return sorted(chosen, key=lambda c: (-c.best_similarity, c.id))


def candidates_from_docs(docs: Iterable[DocInfo]) -> list[Candidate]:
    """One candidate per failure among `docs` (an earlier answer's cited incidents and logs),
    in their order, for a follow-up that could mean any of them."""
    groups: dict[tuple, list[DocInfo]] = {}
    for d in docs:
        if d.source_type in EVENT_TYPES:
            groups.setdefault(_event_key(d), []).append(d)
    return [_candidate(k, ds, None) for k, ds in groups.items()]


# ---- a picked candidate (AskRequest.subject) ---------------------------------------------


@dataclass(frozen=True)
class Selection:
    label: str
    retrieval_query: str
    run_key: tuple = ()  # contextualize._event_key of the picked failure
    subject: dict = field(default_factory=dict)  # as validated, for the next turn
    paths: tuple[str, ...] = ()  # the picked failure's indexed incident reports and logs


def describe_selection(question: str, subject: Mapping[str, str | None],
                       docs: Iterable[DocInfo]) -> Selection | None:
    """The picked failure as the index describes it, or None if no indexed incident or
    failed run matches every field of `subject`. `docs` are the indexed incidents and logs
    that may match (load_subject_documents)."""
    docs = list(docs)
    want = {k: v for k, v in subject.items() if v}

    def matches(d: DocInfo) -> bool:
        return all({"incident_id": d.incident_id, "pipeline": d.pipeline, "task": d.task,
                    "date": d.date}.get(k) == v for k, v in want.items())

    hit = next((d for d in sorted(docs, key=lambda d: d.path) if matches(d)), None)
    if hit is None or not want:
        return None
    key = _event_key(hit)
    group = [d for d in docs if _event_key(d) == key]
    label, _, incident_id = _describe(key, group)
    subject = ({"pipeline": key[1], "task": key[2], "date": key[3]} if key[0] == "run" else {})
    if incident_id:
        subject["incident_id"] = incident_id
    return Selection(label=label, retrieval_query=f"{question} (Selected: {label})",
                     run_key=key, subject=subject,
                     paths=tuple(sorted(d.path for d in group)))


def scope_hits(hits: Sequence[Hit], docs: Mapping[str, DocInfo], run_key: tuple) -> list[Hit]:
    """The evidence for one picked failure: drops the incident reports and task logs of OTHER
    failures, and the chunks added because of them (a runbook linked from another run's error
    line, that runbook's Resolution companion). Everything else keeps its order: runbooks and
    schema docs found by the question itself, and the picked failure's own documents."""
    other = {p for p, d in docs.items()
             if d.source_type in EVENT_TYPES and _event_key(d) != run_key}
    def companion_of(h: Hit) -> str | None:
        return (h.metadata.get("provenance") or {}).get("companion_of")

    kept = [h for h in hits if h.path not in other and h.linked_from not in other]
    # A runbook that was there only through a dropped link takes its companions along (a
    # Resolution companion is a chunk of that runbook, so look at its other chunks).
    still_there = {h.path for h in kept if not companion_of(h)}
    return [h for h in kept if not companion_of(h) or companion_of(h) in still_there]


# ---- index access (the only database reads in this module) -------------------------------


def load_subject_documents(subject: Mapping[str, str | None]) -> list[DocInfo]:
    """Indexed incidents and logs that share the subject's incident id or run."""
    from incident_copilot.db import connect
    from incident_copilot.retrieval.contextualize import _short

    with connect() as conn:
        rows = conn.execute(
            "SELECT path, source_type, title, pipeline_name, metadata->>'incident_id', "
            "coalesce(metadata->>'task', metadata->>'failed_task'), metadata->>'date', "
            "metadata->>'error' FROM documents WHERE source_type IN ('incident', 'log') "
            "AND ((%(incident)s::text IS NOT NULL AND metadata->>'incident_id' = %(incident)s) "
            "OR (pipeline_name = %(pipeline)s::text AND metadata->>'date' = %(date)s::text "
            "AND coalesce(metadata->>'task', metadata->>'failed_task') = %(task)s::text))",
            {"incident": subject.get("incident_id"), "pipeline": subject.get("pipeline"),
             "task": subject.get("task"), "date": subject.get("date")}).fetchall()
    return [DocInfo(path=r[0], source_type=r[1], title=r[2], pipeline=r[3], incident_id=r[4],
                    task=r[5], date=r[6], error=(r[7] or None) and _short(r[7])) for r in rows]


Clarifier = Callable[[str, Sequence[Hit]], list[Candidate]]


def make_clarifier(retriever, *, min_similarity: float, max_candidates: int) -> Clarifier:
    """The clarifier /ask and the answer evaluation use: index metadata for the hits, the
    question's own anchors, and the date scope's unknown identifiers (read after retrieve)."""
    def clarify(question: str, hits: Sequence[Hit]) -> list[Candidate]:
        from incident_copilot.retrieval.contextualize import extract_anchors, load_document_info
        from incident_copilot.retrieval.date_aware import load_log_catalog

        scope = getattr(retriever, "last_scope", None)
        return find_candidates(
            hits, load_document_info(sorted({h.path for h in hits})),
            own_anchors=extract_anchors(question, load_log_catalog()),
            unknown_identifiers=tuple(getattr(scope, "unknown_identifiers", None) or ()),
            min_similarity=min_similarity, max_candidates=max_candidates, question=question)
    return clarify


# ---- near misses: the index's own list of failures (module docstring) --------------------

FAILURE_CATALOG_TTL_SECONDS = 60.0
_failure_cache: dict[str, tuple[float, tuple[DocInfo, ...]]] = {}


def load_failure_catalog() -> tuple[DocInfo, ...]:
    """Every indexed incident report and failed task log (one with a logged error), metadata
    only, cached briefly per database like the log catalogue (retrieval/date_aware.py)."""
    import time

    from incident_copilot.config import get_settings
    from incident_copilot.db import connect
    from incident_copilot.retrieval.contextualize import _short

    url = get_settings().database_url
    cached = _failure_cache.get(url)
    if cached and time.monotonic() - cached[0] < FAILURE_CATALOG_TTL_SECONDS:
        return cached[1]
    with connect() as conn:
        rows = conn.execute(
            "SELECT path, source_type, title, pipeline_name, metadata->>'incident_id', "
            "coalesce(metadata->>'task', metadata->>'failed_task'), metadata->>'date', "
            "metadata->>'error' FROM documents WHERE source_type = 'incident' "
            "OR (source_type = 'log' AND coalesce(metadata->>'error', '') <> '')").fetchall()
    docs = tuple(DocInfo(path=r[0], source_type=r[1], title=r[2], pipeline=r[3],
                         incident_id=r[4], task=r[5], date=r[6],
                         error=(r[7] or None) and _short(r[7])) for r in rows)
    _failure_cache[url] = (time.monotonic(), docs)
    return docs


def clear_failure_catalog_cache() -> None:
    _failure_cache.clear()


@dataclass(frozen=True)
class NearMiss:
    """What to do with a question the gate refused: ask (`candidates`), answer about the one
    failure it matches (`resolved`, with `hits` scoped to it), or neither (refuse)."""

    candidates: tuple[Candidate, ...] = ()
    resolved: Candidate | None = None
    hits: tuple[Hit, ...] = ()


def _on_topic(hits: Sequence[Hit], docs: Mapping[str, DocInfo], wanted: set[str],
              min_similarity: float) -> bool:
    """Some hit at or above `min_similarity` is about the question's topic (its pipeline,
    task or title shares a topic word), so the evidence is not merely generic."""
    for h in hits:
        if h.score < min_similarity:
            continue
        d = docs.get(h.path)
        parts = (d.pipeline, d.task, d.title) if d else ()
        if wanted & _candidate_terms((*parts, h.pipeline_name)):
            return True
    return False


def resolve_near_miss(question: str, hits: Sequence[Hit], docs: Mapping[str, DocInfo],
                      catalog: Iterable[DocInfo], *, own_anchors: Anchors,
                      unknown_identifiers: Sequence[str] = (), min_similarity: float,
                      max_candidates: int) -> NearMiss:
    """The near-miss decision (module docstring). Pure: `docs` is the index metadata of the
    hit paths, `catalog` every indexed failure (load_failure_catalog)."""
    wanted = topic_terms(question)
    if (own_anchors.has_subject or unknown_identifiers or max_candidates < 2 or not wanted
            or not asks_about_one_failure(question)
            or not _on_topic(hits, docs, wanted, min_similarity)):
        return NearMiss()
    groups: dict[tuple, list[DocInfo]] = {}
    for d in catalog:
        groups.setdefault(_event_key(d), []).append(d)
    best: dict[tuple, float] = {}
    for h in hits:
        d = docs.get(h.path)
        if d is not None and d.source_type in EVENT_TYPES:
            best[_event_key(d)] = max(best.get(_event_key(d), 0.0), h.score)
    # A failure matches by what it IS (pipeline, task, incident title), not by the tables its
    # logged error mentions: build_customer_profile's freshness error names loyalty_members,
    # but "why did the loyalty join fail?" is about join_loyalty.
    matching = [(key, _candidate(key, ds, best.get(key))) for key, ds in groups.items()
                if wanted & _candidate_terms(p for d in ds for p in (d.pipeline, d.task,
                                                                     d.title))]
    if 2 <= len(matching) <= max_candidates:
        # Found by this search first (most similar first), then the others by id.
        return NearMiss(candidates=tuple(sorted(
            (c for _, c in matching),
            key=lambda c: (c.best_similarity is None, -(c.best_similarity or 0.0), c.id))))
    if len(matching) == 1 and (matching[0][1].best_similarity or 0.0) >= min_similarity:
        key, only = matching[0]
        return NearMiss(resolved=only, hits=tuple(scope_hits(hits, docs, key)))
    return NearMiss()


NearMissResolver = Callable[[str, Sequence[Hit]], NearMiss]


def make_near_miss_resolver(retriever, *, min_similarity: float,
                            max_candidates: int) -> NearMissResolver:
    """The resolver /ask and the answer evaluation use (like make_clarifier)."""
    def resolve(question: str, hits: Sequence[Hit]) -> NearMiss:
        from incident_copilot.retrieval.contextualize import extract_anchors, load_document_info
        from incident_copilot.retrieval.date_aware import load_log_catalog

        scope = getattr(retriever, "last_scope", None)
        return resolve_near_miss(
            question, hits, load_document_info(sorted({h.path for h in hits})),
            load_failure_catalog(),
            own_anchors=extract_anchors(question, load_log_catalog()),
            unknown_identifiers=tuple(getattr(scope, "unknown_identifiers", None) or ()),
            min_similarity=min_similarity, max_candidates=max_candidates)
    return resolve


# ---- the selected failure's own remediation sections -------------------------------------

# An incident report's documented recovery, then its documented prevention (the first
# heading of each group that the report has).
REMEDIATION_SECTIONS = (("resolution and verification", "resolution"),
                        ("prevention", "follow-ups"))
REMEDIATION_ORIGIN = "subject_remediation"


def pick_remediation_rows(rows: Sequence[tuple], shown: set[str]) -> list[tuple]:
    """From one incident report's chunk rows (evidence.load_document_chunks, document order):
    the first chunk of each REMEDIATION_SECTIONS group that is not already shown."""
    out = []
    for group in REMEDIATION_SECTIONS:
        for wanted in group:
            row = next((r for r in rows if (r[4] or "").lower() == wanted), None)
            if row is not None:
                if row[0] not in shown:
                    out.append(row)
                break
    return out


def remediation_hits(subject_paths: Sequence[str], hits: Sequence[Hit], question_vec,
                     max_added: int = 2) -> list[Hit]:
    """The selected failure's own documented resolution and prevention (from its incident
    reports among `subject_paths`), as extra evidence added after the gate, so they never
    change it. Without them the only remediation the model may see is a generic runbook's
    (INC-2046 got "fix or rerun the export" instead of its own documented recovery)."""
    from incident_copilot.retrieval.evidence import (
        _cosine,
        _related_hit,
        load_document_chunks,
    )

    if not subject_paths or max_added <= 0 or question_vec is None:
        return []
    shown = {h.chunk_id for h in hits}
    out: list[Hit] = []
    rows_by_path = load_document_chunks(sorted(set(subject_paths)))
    for path in sorted(rows_by_path):
        rows = rows_by_path[path]
        if not rows or rows[0][3] != "incident":  # logs and others document no fix
            continue
        for row in pick_remediation_rows(rows, shown):
            hit = _related_hit(row, round(_cosine(question_vec, row[12]), 6), path)
            out.append(replace(hit, origin=REMEDIATION_ORIGIN, metadata={
                **hit.metadata, "provenance": {"remediation_of": path}}))
            shown.add(row[0])
    return out[:max_added]
