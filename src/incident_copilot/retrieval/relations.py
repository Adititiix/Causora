"""Related documents: follow explicit references between indexed documents.

Why (measured on eval/questions.multidoc.jsonl, retrieval run 20261031T175307Z): questions
that need several documents got every one of them into the top 8 for 6 of 12. Similarity
alone missed documents that the question or the retrieved evidence names outright: an
incident id in the question (rank 10 and 28 by similarity), a table named in the question
or in a retrieved log ("Loaded ... rows from shipments_raw"; not in the top 100).

What counts as a reference (exact identifiers only, never ordinary words):
  * an incident id (INC-2047) or runbook id (RB-006) that some indexed document carries;
  * the exact name of a table that has a schema doc (shipments_raw);
  * in the question only: the name of a pipeline that has a pipeline definition doc;
  * an incident report and a task log of the SAME date, where the report names both the
    log's pipeline and its task (its "Failed task", or an upstream "(task ...)").
A document's references are read from its full indexed text and metadata.

Expansion is one level deep: references from the question, then from the first
`max_seed_documents` documents of the evidence, never from a document that was itself added.
Candidates the request's filters would exclude are skipped, as are documents already in the
evidence. They are ordered by origin (the question, then the evidence documents in rank
order) and then by the similarity of their best chunk to the question, and at most
`max_related` are added, one chunk each. A related chunk only takes a free slot or the place
of an extra chunk of a document that keeps another one (evidence.select_evidence), so it never
removes a document from the evidence. The pure functions here decide; evidence.py loads the
rows and places the chunks.
"""

import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.retrieval.base import Filters

INCIDENT_ID_RE = re.compile(r"\bINC-\d{3,6}\b")
RUNBOOK_ID_RE = re.compile(r"\bRB-\d{3,6}\b")
# snake_case identifiers with at least one underscore: table, pipeline and task names.
IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9_])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![A-Za-z0-9_])")
TABLE_TITLE_RE = re.compile(r"^Table:\s*(\S+)")
PIPELINE_TITLE_RE = re.compile(r"^Pipeline:\s*(\S+)")

CATALOG_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, "RelationCatalog"]] = {}

_CATALOG_SQL = """
SELECT d.path, d.source_type, d.title, d.pipeline_name, d.metadata,
       string_agg(c.content, E'\\n' ORDER BY c.chunk_index)
FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY d.id
ORDER BY d.path
"""


@dataclass(frozen=True)
class DocumentInfo:
    path: str
    source_type: str
    pipeline: str | None
    metadata: dict
    identifiers: frozenset[str]  # snake_case identifiers in the text
    incident_ids: tuple[str, ...]
    runbook_ids: tuple[str, ...]


@dataclass
class RelationCatalog:
    documents: dict[str, DocumentInfo] = field(default_factory=dict)
    incidents: dict[str, str] = field(default_factory=dict)      # INC-2045 -> path
    runbooks: dict[str, str] = field(default_factory=dict)       # RB-006 -> path
    tables: dict[str, str] = field(default_factory=dict)         # table -> schema doc path
    pipeline_docs: dict[str, str] = field(default_factory=dict)  # pipeline -> definition path
    logs: list[tuple[str, str, str, str]] = field(default_factory=list)  # path, pipe, task, date


def _ordered_unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def build_catalog(rows: Iterable[tuple]) -> RelationCatalog:
    """Rows: (path, source_type, title, pipeline_name, metadata, text). Malformed rows and
    missing fields are skipped, never guessed."""
    cat = RelationCatalog()
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 6 or not row[0] or not row[1]:
            continue
        path, source_type, title, pipeline, meta, text = row
        meta = meta if isinstance(meta, dict) else {}
        text = text or ""
        title = title or ""
        cat.documents[path] = DocumentInfo(
            path=path, source_type=source_type, pipeline=pipeline, metadata=meta,
            identifiers=frozenset(IDENTIFIER_RE.findall(text)),
            incident_ids=_ordered_unique(INCIDENT_ID_RE.findall(text)),
            runbook_ids=_ordered_unique(RUNBOOK_ID_RE.findall(text)))
        if source_type == "incident" and (m := INCIDENT_ID_RE.match(title)):
            cat.incidents.setdefault(m.group(0), path)
        elif source_type == "runbook" and isinstance(meta.get("runbook_id"), str):
            cat.runbooks.setdefault(meta["runbook_id"], path)
        elif source_type == "schema":
            if m := TABLE_TITLE_RE.match(title):
                cat.tables.setdefault(m.group(1), path)
            elif m := PIPELINE_TITLE_RE.match(title):
                cat.pipeline_docs.setdefault(m.group(1), path)
        elif source_type == "log":
            task, day = meta.get("task"), meta.get("date")
            if pipeline and isinstance(task, str) and isinstance(day, str):
                cat.logs.append((path, pipeline, task, day))
    return cat


def _id_targets(incident_ids: Iterable[str], runbook_ids: Iterable[str],
                cat: RelationCatalog) -> list[str]:
    return [cat.incidents[i] for i in incident_ids if i in cat.incidents] + \
           [cat.runbooks[i] for i in runbook_ids if i in cat.runbooks]


def question_references(question: str, cat: RelationCatalog) -> list[str]:
    """Documents the question names: incident and runbook ids, tables, pipeline definitions."""
    words = IDENTIFIER_RE.findall(question or "")
    out = _id_targets(INCIDENT_ID_RE.findall(question or ""),
                      RUNBOOK_ID_RE.findall(question or ""), cat)
    out += [cat.tables[w] for w in words if w in cat.tables]
    out += [cat.pipeline_docs[w] for w in words if w in cat.pipeline_docs]
    return list(_ordered_unique(out))


def document_references(path: str, cat: RelationCatalog) -> list[str]:
    """Documents that `path` names explicitly (module docstring), itself excluded."""
    doc = cat.documents.get(path)
    if doc is None:
        return []
    out = _id_targets(doc.incident_ids, doc.runbook_ids, cat)
    out += [cat.tables[w] for w in sorted(doc.identifiers) if w in cat.tables]
    day = doc.metadata.get("date")
    if doc.source_type == "incident" and isinstance(day, str):
        out += [p for p, pipe, task, d in cat.logs
                if d == day and pipe in doc.identifiers and task in doc.identifiers]
    elif doc.source_type == "log" and isinstance(day, str):
        task = doc.metadata.get("task")
        out += [p for p, other in cat.documents.items()
                if other.source_type == "incident" and other.metadata.get("date") == day
                and doc.pipeline in other.identifiers and task in other.identifiers]
    return [p for p in _ordered_unique(out) if p != path]


def allowed_by(filters: Filters, doc: DocumentInfo) -> bool:
    """The vector search's WHERE clause (vector.py), applied to a whole document."""
    if filters.source_types and doc.source_type not in filters.source_types:
        return False
    if filters.pipeline and doc.pipeline != filters.pipeline \
            and filters.pipeline not in (doc.metadata.get("pipelines") or []):
        return False
    if doc.source_type == "log":
        if filters.log_date and doc.metadata.get("date") != filters.log_date:
            return False
        if filters.log_pipeline and doc.pipeline != filters.log_pipeline:
            return False
        if filters.log_task and doc.metadata.get("task") != filters.log_task:
            return False
    return True


@dataclass(frozen=True)
class Candidate:
    path: str
    source: str | None  # the evidence document that names it; None = the question
    origin_rank: int = 0  # 0 = the question, n = the n-th evidence document


def related_candidates(question: str, evidence_paths: Sequence[str], cat: RelationCatalog,
                       filters: Filters, max_seed_documents: int) -> list[Candidate]:
    """Every allowed, not-yet-present document named by the question or by one of the first
    `max_seed_documents` evidence documents, question first, each path once."""
    present = set(evidence_paths)
    seeds = list(_ordered_unique(evidence_paths))[: max(0, max_seed_documents)]
    found: list[Candidate] = [Candidate(p, None) for p in question_references(question, cat)]
    for n, s in enumerate(seeds, start=1):
        found += [Candidate(p, s, n) for p in document_references(s, cat)]
    out: list[Candidate] = []
    seen: set[str] = set()
    for c in found:
        doc = cat.documents.get(c.path)
        if c.path in seen or c.path in present or doc is None or not allowed_by(filters, doc):
            continue
        seen.add(c.path)
        out.append(c)
    return out


def rank_candidates(candidates: Sequence[Candidate],
                    similarity: Callable[[str], float | None],
                    max_related: int) -> list[Candidate]:
    """Question references first, then references of the top evidence document, then of the
    next; within each, by the similarity of the document's best usable chunk to the question
    (ties by path). Candidates without a usable chunk (similarity None) are dropped."""
    scored = [(c, similarity(c.path)) for c in candidates]
    scored = [(c, s) for c, s in scored if s is not None]
    scored.sort(key=lambda cs: (cs[0].origin_rank, -cs[1], cs[0].path))
    return [c for c, _ in scored[: max(0, max_related)]]


def load_relation_catalog() -> RelationCatalog:
    """The catalogue of the current index, cached briefly per database (like evidence.py's
    section caches)."""
    url = get_settings().database_url
    hit = _cache.get(url)
    if hit and time.monotonic() - hit[0] < CATALOG_TTL_SECONDS:
        return hit[1]
    with connect() as conn:
        catalog = build_catalog(conn.execute(_CATALOG_SQL).fetchall())
    _cache[url] = (time.monotonic(), catalog)
    return catalog


def clear_relation_cache() -> None:
    _cache.clear()
