"""Evidence selection: candidate retrieval, then a bounded choice of the k chunks to show.

Why (measured on the 50 seed questions, retrieval run 20261022T053248Z and answer runs
20261022T214719Z / 20261023T113853Z):
  * ErrorLinkRetriever makes room for a linked runbook by cutting the lowest-ranked hits, and
    that can be the very log that produced the link (Q015: the target log was rank 8).
  * The linked runbook chunk is the one nearest the exception line, i.e. its Symptoms or
    Overview section; the Resolution section ranks 9th-189th by question similarity, so it
    was in the top 8 for 5 of the 36 questions that expect a runbook.
  * Several chunks of the same few documents can fill every slot (Q016/Q018: 8 slots, 3
    incident reports, the target log at rank 9-10).

How (deterministic; no LLM, no schema change):
  1. Candidates: vector search to depth `candidate_factor * k` with the request's filters.
  2. Diversity: at most `max_chunks_per_document` chunks per document, in rank order; if
     that leaves fewer than k, the skipped chunks fill the gap (still in rank order). With
     `cap_only_when_dated`, only for questions the date filter scoped to one dated log.
  3. Error links, exactly as ErrorLinkRetriever finds them.
  4. Resolution companions: for at most `max_companions` runbooks in the selection, add the
     runbook's Resolution section, only if the question is about a failure or a fix
     (`asks_about_failure`) and the runbook was linked from a failed log or matched the
     question at >= `companion_min_similarity` (default MIN_SIMILARITY).
  4b. Log error contexts (`log_context_companions`, off by default): when the date filter
     scoped the request to one dated log, a failed log shown only through its "run summary"
     chunk (which elides the traceback) also gets its "error context" chunk, for at most
     `max_links` logs (Q016: the traceback and the new worker-image tag were elided).
  4c. Related documents (`related_documents`, off by default): documents the question or
     the first evidence documents name explicitly (an incident or runbook id, a table, a
     pipeline definition, the log of an incident; retrieval/relations.py) and that the
     search missed. At most `max_related`, one level deep, one chunk each (the document's
     chunk most similar to the question), and only where the request's filters allow them.
     With the same switch, when the question asks about a cause (`CAUSE_INTENT_RE`) and the
     top hit is an incident report, that report's cause section ("Inferred cause" or "Root
     cause") comes first: it states the cause and how certain the report is of it.
  5. Trim to k: the top question hit is always kept, then protected chunks (links,
     companions and, with `protect_anchors`, the logs they were linked from), then the rest
     by rank. Related chunks then take any free slot, or replace the lowest-ranked
     unprotected chunk (never a runbook's Resolution) of a document that keeps another
     chunk, so they never remove a document. Chunks keep their order; a link follows its
     log, a companion its runbook, and related chunks come last.

The top question hit is never dropped, every added chunk is scored against the question, and
a companion more similar than the top hit is left out (possible only if the approximate
index missed it), so the best similarity, and with it the MIN_SIMILARITY refusal gate, is the
same as with the candidate search alone. `Hit.metadata["provenance"]` records why each chunk
was chosen; it is internal and not part of the /ask response.
"""

import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.embeddings import Embedder
from incident_copilot.retrieval.base import Filters, Hit
from incident_copilot.retrieval.error_link import ErrorLinkRetriever
from incident_copilot.retrieval.relations import (
    clear_relation_cache,
    load_relation_catalog,
    rank_candidates,
    related_candidates,
)

# Runbook headings that say how to fix the problem (compared case-insensitively).
RESOLUTION_SECTIONS = ("resolution", "remediation", "fix")
# The question is about something going wrong, or asks how to fix it.
FAILURE_INTENT_RE = re.compile(
    r"(?i)\b(?:fail\w*|error\w*|exception\w*|crash\w*|killed|timed\s+out|timeout\w*"
    r"|broken|fix\w*|resolv\w*|recover\w*|remediat\w*|what\s+should\s+i\s+do)\b")
COMPANION_ORIGIN = "resolution_companion"
# Incident-report headings that state the cause (generated reports: "Root cause"; reports
# that separate observation from inference: "Inferred cause", with its confidence).
CAUSE_SECTIONS = ("inferred cause", "root cause")
# The question asks what caused something, or how sure that cause is.
CAUSE_INTENT_RE = re.compile(
    r"(?i)\b(?:caus\w*|why|reason|how\s+(?:sure|certain|confident)|certain\w*|confiden\w*)\b")
CAUSE_ORIGIN = "cause_section"
# The log chunker's section with the failing entry and its full traceback
# (chunking/logs.py); the "run summary" chunk elides exactly those entries.
ERROR_CONTEXT_SECTION = "error context"
LOG_CONTEXT_ORIGIN = "error_context_companion"
RELATED_ORIGIN = "related_document"

SECTION_CACHE_TTL_SECONDS = 60.0
_section_cache: dict[tuple, tuple[float, dict[str, tuple]]] = {}

_SECTION_SQL = """
SELECT c.id::text, d.id::text, d.path, d.source_type, c.section, c.ts_start, c.ts_end,
       c.content, c.chunk_index, c.content_hash, d.pipeline_name, c.metadata, c.embedding
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.source_type = %(source_type)s AND c.embedding IS NOT NULL
  AND lower(c.section) = ANY(%(sections)s::text[])
ORDER BY d.path, c.chunk_index
"""

# Same columns as _SECTION_SQL, for every chunk of the given documents.
_DOCUMENT_CHUNKS_SQL = """
SELECT c.id::text, d.id::text, d.path, d.source_type, c.section, c.ts_start, c.ts_end,
       c.content, c.chunk_index, c.content_hash, d.pipeline_name, c.metadata, c.embedding
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.path = ANY(%(paths)s::text[]) AND c.embedding IS NOT NULL
ORDER BY d.path, c.chunk_index
"""


def asks_about_failure(question: str) -> bool:
    return FAILURE_INTENT_RE.search(question) is not None


def load_sections(source_type: str, sections: Sequence[str]) -> dict[str, tuple]:
    """Document path -> its first chunk row (with embedding) whose section is one of
    `sections`, cached briefly per database like the log catalogue in date_aware.py."""
    key = (get_settings().database_url, source_type, tuple(sections))
    hit = _section_cache.get(key)
    if hit and time.monotonic() - hit[0] < SECTION_CACHE_TTL_SECONDS:
        return hit[1]
    with connect() as conn:
        rows = conn.execute(_SECTION_SQL, {"source_type": source_type,
                                           "sections": list(sections)}).fetchall()
    by_path: dict[str, tuple] = {}
    for row in rows:
        by_path.setdefault(row[2], row)
    _section_cache[key] = (time.monotonic(), by_path)
    return by_path


def load_resolution_sections() -> dict[str, tuple]:
    return load_sections("runbook", RESOLUTION_SECTIONS)


def load_log_error_contexts() -> dict[str, tuple]:
    return load_sections("log", (ERROR_CONTEXT_SECTION,))


def load_document_chunks(paths: Sequence[str]) -> dict[str, list[tuple]]:
    """Document path -> its chunk rows (with embeddings), for the related-document step."""
    if not paths:
        return {}
    with connect() as conn:
        rows = conn.execute(_DOCUMENT_CHUNKS_SQL, {"paths": list(paths)}).fetchall()
    out: dict[str, list[tuple]] = {}
    for row in rows:
        out.setdefault(row[2], []).append(row)
    return out


def clear_resolution_cache() -> None:
    """Clears every cached lookup (resolutions, log error contexts, document relations)."""
    _section_cache.clear()
    clear_relation_cache()


def _as_array(v) -> np.ndarray:
    # pgvector returns its own Vector type for `embedding`; embedders return numpy arrays.
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


def _cosine(a, b) -> float:
    a, b = _as_array(a), _as_array(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom else 0.0


def _with_provenance(hit: Hit, **info) -> Hit:
    prov = {**hit.metadata.get("provenance", {}), **info}
    return replace(hit, metadata={**hit.metadata, "provenance": prov})


# ---- pure selection steps (unit-tested without a database) ---------------------------------


def diversify(candidates: Sequence[Hit], k: int, max_per_document: int | None) -> list[Hit]:
    """The first k candidates with at most `max_per_document` chunks per document; when that
    leaves fewer than k, skipped chunks fill the gap. Rank order is kept. None = no cap."""
    if not max_per_document:
        return list(candidates[:k])
    counts: Counter = Counter()
    chosen: list[int] = []
    skipped: list[int] = []
    for i, h in enumerate(candidates):
        if len(chosen) >= k:
            break
        if counts[h.document_id] < max_per_document:
            counts[h.document_id] += 1
            chosen.append(i)
        else:
            skipped.append(i)
    # Too few distinct documents: show more of the same ones rather than fewer chunks.
    chosen += skipped[: max(0, k - len(chosen))]
    return [candidates[i] for i in sorted(chosen)]


def companion_targets(ordered: Sequence[Hit], question: str, min_similarity: float,
                      max_companions: int) -> list[str]:
    """Runbook paths that get a Resolution companion, in evidence order."""
    if max_companions <= 0 or not asks_about_failure(question):
        return []
    targets: list[str] = []
    for h in ordered:
        if h.source_type != "runbook" or h.path in targets:
            continue
        if h.origin == "error_link" or h.score >= min_similarity:
            targets.append(h.path)
            if len(targets) >= max_companions:
                break
    return targets


def select_evidence(primary: Sequence[Hit], links: dict[str, Hit], companions: dict[str, Hit],
                    k: int, protect_anchors: bool,
                    protected: frozenset[str] = frozenset(),
                    related: Sequence[Hit] = ()) -> list[Hit]:
    """Merge and trim to k (see the module docstring, step 5). `links` maps an anchor log path
    to its linked runbook hit; `companions` maps a document path to the companion chunk that
    follows its first chunk. `protected` chunk ids in `primary` are kept like companions.

    `related` chunks, in priority order, come after everything else. One is kept only if
    there is a free slot, or by replacing the lowest-ranked unprotected chunk of a document
    that still keeps another chunk (never a runbook's Resolution section): a related chunk
    never removes a document and never moves the chunks ranked above it."""
    # (hit, tier): 0 always kept, 1 protected, 2 rest, 3 related (placed after trimming)
    ordered: list[tuple[Hit, int]] = []
    seen: set[str] = set()
    linked: set[str] = set()
    accompanied: set[str] = set()

    def add(hit: Hit, tier: int) -> None:
        if hit.chunk_id in seen:
            return
        seen.add(hit.chunk_id)
        ordered.append((hit, tier))
        companion = companions.get(hit.path)
        if companion is not None and hit.path not in accompanied:
            accompanied.add(hit.path)
            add(companion, 1)

    for i, h in enumerate(primary):
        is_anchor = h.path in links and h.path not in linked
        keep_it = (protect_anchors and is_anchor) or h.chunk_id in protected
        add(h, 0 if i == 0 else 1 if keep_it else 2)
        if is_anchor:
            linked.add(h.path)
            add(links[h.path], 1)

    keep = set(sorted(range(len(ordered)), key=lambda i: (ordered[i][1], i))[:max(0, k)])
    per_document = Counter(ordered[i][0].document_id for i in keep)
    for rel in related:
        if rel.chunk_id in seen:
            continue
        seen.add(rel.chunk_id)
        ordered.append((rel, 3))
        i = len(ordered) - 1
        if len(keep) >= max(0, k):
            spare = [j for j in keep if ordered[j][1] == 2
                     and per_document[ordered[j][0].document_id] > 1
                     and not _is_resolution(ordered[j][0])]
            if not spare:
                break
            j = max(spare)  # the lowest-ranked extra chunk
            keep.discard(j)
            per_document[ordered[j][0].document_id] -= 1
        keep.add(i)
        per_document[ordered[i][0].document_id] += 1
    return [replace(ordered[i][0], rank=r) for r, i in enumerate(sorted(keep), start=1)]


# ---- the retriever ------------------------------------------------------------------------


class EvidenceRetriever(ErrorLinkRetriever):
    """ErrorLinkRetriever with candidate retrieval and evidence selection (module docstring).
    Each step can be switched off, so the steps can be measured separately."""

    def __init__(self, embedder: Embedder, *, name: str = "vector+error_link+evidence",
                 protect_anchors: bool = True, resolution_companions: bool = True,
                 max_chunks_per_document: int | None = 2, candidate_factor: int = 3,
                 max_links: int = 2, max_companions: int = 2,
                 companion_min_similarity: float | None = None,
                 log_context_companions: bool = False, cap_only_when_dated: bool = False,
                 related_documents: bool = False, max_related: int = 2,
                 max_seed_documents: int = 3):
        super().__init__(embedder, max_links=max_links)
        self.name = name
        self.related_documents = related_documents
        self.max_related = max_related
        self.max_seed_documents = max_seed_documents
        self.protect_anchors = protect_anchors
        self.resolution_companions = resolution_companions
        self.log_context_companions = log_context_companions
        self.cap_only_when_dated = cap_only_when_dated
        self.max_chunks_per_document = max_chunks_per_document
        self.candidate_factor = max(1, candidate_factor)
        self.max_companions = max_companions
        self.companion_min_similarity = (get_settings().min_similarity
                                         if companion_min_similarity is None
                                         else companion_min_similarity)

    def retrieve(self, question: str, k: int, filters: Filters | None = None) -> list[Hit]:
        filters = filters or Filters()
        question_vec = self.embedder.embed_query(question)
        self.last_question_vec = question_vec  # reused after retrieval (no second embedding)
        # With cap_only_when_dated, the cap applies only to questions scoped to one dated log
        # (other documents are context there); a question about one incident, runbook or
        # schema may need several of its sections (Q037: the duration is in the Overview).
        cap = (self.max_chunks_per_document
               if not self.cap_only_when_dated or filters.log_date else None)
        depth = k * self.candidate_factor if cap else k
        candidates = self.vector.search(question_vec, depth, filters)
        # Equal distances are ordered by chunk id, so identical inputs give identical output.
        candidates.sort(key=lambda h: (h.scores["vector_distance"], h.chunk_id))
        candidates = [_with_provenance(h, vector_rank=i)
                      for i, h in enumerate(candidates, start=1)]
        primary = diversify(candidates, k, cap)
        related = (self._related(question, question_vec, primary, filters)
                   if self.related_documents and primary else [])

        if filters.source_types and "runbook" not in filters.source_types:
            if related:
                return select_evidence(primary, {}, {}, k, False, related=related)
            return [replace(h, rank=i) for i, h in enumerate(primary, start=1)]
        links = {a: _with_provenance(h, anchor=a)
                 for a, h in self.find_links(primary, question_vec, filters).items()}
        companions = (self._companions(primary, links, question, question_vec)
                      if self.resolution_companions else {})
        # Only when the question is scoped to one dated log (set by DateAwareRetriever):
        # then the failed logs shown are the incident being asked about, not examples.
        protected: frozenset[str] = frozenset()
        if self.log_context_companions and filters.log_date:
            contexts, protected = self._log_contexts(primary, question_vec)
            companions |= contexts
        return select_evidence(primary, links, companions, k, self.protect_anchors, protected,
                               related)

    def _related(self, question: str, question_vec, primary: Sequence[Hit],
                 filters: Filters) -> list[Hit]:
        """One chunk per related document, in priority order (relations.rank_candidates). Each
        is the document's chunk most similar to the question among those not above the top
        hit, so the refusal gate is unaffected."""
        best = max(h.score for h in primary)
        cause = self._cause_section(question, question_vec, primary, best)
        candidates = related_candidates(question, [h.path for h in primary],
                                        load_relation_catalog(), filters,
                                        self.max_seed_documents)
        if not candidates:
            return cause
        rows = load_document_chunks([c.path for c in candidates])
        chosen: dict[str, tuple[float, tuple]] = {}
        for c in candidates:
            scored = [(round(_cosine(question_vec, r[12]), 6), r) for r in rows.get(c.path, [])]
            usable = [(s, r) for s, r in scored if s <= best]
            if usable:  # most similar chunk; equal scores -> the earlier chunk
                chosen[c.path] = max(usable, key=lambda sr: (sr[0], -sr[1][8]))
        return cause + [_related_hit(chosen[c.path][1], chosen[c.path][0], c.source)
                        for c in rank_candidates(candidates,
                                                 lambda p: chosen[p][0] if p in chosen else None,
                                                 self.max_related)]

    @staticmethod
    def _cause_section(question: str, question_vec, primary: Sequence[Hit],
                       best: float) -> list[Hit]:
        """When the question asks about a cause and the top hit is an incident report, that
        report's cause section (the first of CAUSE_SECTIONS it has), unless already shown."""
        top = primary[0]
        if top.source_type != "incident" or not CAUSE_INTENT_RE.search(question):
            return []
        shown = {h.chunk_id for h in primary}
        rows = load_document_chunks([top.path]).get(top.path, [])
        for wanted in CAUSE_SECTIONS:
            for row in rows:
                if (row[4] or "").lower() == wanted:
                    if row[0] in shown:
                        return []
                    sim = round(_cosine(question_vec, row[12]), 6)
                    if sim > best:  # only if the index missed it; never raise the gate
                        return []
                    hit = _related_hit(row, sim, top.path)
                    return [replace(hit, origin=CAUSE_ORIGIN, metadata={
                        **hit.metadata, "provenance": {"cause_of": top.path}})]
        return []

    def _log_contexts(self, primary: Sequence[Hit],
                      question_vec) -> tuple[dict[str, Hit], frozenset[str]]:
        """For the first `max_links` failed logs in `primary`: log path -> its error-context
        chunk when it is not shown yet, and the ids of those already shown (protected when
        trimming, so a retrieved traceback is not dropped for lower-priority chunks; Q008)."""
        failed = list(dict.fromkeys(h.path for h in primary
                                    if h.source_type == "log" and h.metadata.get("error")))
        contexts = load_log_error_contexts()
        best = max(h.score for h in primary)
        out: dict[str, Hit] = {}
        already: set[str] = set()
        for path in failed[: self.max_links]:
            # Read from the evidence itself, so it does not depend on the cache being fresh.
            shown = {h.chunk_id for h in primary if h.path == path
                     and (h.section or "").lower() == ERROR_CONTEXT_SECTION}
            if shown:
                already |= shown
                continue
            row = contexts.get(path)
            if row is None:
                continue
            companion = _companion_hit(row, question_vec, best, LOG_CONTEXT_ORIGIN, path)
            if companion is not None:
                out[path] = companion
        return out, frozenset(already)

    def _companions(self, primary: Sequence[Hit], links: dict[str, Hit], question: str,
                    question_vec) -> dict[str, Hit]:
        preliminary: list[Hit] = []
        for h in primary:
            preliminary.append(h)
            if h.path in links and links[h.path] not in preliminary:
                preliminary.append(links[h.path])
        targets = companion_targets(preliminary, question, self.companion_min_similarity,
                                    self.max_companions)
        if not targets:
            return {}
        present = {h.chunk_id for h in preliminary}
        best = max(h.score for h in primary)
        sections = load_resolution_sections()
        out: dict[str, Hit] = {}
        for path in targets:
            row = sections.get(path)
            if row is None or row[0] in present:  # no such section, or already shown
                continue
            companion = _companion_hit(row, question_vec, best, COMPANION_ORIGIN, path)
            if companion is not None:
                out[path] = companion
        return out


def _is_resolution(hit: Hit) -> bool:
    return hit.source_type == "runbook" and (hit.section or "").lower() in RESOLUTION_SECTIONS


def _related_hit(row: tuple, sim: float, named_by: str | None) -> Hit:
    """A chunk row of a related document as a Hit (named_by None: the question names it)."""
    return Hit(chunk_id=row[0], document_id=row[1], path=row[2], source_type=row[3],
               section=row[4], ts_start=row[5], ts_end=row[6], text=row[7],
               chunk_index=row[8], content_hash=row[9], pipeline_name=row[10],
               metadata={**(row[11] or {}),
                         "provenance": {"related_to": named_by or "question"}},
               rank=0, scores={"vector_similarity": sim, "vector_distance": round(1 - sim, 6)},
               origin=RELATED_ORIGIN, linked_from=named_by)


def _companion_hit(row: tuple, question_vec, best: float, origin: str,
                   companion_of: str) -> Hit | None:
    """A cached section row as a Hit scored against the question, or None if it would score
    above the top question hit (only possible when the approximate index missed the chunk;
    adding it would raise the best similarity the refusal gate sees)."""
    sim = round(_cosine(question_vec, row[12]), 6)
    if sim > best:
        return None
    return Hit(chunk_id=row[0], document_id=row[1], path=row[2], source_type=row[3],
               section=row[4], ts_start=row[5], ts_end=row[6], text=row[7],
               chunk_index=row[8], content_hash=row[9], pipeline_name=row[10],
               metadata={**(row[11] or {}), "provenance": {"companion_of": companion_of}},
               rank=0, scores={"vector_similarity": sim, "vector_distance": round(1 - sim, 6)},
               origin=origin)
