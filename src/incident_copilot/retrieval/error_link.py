"""Vector retrieval plus an error-line link from failed logs to runbooks.

Why: a question like "Why did transform_orders fail on 2026-09-04?" names a task and a date,
not a failure mode, so the matching runbook ("Column renamed upstream") shares almost no
vocabulary with it and ranks 30th-100th by question similarity alone (measured on the seed
questions). The log chunk, which is retrieved, carries the exception line
(`KeyError: 'customer_tier'`), and that line is what an engineer would search the runbooks
with.

How (deterministic, no LLM, no keyword search, no reranking):
  1. Vector search with the question (k results).
  2. For the first `max_links` retrieved failed-log documents, embed the log's exception
     line and take the nearest runbook chunk (respecting the request's filters).
  3. Insert each linked runbook right after its log's first chunk; drop the lowest-ranked
     question hits to stay within k. Ranks are renumbered.

Linked hits keep `vector_similarity` = similarity to the QUESTION (so the refusal gate is not
inflated) and add `link_similarity` = similarity to the exception line.
"""

from dataclasses import replace

from incident_copilot.embeddings import Embedder
from incident_copilot.retrieval.base import Filters, Hit
from incident_copilot.retrieval.vector import VectorRetriever


class ErrorLinkRetriever:
    name = "vector+error_link"

    def __init__(self, embedder: Embedder, max_links: int = 2):
        self.embedder = embedder
        self.vector = VectorRetriever(embedder)
        self.max_links = max_links

    def retrieve(self, question: str, k: int, filters: Filters | None = None) -> list[Hit]:
        filters = filters or Filters()
        question_vec = self.embedder.embed_query(question)
        self.last_question_vec = question_vec
        hits = self.vector.search(question_vec, k, filters)
        if filters.source_types and "runbook" not in filters.source_types:
            return hits
        links = self.find_links(hits, question_vec, filters)
        if not links:
            return hits
        keep = hits[: max(0, k - len(links))]
        merged: list[Hit] = []
        placed: set[str] = set()
        for h in keep:
            merged.append(h)
            link = links.get(h.path)
            if link and h.path not in placed:
                merged.append(link)
                placed.add(h.path)
        # An anchor that was cut from `keep` still contributes its runbook, at the end.
        # (The anchor itself is lost in that case; EvidenceRetriever keeps it.)
        merged.extend(link for path, link in links.items() if path not in placed)
        return [replace(h, rank=i) for i, h in enumerate(merged[:k], start=1)]

    def find_links(self, hits: list[Hit], question_vec, filters: Filters) -> dict[str, Hit]:
        """Anchor log path -> runbook chunk nearest to that log's exception line, for the first
        `max_links` failed logs in `hits`. Runbooks already in `hits` are not linked again."""
        anchors: list[Hit] = []
        for h in hits:
            if (h.source_type == "log" and h.metadata.get("error")
                    and h.path not in {a.path for a in anchors}):
                anchors.append(h)
            if len(anchors) >= self.max_links:
                break

        present = {h.path for h in hits}  # a runbook already retrieved is not linked again
        links: dict[str, Hit] = {}  # anchor path -> linked runbook hit
        runbook_filter = Filters(source_types=("runbook",), pipeline=filters.pipeline)
        searched: dict[str, list[Hit]] = {}  # two logs often fail with the same error line
        for anchor in anchors:
            error = anchor.metadata["error"]
            if error not in searched:
                error_vec = self.embedder.embed_query(error)
                searched[error] = self.vector.search(error_vec, 1, runbook_filter,
                                                     score_vec=question_vec)
            found = searched[error]
            if not found or found[0].path in present:
                continue
            link = found[0]
            scores = {"vector_similarity": link.scores["vector_similarity"],
                      "vector_distance": link.scores["vector_distance"],
                      "link_similarity": link.scores["search_similarity"]}
            links[anchor.path] = replace(link, scores=scores, origin="error_link",
                                         linked_from=anchor.path)
            present.add(link.path)
        return links
