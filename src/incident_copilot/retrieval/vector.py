"""Dense retrieval: cosine distance in pgvector (`embedding <=> query`).

similarity = 1 - cosine distance, in [-1, 1]; with normalized embeddings from the same model
it is comparable across questions, which is what the pre-LLM refusal gate relies on.
"""

import numpy as np

from incident_copilot.db import connect
from incident_copilot.embeddings import Embedder
from incident_copilot.retrieval.base import Filters, Hit

_SQL = """
SELECT c.id::text, d.id::text, d.path, d.source_type, c.section, c.ts_start, c.ts_end,
       c.content, c.chunk_index, c.content_hash, d.pipeline_name, c.metadata,
       c.embedding <=> %(q)s AS distance,
       c.embedding <=> %(score)s AS score_distance
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL
  AND (%(types)s::text[] IS NULL OR d.source_type = ANY(%(types)s::text[]))
  AND (%(pipeline)s::text IS NULL
       OR d.pipeline_name = %(pipeline)s::text
       OR d.metadata->'pipelines' ? %(pipeline)s::text)
  AND (d.source_type <> 'log' OR (
           (%(log_date)s::text IS NULL OR c.metadata->>'date' = %(log_date)s::text)
       AND (%(log_pipeline)s::text IS NULL OR d.pipeline_name = %(log_pipeline)s::text)
       AND (%(log_task)s::text IS NULL OR c.metadata->>'task' = %(log_task)s::text)))
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s
"""


class VectorRetriever:
    name = "vector"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def retrieve(self, question: str, k: int, filters: Filters | None = None) -> list[Hit]:
        self.last_question_vec = self.embedder.embed_query(question)  # reused after retrieval
        return self.search(self.last_question_vec, k, filters)

    def search(self, query_vec: np.ndarray, k: int, filters: Filters | None = None,
               score_vec: np.ndarray | None = None) -> list[Hit]:
        """Nearest chunks to `query_vec`. `vector_similarity` is reported against
        `score_vec` (default: the same vector), so a follow-up search can still be scored
        against the original question."""
        filters = filters or Filters()
        params = {
            "q": query_vec,
            "score": query_vec if score_vec is None else score_vec,
            "types": list(filters.source_types) if filters.source_types else None,
            "pipeline": filters.pipeline,
            "log_date": filters.log_date,
            "log_pipeline": filters.log_pipeline,
            "log_task": filters.log_task,
            "k": k,
        }
        with connect() as conn, conn.transaction():
            # With a WHERE filter, a plain HNSW scan can return fewer than k rows; iterative
            # scan (pgvector >= 0.8) keeps searching until k matching rows are found.
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(_SQL, params).fetchall()
        # relaxed_order may return rows slightly out of order; sort exactly.
        rows.sort(key=lambda r: r[12])
        hits = []
        for i, r in enumerate(rows, start=1):
            scores = {"vector_similarity": round(1.0 - float(r[13]), 6),
                      "vector_distance": round(float(r[13]), 6)}
            if score_vec is not None:
                scores["search_similarity"] = round(1.0 - float(r[12]), 6)
            hits.append(Hit(
                chunk_id=r[0], document_id=r[1], path=r[2], source_type=r[3], section=r[4],
                ts_start=r[5], ts_end=r[6], text=r[7], chunk_index=r[8], content_hash=r[9],
                pipeline_name=r[10], metadata=r[11] or {}, rank=i, scores=scores,
            ))
        return hits
