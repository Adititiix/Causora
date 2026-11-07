"""The real local embedding model. Marked slow; skipped unless the model is already cached
(download it once with `python -m incident_copilot.embeddings download`). Never downloads.
"""

import numpy as np
import pytest

from incident_copilot.config import Settings
from incident_copilot.embeddings import FastEmbedEmbedder, model_cache_files

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("fastembed")
    s = Settings()
    if not model_cache_files(s.embedding_model, s.embedding_cache_dir):
        pytest.skip(f"{s.embedding_model} not cached in {s.embedding_cache_dir}; run "
                    "`python -m incident_copilot.embeddings download`")
    return FastEmbedEmbedder(s.embedding_model, s.embedding_dim, s.embedding_cache_dir)


def test_real_model_has_schema_dimension_and_unit_vectors(embedder):
    vecs = embedder.embed_documents(["one", "two"])
    assert [v.shape for v in vecs] == [(384,), (384,)]
    assert all(np.isclose(np.linalg.norm(v), 1.0, atol=1e-5) for v in vecs)


def test_real_model_ranks_the_relevant_log_first(embedder):
    docs = embedder.embed_documents([
        "Log: orders_etl / transform_orders / 2026-09-04 / FAILED / error: KeyError: "
        "'customer_tier'",
        "Runbook: Disk full. No space left on device while writing temporary files.",
        "Table: payments_raw. Columns: payment_id, amount, currency, captured_at.",
    ])
    q = embedder.embed_query("Why did transform_orders fail with a KeyError?")
    scores = [float(np.dot(q, d)) for d in docs]
    assert int(np.argmax(scores)) == 0, scores
