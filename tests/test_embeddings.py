import numpy as np
import pytest

from incident_copilot.config import Settings
from incident_copilot.embeddings import (
    BGE_QUERY_PREFIX,
    Embedder,
    EmbeddingDimensionError,
    FastEmbedEmbedder,
    HashingEmbedder,
    check_schema_dim,
    get_embedder,
)


def cos(a, b):
    return float(np.dot(a, b))  # vectors are unit length


def test_hashing_embedder_shape_norm_and_determinism():
    a, b = HashingEmbedder(), HashingEmbedder()
    v = a.embed_query("KeyError customer_tier in transform_orders")
    assert v.shape == (384,)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.array_equal(v, b.embed_documents(["KeyError customer_tier in transform_orders"])[0])
    assert isinstance(a, Embedder)


def test_hashing_embedder_puts_texts_sharing_words_closer():
    e = HashingEmbedder()
    q = e.embed_query("why did transform_orders fail with KeyError customer_tier")
    docs = e.embed_documents([
        "Disk full: no space left on device while writing parquet",
        "KeyError: 'customer_tier' raised in transform_orders apply_transforms",
        "Kafka offset out of range for the clickstream consumer group",
    ])
    scores = [cos(q, d) for d in docs]
    assert int(np.argmax(scores)) == 1
    assert scores[1] > 0.3 > max(scores[0], scores[2])


def test_hashing_embedder_handles_empty_text():
    v = HashingEmbedder().embed_query("")
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_schema_dimension_is_enforced():
    check_schema_dim(384)
    with pytest.raises(EmbeddingDimensionError, match="vector\\(384\\)"):
        check_schema_dim(768)
    with pytest.raises(EmbeddingDimensionError):
        get_embedder("hashing", Settings(embedding_dim=768))


class StubModel:
    """Stands in for fastembed.TextEmbedding: records inputs, returns fixed-size vectors."""

    def __init__(self, dim):
        self.dim = dim
        self.inputs: list[list[str]] = []

    def embed(self, texts, batch_size=256):
        texts = list(texts)
        self.inputs.append(texts)
        return iter(np.full(self.dim, 2.0, dtype=np.float32) for _ in texts)


def test_fastembed_wrong_model_dimension_fails_clearly():
    e = FastEmbedEmbedder("some/768-dim-model", model=StubModel(768))
    with pytest.raises(EmbeddingDimensionError, match="returns 768-dimensional"):
        e.embed_documents(["x"])


def test_fastembed_normalizes_batches_and_prefixes_bge_queries():
    stub = StubModel(384)
    e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", model=stub)
    assert stub.inputs == []  # nothing runs until first use
    vecs = e.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3 and all(np.isclose(np.linalg.norm(v), 1.0) for v in vecs)
    e.embed_query("why did it fail")
    assert stub.inputs[-1] == [BGE_QUERY_PREFIX + "why did it fail"]
    assert ["a", "b", "c"] in stub.inputs  # documents get no prefix
    assert e.embed_documents([]) == []


def test_get_embedder_kinds():
    assert isinstance(get_embedder("hashing", Settings()), HashingEmbedder)
    real = get_embedder("fastembed", Settings())
    assert isinstance(real, FastEmbedEmbedder) and real._model is None  # lazy
    with pytest.raises(ValueError):
        get_embedder("nope", Settings())
