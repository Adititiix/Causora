"""Test doubles shared by several test modules."""

import numpy as np

from incident_copilot.embeddings import HashingEmbedder


class CountingEmbedder:
    """HashingEmbedder that records every call, so tests can prove "zero embedding calls"."""

    def __init__(self, name: str | None = None):
        self._inner = HashingEmbedder()
        self.dim = self._inner.dim
        self.name = name or self._inner.name
        self.document_calls = 0
        self.query_calls = 0
        self.texts: list[str] = []

    def embed_documents(self, texts):
        self.document_calls += 1
        self.texts.extend(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        self.query_calls += 1
        return self._inner.embed_query(text)


class WrongDimensionEmbedder(CountingEmbedder):
    """Returns 10-dim vectors, which the vector(384) column must reject."""

    def embed_documents(self, texts):
        self.document_calls += 1
        return [np.ones(10, dtype=np.float32) for _ in texts]
