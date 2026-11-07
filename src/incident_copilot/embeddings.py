"""Text embeddings behind a small protocol, so tests use a deterministic fake.

Real:  FastEmbedEmbedder runs BAAI/bge-small-en-v1.5 locally through ONNX Runtime
       (fastembed). No PyTorch, no API key. The model is loaded lazily on first use and its
       output dimension is checked against EMBEDDING_DIM (the database column is vector(384)).
Fake:  HashingEmbedder is a bag-of-words feature-hashing embedder. Texts that share words get
       closer vectors, so ranking tests are meaningful without a network or a model download.

Pre-download the model once (it is cached in EMBEDDING_CACHE_DIR):
    python -m incident_copilot.embeddings download
"""

import hashlib
import math
import re
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from incident_copilot.config import SCHEMA_EMBEDDING_DIM, Settings, get_settings

# BGE's recommended instruction for retrieval queries (passages get no prefix).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingDimensionError(RuntimeError):
    """The configured or actual embedding size does not match the database column."""


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model could not be loaded (not installed, not downloaded, no network)."""


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def check_schema_dim(dim: int) -> None:
    if dim != SCHEMA_EMBEDDING_DIM:
        raise EmbeddingDimensionError(
            f"EMBEDDING_DIM={dim}, but chunks.embedding is vector({SCHEMA_EMBEDDING_DIM}). "
            f"Use a {SCHEMA_EMBEDDING_DIM}-dimensional model (e.g. BAAI/bge-small-en-v1.5) "
            "or write a schema migration first."
        )


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests (feature hashing, signed, sublinear tf).

    Uses blake2b, not Python's hash(), so vectors are identical across processes.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = SCHEMA_EMBEDDING_DIM):
        self.dim = dim
        self.name = f"hashing-bow-{dim}"

    def _embed(self, text: str) -> np.ndarray:
        counts: dict[str, int] = {}
        for token in self._TOKEN_RE.findall(text.lower()):
            counts[token] = counts.get(token, 0) + 1
        vec = np.zeros(self.dim, dtype=np.float32)
        for token, n in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(n))
        if not vec.any():
            vec[0] = 1.0  # empty text: a valid unit vector instead of all zeros
        return l2_normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class FastEmbedEmbedder:
    """Local ONNX embedding model via fastembed. Loaded lazily; batched; L2-normalized."""

    def __init__(
        self,
        model_name: str,
        dim: int = SCHEMA_EMBEDDING_DIM,
        cache_dir: str | None = None,
        batch_size: int = 32,
        model=None,  # injectable for tests
    ):
        check_schema_dim(dim)
        self.name = model_name
        self.dim = dim
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name.lower() else ""
        self._model = model
        self._checked = False
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None and self._checked:
            return self._model
        with self._lock:  # API routes run in threads; load the model only once
            return self._load_locked()

    def _load_locked(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise EmbeddingUnavailableError(
                    "fastembed is not installed. Run: python -m pip install -r requirements-ml.txt"
                ) from exc
            try:
                self._model = TextEmbedding(model_name=self.name, cache_dir=self.cache_dir)
            except Exception as exc:  # noqa: BLE001 - download/IO/ONNX errors
                raise EmbeddingUnavailableError(
                    f"could not load embedding model {self.name!r} ({type(exc).__name__}). "
                    "Download it once with: python -m incident_copilot.embeddings download"
                ) from exc
        if not self._checked:
            probe = np.asarray(next(iter(self._model.embed(["dimension check"]))))
            if probe.shape[-1] != self.dim:
                raise EmbeddingDimensionError(
                    f"Embedding model {self.name!r} returns {probe.shape[-1]}-dimensional "
                    f"vectors, but EMBEDDING_DIM={self.dim} and chunks.embedding is "
                    f"vector({SCHEMA_EMBEDDING_DIM}). Pick a {SCHEMA_EMBEDDING_DIM}-dim model."
                )
            self._checked = True
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        model = self._load()
        return [l2_normalize(v) for v in model.embed(list(texts), batch_size=self.batch_size)]

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        return l2_normalize(next(iter(model.embed([self.query_prefix + text]))))


def model_cache_files(model_name: str, cache_dir: str | Path) -> list[Path]:
    """ONNX files already downloaded for `model_name` (empty list if not cached)."""
    cache = Path(cache_dir)
    if not cache.is_dir():
        return []
    try:
        from fastembed import TextEmbedding

        sources = [m["sources"]["hf"] for m in TextEmbedding.list_supported_models()
                   if m["model"] == model_name and m["sources"].get("hf")]
    except Exception:  # noqa: BLE001 - fastembed missing or its catalogue changed
        return []
    found: list[Path] = []
    for source in sources:
        folder = "models--" + source.replace("/", "--")
        found.extend(p for p in cache.glob(f"{folder}/**/*.onnx") if p.stat().st_size > 0)
    return found


def get_embedder(kind: str = "fastembed", settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    check_schema_dim(settings.embedding_dim)
    if kind == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if kind == "fastembed":
        return FastEmbedEmbedder(
            settings.embedding_model, settings.embedding_dim, settings.embedding_cache_dir
        )
    raise ValueError(f"unknown embedder {kind!r} (expected 'fastembed' or 'hashing')")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["download"]:
        print("usage: python -m incident_copilot.embeddings download")
        return 2
    settings = get_settings()
    embedder = get_embedder("fastembed", settings)
    vec = embedder.embed_query("warm-up")
    files = model_cache_files(settings.embedding_model, settings.embedding_cache_dir)
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"model:     {settings.embedding_model}")
    print(f"dimension: {len(vec)} (EMBEDDING_DIM={settings.embedding_dim})")
    print(f"cache:     {Path(settings.embedding_cache_dir).resolve()} ({size_mb:.0f} MB of ONNX)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
