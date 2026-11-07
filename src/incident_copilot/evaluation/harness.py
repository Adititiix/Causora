"""Run retrievers over evaluation questions and keep raw per-question results.

Retrievers are used only through the public `Retriever.retrieve()` interface, so any future
retriever (hybrid, reranking, ...) is evaluated by registering one more RetrieverConfig;
nothing here re-implements retrieval.

    configs = default_configs()                     # one per entry in retrieval.RETRIEVERS
    rows = run_retrieval_eval(questions, configs[0], embedder, repeats=3)
"""

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from incident_copilot.db import connect
from incident_copilot.embeddings import Embedder
from incident_copilot.evaluation.metrics import DEFAULT_KS, retrieval_metrics
from incident_copilot.retrieval import EXPERIMENTAL_RETRIEVERS, RETRIEVERS, Filters, Retriever

REQUIRED_FIELDS = ("external_id", "question", "ground_truth", "category", "should_abstain",
                   "expected_paths")


@dataclass(frozen=True)
class RetrieverConfig:
    """A named way of building a retriever. `extra` is free-form (window, model, notes)."""

    name: str
    factory: Callable[[Embedder], Retriever]
    description: str = ""
    extra: dict = field(default_factory=dict)


DESCRIPTIONS = {
    "vector": "cosine similarity to the question only",
    "vector+error_link": "vector, plus the runbook found via each retrieved failed log's "
                         "exception line",
    "vector+date": "vector, with log chunks narrowed to an unambiguous date in the question",
    "vector+error_link+date": "vector+error_link with the same date-aware log narrowing",
    "vector+error_link+anchors+date": "vector+error_link+date; a linked runbook never pushes "
                                      "out the log it was linked from",
    "vector+error_link+resolution+date": "vector+error_link+date plus the Resolution section "
                                         "of a matched runbook for failure/fix questions",
    "vector+error_link+diverse+date": "vector+error_link+date with at most 2 chunks per "
                                      "document",
    "vector+error_link+evidence+date": "anchors + resolution + diverse together",
    "vector+error_link+evidence+context+date": "evidence, plus the error-context chunk "
                                               "(traceback) of a failed log the question "
                                               "is dated to",
    "vector+error_link+evidence+context+scoped+date": "evidence+context with the per-document "
                                                      "cap only for dated-log questions",
}


def default_configs() -> list[RetrieverConfig]:
    return [RetrieverConfig(name, cls, DESCRIPTIONS.get(name, ""))
            for name, cls in RETRIEVERS.items()]


def experimental_configs() -> list[RetrieverConfig]:
    """Opt-in retrievers (retrieval.EXPERIMENTAL_RETRIEVERS); never the default."""
    return [RetrieverConfig(name, cls, DESCRIPTIONS.get(name, ""))
            for name, cls in EXPERIMENTAL_RETRIEVERS.items()]


# ---- dataset ----------------------------------------------------------------------------


def load_questions(path: str | Path) -> tuple[list[dict], list[dict]]:
    """(valid records, problems). Never modifies the file."""
    records, problems = [], []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append({"line": lineno, "problem": f"invalid JSON: {exc.msg}"})
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            problems.append({"line": lineno, "id": rec.get("external_id"),
                             "problem": f"missing fields {missing}"})
            continue
        records.append(rec)
    return records, problems


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_type_of(path: str) -> str | None:
    from incident_copilot.ingestion.sources import classify

    return classify(path)


def index_fingerprint() -> dict:
    """What the evaluation ran against: counts plus a hash of every (path, content_hash)."""
    with connect() as conn:
        docs = conn.execute("SELECT path, content_hash, metadata->>'embedding_model', "
                            "metadata->>'chunker_version' FROM documents ORDER BY path"
                            ).fetchall()
        chunks = conn.execute("SELECT count(*), count(embedding) FROM chunks").fetchone()
    digest = hashlib.sha256("\n".join(f"{p}:{h}" for p, h, _, _ in docs).encode()).hexdigest()
    return {"documents": len(docs), "chunks": chunks[0], "embedded_chunks": chunks[1],
            "embedding_models": sorted({m for _, _, m, _ in docs if m}),
            "chunker_versions": sorted({v for _, _, _, v in docs if v}),
            "corpus_sha256": digest}


# ---- running ----------------------------------------------------------------------------


def _row(q: dict, label: str, retriever: Retriever, hits, timings: list[float],
         ks: Sequence[int]) -> dict:
    paths = [h.path for h in hits]
    row = {
        "config": label,
        "retriever": retriever.name,
        "id": q["external_id"],
        "category": q["category"],
        "should_abstain": bool(q["should_abstain"]),
        "question": q["question"],
        "expected_paths": list(q["expected_paths"]),
        "expected_source_types": sorted({source_type_of(p) or "?"
                                         for p in q["expected_paths"]}),
        "retrieved_paths": paths,
        "retrieved_sections": [h.section for h in hits],
        "retrieved_origins": [h.origin for h in hits],
        "retrieved_chars": [len(h.text) for h in hits],
        "retrieved_similarity": [round(h.score, 6) for h in hits],
        "best_similarity": max((h.score for h in hits), default=None),
        "latency_ms": statistics.median(timings),
        "latency_runs_ms": [round(t, 3) for t in timings],
    }
    row.update(retrieval_metrics(q["expected_paths"], paths, ks))
    return row


def run_retrieval_eval_many(questions: Sequence[dict], configs: Sequence[RetrieverConfig],
                            embedder: Embedder, *, max_k: int = 8,
                            ks: Sequence[int] = DEFAULT_KS, repeats: int = 1,
                            filters: Filters | None = None,
                            labels: dict[str, str] | None = None) -> list[dict]:
    """One row per (config, question). Timing is INTERLEAVED: for every question and repeat,
    each config runs once, in an order that rotates, so drifts in machine load (which are
    large on Docker Desktop) affect all configs equally and their latencies are comparable.

    Latency is wall time of `retrieve()` (query embedding + database connection and search,
    + the link searches for error_link), median over `repeats`. Ranked results come from the
    first run (retrieval is deterministic for a fixed index)."""
    labels = labels or {}
    built = [(cfg, cfg.factory(embedder)) for cfg in configs]
    per_config: dict[str, list[dict]] = {cfg.name: [] for cfg in configs}
    for qi, q in enumerate(questions):
        timings: dict[str, list[float]] = {cfg.name: [] for cfg in configs}
        hits: dict[str, list] = {}
        for rep in range(max(1, repeats)):
            shift = (qi + rep) % len(built)
            for cfg, retriever in built[shift:] + built[:shift]:
                t0 = time.perf_counter()
                result = retriever.retrieve(q["question"], max_k, filters)
                timings[cfg.name].append((time.perf_counter() - t0) * 1000)
                hits.setdefault(cfg.name, result)
        for cfg, retriever in built:
            per_config[cfg.name].append(_row(q, labels.get(cfg.name, cfg.name), retriever,
                                             hits[cfg.name], timings[cfg.name], ks))
    return [row for cfg in configs for row in per_config[cfg.name]]


def run_retrieval_eval(questions: Sequence[dict], config: RetrieverConfig, embedder: Embedder,
                       *, max_k: int = 8, ks: Sequence[int] = DEFAULT_KS, repeats: int = 1,
                       filters: Filters | None = None, label: str | None = None) -> list[dict]:
    """Single-config version of `run_retrieval_eval_many`."""
    return run_retrieval_eval_many(questions, [config], embedder, max_k=max_k, ks=ks,
                                   repeats=repeats, filters=filters,
                                   labels={config.name: label or config.name})


def deep_ranks(questions: Sequence[dict], retriever: Retriever, depth: int = 100
               ) -> dict[str, dict[str, int]]:
    """For failure analysis: rank of every expected document in a deep search
    (question id -> path -> first rank; missing = not within `depth`)."""
    out = {}
    for q in questions:
        if not q["expected_paths"]:
            continue
        hits = retriever.retrieve(q["question"], depth)
        ranks: dict[str, int] = {}
        for h in hits:
            ranks.setdefault(h.path, h.rank)
        out[q["external_id"]] = {p: ranks[p] for p in q["expected_paths"] if p in ranks}
    return out
