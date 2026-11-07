"""Follow-up questions: the question alone (baseline, how /ask behaved before history) vs the
question with its earlier turns resolved (retrieval/contextualize.py).

    python -m incident_copilot.evaluation.followup_eval                 # retrieval only
    python -m incident_copilot.evaluation.followup_eval --llm --pause 4 # also generate

Cases live in eval/followups.jsonl, kept apart from the 50 seed questions. Each case carries
the earlier turns as a client would send them (question, answer, cited paths), the expected
resolution (standalone / contextualized / ambiguous / no_context), the expected outcome
(answer / abstain / clarify) and the documents a good answer should rest on.

Measured: resolution status, gate decision, whether an expected document was retrieved, and,
with --llm, the answer status, cited documents, generation latency and LLM attempts. NOT
measured: whether each claim is supported by the chunk it cites (needs a human reader).
Writes eval/results/<RUN_ID>_followups/ (cases.jsonl, summary.md, run_metadata.json).
"""

import argparse
import datetime as dt
import json
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from incident_copilot.config import get_settings
from incident_copilot.embeddings import get_embedder
from incident_copilot.evaluation.harness import file_sha256, index_fingerprint
from incident_copilot.generation import (
    FakeLLMProvider,
    LLMError,
    LLMNotConfiguredError,
    answer_question,
    get_llm_provider,
)
from incident_copilot.retrieval import get_retriever
from incident_copilot.retrieval.contextualize import Turn, contextualize, load_document_info
from incident_copilot.retrieval.date_aware import load_log_catalog

ARMS = ("baseline", "new")


def load_cases(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def outcome_correct(case: dict, row: dict) -> bool | None:
    """Did the arm end the way the case expects? None when nothing was generated."""
    status, reason = row.get("status"), row.get("insufficient_reason")
    if status is None or status == "error":
        return None
    expect = case["expected_outcome"]
    if expect == "clarify":
        return reason == "ambiguous_reference"
    if expect == "abstain":
        return status == "insufficient_evidence"
    cited = set(row.get("cited_paths") or [])
    return status == "answered" and (not case["expected_paths"]
                                     or bool(cited & set(case["expected_paths"])))


def run_case(case: dict, arm: str, *, retriever, provider, k: int, min_similarity: float,
             catalog, lookup, generate: bool) -> dict:
    history = [Turn(t["question"], t.get("answer"), tuple(t.get("cited_paths") or ()))
               for t in case["history"]] if arm == "new" else []
    ctx = contextualize(case["question"], history, catalog, lookup) if history else None
    row = {"id": case["id"], "arm": arm, "category": case["category"],
           "resolution": ctx.status if ctx else "none",
           "retrieval_query": ctx.retrieval_query if ctx else case["question"]}
    t0 = time.perf_counter()
    try:
        outcome = answer_question(case["question"], retriever=retriever,
                                  llm_factory=lambda: provider, k=k,
                                  min_similarity=min_similarity, contextualization=ctx,
                                  retrieval_only=not generate)
        error = None
    except (LLMError, LLMNotConfiguredError) as exc:
        outcome, error = exc.partial_outcome, f"{type(exc).__name__}: {str(exc)[:160]}"
    paths = [h.path for h in outcome.hits]
    expected = set(case["expected_paths"])
    row |= {
        "best_similarity": outcome.gate.get("best_similarity"),
        "gate_passed": outcome.gate.get("passed"),
        "retrieved_paths": paths,
        "expected_retrieved": (bool(expected & set(paths)) if expected else None),
        "retrieval_ms": outcome.retrieval_ms,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "status": "error" if error else (outcome.status if generate or outcome.status !=
                                         "retrieval_only" else None),
        "insufficient_reason": outcome.insufficient_reason,
        "answer": outcome.answer,
        "cited_paths": sorted({c.path for c in outcome.citations}),
        "invalid_ids": (outcome.validation or {}).get("invalid_ids", []),
        "next_steps": outcome.next_steps,
        "llm_called": outcome.timings.get("llm_called", False),
        "llm_attempts": outcome.timings.get("llm_attempts"),
        "generation_ms": outcome.generation_ms,
        "error": error,
    }
    row["outcome_correct"] = outcome_correct(case, row)
    if arm == "new":
        row["resolution_correct"] = row["resolution"] == case["expected_resolution"] or (
            not case["history"] and row["resolution"] == "none")
    return row


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def _frac(rows, key, value=True) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(v == value for v in vals)}/{len(vals)}" if vals else "n/a"


def render_summary(cases: Sequence[dict], rows: Sequence[dict], meta: dict) -> str:
    by = {(r["id"], r["arm"]): r for r in rows}
    lines = [f"# Follow-up evaluation ({meta['run_id']})", "",
             f"Retriever `{meta['retriever']}`, top_k {meta['top_k']}, min_similarity "
             f"{meta['min_similarity']}, generation: {meta['generation']}. "
             f"{len(cases)} cases from `{meta['cases_path']}` (separate from the seed set).",
             "", "| metric | baseline (question alone) | new (with history) |", "|---|---|---|"]
    for label, key, value in (("gate passed", "gate_passed", True),
                              ("an expected document retrieved (cases with one)",
                               "expected_retrieved", True),
                              ("ended as expected (answer / abstain / clarify)",
                               "outcome_correct", True),
                              ("LLM called", "llm_called", True)):
        cells = [_frac([r for r in rows if r["arm"] == arm], key, value) for arm in ARMS]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    for arm_label, ms_key in (("retrieval ms median / p95", "retrieval_ms"),
                              ("generation ms median / p95", "generation_ms")):
        cells = []
        for arm in ARMS:
            v = [r[ms_key] for r in rows if r["arm"] == arm and r.get(ms_key) is not None]
            cells.append(f"{statistics.median(v):.0f} / {_pct(v, 0.95):.0f} (n={len(v)})"
                         if v else "n/a")
        lines.append(f"| {arm_label} | {cells[0]} | {cells[1]} |")
    retries = [r for r in rows if (r.get("llm_attempts") or 1) > 1]
    errors = [r for r in rows if r.get("error")]
    lines += ["", f"Resolution as expected (new): {_frac(rows, 'resolution_correct')}. "
              f"LLM calls with retries: {len(retries)}; errors: {len(errors)}.", "",
              "| id | category | resolution | best sim (base → new) | gate | expected doc "
              "(base → new) | outcome (base → new) | cited (new) |", "|---|---|---|---|---|---|"
              "---|---|"]

    def fmt(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    for c in cases:
        b, n = by[(c["id"], "baseline")], by[(c["id"], "new")]
        outcome = lambda r: (r["status"] or "-") + (f" ({r['insufficient_reason']})"  # noqa: E731
                                                    if r["insufficient_reason"] else "")
        lines.append(
            f"| {c['id']} | {c['category']} | {n['resolution']} (expected "
            f"{c['expected_resolution']}) | {fmt(b['best_similarity'])} → "
            f"{fmt(n['best_similarity'])} | {fmt(b['gate_passed'])} → {fmt(n['gate_passed'])} "
            f"| {fmt(b['expected_retrieved'])} → {fmt(n['expected_retrieved'])} | "
            f"{outcome(b)} → {outcome(n)} | {', '.join(n['cited_paths']) or '-'} |")
    lines += ["", "## Limitations", "",
              "- 16 hand-written cases on the synthetic corpus; earlier turns are given, not "
              "generated, so this measures resolution and retrieval, not a live conversation.",
              "- `ended as expected` for an answer only checks that an expected document was "
              "cited. Whether each claim is supported by its citation is NOT MEASURED.",
              "- One run, temperature 0.1; latency is from one machine on the Gemini free tier."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.evaluation.followup_eval")
    parser.add_argument("--cases", default="eval/followups.jsonl")
    parser.add_argument("--retriever", default=None, help="default: RETRIEVER setting")
    parser.add_argument("--llm", action="store_true", help="also generate answers (Gemini)")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between LLM calls")
    parser.add_argument("--out-root", default="eval/results")
    args = parser.parse_args(argv)

    settings = get_settings()
    cases = load_cases(args.cases)
    name = args.retriever or settings.retriever
    retriever = get_retriever(name, get_embedder("fastembed"))
    provider = get_llm_provider(settings) if args.llm else FakeLLMProvider([])
    catalog = load_log_catalog()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_root) / f"{run_id}_followups"
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        for arm in ARMS:
            row = run_case(case, arm, retriever=retriever, provider=provider,
                           k=settings.ask_top_k, min_similarity=settings.min_similarity,
                           catalog=catalog, lookup=load_document_info, generate=args.llm)
            rows.append(row)
            print(f"{case['id']} {arm:<8} {row['resolution']:<14} best="
                  f"{row['best_similarity'] or 0:.3f} status={row['status']} "
                  f"reason={row['insufficient_reason']} ok={row['outcome_correct']}")
            if args.llm and row["llm_called"]:
                time.sleep(args.pause)
    meta = {"kind": "followup_eval", "run_id": run_id, "retriever": retriever.name,
            "top_k": settings.ask_top_k, "min_similarity": settings.min_similarity,
            "generation": settings.llm_model if args.llm else "none (retrieval only)",
            "cases_path": args.cases, "cases_sha256": file_sha256(args.cases),
            "python": platform.python_version(), "index": index_fingerprint()}
    (run_dir / "cases.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(render_summary(cases, rows, meta), encoding="utf-8")
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
