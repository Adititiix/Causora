"""Retrieval smoke test: run every seed question through VectorRetriever (no LLM).

    python -m incident_copilot.evaluation.retrieval_smoke
    python -m incident_copilot.evaluation.retrieval_smoke --window 4   # compare log windows
    docker compose exec api python -m incident_copilot.evaluation.retrieval_smoke

Reports, per category:
  any@k  at least one expected path is among the top-k retrieved chunks
  all@k  every expected path is among the top-k chunks (all@1 is impossible for questions
         that expect two documents; the report says how many those are)
and the best-hit similarity distribution for answerable vs unanswerable questions, which is
what the MIN_SIMILARITY refusal gate uses.

--window N indexes the corpus with ERROR_CONTEXT_ENTRIES=N into a temporary database
(<db>_eval_wN, dropped afterwards), so the main index is never touched.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from incident_copilot.chunking import logs as log_chunker
from incident_copilot.config import get_settings
from incident_copilot.db import connect
from incident_copilot.embeddings import Embedder, get_embedder
from incident_copilot.evaluation.scratch import temporary_index
from incident_copilot.retrieval import RETRIEVERS, Retriever, get_retriever

KS = (1, 3, 5, 8)
CATEGORY_ORDER = ["log_diagnosis", "runbook", "schema", "incident"]


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evaluate(questions: list[dict], retriever: Retriever, max_k: int = max(KS)) -> list[dict]:
    rows = []
    for q in questions:
        hits = retriever.retrieve(q["question"], max_k)
        expected = set(q["expected_paths"])
        row = {"id": q["external_id"], "category": q["category"], "question": q["question"],
               "should_abstain": q["should_abstain"], "expected": sorted(expected),
               "best_similarity": hits[0].score if hits else None,
               "top_paths": [h.path for h in hits]}
        for k in KS:
            got = {h.path for h in hits[:k]}
            row[f"any@{k}"] = bool(expected & got)
            row[f"all@{k}"] = bool(expected) and expected <= got
        rows.append(row)
    return rows


def _pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):5.1f}%" if values else "    -"


def _quantiles(xs: list[float]) -> str:
    xs = sorted(xs)
    if not xs:
        return "n/a"
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return (f"min {xs[0]:.3f}  p25 {q[0]:.3f}  median {statistics.median(xs):.3f}  "
            f"p75 {q[2]:.3f}  max {xs[-1]:.3f}")


def report(rows: list[dict], label: str) -> dict:
    print(f"\n=== Retrieval smoke: {label} ===")
    answerable = [r for r in rows if not r["should_abstain"]]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in answerable:
        by_cat[r["category"]].append(r)
    cats = [c for c in CATEGORY_ORDER if c in by_cat] + sorted(set(by_cat) - set(CATEGORY_ORDER))

    header = f"{'category':<15}{'n':>3}" + "".join(f"{'any@' + str(k):>8}" for k in KS) + \
        "".join(f"{'all@' + str(k):>8}" for k in KS)
    print(header)
    for name, group in [*[(c, by_cat[c]) for c in cats], ("ALL answerable", answerable)]:
        line = f"{name:<15}{len(group):>3}"
        line += "".join(f"{_pct([r[f'any@{k}'] for r in group]):>8}" for k in KS)
        line += "".join(f"{_pct([r[f'all@{k}'] for r in group]):>8}" for k in KS)
        print(line)
    multi = sum(len(r["expected"]) > 1 for r in answerable)
    print(f"({multi} of {len(answerable)} answerable questions expect 2+ documents, so all@1 "
          "cannot exceed "
          f"{100 * (len(answerable) - multi) / len(answerable):.1f}% by construction)")

    ans = [r["best_similarity"] for r in answerable if r["best_similarity"] is not None]
    una_rows = [r for r in rows if r["should_abstain"]]
    una = [r["best_similarity"] for r in una_rows if r["best_similarity"] is not None]
    print("\nBest-hit similarity (what the MIN_SIMILARITY gate sees)")
    print(f"  answerable   ({len(ans):>2}): {_quantiles(ans)}")
    print(f"  unanswerable ({len(una):>2}): {_quantiles(una)}")
    for r in sorted(una_rows, key=lambda r: -(r["best_similarity"] or 0)):
        print(f"    {r['best_similarity']:.3f}  {r['id']}  {r['question'][:70]}"
              f"  -> {r['top_paths'][0] if r['top_paths'] else '-'}")

    print("\n  threshold  unanswerable blocked  answerable blocked (wrongly)")
    candidates = sorted({round(x, 2) for x in ans + una} | {0.0})
    shown = [t for t in candidates if t >= min(una + ans, default=0) - 0.01]
    for t in shown:
        ub = sum(x < t for x in una)
        ab = sum(x < t for x in ans)
        print(f"  {t:9.2f}  {ub:>2}/{len(una):<2}{'':>17}{ab:>2}/{len(ans)}")
    safe = (int(min(ans) * 100) / 100) if ans else 0.0  # highest 2-dp value <= min answerable
    blocked = sum(x < safe for x in una)
    print(f"\nHighest threshold that blocks no answerable seed question: {safe:.2f} "
          f"(blocks {blocked}/{len(una)} unanswerable).")
    print("Vector similarity alone is a weak refusal signal: unanswerable questions that share "
          "vocabulary with the corpus score like answerable ones. Those must be refused by the "
          "model's insufficient_evidence flag.")

    misses = [r for r in answerable if not r["any@8"]]
    if misses:
        print(f"\nAnswerable questions with no expected document in the top 8 ({len(misses)}):")
        for r in misses:
            print(f"  {r['id']} [{r['category']}] {r['question'][:70]}")
            print(f"      expected {r['expected']}; got {r['top_paths'][:3]}")
    return {"label": label, "safe_threshold": safe, "rows": rows}


def run_all(questions: list[dict], embedder: Embedder, names: list[str], label: str) -> list:
    return [report(evaluate(questions, get_retriever(n, embedder)), f"retriever {n}; {label}")
            for n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.evaluation.retrieval_smoke")
    parser.add_argument("--questions", default="eval/questions.seed.jsonl")
    parser.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--window", type=int, default=None,
                        help="index into a temporary database with ERROR_CONTEXT_ENTRIES=N")
    parser.add_argument("--retriever", action="append", choices=sorted(RETRIEVERS),
                        help="repeatable (default: all retrievers)")
    parser.add_argument("--json", dest="json_out", default=None, help="write per-question rows")
    args = parser.parse_args(argv)

    settings = get_settings()
    questions = load_questions(Path(args.questions))
    embedder = get_embedder(args.embedder, settings)
    names = args.retriever or list(RETRIEVERS)
    if args.embedder == "hashing":
        print("WARNING: hashing embedder is a test fake; these numbers mean nothing.")

    if args.window is None:
        with connect() as conn:
            n = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()
        label = (f"{embedder.name}, main index ({n[0]} chunks, log window "
                 f"{log_chunker.ERROR_CONTEXT_ENTRIES})")
        results = run_all(questions, embedder, names, label)
    else:
        with temporary_index(embedder=embedder, window=args.window) as idx:
            label = (f"{embedder.name}, temporary index {idx.name} "
                     f"({idx.summary.chunks_written} chunks, log window {idx.window})")
            results = run_all(questions, embedder, names, label)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
