r"""Retrieval-only benchmark: the default retriever against the experimental ones. No LLM.

    python -m incident_copilot.evaluation.retrieval_benchmark
    python -m incident_copilot.evaluation.retrieval_benchmark \
        --retriever vector+error_link+evidence+date
    python -m incident_copilot.evaluation.retrieval_benchmark \
        --reference eval/results/20261022T053248Z

Runs every seed question through the baseline (`--baseline`, default vector+error_link+date)
and each candidate with harness.run_retrieval_eval_many (interleaved timing), and adds, per
question:
  resolution_in_top_k  a Resolution chunk of an expected runbook is in the top k
                       (questions whose expected_paths contain a runbook)
  logs_in_top_k        every expected log is in the top k (questions that expect a log)
  distinct_documents / max_chunks_per_document   crowding
  evidence_tokens      sum of chunk characters / 4 (rough estimate)
  gate_pass            best similarity >= MIN_SIMILARITY (what the refusal gate decides)
Hit@K, Recall@K, All@K and MRR are the harness metrics (evaluation/metrics.py), unchanged.

`--reference` compares the baseline's ranked paths and sections with an earlier run's
per_question_retrieval.csv and reports every difference (a baseline that does not reproduce
makes the comparison meaningless). Writes a NEW folder eval/results/<RUN_ID>_retrieval/:
per_question.csv, comparison.csv, changes.csv, summary.md, run_metadata.json.
"""

import argparse
import csv
import datetime as dt
import json
import platform
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from incident_copilot.config import get_settings
from incident_copilot.db import mask_dsn
from incident_copilot.embeddings import get_embedder
from incident_copilot.evaluation.answers import EVIDENCE_METRICS, evidence_text_metrics
from incident_copilot.evaluation.harness import (
    DESCRIPTIONS,
    RetrieverConfig,
    file_sha256,
    index_fingerprint,
    load_questions,
    run_retrieval_eval_many,
    source_type_of,
)
from incident_copilot.evaluation.metrics import first_ranks, latency_summary
from incident_copilot.retrieval import EXPERIMENTAL_RETRIEVERS, RETRIEVERS
from incident_copilot.retrieval.evidence import (
    COMPANION_ORIGIN,
    ERROR_CONTEXT_SECTION,
    LOG_CONTEXT_ORIGIN,
    RESOLUTION_SECTIONS,
    load_resolution_sections,
)

CHARS_PER_TOKEN = 4  # rough estimate for English/log text; only ratios are compared
TARGET_LOG_QUESTIONS = ("Q015", "Q016", "Q018")
# Experiment guardrails for a candidate (agreed before the run; not tuned afterwards).
GUARDRAILS = {
    "resolution_in_top_k": ("min_count", 30),
    "target_logs_in_top_k": ("min_count", 3),
    # "No lower than the baseline" (93.2% / 86.4% are 41/44 and 38/44 rounded): compared with
    # the baseline's measured value, so a rounded literal cannot fail an identical result.
    "all@8": ("not_below_baseline", None),
    "hit@5": ("not_below_baseline", None),
    "MRR": ("min_value", 0.680),
    "unanswerable_gate_changes": ("max_value", 0),
    "evidence_token_growth": ("max_value", 0.25),
    "median_latency_increase_ms": ("max_value", 50.0),
}
METRICS = ("hit@1", "hit@3", "hit@5", "hit@8", "recall@5", "all@5", "all@8")


# ---- per question ---------------------------------------------------------------------------


def section_metrics(row: dict, k: int, min_similarity: float, ground_truth: str | None = None,
                    evidence_text: str | None = None) -> dict:
    """Section-level and evidence-size measures for one harness row (top k only). With the
    ground truth and the evidence text, also the evidence-level ground-truth term checks
    (answers.evidence_text_metrics: heuristics, not correctness)."""
    paths, sections = row["retrieved_paths"][:k], row["retrieved_sections"][:k]
    origins = row.get("retrieved_origins", [])[:k]
    chars = row.get("retrieved_chars", [])[:k]
    expected = row["expected_paths"]
    runbooks = [p for p in expected if source_type_of(p) == "runbook"]
    logs = [p for p in expected if source_type_of(p) == "log"]
    counts = Counter(paths)
    best = row["best_similarity"]
    return {
        "expects_runbook": bool(runbooks),
        "resolution_in_top_k": (any(p in runbooks and (s or "").lower() in RESOLUTION_SECTIONS
                                    for p, s in zip(paths, sections, strict=False))
                                if runbooks else None),
        "expects_log": bool(logs),
        "logs_in_top_k": all(p in paths for p in logs) if logs else None,
        # The traceback of every expected log is shown (its "error context" chunk).
        "log_error_context_in_top_k": (all(any(pp == p and s == ERROR_CONTEXT_SECTION
                                               for pp, s in zip(paths, sections, strict=False))
                                           for p in logs) if logs else None),
        "n_chunks": len(paths),
        "distinct_documents": len(counts),
        "max_chunks_per_document": max(counts.values(), default=0),
        "companions": sum(o == COMPANION_ORIGIN for o in origins),
        "log_context_companions": sum(o == LOG_CONTEXT_ORIGIN for o in origins),
        "evidence_tokens": sum(chars) / CHARS_PER_TOKEN,
        "gate_pass": best is not None and best >= min_similarity,
        "expected_ranks": {p: first_ranks(paths).get(p) for p in expected},
        **(evidence_text_metrics(category=row["category"], question=row["question"],
                                 ground_truth=ground_truth, evidence=evidence_text)
           if ground_truth is not None and evidence_text is not None and expected
           else dict.fromkeys(EVIDENCE_METRICS)),
    }


def by_type_recall(rows: Sequence[dict], k: int) -> dict[str, float | None]:
    """Share of expected documents of each source type found in the top k."""
    found: Counter = Counter()
    total: Counter = Counter()
    for r in rows:
        top = set(r["retrieved_paths"][:k])
        for p in r["expected_paths"]:
            t = source_type_of(p) or "?"
            total[t] += 1
            found[t] += p in top
    return {t: found[t] / total[t] for t in sorted(total)}


# ---- per configuration --------------------------------------------------------------------


def _mean_known(values) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return statistics.fmean(xs) if xs else None


def aggregate(rows: Sequence[dict], k: int) -> dict:
    """Metrics of one configuration's rows (each already carrying section_metrics)."""
    ans = [r for r in rows if not r["should_abstain"]]
    una = [r for r in rows if r["should_abstain"]]
    res = [r for r in ans if r["expects_runbook"]]
    targets = [r for r in ans if r["id"] in TARGET_LOG_QUESTIONS]
    lat = latency_summary(r["latency_ms"] for r in rows)
    out = {m: statistics.fmean(r[m] for r in ans) if ans else None for m in METRICS}
    out |= {
        "MRR": statistics.fmean(r["rr"] for r in ans) if ans else None,
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "resolution_in_top_k": sum(bool(r["resolution_in_top_k"]) for r in res),
        "resolution_questions": len(res),
        "logs_in_top_k": sum(bool(r["logs_in_top_k"]) for r in ans if r["expects_log"]),
        "log_error_context_in_top_k": sum(bool(r["log_error_context_in_top_k"])
                                          for r in ans if r["expects_log"]),
        "log_questions": sum(r["expects_log"] for r in ans),
        "target_logs_in_top_k": sum(bool(r["logs_in_top_k"]) for r in targets),
        "target_log_questions": len(targets),
        "unanswerable_gate_pass": sum(r["gate_pass"] for r in una),
        "mean_distinct_documents": statistics.fmean(r["distinct_documents"] for r in rows),
        "mean_max_chunks_per_document": statistics.fmean(r["max_chunks_per_document"]
                                                         for r in rows),
        "companions": sum(r["companions"] for r in rows),
        "log_context_companions": sum(r["log_context_companions"] for r in rows),
        **{f"mean_{m}": _mean_known(r[m] for r in ans) for m in EVIDENCE_METRICS},
        "mean_evidence_tokens": statistics.fmean(r["evidence_tokens"] for r in rows),
        "median_latency_ms": lat["median"], "p95_latency_ms": lat["p95"],
        "recall@8_by_type": by_type_recall(ans, k),
    }
    return out


def compare(base_rows: Sequence[dict], cand_rows: Sequence[dict], base: dict,
            cand: dict) -> tuple[dict, list[dict]]:
    """Guardrail inputs that need both runs, and every per-question change."""
    b, c = {r["id"]: r for r in base_rows}, {r["id"]: r for r in cand_rows}
    gate_changes = [i for i in b if b[i]["gate_pass"] != c[i]["gate_pass"]]
    best_changes = [i for i in b if b[i]["best_similarity"] != c[i]["best_similarity"]]
    extra = {
        "unanswerable_gate_changes": sum(b[i]["should_abstain"] for i in gate_changes),
        "gate_changes": gate_changes,
        "best_similarity_changes": best_changes,
        "evidence_token_growth": (cand["mean_evidence_tokens"] / base["mean_evidence_tokens"] - 1
                                  if base["mean_evidence_tokens"] else None),
        "median_latency_increase_ms": cand["median_latency_ms"] - base["median_latency_ms"],
    }
    changes = []
    # Every harness metric at every cut-off, so no regression can hide between them.
    fields = (*sorted({m for r in base_rows for m in r if "@" in m and m[0] != "_"}),
              "rr", "resolution_in_top_k", "logs_in_top_k")
    for i, rb in b.items():
        rc = c[i]
        diff = {f: (rb.get(f), rc.get(f)) for f in fields if rb.get(f) != rc.get(f)}
        if not diff:
            continue
        worse = any(f in diff and (rc.get(f) or 0) < (rb.get(f) or 0) for f in fields)
        changes.append({"id": i, "category": rb["category"], "direction":
                        "regression" if worse else "improvement", "changes": diff,
                        "expected_ranks_baseline": rb["expected_ranks"],
                        "expected_ranks_candidate": rc["expected_ranks"]})
    return extra, changes


def guardrail_limit(key: str, baseline: dict | None) -> float | None:
    kind, limit = GUARDRAILS[key]
    if kind == "not_below_baseline":
        return None if baseline is None else baseline.get(key)
    return limit


def check_guardrails(metrics: dict, baseline: dict | None = None) -> dict[str, bool | None]:
    out = {}
    for key, (kind, _) in GUARDRAILS.items():
        value, limit = metrics.get(key), guardrail_limit(key, baseline)
        if value is None or limit is None:
            out[key] = None
        elif kind in ("min_count", "min_value", "not_below_baseline"):
            out[key] = value >= limit
        else:
            out[key] = value <= limit
    return out


def check_reference(rows: Sequence[dict], reference_csv: Path, label: str) -> list[dict]:
    """Questions whose ranked paths/sections differ from an earlier run's baseline rows."""
    with open(reference_csv, encoding="utf-8") as fh:
        ref = {r["id"]: r for r in csv.DictReader(fh) if r["config"] == label}
    if not ref:
        return [{"id": None, "problem": f"no rows labelled {label!r} in {reference_csv}"}]
    out = []
    for r in rows:
        old = ref.get(r["id"])
        if old is None:
            out.append({"id": r["id"], "problem": "missing from the reference"})
        elif (json.loads(old["retrieved_paths"]) != r["retrieved_paths"]
              or json.loads(old["retrieved_sections"]) != r["retrieved_sections"]):
            out.append({"id": r["id"], "problem": "ranked paths or sections differ"})
    return out


# ---- output -------------------------------------------------------------------------------


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_summary(run_id: str, labels: Sequence[str], aggregates: dict, extras: dict,
                   guardrails: dict, changes: dict, missing_sections: list[str],
                   reproduction: list[dict] | None, baseline: str) -> str:
    rows = [("Hit@1", "hit@1"), ("Hit@3", "hit@3"), ("Hit@5", "hit@5"), ("Hit@8", "hit@8"),
            ("Recall@5", "recall@5"), ("All@5", "all@5"), ("All@8", "all@8"), ("MRR", "MRR"),
            ("Resolution of an expected runbook in top k", "resolution_in_top_k"),
            ("All expected logs in top k (log questions)", "logs_in_top_k"),
            ("Target log in top k (Q015/Q016/Q018)", "target_logs_in_top_k"),
            ("Unanswerable passing the gate", "unanswerable_gate_pass"),
            ("Mean distinct documents in top k", "mean_distinct_documents"),
            ("Mean max chunks from one document", "mean_max_chunks_per_document"),
            ("Resolution companions added (all questions)", "companions"),
            ("Error-context companions added (all questions)", "log_context_companions"),
            ("Traceback of every expected log in top k (log questions)",
             "log_error_context_in_top_k"),
            ("Heuristic: error-line term recall in evidence",
             "mean_error_term_recall_in_evidence"),
            ("Heuristic: root-cause term recall in evidence",
             "mean_root_cause_term_recall_in_evidence"),
            ("Heuristic: remediation coverage in evidence",
             "mean_remediation_covered_in_evidence"),
            ("Heuristic: fact term recall in evidence", "mean_fact_term_recall_in_evidence"),
            ("Mean evidence tokens (chars/4)", "mean_evidence_tokens"),
            ("Median retrieval latency ms", "median_latency_ms"),
            ("p95 retrieval latency ms", "p95_latency_ms")]
    lines = [f"# Retrieval benchmark ({run_id})", "",
             f"Baseline: `{baseline}`. No LLM was called. Hit/Recall/All/MRR are the harness "
             "metrics over answerable questions; resolution and log coverage are section/"
             "document checks on the top k.", ""]
    if reproduction is not None:
        lines += [("Baseline reproduces the reference run: yes (identical ranked paths and "
                   "sections for every question)." if not reproduction else
                   f"**Baseline does NOT reproduce the reference run ({len(reproduction)} "
                   "questions differ); the comparison below is not trustworthy.**"), ""]
    head = "| Metric | " + " | ".join(f"`{x}`" for x in labels) + " |"
    lines += [head, "|---|" + "---:|" * len(labels)]
    for title, key in rows:
        vals = []
        for x in labels:
            a = aggregates[x]
            v = a[key]
            if key == "resolution_in_top_k":
                v = f"{v}/{a['resolution_questions']}"
            elif key in ("logs_in_top_k", "log_error_context_in_top_k"):
                v = f"{v}/{a['log_questions']}"
            elif key == "target_logs_in_top_k":
                v = f"{v}/{a['target_log_questions']}"
            elif key == "unanswerable_gate_pass":
                v = f"{v}/{a['n_unanswerable']}"
            vals.append(_fmt(v))
        lines.append(f"| {title} | " + " | ".join(vals) + " |")
    types = sorted({t for x in labels for t in aggregates[x]["recall@8_by_type"]})
    for t in types:
        lines.append(f"| Recall@8 of expected {t} documents | " + " | ".join(
            _fmt(aggregates[x]["recall@8_by_type"].get(t)) for x in labels) + " |")
    lines += ["", "## Guardrails (candidate vs baseline)", ""]
    cands = [x for x in labels if x != baseline]
    lines += ["| Guardrail | Limit | " + " | ".join(f"`{x}`" for x in cands) + " |",
              "|---|---|" + "---|" * len(cands)]
    for key, (kind, limit) in GUARDRAILS.items():
        sign = "<=" if kind == "max_value" else ">="
        shown = guardrail_limit(key, aggregates[baseline])
        shown = (f"baseline ({_fmt(shown)})" if kind == "not_below_baseline"
                 else _fmt(limit))
        cells = []
        for x in cands:
            value = extras[x].get(key, aggregates[x].get(key))
            ok = guardrails[x][key]
            cells.append(f"{_fmt(value)} {'pass' if ok else 'FAIL' if ok is False else '?'}")
        lines.append(f"| {key} | {sign} {shown} | " + " | ".join(cells) + " |")
    for x in cands:
        e = extras[x]
        lines += ["", f"## `{x}`: per-question changes vs baseline", "",
                  f"Gate decisions changed: {', '.join(e['gate_changes']) or 'none'}. "
                  f"Best similarity changed: {', '.join(e['best_similarity_changes']) or 'none'}.",
                  ""]
        if changes[x]:
            lines += ["| id | category | direction | changed (baseline -> candidate) | "
                      "expected doc ranks baseline -> candidate |", "|---|---|---|---|---|"]
            for ch in changes[x]:
                diff = "; ".join(f"{f} {_fmt(a)}->{_fmt(b)}" for f, (a, b) in ch["changes"].items())
                ranks = "; ".join(
                    f"{p.split('/')[-1]} {ch['expected_ranks_baseline'].get(p) or '-'}->"
                    f"{ch['expected_ranks_candidate'].get(p) or '-'}"
                    for p in ch["expected_ranks_baseline"])
                lines.append(f"| {ch['id']} | {ch['category']} | {ch['direction']} | {diff} | "
                             f"{ranks} |")
        else:
            lines.append("no per-question changes")
    lines += ["", "## Notes", "",
              "- Expected runbooks without a Resolution section in the index (counted as "
              f"misses, not skipped): {', '.join(missing_sections) or 'none'}.",
              "- A Resolution chunk counts only if it belongs to an EXPECTED runbook; another "
              "runbook's Resolution does not.",
              "- Heuristic rows apply answers.evidence_text_metrics to the evidence text: "
              "ground-truth terms present in what the model would see, not correctness.",
              "- Evidence tokens are characters / 4; latency is wall time of retrieve() on one "
              "machine (median of repeats, interleaved), not a production benchmark.",
              "- Synthetic corpus and seed questions: a regression check, not proof of "
              "accuracy."]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()})


class _Recorder:
    """Passes retrieve() through and keeps the first result's text per question (the harness
    keeps ranked paths only)."""

    def __init__(self, inner, label: str, store: dict):
        self.inner, self.label, self.store = inner, label, store
        self.name = inner.name

    def retrieve(self, question, k, filters=None):
        hits = self.inner.retrieve(question, k, filters)
        self.store.setdefault((self.label, question), "\n".join(h.text for h in hits[:k]))
        return hits


def _recording(config: RetrieverConfig, store: dict):
    return lambda embedder: _Recorder(config.factory(embedder), config.name, store)


def run_benchmark(questions: Sequence[dict], configs: Sequence[RetrieverConfig], embedder, *,
                  baseline: str, k: int, min_similarity: float, repeats: int,
                  resolution_paths: set[str]) -> dict:
    """Pure orchestration over the harness (no files). Every question is run exactly once per
    configuration; that is checked, not assumed."""
    texts: dict[tuple[str, str], str] = {}  # (config, question) -> first run's evidence text
    recorded = [RetrieverConfig(c.name, _recording(c, texts), c.description, c.extra)
                for c in configs]
    rows = run_retrieval_eval_many(questions, recorded, embedder, max_k=k, repeats=repeats)
    truth = {q["external_id"]: q["ground_truth"] for q in questions}
    by_config: dict[str, list[dict]] = {}
    for r in rows:
        r |= section_metrics(r, k, min_similarity, truth.get(r["id"]),
                             texts.get((r["config"], r["question"])))
        by_config.setdefault(r["config"], []).append(r)
    ids = [q["external_id"] for q in questions]
    for name, rs in by_config.items():
        if sorted(r["id"] for r in rs) != sorted(ids):
            raise RuntimeError(f"{name}: questions were not each evaluated exactly once")
    labels = [c.name for c in configs]
    aggregates = {x: aggregate(by_config[x], k) for x in labels}
    extras, changes, guardrails = {}, {}, {}
    for x in labels:
        if x == baseline:
            continue
        extras[x], changes[x] = compare(by_config[baseline], by_config[x],
                                        aggregates[baseline], aggregates[x])
        guardrails[x] = check_guardrails(aggregates[x] | extras[x], aggregates[baseline])
    expected_runbooks = {p for q in questions for p in q["expected_paths"]
                         if source_type_of(p) == "runbook"}
    return {"rows": rows, "by_config": by_config, "labels": labels, "aggregates": aggregates,
            "extras": extras, "changes": changes, "guardrails": guardrails,
            "missing_sections": sorted(expected_runbooks - resolution_paths)}


def new_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    known = list(RETRIEVERS) + list(EXPERIMENTAL_RETRIEVERS)
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.evaluation."
                                          "retrieval_benchmark")
    parser.add_argument("--questions", default="eval/questions.seed.jsonl")
    parser.add_argument("--baseline", default="vector+error_link+date", choices=known)
    parser.add_argument("--retriever", action="append", choices=known,
                        help="candidate (repeatable; default: every experimental retriever)")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per question")
    parser.add_argument("--reference", default=None,
                        help="earlier retrieval run folder to check the baseline against")
    parser.add_argument("--reference-label", default="vector+error_link+date | w8 "
                                                     "(production index)")
    parser.add_argument("--out-root", default="eval/results")
    args = parser.parse_args(argv)

    settings = get_settings()
    questions, problems = load_questions(args.questions)
    if problems:
        print(f"{args.questions}: {len(problems)} malformed record(s), e.g. {problems[0]}")
        return 2
    candidates = [c for c in (args.retriever or list(EXPERIMENTAL_RETRIEVERS))
                  if c != args.baseline]
    registry = RETRIEVERS | EXPERIMENTAL_RETRIEVERS
    configs = [RetrieverConfig(n, registry[n], DESCRIPTIONS.get(n, ""))
               for n in [args.baseline, *candidates]]
    run_id = new_run_id()
    out_dir = Path(args.out_root) / f"{run_id}_retrieval"
    out_dir.mkdir(parents=True, exist_ok=False)  # never reuse an existing folder

    embedder = get_embedder("fastembed", settings)
    k = settings.ask_top_k
    result = run_benchmark(questions, configs, embedder, baseline=args.baseline, k=k,
                           min_similarity=settings.min_similarity, repeats=args.repeats,
                           resolution_paths=set(load_resolution_sections()))
    reproduction = None
    if args.reference:
        reproduction = check_reference(result["by_config"][args.baseline],
                                       Path(args.reference) / "per_question_retrieval.csv",
                                       args.reference_label)

    _write_csv(out_dir / "per_question.csv", result["rows"])
    _write_csv(out_dir / "comparison.csv",
               [{"config": x, **result["aggregates"][x], **result["extras"].get(x, {})}
                for x in result["labels"]])
    _write_csv(out_dir / "changes.csv", [{"config": x, **ch} for x in result["changes"]
                                         for ch in result["changes"][x]])
    (out_dir / "summary.md").write_text(render_summary(
        run_id, result["labels"], result["aggregates"], result["extras"],
        result["guardrails"], result["changes"], result["missing_sections"], reproduction,
        args.baseline), encoding="utf-8")
    meta = {"kind": "retrieval_benchmark", "run_id": run_id,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "database": mask_dsn(settings.database_url), "baseline": args.baseline,
            "candidates": candidates, "top_k": k, "min_similarity": settings.min_similarity,
            "repeats": args.repeats, "embedding_model": settings.embedding_model,
            "seed_questions_sha256": file_sha256(args.questions), "index": index_fingerprint(),
            "reference": args.reference, "reference_differences": reproduction,
            "guardrails": GUARDRAILS, "guardrail_results": result["guardrails"]}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                               encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"written to {out_dir}")
    return 1 if reproduction else 0


if __name__ == "__main__":
    raise SystemExit(main())
