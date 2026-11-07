"""Real-LLM answer evaluation: seed questions through the /ask pipeline, scored and saved.

    python -m incident_copilot.evaluation.answer_eval --ids Q001,Q045      # chosen questions
    python -m incident_copilot.evaluation.answer_eval --limit 5            # the first five
    python -m incident_copilot.evaluation.answer_eval                      # all 50
    python -m incident_copilot.evaluation.answer_eval --resume eval/results/<RUN_ID>_answers

Every question goes through generation.answer_question() with the retriever, top-k, gate
threshold and LLM provider that /ask uses, all read from settings; nothing here overrides
them. Unlike /ask, nothing is written to the database: the run folder is the record.

A new folder eval/results/<RUN_ID>_answers/ is created (existing results are never touched):
  answers.jsonl       one JSON line per attempt, appended and flushed as each one finishes;
                      the last line for a question id is the current one
  answer_quality.csv  one row per question
  summary.md          aggregates, failures, abstentions, metric definitions, limitations
  run_metadata.json   model, settings, prompt and corpus fingerprints; never a key or password
The CSV, summary and metadata are rewritten after every question, so an interrupted run
leaves consistent files behind.

A generation failure (upstream error, malformed JSON, unexpected error) is recorded and the
run moves on; after --max-consecutive-errors failures in a row it stops (an exhausted quota is
the usual cause). A database or embedding failure stops the run. --resume continues a run in
its own folder, re-running only questions without a successful record, and refuses when the
model, settings, prompt module, questions or corpus differ from the original run.

Exit codes: 0 every selected question has a successful record; 1 some failed, or the run
stopped or aborted; 2 usage or configuration error (nothing was run); 130 interrupted.

Metrics are deterministic (evaluation/answers.py). The ground-truth term checks are surface
heuristics, not correctness, and human-judged groundedness stays null.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg

from incident_copilot.config import Settings, get_settings
from incident_copilot.embeddings import (
    EmbeddingDimensionError,
    EmbeddingUnavailableError,
    get_embedder,
)
from incident_copilot.evaluation.answers import (
    EVIDENCE_METRICS,
    QUALIFICATION_METRICS,
    TEXT_METRICS,
    answer_text_metrics,
    evidence_text_metrics,
    parse_ground_truth,
    provider_kind,
    qualification_metrics,
    score_answer,
)
from incident_copilot.evaluation.harness import file_sha256, index_fingerprint, load_questions
from incident_copilot.evaluation.metrics import latency_summary, refusal_confusion
from incident_copilot.generation import (
    SETUP_INSTRUCTIONS,
    AskOutcome,
    LLMBadResponseError,
    LLMError,
    LLMNotConfiguredError,
    LLMProvider,
    answer_question,
    get_llm_provider,
)
from incident_copilot.generation import prompt as prompt_module
from incident_copilot.generation.clarify import (
    Clarifier,
    NearMissResolver,
    make_clarifier,
    make_near_miss_resolver,
    remediation_hits,
)
from incident_copilot.generation.prompt import evidence_id
from incident_copilot.redaction import redact
from incident_copilot.retrieval import Retriever, get_retriever

KIND = "answer_eval"
ANSWERS, CSV_NAME, SUMMARY, METADATA = ("answers.jsonl", "answer_quality.csv", "summary.md",
                                        "run_metadata.json")
GATE_REASONS = frozenset({"no_hits", "below_min_similarity"})
# A gate refusal that offers the failures a vague question may mean (generation/clarify.py).
CLARIFICATION_REASON = "ambiguous_query"
# Failures that would hit every remaining question: stop instead of recording 50 copies.
ABORT_ERRORS = (psycopg.Error, EmbeddingUnavailableError, EmbeddingDimensionError,
                LLMNotConfiguredError)
# What must match for --resume to append to an existing run.
FINGERPRINT_KEYS = ("model", "retriever", "ask_top_k", "min_similarity", "embedding_model",
                    "prompt_module_sha256", "seed_questions_sha256", "corpus_sha256")
METRIC_KEYS = ("expected_paths_retrieved", "abstained", "abstained_by", "abstention_correct",
               "unnecessary_abstention", "missed_abstention", "n_citations", "cited_paths",
               "cited_path_precision", "cited_path_recall", "uncited_expected_paths",
               "extra_cited_paths", "citation_validity", "invalid_ids", "dropped_claims",
               "remediation_expected", "next_steps_count", *TEXT_METRICS, *EVIDENCE_METRICS,
               *QUALIFICATION_METRICS, "groundedness_human")
CSV_FIELDS = ("id", "category", "should_abstain", "status", "insufficient_reason",
              "refusal_note", "error_category", "best_similarity", "gate_passed",
              "expected_paths",
              *METRIC_KEYS, "model", "latency_ms", "retrieval_ms", "generation_ms",
              "tokens_in", "tokens_out")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


# ---- one question -------------------------------------------------------------------------


def _error(category: str, exc: BaseException) -> dict:
    return {"category": category, "type": type(exc).__name__,
            "message": redact(str(exc))[:300]}


def score_record(rec: dict, outcome: AskOutcome | None) -> dict:
    """Deterministic metrics for one record (keys in METRIC_KEYS; None = not applicable)."""
    expected = set(rec["expected_paths"])
    retrieved = {r["path"] for r in rec["retrieved"]}
    m: dict = dict.fromkeys(METRIC_KEYS)
    # Retrieval ceiling: an expected document that was never retrieved cannot be cited.
    m["expected_paths_retrieved"] = len(expected & retrieved) / len(expected) if expected else None
    answerable = not rec["should_abstain"]
    if answerable:
        m["remediation_expected"] = bool(
            parse_ground_truth(rec["category"], rec["ground_truth"])["remediation"])
    if rec["status"] == "error" or outcome is None:
        return m
    if answerable:  # also for refusals: was the ground truth in what the model was shown?
        m.update(evidence_text_metrics(
            category=rec["category"], question=rec["question"],
            ground_truth=rec["ground_truth"], evidence="\n".join(h.text for h in outcome.hits)))
    scoring = {"model": outcome.model, "abstained": bool(outcome.insufficient_evidence),
               "evidence_ids": [evidence_id(h) for h in outcome.hits],
               "evidence_paths": [h.path for h in outcome.hits],
               "metadata": {"response": {"citations": [c.evidence_id for c in outcome.citations]},
                            "validation": outcome.validation}}
    base = score_answer(scoring, sorted(expected), rec["should_abstain"])
    refused = base["refused"]
    cited = set(base["cited_paths"])
    m.update({
        # Document level, like the recall/precision they explain.
        "uncited_expected_paths": (sorted(expected - cited)
                                   if answerable and rec["status"] == "answered" else None),
        "extra_cited_paths": sorted(cited - expected),
        "abstained": refused,
        "abstained_by": (None if not refused else
                         "clarification" if rec["insufficient_reason"] == CLARIFICATION_REASON
                         else "gate" if rec["insufficient_reason"] in GATE_REASONS
                         else "model"),
        "abstention_correct": base["abstention_correct"],
        "unnecessary_abstention": refused and not rec["should_abstain"],
        "missed_abstention": rec["should_abstain"] and not refused,
        "n_citations": base["n_citations"],
        "cited_paths": base["cited_paths"],
        "cited_path_precision": base["cited_path_precision"],
        "cited_path_recall": base["cited_path_recall"],
        "citation_validity": base["citation_validity"],
        "invalid_ids": len((outcome.validation or {}).get("invalid_ids", [])),
        "dropped_claims": base["dropped_claims"],
        "next_steps_count": len(rec["next_steps"]),
    })
    if rec["status"] == "answered":
        m.update(qualification_metrics(
            cited={c["evidence_id"]: c["text"] or "" for c in rec["citations"]},
            summary=rec["answer"],
            claims=[c["claim"] for c in rec["evidence"] + rec["inference"]],
            inferences=rec["inference"]))
    if rec["status"] == "answered" and not rec["should_abstain"]:
        m.update(answer_text_metrics(
            category=rec["category"], question=rec["question"],
            ground_truth=rec["ground_truth"], summary=rec["answer"],
            claims=[c["claim"] for c in rec["evidence"] + rec["inference"]],
            next_steps=rec["next_steps"]))
    return m


def build_record(q: dict, outcome: AskOutcome | None, error: dict | None) -> dict:
    hits = outcome.hits if outcome else []
    text_by_chunk = {h.chunk_id: h.text for h in hits}
    used = outcome.used_chunk_ids if outcome and not error else set()
    ok = outcome is not None and error is None
    rec = {
        "id": q["external_id"], "category": q["category"],
        "should_abstain": bool(q["should_abstain"]), "question": q["question"],
        "ground_truth": q["ground_truth"], "expected_paths": list(q["expected_paths"]),
        "status": outcome.status if ok else "error",
        "insufficient_reason": outcome.insufficient_reason if ok else None,
        # Unverified model text, kept only to diagnose refusals.
        "refusal_note": outcome.refusal_note if ok else None,
        "answer": outcome.answer if ok else None,
        "evidence": outcome.evidence if ok else [],
        "inference": outcome.inference if ok else [],
        "next_steps": outcome.next_steps if ok else [],
        # The cited chunk text is kept so groundedness can be judged from this file alone.
        "citations": [{"evidence_id": c.evidence_id, "path": c.path, "source_type": c.source_type,
                       "section": c.section, "chunk_index": c.chunk_index,
                       "ts_start": c.ts_start, "ts_end": c.ts_end,
                       "text": text_by_chunk.get(c.chunk_id)}
                      for c in (outcome.citations if ok else [])],
        "retrieved": [{"evidence_id": evidence_id(h), "path": h.path,
                       "source_type": h.source_type, "section": h.section, "origin": h.origin,
                       "linked_from": h.linked_from, "similarity": h.score,
                       "used_in_answer": h.chunk_id in used} for h in hits],
        "gate": outcome.gate if outcome else {},
        "validation": outcome.validation if outcome else {},
        "dropped_claims": outcome.dropped_claims if outcome else [],
        "retrieval_info": outcome.retrieval_info if outcome else {},
        "model": outcome.model if outcome else None,
        "tokens_in": outcome.tokens_in if outcome else None,
        "tokens_out": outcome.tokens_out if outcome else None,
        "retrieval_ms": outcome.retrieval_ms if outcome else None,
        "generation_ms": outcome.generation_ms if outcome else None,
        "latency_ms": outcome.latency_ms if outcome else None,
        "error": error,
        "evaluated_utc": _now(),
    }
    rec["metrics"] = score_record(rec, outcome if ok else None)
    return rec


def evaluate_question(q: dict, *, retriever: Retriever, provider: LLMProvider, k: int,
                      min_similarity: float, clarifier: Clarifier | None = None,
                      near_miss: NearMissResolver | None = None) -> dict:
    """One question through answer_question(); a generation failure becomes a record. With
    `near_miss`, a refused question is resolved like /ask does (clarify.resolve_near_miss),
    including the matched failure's own remediation sections."""
    outcome, error = None, None
    remediation = None if near_miss is None else (lambda paths, hits: remediation_hits(
        paths, hits, getattr(retriever, "last_question_vec", None)))
    try:
        outcome = answer_question(q["question"], retriever=retriever,
                                  llm_factory=lambda: provider, k=k,
                                  min_similarity=min_similarity, clarifier=clarifier,
                                  near_miss=near_miss, remediation=remediation)
    except ABORT_ERRORS:
        raise
    except LLMError as exc:
        outcome = getattr(exc, "partial_outcome", None)
        category = "llm_bad_response" if isinstance(exc, LLMBadResponseError) else \
            "llm_upstream_error"
        error = _error(category, exc)
    except Exception as exc:  # noqa: BLE001 - one broken question must not end the run
        outcome = getattr(exc, "partial_outcome", None)
        error = _error("internal_error", exc)
    return build_record(q, outcome, error)


# ---- files --------------------------------------------------------------------------------


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_record(run_dir: Path, rec: dict) -> None:
    with open(run_dir / ANSWERS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_records(run_dir: Path) -> list[dict]:
    """Every record of a run folder, in file order. A truncated last line (the process died
    mid-write) is skipped; that question simply runs again on --resume."""
    path = run_dir / ANSWERS
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _csv_value(value):
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def write_outputs(run_dir: Path, records: Sequence[dict], meta: dict) -> None:
    rows = []
    for r in records:
        row = {k: r.get(k) for k in CSV_FIELDS if k in r}
        row |= r["metrics"]
        row["error_category"] = (r["error"] or {}).get("category")
        row["best_similarity"] = (r["gate"] or {}).get("best_similarity")
        row["gate_passed"] = (r["gate"] or {}).get("passed")
        rows.append({k: _csv_value(row.get(k)) for k in CSV_FIELDS})
    tmp = run_dir / (CSV_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, run_dir / CSV_NAME)
    _write_atomic(run_dir / SUMMARY, render_summary(records, meta))
    _write_atomic(run_dir / METADATA, json.dumps(meta, indent=2, default=str))


# ---- summary ------------------------------------------------------------------------------


def _frac(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "n/a (0 questions)"


def _mean(values) -> str:
    xs = [float(v) for v in values if v is not None]
    return f"{statistics.fmean(xs):.3f} (n={len(xs)})" if xs else "n/a (n=0)"


def _num(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def _cell(text: str | None, limit: int = 160) -> str:
    text = " ".join((text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _ids(records) -> str:
    return ", ".join(r["id"] for r in records) or "none"


def _ms(summary: dict) -> str:
    if not summary["n"]:
        return "n/a (no samples)"
    p95 = f"{summary['p95']:.0f}" if summary["p95"] is not None else "n/a"
    note = f"; {summary['note']}" if summary["note"] else ""
    return f"{summary['median']:.0f} / {p95} ms (n={summary['n']}{note})"


LIMITATIONS = [
    "Synthetic corpus and 50 seed questions written alongside it: small, and not independent "
    "of the system's design. Only 6 questions are unanswerable.",
    "One run at temperature 0.1: answers can differ between runs; nothing here measures that "
    "variance.",
    "Ground-truth term checks (error class/terms, root-cause terms, remediation coverage, fact "
    "terms, incident id) are SURFACE HEURISTICS: a correct paraphrase can score low and a wrong "
    "answer that repeats the right words can score high. They are not correctness.",
    "Groundedness (is every claim supported by the chunk it cites?) is NOT MEASURED: "
    "`groundedness_human` is null until a human labels it. Citation validation only proves "
    "that cited ids were retrieved, not that they support the claim.",
    "Cited-path precision counts only `expected_paths` as relevant; citing another genuinely "
    "relevant document (an incident report, a schema doc) lowers it.",
    "For incident questions the ground-truth fix is the runbook's generic fix, not the incident "
    "report's own resolution, so an answer that reports what was actually done can score low.",
    "`next_steps` carry no evidence ids, so validation cannot check them; only the prompt asks "
    "that they come from the evidence.",
    "Latency includes free-tier queueing and retries and comes from one machine; it is not a "
    "production benchmark. Tokens are as reported by the provider.",
]

DEFINITIONS = [
    "Questions whose latest record is an error are listed under Errors and excluded from every "
    "abstention, citation, validation and heuristic metric (they count in latency only).",
    "`cited_path_recall` = |cited documents ∩ expected_paths| / |expected_paths|, for "
    "answerable questions that were answered (None if refused).",
    "`cited_path_precision` = |cited documents ∩ expected_paths| / |cited documents| (None "
    "when nothing is cited).",
    "`expected_paths_retrieved` = share of expected_paths among the retrieved chunks: the "
    "ceiling for citation recall.",
    "Unnecessary abstention: an answerable question refused. Missed abstention: an "
    "unanswerable question answered. `abstained_by`: gate (no LLM call), model, or "
    "clarification (a gate refusal that offered the failures a vague question may mean; no "
    "LLM call; still not an answer, counted apart from the two refusals).",
    "Term checks (`answers.answer_text_metrics`) ignore terms that already appear in the "
    "question; remediation actions are the ground-truth fix split at commas/semicolons, and an "
    "action is covered when at least half of its terms appear.",
    "`*_in_evidence` (`answers.evidence_text_metrics`): the same checks on all retrieved chunk "
    "text. A rough ceiling for the answer value; still a heuristic (unrelated chunks can share "
    "words with the ground truth).",
    "Empty next_steps is counted only for answered questions whose ground truth contains a fix "
    "(log_diagnosis, runbook, incident); schema questions expect none.",
]


def render_summary(records: Sequence[dict], meta: dict) -> str:
    ok = [r for r in records if r["status"] != "error"]
    errors = [r for r in records if r["status"] == "error"]
    m = [r["metrics"] for r in ok]
    una = [r for r in ok if r["should_abstain"]]
    ans = [r for r in ok if not r["should_abstain"]]
    answered = [r for r in ans if r["status"] == "answered"]
    qualified = [r for r in ok if r["status"] == "answered"
                 and r["metrics"].get("cites_qualified_source")]
    by_cat = {c: [r for r in answered if r["category"] == c]
              for c in ("log_diagnosis", "runbook", "incident", "schema")}
    remediation = [r for r in answered if r["metrics"]["remediation_actions"]]
    fixable = [r for r in answered if r["metrics"].get("remediation_expected")]
    facts = by_cat["schema"] + by_cat["incident"]
    logs = by_cat["log_diagnosis"]
    conf = refusal_confusion([r["should_abstain"] for r in ok],
                             [r["metrics"]["abstained"] for r in ok]) if ok else {}
    llm_calls = [r for r in records if r["generation_ms"] is not None]
    counted = [r for r in llm_calls if r["tokens_in"] is not None]
    kinds = Counter(provider_kind(r["model"]) for r in records if r["model"])
    statuses = Counter(r["status"] for r in records)

    rows = [
        ("Run", "model (provider kind)",
         f"{meta.get('model')} ({', '.join(sorted(kinds)) or 'no LLM call yet'})"),
        ("Run", "settings", f"retriever {meta.get('retriever')}, top_k {meta.get('ask_top_k')}, "
                            f"min_similarity {meta.get('min_similarity')}"),
        ("Run", "state", meta.get("state")),
        ("Run", "questions evaluated / selected",
         f"{len(records)} / {len(meta.get('selection', {}).get('question_ids', []))}"),
        ("Run", "answered / insufficient_evidence / error",
         f"{statuses['answered']} / {statuses['insufficient_evidence']} / {statuses['error']}"),
        ("Abstention", "unanswerable refused (by gate / by model)",
         f"{_frac(sum(r['metrics']['abstained'] for r in una), len(una))} "
         f"({sum(r['metrics']['abstained_by'] == 'gate' for r in una)} / "
         f"{sum(r['metrics']['abstained_by'] == 'model' for r in una)})"),
        ("Abstention", "clarification asked instead (unanswerable / answerable)",
         f"{_ids([r for r in una if r['metrics']['abstained_by'] == 'clarification'])} / "
         f"{_ids([r for r in ans if r['metrics']['abstained_by'] == 'clarification'])}"),
        ("Abstention", "missed abstention (unanswerable answered)",
         _ids([r for r in una if r["metrics"]["missed_abstention"]])),
        ("Abstention", "unnecessary abstention (answerable refused)",
         f"{_frac(sum(r['metrics']['unnecessary_abstention'] for r in ans), len(ans))}: "
         f"{_ids([r for r in ans if r['metrics']['unnecessary_abstention']])}"),
        ("Abstention", "false-refusal rate / missed-refusal rate",
         f"{conf.get('false_refusal_rate')} / {conf.get('missed_refusal_rate')}"),
        ("Citations", "mean cited-path recall (answered answerable)",
         _mean(r["metrics"]["cited_path_recall"] for r in answered)),
        ("Citations", "all expected paths cited",
         _frac(sum(r["metrics"]["cited_path_recall"] == 1 for r in answered), len(answered))),
        ("Citations", "mean cited-path precision",
         _mean(r["metrics"]["cited_path_precision"] for r in answered)),
        ("Citations", "mean expected paths retrieved (ceiling)",
         _mean(r["metrics"]["expected_paths_retrieved"] for r in ans)),
        ("Validation", "invalid cited ids / dropped claims",
         f"{sum(x['invalid_ids'] or 0 for x in m)} / {sum(x['dropped_claims'] or 0 for x in m)}"),
        ("Validation", "questions with dropped claims",
         _ids([r for r in ok if r["metrics"]["dropped_claims"]])),
        ("Next steps", "empty next_steps, answered questions whose ground truth has a fix",
         f"{_frac(sum(not r['next_steps'] for r in fixable), len(fixable))}: "
         f"{_ids([r for r in fixable if not r['next_steps']])}"),
        ("Next steps", "answered questions with no fix expected (schema)",
         f"{len(answered) - len(fixable)} (not counted above)"),
        ("Heuristic", "error class mentioned (log_diagnosis)",
         _frac(sum(bool(r["metrics"]["error_class_mentioned"]) for r in by_cat["log_diagnosis"]),
               len(by_cat["log_diagnosis"]))),
        ("Heuristic", "mean error-line term recall (log_diagnosis): answer / evidence",
         f"{_mean(r['metrics']['error_term_recall'] for r in logs)} / "
         f"{_mean(r['metrics']['error_term_recall_in_evidence'] for r in logs)}"),
        ("Heuristic", "mean root-cause term recall (log_diagnosis): answer / evidence",
         f"{_mean(r['metrics']['root_cause_term_recall'] for r in logs)} / "
         f"{_mean(r['metrics']['root_cause_term_recall_in_evidence'] for r in logs)}"),
        ("Heuristic", "mean remediation coverage: next_steps / whole answer / evidence",
         f"{_mean(r['metrics']['remediation_covered_next_steps'] for r in remediation)} / "
         f"{_mean(r['metrics']['remediation_covered_any'] for r in remediation)} / "
         f"{_mean(r['metrics']['remediation_covered_in_evidence'] for r in remediation)}"),
        ("Heuristic", "mean fact term recall (schema, incident): answer / evidence",
         f"{_mean(r['metrics']['fact_term_recall'] for r in facts)} / "
         f"{_mean(r['metrics']['fact_term_recall_in_evidence'] for r in facts)}"),
        ("Heuristic", "incident id mentioned",
         _frac(sum(bool(r["metrics"]["incident_id_mentioned"]) for r in by_cat["incident"]),
               len(by_cat["incident"]))),
        ("Heuristic", "source qualification kept (answers citing a medium/low-confidence or "
                      "open-alternative chunk)",
         _frac(sum(bool(r["metrics"]["qualification_kept"]) for r in qualified), len(qualified))
         + f"; not kept: {_ids([r for r in qualified if not r['metrics']['qualification_kept']])}"),
        ("Groundedness", "human-judged groundedness", "NOT MEASURED (groundedness_human is null)"),
        ("Latency", "end-to-end median / p95",
         _ms(latency_summary(r["latency_ms"] for r in records if r["latency_ms"] is not None))),
        ("Latency", "generation median / p95 (LLM calls)",
         _ms(latency_summary(r["generation_ms"] for r in llm_calls))),
        ("Tokens", "total in / out (calls that reported tokens)",
         f"{sum(r['tokens_in'] for r in counted)} / {sum(r['tokens_out'] or 0 for r in counted)} "
         f"over {len(counted)} of {len(llm_calls)} calls" if counted else "not reported"),
    ]
    lines = [f"# Incident Copilot answer evaluation ({meta.get('run_id')})", ""]
    if kinds.get("fake_scripted"):
        lines += ["**Answers came from the scripted FakeLLMProvider: these numbers are not "
                  "model quality.**", ""]
    lines += ["| Area | Metric | Value |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {v} |" for a, b, v in rows]
    lines += ["", "## Errors", ""]
    if errors:
        lines += ["| id | category | message |", "|---|---|---|"]
        lines += [f"| {r['id']} | {r['error']['category']} | "
                  f"{r['error']['message'].replace('|', '/')} |" for r in errors]
    else:
        lines.append("none")
    lines += ["", "## Abstentions", "",
              "`expected retrieved` and `fix in evidence` say whether the model was shown what "
              "the ground truth needs (heuristic); the note is the model's own unverified text.",
              ""]
    refused = [r for r in ok if r["metrics"]["abstained"]]
    if refused:
        lines += ["| id | should abstain | by | reason | best similarity | expected retrieved | "
                  "fix in evidence | model's note |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['should_abstain']} | {r['metrics']['abstained_by']} | "
                  f"{r['insufficient_reason']} | {(r['gate'] or {}).get('best_similarity')} | "
                  f"{_num(r['metrics']['expected_paths_retrieved'])} | "
                  f"{_num(r['metrics'].get('remediation_covered_in_evidence'))} | "
                  f"{_cell(r.get('refusal_note'))} |" for r in refused]
    else:
        lines.append("none")
    lines += ["", "## Dropped claims", ""]
    dropped = [(r["id"], d) for r in ok for d in r.get("dropped_claims") or []]
    if dropped:
        lines += ["| id | field | cited ids | invalid ids | claim |", "|---|---|---|---|---|"]
        lines += [f"| {i} | {d['field']} | {', '.join(d['ids']) or '(none)'} | "
                  f"{', '.join(d['invalid_ids']) or '(no ids)'} | {_cell(d['claim'])} |"
                  for i, d in dropped]
    else:
        lines.append("none")
    lines += ["", "## Metric definitions", ""] + [f"- {d}" for d in DEFINITIONS]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines) + "\n"


# ---- running ------------------------------------------------------------------------------


def _progress(rec: dict) -> str:
    m = rec["metrics"]
    if rec["status"] == "error":
        return f"{rec['id']} ERROR {rec['error']['category']}: {rec['error']['message'][:120]}"
    recall = m["cited_path_recall"]
    return (f"{rec['id']} {rec['status']:<21} reason={rec['insufficient_reason'] or '-':<26} "
            f"cited={m['n_citations']} recall={'-' if recall is None else f'{recall:.2f}'} "
            f"next_steps={m['next_steps_count']} gen_ms={rec['generation_ms']}")


def run_answer_eval(questions: Sequence[dict], *, retriever: Retriever, provider: LLMProvider,
                    run_dir: Path, meta: dict, k: int, min_similarity: float,
                    pause_s: float = 4.0, previous: Sequence[dict] = (),
                    max_consecutive_errors: int = 5, sleep: Callable[[float], None] = time.sleep,
                    log: Callable[[str], None] = print,
                    clarifier: Clarifier | None = None,
                    near_miss: NearMissResolver | None = None) -> list[dict]:
    """Evaluate every question without a successful record in `previous`; return the current
    record per question, in question order. `meta["state"]` ends as complete,
    stopped_after_errors, aborted or interrupted, and the files always reflect it."""
    latest = {r["id"]: r for r in previous}
    order = [q["external_id"] for q in questions]

    def current() -> list[dict]:
        return [latest[i] for i in order if i in latest]

    state, consecutive, called_llm = "complete", 0, False
    try:
        for q in questions:
            prev = latest.get(q["external_id"])
            if prev is not None and prev["status"] != "error":
                continue
            if called_llm and pause_s > 0:  # space out real requests only
                sleep(pause_s)
            rec = evaluate_question(q, retriever=retriever, provider=provider, k=k,
                                    min_similarity=min_similarity, clarifier=clarifier,
                                    near_miss=near_miss)
            append_record(run_dir, rec)
            latest[rec["id"]] = rec
            called_llm = rec["generation_ms"] is not None
            log(_progress(rec))
            consecutive = consecutive + 1 if rec["status"] == "error" else 0
            write_outputs(run_dir, current(), meta)
            if max_consecutive_errors and consecutive >= max_consecutive_errors:
                state = "stopped_after_errors"
                log(f"Stopped after {consecutive} consecutive failures. Fix the cause, then "
                    f"continue with --resume {run_dir}")
                break
    except KeyboardInterrupt:
        state = "interrupted"
        raise
    except ABORT_ERRORS as exc:
        state = "aborted"
        meta["abort_reason"] = f"{type(exc).__name__}: {redact(str(exc))[:300]}"
        raise
    finally:
        meta["state"], meta["finished_utc"] = state, _now()
        write_outputs(run_dir, current(), meta)
    return current()


# ---- metadata and selection ---------------------------------------------------------------


def config_fingerprint(meta: dict) -> str:
    key = json.dumps({k: meta.get(k) for k in FINGERPRINT_KEYS}, sort_keys=True, default=str)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_metadata(*, run_id: str, settings: Settings, provider: LLMProvider,
                   questions_path: Path, selection: dict, pause_s: float, index: dict) -> dict:
    """Everything needed to reproduce or compare the run. No key, password or DSN."""
    manifest = Path(settings.data_dir) / "manifest.json"
    meta = {
        "kind": KIND, "run_id": run_id, "started_utc": _now(), "finished_utc": None,
        "state": "running", "python": platform.python_version(), "platform": platform.platform(),
        "model": provider.model, "llm_timeout_seconds": settings.llm_timeout_seconds,
        "llm_max_retries": settings.llm_max_retries, "retriever": settings.retriever,
        "ask_top_k": settings.ask_top_k, "min_similarity": settings.min_similarity,
        "embedding_model": settings.embedding_model, "embedding_dim": settings.embedding_dim,
        # prompt.py holds the system prompt, the evidence template and the answer schema.
        "prompt_module_sha256": file_sha256(prompt_module.__file__),
        "questions_path": str(questions_path),
        "seed_questions_sha256": file_sha256(questions_path),
        "data_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "index": index, "corpus_sha256": index.get("corpus_sha256"),
        "selection": selection, "pause_seconds": pause_s,
    }
    meta["config_fingerprint"] = config_fingerprint(meta)
    return meta


def select_questions(records: Sequence[dict], ids: Sequence[str] | None = None,
                     limit: int | None = None) -> list[dict]:
    """Questions in seed order, optionally only `ids` (external_id) and at most `limit`."""
    chosen = list(records)
    if ids:
        wanted = {i.strip().upper() for i in ids if i.strip()}
        unknown = wanted - {q["external_id"] for q in records}
        if unknown:
            raise ValueError(f"unknown question ids: {', '.join(sorted(unknown))}")
        chosen = [q for q in chosen if q["external_id"] in wanted]
    return chosen[:limit] if limit is not None else chosen


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _non_negative_float(value: str) -> float:
    x = float(value)
    if x < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.evaluation.answer_eval",
                                     description="Real-LLM answer evaluation of the seed set.")
    parser.add_argument("--questions", default="eval/questions.seed.jsonl")
    parser.add_argument("--ids", default=None, help="comma-separated external ids, e.g. Q001,Q045")
    parser.add_argument("--limit", type=_positive_int, default=None,
                        help="evaluate at most N questions (after --ids)")
    parser.add_argument("--pause", type=_non_negative_float, default=4.0,
                        help="seconds between real LLM calls (default 4)")
    parser.add_argument("--max-consecutive-errors", type=int, default=5,
                        help="stop after this many failures in a row (0 = never)")
    parser.add_argument("--out-root", default="eval/results",
                        help="parent folder of the new run folder")
    parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                        help="continue an earlier answer-eval run in its own folder")
    args = parser.parse_args(argv)

    settings = get_settings()
    questions_path = Path(args.questions)
    records, problems = load_questions(questions_path)
    if problems:
        print(f"{questions_path}: {len(problems)} malformed record(s), e.g. {problems[0]}")
        return 2

    previous: list[dict] = []
    old_meta: dict | None = None
    if args.resume:
        run_dir = Path(args.resume)
        meta_path = run_dir / METADATA
        old_meta = json.loads(meta_path.read_text("utf-8")) if meta_path.is_file() else None
        if not old_meta or old_meta.get("kind") != KIND:
            print(f"{run_dir} is not an answer-eval run folder; refusing to write into it.")
            return 2
        if args.ids or args.limit:
            print("--resume reuses the run's own question selection; drop --ids/--limit.")
            return 2
        wanted = old_meta["selection"]["question_ids"]
        selected = select_questions(records, wanted)
        selection = old_meta["selection"]
    else:
        try:
            selected = select_questions(records, args.ids.split(",") if args.ids else None,
                                        args.limit)
        except ValueError as exc:
            print(exc)
            return 2
        selection = {"ids": args.ids, "limit": args.limit,
                     "question_ids": [q["external_id"] for q in selected]}
    if not selected:
        print("No questions selected.")
        return 2

    try:
        provider = get_llm_provider(settings)
    except LLMNotConfiguredError:
        print(SETUP_INSTRUCTIONS)
        return 2
    embedder = get_embedder("fastembed", settings)
    retriever = get_retriever(settings.retriever, embedder)
    index = index_fingerprint()

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = build_metadata(run_id=run_id, settings=settings, provider=provider,
                          questions_path=questions_path, selection=selection,
                          pause_s=args.pause, index=index)
    if old_meta is not None:
        if old_meta.get("config_fingerprint") != meta["config_fingerprint"]:
            changed = [k for k in FINGERPRINT_KEYS if old_meta.get(k) != meta.get(k)]
            print(f"Refusing to resume: {', '.join(changed)} changed since the original run. "
                  "Start a new run instead.")
            return 2
        meta |= {"run_id": old_meta["run_id"], "started_utc": old_meta["started_utc"],
                 "resumed_utc": [*old_meta.get("resumed_utc", []), _now()]}
        previous = load_records(run_dir)
    else:
        run_dir = Path(args.out_root) / f"{run_id}_answers"
        run_dir.mkdir(parents=True, exist_ok=False)  # never reuse an existing folder

    print(f"model={meta['model']} retriever={meta['retriever']} top_k={meta['ask_top_k']} "
          f"min_similarity={meta['min_similarity']} questions={len(selected)} "
          f"pause={args.pause}s -> {run_dir}", flush=True)
    try:
        final = run_answer_eval(selected, retriever=retriever, provider=provider,
                                run_dir=run_dir, meta=meta, k=settings.ask_top_k,
                                min_similarity=settings.min_similarity, pause_s=args.pause,
                                previous=previous,
                                max_consecutive_errors=args.max_consecutive_errors,
                                log=lambda s: print(s, flush=True),
                                # As /ask does: vague questions get a clarification.
                                clarifier=make_clarifier(
                                    retriever, min_similarity=settings.clarify_min_similarity,
                                    max_candidates=settings.clarify_max_candidates),
                                near_miss=make_near_miss_resolver(
                                    retriever, min_similarity=settings.clarify_min_similarity,
                                    max_candidates=settings.clarify_max_candidates))
    except KeyboardInterrupt:
        print(f"Interrupted. Finished questions are saved; continue with --resume {run_dir}")
        return 130
    except ABORT_ERRORS as exc:
        print(f"Aborted: {meta.get('abort_reason', type(exc).__name__)}. Finished questions "
              f"are saved in {run_dir}")
        return 1
    failed = [r["id"] for r in final if r["status"] == "error"]
    print(f"{len(final)} question(s) recorded in {run_dir}; failed: {', '.join(failed) or 'none'}")
    return 0 if meta["state"] == "complete" and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
