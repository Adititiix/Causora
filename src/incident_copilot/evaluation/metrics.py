"""Evaluation metrics. Pure functions, no I/O, so they are unit-tested on hand-made inputs.

Retrieval metrics work on DOCUMENT paths. A retriever returns chunks; a document counts as
retrieved at rank r if any of its chunks is at rank r (its first occurrence). With
E = expected_paths and top-K = the first K chunks:

  hit@K      1 if at least one document of E is in the top-K, else 0   ("any expected")
  recall@K   |E ∩ top-K documents| / |E|
  all@K      1 if every document of E is in the top-K, else 0          ("all required")
  RR         1 / rank of the first expected document within the retrieved depth, else 0
  MRR        mean RR over answerable questions

Unanswerable questions (E empty) have no retrieval metrics; they are evaluated as refusals.

Refusal is a binary decision whose POSITIVE class is "refuse":
  TP unanswerable refused   FN unanswerable answered (missed refusal)
  FP answerable refused     TN answerable answered
"""

import math
import re
import statistics
from collections.abc import Iterable, Sequence

from incident_copilot.chunking.logs import PATH_RE

DEFAULT_KS = (1, 3, 5, 8)


# ---- retrieval --------------------------------------------------------------------------


def first_ranks(retrieved_paths: Sequence[str]) -> dict[str, int]:
    """Document path -> 1-based rank of its first chunk."""
    ranks: dict[str, int] = {}
    for i, path in enumerate(retrieved_paths, start=1):
        ranks.setdefault(path, i)
    return ranks


def retrieval_metrics(expected_paths: Iterable[str], retrieved_paths: Sequence[str],
                      ks: Sequence[int] = DEFAULT_KS) -> dict:
    expected = list(dict.fromkeys(expected_paths))
    if not expected:
        return {}
    ranks = first_ranks(retrieved_paths)
    found = sorted(ranks[p] for p in expected if p in ranks)
    out: dict = {"first_relevant_rank": found[0] if found else None,
                 "rr": 1.0 / found[0] if found else 0.0}
    for k in ks:
        n_in = sum(r <= k for r in found)
        out[f"hit@{k}"] = int(n_in > 0)
        out[f"recall@{k}"] = n_in / len(expected)
        out[f"all@{k}"] = int(n_in == len(expected))
    return out


# ---- latency ----------------------------------------------------------------------------


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of already sorted values (same as numpy's default)."""
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def latency_summary(values_ms: Iterable[float], min_n_p95: int = 20,
                    min_n_p99: int = 100) -> dict:
    """mean/median always; p95 and p99 only when the sample is large enough to mean
    something (otherwise None, with the reason in `note`)."""
    xs = sorted(float(v) for v in values_ms)
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p95": None, "p99": None,
                "note": "no samples"}
    notes = []
    p95 = _quantile(xs, 0.95) if len(xs) >= min_n_p95 else None
    p99 = _quantile(xs, 0.99) if len(xs) >= min_n_p99 else None
    if p95 is None:
        notes.append(f"p95 needs >= {min_n_p95} samples")
    if p99 is None:
        notes.append(f"p99 needs >= {min_n_p99} samples")
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
            "p95": p95, "p99": p99, "note": "; ".join(notes)}


# ---- refusal ----------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def refusal_confusion(should_abstain: Sequence[bool], refused: Sequence[bool]) -> dict:
    pairs = list(zip(should_abstain, refused, strict=True))
    tp = sum(a and r for a, r in pairs)
    fn = sum(a and not r for a, r in pairs)
    fp = sum(not a and r for a, r in pairs)
    tn = sum(not a and not r for a, r in pairs)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "specificity": _ratio(tn, tn + fp), "f1": f1,
            "false_refusal_rate": _ratio(fp, fp + tn),   # answerable questions refused
            "missed_refusal_rate": _ratio(fn, fn + tp)}  # unanswerable questions answered


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; honest for small n (unlike +/- 1.96*se)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def gate_refuses(best_similarity: float | None, threshold: float) -> bool:
    """Mirror of the pre-LLM gate in generation/answer.py: refuse unless best >= threshold."""
    return best_similarity is None or best_similarity < threshold


def threshold_sweep(best_similarity: Sequence[float | None], should_abstain: Sequence[bool],
                    thresholds: Iterable[float]) -> list[dict]:
    rows = []
    for t in thresholds:
        refused = [gate_refuses(s, t) for s in best_similarity]
        rows.append({"threshold": t, **refusal_confusion(should_abstain, refused)})
    return rows


def midpoint_threshold(best_similarity: Sequence[float], should_abstain: Sequence[bool]) -> float:
    """The calibration rule used for MIN_SIMILARITY: midpoint between the highest
    unanswerable score that sits below every answerable score and the lowest answerable
    score. Falls back to the lowest answerable score if no unanswerable one is below it."""
    answerable = [s for s, a in zip(best_similarity, should_abstain, strict=True) if not a]
    lowest = min(answerable)
    below = [s for s, a in zip(best_similarity, should_abstain, strict=True) if a and s < lowest]
    return (max(below) + lowest) / 2 if below else lowest


def leave_one_out_refusal(best_similarity: Sequence[float], should_abstain: Sequence[bool]
                          ) -> list[dict]:
    """Exploratory: choose the threshold with `midpoint_threshold` WITHOUT question i, then
    apply it to question i. Estimates how the calibration rule behaves on an unseen
    question; with 6 unanswerable questions it is still a very small sample."""
    out = []
    for i in range(len(best_similarity)):
        rest_s = [s for j, s in enumerate(best_similarity) if j != i]
        rest_a = [a for j, a in enumerate(should_abstain) if j != i]
        t = midpoint_threshold(rest_s, rest_a)
        out.append({"index": i, "threshold": t,
                    "refused": gate_refuses(best_similarity[i], t),
                    "should_abstain": should_abstain[i]})
    return out


def roc_points(best_similarity: Sequence[float], should_abstain: Sequence[bool]) -> list[dict]:
    """ROC for "refuse" (score = -similarity: lower similarity means refuse). One point per
    distinct threshold, plus the (0,0) and (1,1) ends; AUC by the trapezoid rule."""
    thresholds = sorted({s for s in best_similarity}) + [math.inf]
    points = []
    for t in thresholds:
        refused = [gate_refuses(s, t) for s in best_similarity]
        c = refusal_confusion(should_abstain, refused)
        points.append({"threshold": t, "fpr": c["false_refusal_rate"] or 0.0,
                       "tpr": c["recall"] or 0.0})
    points.append({"threshold": -math.inf, "fpr": 0.0, "tpr": 0.0})
    return sorted(points, key=lambda p: (p["fpr"], p["tpr"]))


def auc(points: Sequence[dict]) -> float:
    return sum((b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2
               for a, b in zip(points, points[1:], strict=False))


# ---- failure classification -------------------------------------------------------------

FAILURE_LABELS = {
    "wrong_task_date": "a log of the same pipeline and task, but another date, was retrieved "
                       "instead of the expected log",
    "missing_runbook": "the expected log was retrieved but the expected runbook was not",
    "incorrect_error_link": "error_link added a runbook other than the expected one",
    "outside_top_k": "an expected document exists in the index but ranks below K",
    "not_retrieved": "an expected document was not found even in the deep search",
    "gate_false_refusal": "answerable question refused by the similarity gate",
    "gate_missed_refusal": "unanswerable question passed the similarity gate",
}


def parse_log_path(path: str) -> dict | None:
    m = PATH_RE.search(path)
    return {"pipeline": m["pipeline"], "date": m["date"], "task": m["task"]} if m else None


def classify_failure(*, expected_paths: Sequence[str], retrieved_paths: Sequence[str],
                     retrieved_origins: Sequence[str], deep_ranks: dict[str, int],
                     k: int, should_abstain: bool, refused: bool) -> list[str]:
    """Labels (possibly several) for why a question failed at K, each backed by the
    retrieved paths. Empty list = no failure."""
    if should_abstain:
        return [] if refused else ["gate_missed_refusal"]
    labels: list[str] = []
    if refused:
        labels.append("gate_false_refusal")
    top = retrieved_paths[:k]
    top_set = set(top)
    missing = [p for p in expected_paths if p not in top_set]
    if not missing:
        return labels
    expected_logs = [p for p in expected_paths if p.startswith(("logs/", "extra/logs/"))]
    expected_runbooks = [p for p in expected_paths if "runbooks/" in p]
    for log in (p for p in expected_logs if p in missing):
        want = parse_log_path(log)
        same_task = [p for p in top if (got := parse_log_path(p)) and want
                     and (got["pipeline"], got["task"]) == (want["pipeline"], want["task"])
                     and got["date"] != want["date"]]
        if same_task:
            labels.append("wrong_task_date")
    if expected_logs and all(p in top_set for p in expected_logs) and \
            any(p in missing for p in expected_runbooks):
        labels.append("missing_runbook")
    linked = [p for p, o in zip(top, retrieved_origins[:k], strict=False) if o == "error_link"]
    if linked and expected_runbooks and not set(linked) & set(expected_runbooks):
        labels.append("incorrect_error_link")
    for p in missing:
        rank = deep_ranks.get(p)
        labels.append("outside_top_k" if rank is not None else "not_retrieved")
    return list(dict.fromkeys(labels))


def short_date(path: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    return m.group(1) if m else None
