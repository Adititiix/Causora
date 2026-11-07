"""Answer-quality data: load logged /ask results and score them deterministically.

Only rows produced by a real LLM count as model quality. Rows whose model is the scripted
FakeLLMProvider (or empty: gate refusals and retrieval-only calls, where no model ran) are
separated by `split_by_provider`, never mixed into quality numbers.

Deterministic scores (no judge model):
  citation_validity      every cited id was a retrieved id (server-side validation drops
                         the others, so this also reports how many claims were dropped)
  cited_path_precision   cited documents that are in expected_paths / cited documents
  cited_path_recall      expected documents that were cited / expected documents
  abstention_correct     refused == should_abstain
Groundedness / faithfulness need a judge (human or LLM); they are not computed here.

Ground-truth term checks (`answer_text_metrics`) test whether TERMS from the seed ground truth
appear in an answer. They are surface heuristics: a correct paraphrase can miss them and a
wrong answer can contain them. They are never reported as correctness or groundedness.
"""

import re
from collections.abc import Sequence

from incident_copilot.db import connect

FAKE_MODEL_PREFIX = "fake-llm"


def load_logged_answers() -> list[dict]:
    """Every /ask logged in `queries`, with its citations' document paths."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT q.id::text, q.question, q.answer, q.abstained, q.model, q.latency_ms,
                   q.retrieval_ms, q.generation_ms, q.tokens_in, q.tokens_out, q.cost_usd,
                   q.metadata, q.created_at,
                   array_agg(r.evidence_id ORDER BY r.rank) FILTER (WHERE r.id IS NOT NULL),
                   array_agg(r.document_path ORDER BY r.rank) FILTER (WHERE r.id IS NOT NULL)
            FROM queries q LEFT JOIN retrievals r ON r.query_id = q.id
            GROUP BY q.id ORDER BY q.created_at
            """
        ).fetchall()
    keys = ["query_id", "question", "answer", "abstained", "model", "latency_ms",
            "retrieval_ms", "generation_ms", "tokens_in", "tokens_out", "cost_usd", "metadata",
            "created_at", "evidence_ids", "evidence_paths"]
    return [dict(zip(keys, r, strict=True)) for r in rows]


def provider_kind(model: str | None) -> str:
    if not model:
        return "no_llm_call"
    if model.startswith(FAKE_MODEL_PREFIX):
        return "fake_scripted"
    return "real_llm"


def split_by_provider(records: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"real_llm": [], "fake_scripted": [], "no_llm_call": []}
    for r in records:
        out[provider_kind(r.get("model"))].append(r)
    return out


def score_answer(record: dict, expected_paths: Sequence[str], should_abstain: bool) -> dict:
    """Deterministic scores for one logged answer (see module docstring)."""
    meta = record.get("metadata") or {}
    cited_ids = (meta.get("response") or {}).get("citations") or []
    id_to_path = dict(zip(record.get("evidence_ids") or [], record.get("evidence_paths") or [],
                          strict=False))
    cited_paths = sorted({id_to_path[i] for i in cited_ids if i in id_to_path})
    validation = meta.get("validation") or {}
    expected = set(expected_paths)
    refused = bool(record.get("abstained"))
    return {
        "query_id": record.get("query_id"),
        "provider": provider_kind(record.get("model")),
        "refused": refused,
        "abstention_correct": refused == bool(should_abstain),
        "n_citations": len(cited_ids),
        "citation_validity": all(i in id_to_path for i in cited_ids),
        "dropped_claims": (validation.get("dropped_evidence_claims", 0)
                           + validation.get("dropped_inference_claims", 0)),
        "cited_paths": cited_paths,
        "cited_path_precision": (len(set(cited_paths) & expected) / len(cited_paths)
                                 if cited_paths else None),
        "cited_path_recall": (len(set(cited_paths) & expected) / len(expected)
                              if expected and not refused else None),
        "groundedness": None,  # needs a judge; not measured
    }


# ---- ground-truth term checks (heuristics, see module docstring) --------------------------
#
# The seed ground truth comes from fixed templates (scripts/generate_synthetic_data.py,
# build_eval), so its parts can be recovered exactly:
#   log_diagnosis  "It failed with: <error>. Root cause: <root cause> Fix: <remediation>"
#   runbook        "<remediation>"
#   incident       "<INC-id> on <date>[; it lasted <n> minutes]. <remediation>"
#   schema         "<fact>"
#   unanswerable   "NOT_ANSWERABLE"

_LOG_GT = re.compile(
    r"^It failed with: (?P<error>.*)\. Root cause: (?P<root_cause>.*?) Fix: (?P<remediation>.*)$",
    re.S)
_INCIDENT_GT = re.compile(r"^(?P<fact>(?P<incident_id>INC-\d+) on [^.]*)\. (?P<remediation>.*)$",
                          re.S)
_TOKEN = re.compile(r"[a-z0-9_]+(?:[-'][a-z0-9_]+)*")
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how if in into is
it its it's of on or our so such than that the their them then there these this those to too
was we were what when where which while who why will with would you your should not no only
also any all each other once after before until via per out up off over under about again
""".split())
# An action counts as covered when at least this share of its terms appear in the answer.
REMEDIATION_COVERAGE_THRESHOLD = 0.5

TEXT_METRICS = ("error_class", "error_class_mentioned", "error_term_recall",
                "root_cause_term_recall", "remediation_actions", "remediation_covered_next_steps",
                "remediation_covered_any", "fact_term_recall", "incident_id_mentioned")


def parse_ground_truth(category: str, text: str) -> dict:
    """The template parts of one ground-truth string; parts a category lacks are None."""
    out = dict.fromkeys(("error", "root_cause", "remediation", "fact", "incident_id"))
    match = None
    if category == "log_diagnosis":
        match = _LOG_GT.match(text)
    elif category == "incident":
        match = _INCIDENT_GT.match(text)
    elif category == "runbook":
        out["remediation"] = text.strip()
    elif category == "schema":
        out["fact"] = text.strip()
    if match:
        out.update({k: v.strip() for k, v in match.groupdict().items() if v})
    return out


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> set[str]:
    """Lower-cased, crudely stemmed words without stopwords. Identifiers such as
    customer_tier or 2026-08-28 stay whole; numbers of any length are kept."""
    terms = set()
    for tok in _TOKEN.findall((text or "").lower()):
        if tok in _STOPWORDS or (len(tok) < 3 and not any(c.isdigit() for c in tok)):
            continue
        terms.add(_stem(tok))
    return terms


def term_recall(reference: str, candidate: str, exclude: str = "") -> float | None:
    """Share of the reference's terms (minus terms already in `exclude`, usually the
    question) that occur in the candidate. None when nothing is left to look for."""
    wanted = content_terms(reference) - content_terms(exclude)
    if not wanted:
        return None
    return len(wanted & content_terms(candidate)) / len(wanted)


def mentions(term: str, text: str) -> bool:
    """Case-insensitive whole-token match (KeyError is not found inside MyKeyErrorX)."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text or "",
                     re.I) is not None


def error_class(error: str) -> str | None:
    """The exception class of an error line: `psycopg2.errors.UniqueViolation: ...` ->
    UniqueViolation. Lines without one (`Pod ... terminated: reason=OOMKilled ...`) fall back
    to the first CamelCase / acronym token (OOMKilled)."""
    head = error.split(":", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z_][\w.]*", head):
        return head.rsplit(".", 1)[-1]
    match = re.search(r"\b(?:[A-Z]{2,}[a-z]|[A-Z][a-z]+[A-Z])\w*", error)
    return match.group(0) if match else None


def remediation_actions(remediation: str, exclude: str = "") -> list[str]:
    """Split a fix into actions at commas, semicolons and sentence ends; actions whose terms
    all occur in `exclude` (the question) are dropped, as they test nothing."""
    parts = (p.strip() for p in re.split(r"[;,]|\.\s", remediation))
    parts = (re.sub(r"^(?:then|and|or)\s+", "", p, flags=re.I) for p in parts)
    return [p for p in parts if content_terms(p) - content_terms(exclude)]


def action_coverage(actions: Sequence[str], candidate: str, exclude: str = "") -> float | None:
    if not actions:
        return None
    covered = sum((term_recall(a, candidate, exclude) or 0.0) >= REMEDIATION_COVERAGE_THRESHOLD
                  for a in actions)
    return covered / len(actions)


def answer_text_metrics(*, category: str, question: str, ground_truth: str, summary: str | None,
                        claims: Sequence[str], next_steps: Sequence[str]) -> dict:
    """Ground-truth term checks for one answer (keys in TEXT_METRICS; None = not applicable).

    error_class_mentioned          log_diagnosis: the exception class appears in the answer
    error_term_recall              log_diagnosis: terms of the error line found in the answer
    root_cause_term_recall         log_diagnosis: terms of the root cause found in the answer
    remediation_covered_next_steps actions of the fix covered by next_steps alone
    remediation_covered_any        ... covered anywhere in the answer
    fact_term_recall               schema / incident: terms of the stated fact found
    incident_id_mentioned          incident: the INC id appears in the answer
    Terms already in the question are excluded, so repeating the question scores nothing."""
    gt = parse_ground_truth(category, ground_truth)
    everything = "\n".join([summary or "", *claims, *next_steps])
    steps = "\n".join(next_steps)
    out: dict = dict.fromkeys(TEXT_METRICS)
    if gt["error"]:
        cls = error_class(gt["error"])
        out["error_class"] = cls
        out["error_class_mentioned"] = mentions(cls, everything) if cls else None
        out["error_term_recall"] = term_recall(gt["error"], everything, question)
    if gt["root_cause"]:
        out["root_cause_term_recall"] = term_recall(gt["root_cause"], everything, question)
    if gt["remediation"]:
        actions = remediation_actions(gt["remediation"], question)
        out["remediation_actions"] = len(actions)
        out["remediation_covered_next_steps"] = action_coverage(actions, steps, question)
        out["remediation_covered_any"] = action_coverage(actions, everything, question)
    if gt["fact"]:
        out["fact_term_recall"] = term_recall(gt["fact"], everything, question)
    if gt["incident_id"]:
        out["incident_id_mentioned"] = mentions(gt["incident_id"], everything)
    return out


EVIDENCE_METRICS = ("error_term_recall_in_evidence", "root_cause_term_recall_in_evidence",
                    "remediation_covered_in_evidence", "fact_term_recall_in_evidence")


def evidence_text_metrics(*, category: str, question: str, ground_truth: str,
                          evidence: str) -> dict:
    """The same term checks run on the retrieved evidence text instead of the answer (keys in
    EVIDENCE_METRICS). A rough reference, still a heuristic: a low value means the ground-truth
    terms were mostly absent from what the model was shown, so a low answer score is then a
    retrieval limit rather than an omission. Unrelated chunks can share words with the ground
    truth, so a non-zero value does not prove the fact was in the evidence."""
    gt = parse_ground_truth(category, ground_truth)
    out: dict = dict.fromkeys(EVIDENCE_METRICS)
    if gt["error"]:
        out["error_term_recall_in_evidence"] = term_recall(gt["error"], evidence, question)
    if gt["root_cause"]:
        out["root_cause_term_recall_in_evidence"] = term_recall(gt["root_cause"], evidence,
                                                                question)
    if gt["remediation"]:
        out["remediation_covered_in_evidence"] = action_coverage(
            remediation_actions(gt["remediation"], question), evidence, question)
    if gt["fact"]:
        out["fact_term_recall_in_evidence"] = term_recall(gt["fact"], evidence, question)
    return out


# ---- source qualification (does the answer keep the uncertainty its sources state?) -------

QUALIFICATION_METRICS = ("cites_qualified_source", "qualification_kept")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
# "Confidence: medium. ..." / "Confidence: high that ...; medium that ..." (to end of line).
_STATED_CONFIDENCE_LINE = re.compile(r"(?im)\bconfidence:([^\n]*)")
_LEVEL = re.compile(r"(?i)\b(low|medium|high)\b")
_OPEN_ALTERNATIVE = re.compile(r"(?i)\bnot\s+(?:been\s+)?ruled\s+out\b")
# Wording that keeps a claim qualified in an answer.
_ANSWER_QUALIFIER = re.compile(
    r"(?i)\b(?:low|medium)[- ]confidence\b|\bconfidence\s+(?:is\s+|was\s+)?(?:low|medium)\b"
    r"|\bnot\s+(?:been\s+)?(?:ruled\s+out|confirmed|proven|established)\b|\bunconfirmed\b"
    r"|\binfer(?:s|red)?\b|\blikely\b|\bpossibl[ey]\b|\bmay\b|\bmight\b|\bsuggests?\b"
    r"|\buncertain\b|\bnot\s+certain\b")


# A sentence that states a cause.
_CAUSAL = re.compile(
    r"(?i)\b(?:caus\w*|because|due\s+to|led\s+to|leads?\s+to|result(?:ed|s)?\s+(?:in|from)"
    r"|triggered|root\s+cause)\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def first_cause_sentence(text: str | None) -> str | None:
    """The first sentence of `text` that states a cause, or None."""
    return next((s for s in _SENTENCE_END.split(text or "") if _CAUSAL.search(s)), None)


def stated_confidence(text: str) -> str | None:
    """The highest confidence a chunk states on its "Confidence:" lines, or None."""
    levels = [lv.lower() for line in _STATED_CONFIDENCE_LINE.findall(text or "")
              for lv in _LEVEL.findall(line)]
    return max(levels, key=CONFIDENCE_ORDER.__getitem__) if levels else None


def is_qualified_source(text: str) -> bool:
    """A chunk whose own conclusion is uncertain: every confidence it states is below high,
    or it leaves an alternative explanation open ("not ruled out")."""
    level = stated_confidence(text)
    return (level is not None and level != "high") or bool(_OPEN_ALTERNATIVE.search(text or ""))


def qualification_metrics(*, cited: dict[str, str], summary: str | None, claims: Sequence[str],
                          inferences: Sequence[dict]) -> dict:
    """Keys in QUALIFICATION_METRICS. `cited` maps a cited evidence id to its chunk text.

    qualification_kept is None when no cited chunk is qualified; otherwise True only if the
    answer (summary and claims) carries qualifying wording, the summary's first sentence that
    states a cause carries it too (a later caveat does not qualify an opening claim), AND no
    inference based on a qualified chunk claims more confidence than that chunk states. A
    surface check, like the term checks above."""
    qualified = {i: t for i, t in cited.items() if is_qualified_source(t)}
    out: dict = {"cites_qualified_source": bool(qualified), "qualification_kept": None}
    if not qualified:
        return out
    text = " ".join([summary or "", *claims])
    upgraded = False
    for inf in inferences:
        cap = [stated_confidence(qualified[i]) for i in inf.get("based_on") or ()
               if i in qualified]
        cap = [c for c in cap if c is not None]
        if cap and CONFIDENCE_ORDER.get(str(inf.get("confidence")), 0) > \
                min(CONFIDENCE_ORDER[c] for c in cap):
            upgraded = True
    opening = first_cause_sentence(summary)
    opening_ok = opening is None or bool(_ANSWER_QUALIFIER.search(opening))
    out["qualification_kept"] = (bool(_ANSWER_QUALIFIER.search(text)) and opening_ok
                                 and not upgraded)
    return out
