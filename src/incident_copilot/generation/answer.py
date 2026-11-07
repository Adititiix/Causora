"""Retrieve -> gate -> generate -> validate -> cite. No HTTP and no database writes here.

Guarantees enforced in code, not left to the model:
  * No hits, or best similarity < MIN_SIMILARITY: insufficient evidence, zero LLM calls.
  * Every cited id must be one of the retrieved evidence ids; a claim citing any unknown id
    is dropped. If no valid evidence claim remains, the answer is insufficient evidence.
  * Citations are built from chunk metadata (path, section, timestamps), never from model text.

For diagnosis only (never returned as an answer by /ask): `refusal_note` keeps what the model
said was missing when it refused, and `dropped_claims` keeps the claims validation removed.

Follow-ups (`contextualization`, retrieval/contextualize.py): retrieval searches with the
resolved standalone query and the gate judges that query's evidence like any other; the model
answers the user's own question, with the resolved subject as a note that is not evidence. An
ambiguous reference is answered with a clarifying question: no retrieval, no LLM call.

Vague questions (`clarifier`, generation/clarify.py): a standalone question whose evidence
points at several distinct failures gets those failures as `clarification` (insufficient_reason
"ambiguous_query", no LLM call) so the user can pick one: when the gate refuses it, or when it
passes but asks about one particular failure (so one answer cannot mix several failures). It is
still not an answer: nothing is answered until a later request passes the gate.

A picked failure (`scope`): evidence about OTHER failures is removed right after retrieval
(clarify.scope_hits), before the gate, so it can neither be cited nor lift the gate.

Next steps cite evidence ids like claims; a step citing an id that was not retrieved is dropped,
a step citing none is returned with no ids (shown as unsupported). Each kept step also gets,
from its cited chunks and not from the model, a `support`: "incident_report" (an incident
report about the failure states it), "general_guidance" (only runbooks, schemas or logs: not
documented as this failure's fix) or "unsupported" (cites nothing); and a `kind`, "recovery"
or "prevention": the cited sections decide when they are all of one kind, else the model's.

Near misses (`near_miss`, clarify.resolve_near_miss): a question the gate refused may still be
asked back (several indexed failures match its topic) or answered about the ONE indexed
failure it matches, with the evidence scoped to that failure; the gate record keeps the real
similarity and says `accepted_by`. The selected failure (picked, kept or the one matched)
gets its own incident report's resolution and prevention sections as extra evidence after the
gate (`remediation`, clarify.remediation_hits), so its documented fix is not crowded out by a
generic runbook.
"""

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from pydantic import ValidationError

from incident_copilot.generation.clarify import (
    CLARIFICATION_TEXT,
    Clarifier,
    NearMissResolver,
    asks_about_one_failure,
    candidates_from_docs,
)
from incident_copilot.generation.llm import (
    LLMBadResponseError,
    LLMError,
    LLMNotConfiguredError,
    LLMProvider,
)
from incident_copilot.generation.prompt import (
    SYSTEM_PROMPT,
    ModelAnswer,
    build_user_prompt,
    evidence_id,
)
from incident_copilot.retrieval import Filters, Hit, Retriever
from incident_copilot.retrieval.contextualize import (
    AMBIGUOUS,
    NO_CONTEXT,
    STANDALONE,
    Contextualization,
)

INSUFFICIENT_EVIDENCE_TEXT = "Not enough evidence to determine the cause"


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    chunk_id: str
    path: str
    source_type: str
    section: str | None
    chunk_index: int
    ts_start: object
    ts_end: object

    @classmethod
    def from_hit(cls, hit: Hit) -> "Citation":
        return cls(evidence_id(hit), hit.chunk_id, hit.path, hit.source_type, hit.section,
                   hit.chunk_index, hit.ts_start, hit.ts_end)


@dataclass
class AskOutcome:
    question: str
    hits: list[Hit]
    status: str  # retrieval_only | answered | insufficient_evidence
    answer: str | None = None
    insufficient_evidence: bool | None = None
    insufficient_reason: str | None = None
    evidence: list[dict] = field(default_factory=list)
    inference: list[dict] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    retrieval_ms: int = 0
    generation_ms: int | None = None
    latency_ms: int = 0
    gate: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    # What the retriever decided beyond ranking, e.g. whether a date filter was applied.
    retrieval_info: dict = field(default_factory=dict)
    # The model's own "summary" when it set insufficient_evidence (prompt rule 4 asks it to say
    # what is missing). Unverified model text: kept for diagnosis, never used as `answer`.
    refusal_note: str | None = None
    # Claims removed by validation: {"field", "claim", "ids", "invalid_ids"} each.
    dropped_claims: list[dict] = field(default_factory=list)
    # Stage latencies and LLM call counts: retrieval_ms, generation_ms, validation_ms,
    # llm_called, llm_attempts (the API adds embedding_ms).
    timings: dict = field(default_factory=dict)
    # For insufficient_reason "ambiguous_query": {"reason", "candidates": [...]} (clarify.py).
    clarification: dict | None = None
    # The evidence ids each next step cites, aligned with next_steps ([] = unsupported).
    next_step_evidence: list[list[str]] = field(default_factory=list)
    # Aligned with next_steps: "recovery" | "prevention", and where the step comes from
    # ("incident_report" | "general_guidance" | "unsupported"), see the module docstring.
    next_step_kinds: list[str] = field(default_factory=list)
    next_step_support: list[str] = field(default_factory=list)
    # What the evidence leaves open (prompt rule 8). Model text, not validated against ids.
    unknowns: list[str] = field(default_factory=list)

    @property
    def used_chunk_ids(self) -> set[str]:
        return {c.chunk_id for c in self.citations}


_ID_RE = re.compile(r"^\[?\s*(E\d+)\s*\]?$", re.IGNORECASE)


def _norm_id(raw: str) -> str:
    m = _ID_RE.match(str(raw).strip())
    return m.group(1).upper() if m else str(raw).strip()


def parse_model_answer(text: str) -> ModelAnswer:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
        for key in ("evidence", "inference", "next_steps", "unknowns"):
            data.setdefault(key, [])
        # A step given as a plain string (older answers, scripted fakes) cites nothing. A step
        # without a kind is "recovery" until its cited sections say otherwise
        # (classify_next_steps).
        if isinstance(data["next_steps"], list):
            data["next_steps"] = [{"step": s, "evidence_ids": []} if isinstance(s, str) else s
                                  for s in data["next_steps"]]
            data["next_steps"] = [{"kind": "recovery", **s} if isinstance(s, dict) else s
                                  for s in data["next_steps"]]
        return ModelAnswer.model_validate(data)
    except (ValueError, ValidationError) as exc:
        raise LLMBadResponseError(
            f"LLM response is not valid answer JSON ({type(exc).__name__})"
        ) from None


def validate_claims(answer: ModelAnswer,
                    valid_ids: set[str]) -> tuple[list, list, dict, list[dict]]:
    """Keep only claims whose every cited id was actually retrieved. Returns the kept evidence
    and inference claims, the counts report, and the dropped claims themselves."""
    evidence, inference, invalid, dropped = [], [], set(), []
    dropped_evidence = dropped_inference = 0
    for claim in answer.evidence:
        ids = [_norm_id(i) for i in claim.evidence_ids]
        bad = [i for i in ids if i not in valid_ids]
        if not ids or bad:
            dropped_evidence += 1
            invalid.update(bad)
            dropped.append({"field": "evidence", "claim": claim.claim, "ids": ids,
                            "invalid_ids": bad})
            continue
        evidence.append({"claim": claim.claim, "evidence_ids": list(dict.fromkeys(ids))})
    for claim in answer.inference:
        ids = [_norm_id(i) for i in claim.based_on]
        bad = [i for i in ids if i not in valid_ids]
        if not ids or bad:
            dropped_inference += 1
            invalid.update(bad)
            dropped.append({"field": "inference", "claim": claim.claim, "ids": ids,
                            "invalid_ids": bad})
            continue
        inference.append({"claim": claim.claim, "based_on": list(dict.fromkeys(ids)),
                          "confidence": claim.confidence})
    report = {"dropped_evidence_claims": dropped_evidence,
              "dropped_inference_claims": dropped_inference,
              "invalid_ids": sorted(invalid)}
    return evidence, inference, report, dropped


def validate_next_steps(answer: ModelAnswer,
                        valid_ids: set[str]) -> tuple[list[str], list[list[str]], list[dict]]:
    """(step texts, the evidence ids of each, dropped steps). A step that cites an id that was
    not retrieved is dropped, like a claim. A step with no id is kept with an empty list: the
    page shows it as unsupported, never as cited."""
    steps, ids_of, dropped = [], [], []
    for item in answer.next_steps:
        text = item.step.strip()
        if not text:
            continue
        ids = list(dict.fromkeys(_norm_id(i) for i in item.evidence_ids))
        bad = [i for i in ids if i not in valid_ids]
        if bad:
            dropped.append({"field": "next_steps", "claim": text, "ids": ids,
                            "invalid_ids": bad})
            continue
        steps.append(text)
        ids_of.append(ids)
    return steps, ids_of, dropped


# Section headings whose steps are of one kind (runbooks and incident reports).
PREVENTION_SECTIONS = frozenset({"prevention", "follow-ups"})
RECOVERY_SECTIONS = frozenset({"resolution", "resolution and verification", "diagnosis"})


def classify_next_steps(answer: ModelAnswer, steps: Sequence[str],
                        ids_of: Sequence[Sequence[str]], by_id: dict[str, Hit],
                        subject_paths: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """(kinds, support) aligned with the validated `steps` (module docstring). `support` is
    decided from the cited chunks only: "incident_report" when one is an incident report (of
    the selected failure, when `subject_paths` names it), else "general_guidance", or
    "unsupported" without ids. The kind follows the cited sections when they are all
    prevention or all recovery sections, else the model's."""
    model_kind = {s.step.strip(): s.kind for s in answer.next_steps}
    subject = set(subject_paths)
    kinds, support = [], []
    for text, ids in zip(steps, ids_of, strict=True):
        cited = [by_id[i] for i in ids if i in by_id]
        sections = {(h.section or "").lower() for h in cited}
        if sections and sections <= PREVENTION_SECTIONS:
            kinds.append("prevention")
        elif sections and sections <= RECOVERY_SECTIONS:
            kinds.append("recovery")
        else:
            kinds.append(model_kind.get(text, "recovery"))
        reports = [h for h in cited if h.source_type == "incident"
                   and (not subject or h.path in subject)]
        support.append("unsupported" if not cited
                       else "incident_report" if reports else "general_guidance")
    return kinds, support


def build_citations(evidence: list[dict], inference: list[dict],
                    by_id: dict[str, Hit]) -> list[Citation]:
    used = {i for c in evidence for i in c["evidence_ids"]}
    used |= {i for c in inference for i in c["based_on"]}
    return [Citation.from_hit(by_id[i]) for i in sorted(used, key=lambda x: int(x[1:]))]


def _ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def answer_question(
    question: str,
    *,
    retriever: Retriever,
    llm_factory: Callable[[], LLMProvider],
    k: int,
    min_similarity: float,
    filters: Filters | None = None,
    retrieval_only: bool = False,
    contextualization: Contextualization | None = None,
    clarifier: Clarifier | None = None,
    scope: Callable[[list[Hit]], list[Hit]] | None = None,
    near_miss: NearMissResolver | None = None,
    remediation: Callable[[Sequence[str], list[Hit]], list[Hit]] | None = None,
    subject_paths: Sequence[str] = (),
) -> AskOutcome:
    """Raises LLMNotConfiguredError (only when an LLM is actually needed) and LLMError.
    Either exception carries the retrieval-so-far as `exc.partial_outcome`, so the caller
    can log a failed request together with the evidence that was retrieved."""
    t0 = time.perf_counter()
    ctx = contextualization
    if ctx is not None and ctx.status == AMBIGUOUS:
        outcome = AskOutcome(question=question, hits=[], status="insufficient_evidence",
                             answer=ctx.clarification(), insufficient_evidence=True,
                             insufficient_reason="ambiguous_reference",
                             gate={"best_similarity": None, "min_similarity": min_similarity,
                                   "passed": False},
                             retrieval_info={"conversation": ctx.as_dict()},
                             timings={"retrieval_ms": 0, "llm_called": False})
        # The same failures as choices, so the user can pick one instead of retyping it.
        choices = candidates_from_docs(ctx.candidate_docs)
        if len(choices) >= 2:
            outcome.clarification = {"reason": "earlier_answer_covered_several",
                                     "candidates": [c.as_dict() for c in choices]}
        outcome.latency_ms = _ms(t0)
        return outcome
    query = ctx.retrieval_query if ctx is not None else question
    hits = retriever.retrieve(query, k, filters)
    if scope is not None:
        # A picked failure: evidence about other failures is left out, order kept.
        hits = [replace(h, rank=i) for i, h in enumerate(scope(hits), start=1)]
    outcome = AskOutcome(question=question, hits=hits, status="retrieval_only",
                         retrieval_ms=_ms(t0),
                         retrieval_info=dict(getattr(retriever, "last_info", None) or {}))
    if ctx is not None:
        outcome.retrieval_info["conversation"] = ctx.as_dict()
    outcome.timings = {"retrieval_ms": outcome.retrieval_ms, "llm_called": False}
    best = max((h.score for h in hits), default=None)
    passed = best is not None and best >= min_similarity
    outcome.gate = {"best_similarity": best, "min_similarity": min_similarity,
                    "passed": passed}

    def insufficient(reason: str) -> AskOutcome:
        outcome.status = "insufficient_evidence"
        outcome.answer = INSUFFICIENT_EVIDENCE_TEXT
        outcome.insufficient_evidence = True
        outcome.insufficient_reason = reason
        outcome.evidence, outcome.inference, outcome.next_steps = [], [], []
        outcome.next_step_evidence = []
        outcome.next_step_kinds, outcome.next_step_support, outcome.unknowns = [], [], []
        outcome.citations = []
        outcome.latency_ms = _ms(t0)
        return outcome

    if retrieval_only:
        outcome.latency_ms = _ms(t0)
        return outcome
    if not hits:
        return insufficient("no_hits")
    # A vague standalone question whose evidence covers several failures: ask which one
    # instead of refusing (gate failed) or answering about all of them at once (gate passed,
    # only for a question about one particular failure; clarify.asks_about_one_failure).
    # Never for a follow-up resolved from history or a picked subject: that is decided.
    undecided = ctx is None or ctx.status in (STANDALONE, NO_CONTEXT)
    can_clarify = clarifier is not None and undecided
    if can_clarify and (not passed or asks_about_one_failure(question)):
        candidates = clarifier(question, hits)
        if candidates:
            insufficient("ambiguous_query")
            outcome.answer = CLARIFICATION_TEXT
            outcome.clarification = {"reason": "several_failures_match",
                                     "candidates": [c.as_dict() for c in candidates]}
            return outcome
    context_note = ctx.context_note if ctx is not None else None
    if not passed:
        # A near miss (module docstring): ask among the indexed failures it matches, or
        # answer about the only one, scoped to it; otherwise the refusal stands.
        decision = near_miss(question, hits) if near_miss is not None and undecided else None
        if decision is not None and decision.candidates:
            insufficient("ambiguous_query")
            outcome.answer = CLARIFICATION_TEXT
            outcome.clarification = {"reason": "several_failures_in_index",
                                     "candidates": [c.as_dict() for c in decision.candidates]}
            return outcome
        if decision is None or decision.resolved is None or not decision.hits:
            return insufficient("below_min_similarity")
        only = decision.resolved
        hits = [replace(h, rank=i) for i, h in enumerate(decision.hits, start=1)]
        outcome.hits = hits
        outcome.gate["accepted_by"] = {"rule": "sole_matching_failure", "candidate": only.id}
        outcome.retrieval_info["conversation"] = {
            **outcome.retrieval_info.get("conversation", {}), "status": "matched_in_index",
            "reason": "the question matches exactly one indexed failure",
            "subjects": [only.label], "subject": only.subject}
        subject_paths = only.paths
        context_note = (f"The question matches exactly one failure in the index: "
                        f"{only.label}. Answer about that failure only.")

    if remediation is not None and subject_paths:
        # After the gate, so these never decide it (clarify.remediation_hits).
        t_r = time.perf_counter()
        extra = remediation(subject_paths, hits)
        if extra:
            hits = hits + [replace(h, rank=len(hits) + i) for i, h in enumerate(extra, 1)]
            outcome.hits = hits
        outcome.timings["remediation_ms"] = _ms(t_r)

    t1 = time.perf_counter()
    try:
        llm = llm_factory()  # raises LLMNotConfiguredError if no key
        outcome.model = getattr(llm, "model", None)
        outcome.timings["llm_called"] = True
        t_p = time.perf_counter()
        prompt = build_user_prompt(question, hits, context_note)
        outcome.timings["prompt_ms"] = _ms(t_p)
        outcome.timings["prompt_chars"] = len(SYSTEM_PROMPT) + len(prompt)
        response = llm.generate(SYSTEM_PROMPT, prompt, ModelAnswer)
        outcome.generation_ms = _ms(t1)
        outcome.model = response.model
        outcome.tokens_in, outcome.tokens_out = response.tokens_in, response.tokens_out
        outcome.timings |= {"generation_ms": outcome.generation_ms,
                            "llm_attempts": getattr(response, "attempts", 1)}
        answer = parse_model_answer(response.text)
    except (LLMError, LLMNotConfiguredError) as exc:
        if outcome.model is not None:
            outcome.generation_ms = _ms(t1)
            outcome.timings |= {"generation_ms": outcome.generation_ms,
                                "llm_attempts": getattr(exc, "attempts", 1)}
        else:
            outcome.timings["llm_called"] = False
        outcome.latency_ms = _ms(t0)
        exc.partial_outcome = outcome
        raise

    t2 = time.perf_counter()
    by_id = {evidence_id(h): h for h in hits}
    evidence, inference, report, dropped = validate_claims(answer, set(by_id))
    outcome.validation = report
    outcome.dropped_claims = dropped
    outcome.timings["validation_ms"] = _ms(t2)
    if answer.insufficient_evidence:
        insufficient("model_insufficient_evidence")
        outcome.refusal_note = answer.summary.strip() or None
        return outcome
    if not evidence:
        return insufficient("no_valid_citations")

    outcome.status = "answered"
    outcome.insufficient_evidence = False
    outcome.answer = answer.summary.strip()
    outcome.evidence, outcome.inference = evidence, inference
    steps, step_ids, dropped_steps = validate_next_steps(answer, set(by_id))
    outcome.next_steps, outcome.next_step_evidence = steps, step_ids
    outcome.next_step_kinds, outcome.next_step_support = classify_next_steps(
        answer, steps, step_ids, by_id, subject_paths)
    outcome.dropped_claims += dropped_steps
    outcome.unknowns = [u.strip() for u in answer.unknowns if u.strip()]
    outcome.citations = build_citations(evidence, inference, by_id)
    outcome.latency_ms = _ms(t0)
    return outcome
