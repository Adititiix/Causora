"""Source confidence: the prompt asks the model to keep the certainty the evidence states, the
evidence carries what the model needs for that, and the answer evaluation checks whether an
answer kept it (evaluation/answers.qualification_metrics). Offline and deterministic: the
model's behaviour itself is measured only by the real-LLM answer evaluation.
"""

import pytest

from incident_copilot.evaluation.answer_eval import METRIC_KEYS
from incident_copilot.evaluation.answers import (
    QUALIFICATION_METRICS,
    first_cause_sentence,
    is_qualified_source,
    qualification_metrics,
    stated_confidence,
)
from incident_copilot.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from incident_copilot.retrieval import Hit

# Shaped like the "Inferred cause" sections of the incident reports in data/extra.
MEDIUM = ("# Incident report INC-2046\n## Inferred cause\n- Proposed explanation: the backfill "
          "and the scheduled pull exceeded the rate limit together.\n- Confidence: medium. "
          "Per-client request counts were not available.\n- Alternatives considered: the "
          "provider lowered our limit. Not ruled out.")
HIGH = ("## Inferred cause\n- Proposed explanation: the rename broke the transform.\n"
        "- Confidence: high. Every link is in a log.\n- Alternatives considered: a stalled "
        "connector. Ruled out: the raw table was fresh.")
MIXED = "- Confidence: high that the replay caused it; medium that replays are the only source."
OBSERVED = "## Observed evidence\n- The backfill ran from 01:40 to 07:10; 29 HTTP 429 responses."


def flat_prompt() -> str:
    return " ".join(SYSTEM_PROMPT.split())


# ---- the prompt ---------------------------------------------------------------------------


def test_prompt_asks_to_keep_the_stated_confidence_and_open_alternatives():
    p = flat_prompt()
    assert "Keep the certainty the evidence gives" in p
    assert "with its stated confidence and the alternatives it leaves open, never as " \
           "confirmed" in p
    assert "must not have a higher confidence than that block states" in p
    assert "If blocks disagree, say so and cite each side" in p
    # A cause the model derives from co-occurring observations is labelled as its inference.
    assert "is your inference: say so in \"summary\" and do not state it as fact" in p


def test_existing_rules_are_unchanged():
    p = flat_prompt()
    assert "Only cite ids that appear in the user message" in p  # citation rule
    assert 'If the evidence does not answer the question, set "insufficient_evidence" to ' \
           "true" in p  # abstention rule
    assert "Do not fill gaps with general knowledge" in p
    assert "a confidence of low, medium or high" in p  # the schema's levels, no numeric score


def test_the_section_and_the_stated_confidence_reach_the_model():
    hit = Hit(chunk_id="c", document_id="d", path="extra/incidents/INC-2046.md",
              source_type="incident", section="Inferred cause", ts_start=None, ts_end=None,
              text=MEDIUM, chunk_index=4, content_hash="h", pipeline_name=None, rank=1)
    prompt = build_user_prompt("What caused it?", [hit])
    assert 'section="Inferred cause"' in prompt
    assert "Confidence: medium" in prompt and "Not ruled out" in prompt


# ---- what counts as a qualified source -----------------------------------------------------


@pytest.mark.parametrize("text, level, qualified", [
    (MEDIUM, "medium", True),
    (HIGH, "high", False),           # "Ruled out:" is not an open alternative
    (MIXED, "high", False),          # the strongest stated confidence counts
    ("- Confidence: low.", "low", True),
    ("Alternative: a quota change, not ruled out.", None, True),
    (OBSERVED, None, False),
    ("", None, False),
    (None, None, False),
])
def test_stated_confidence_and_qualified_sources(text, level, qualified):
    assert stated_confidence(text) == level
    assert is_qualified_source(text) is qualified


# ---- does an answer keep the qualification? -----------------------------------------------


def check(summary, *, cited=None, claims=(), inferences=()):
    return qualification_metrics(cited=cited if cited is not None else {"E1": MEDIUM},
                                 summary=summary, claims=list(claims), inferences=inferences)


def test_a_medium_confidence_cause_stated_as_fact_is_not_kept():
    out = check("The 429s were caused by the backfill running alongside the scheduled run.")
    assert out == {"cites_qualified_source": True, "qualification_kept": False}


@pytest.mark.parametrize("summary", [
    "The report infers, with medium confidence, that the backfill caused the 429s.",
    "The backfill likely caused the 429s; a provider quota change was not ruled out.",
    "The cause is not confirmed: the report's confidence is medium.",
])
def test_qualified_wording_keeps_the_qualification(summary):
    assert check(summary)["qualification_kept"] is True


def test_an_inference_may_not_claim_more_confidence_than_its_source():
    summary = "The report infers, with medium confidence, that the backfill caused the 429s."
    upgraded = [{"claim": "The backfill caused it.", "based_on": ["E1"], "confidence": "high"}]
    kept = [{"claim": "The backfill caused it.", "based_on": ["E1"], "confidence": "medium"}]
    assert check(summary, inferences=upgraded)["qualification_kept"] is False
    assert check(summary, inferences=kept)["qualification_kept"] is True
    # Based on another (unqualified) block only: the source's cap does not apply.
    other = [{"claim": "x", "based_on": ["E2"], "confidence": "high"}]
    assert check(summary, cited={"E1": MEDIUM, "E2": OBSERVED},
                 inferences=other)["qualification_kept"] is True


def test_qualification_is_not_applicable_without_a_qualified_cited_source():
    # A high-confidence source, or a qualified chunk that was retrieved but not cited.
    assert check("The rename broke it.", cited={"E1": HIGH}) == \
        {"cites_qualified_source": False, "qualification_kept": None}
    assert check("x", cited={})["qualification_kept"] is None


def test_malformed_inferences_do_not_break_the_check():
    out = check("It likely caused it.",
                inferences=[{}, {"based_on": None}, {"based_on": ["E9"], "confidence": "?"}])
    assert out["qualification_kept"] is True


def test_the_answer_evaluation_reports_the_check():
    assert set(QUALIFICATION_METRICS) <= set(METRIC_KEYS)


# ---- the first statement of an uncertain cause carries the qualifier -----------------------


def test_prompt_puts_the_qualifier_in_the_first_cause_sentence_and_not_on_facts():
    p = flat_prompt()
    assert "The first sentence that states such a cause must itself say it is inferred or " \
           "suspected and give the stated confidence" in p
    assert "a caveat in a later sentence is not enough" in p
    assert "State directly observed facts (log lines, times, counts) plainly, without " \
           "hedging" in p


@pytest.mark.parametrize("text, expected", [
    ("The task failed at 07:30. It was caused by the backfill. More text.",
     "It was caused by the backfill."),
    ("The pull ran for 5h 24m because of 29 HTTP 429 responses!",
     "The pull ran for 5h 24m because of 29 HTTP 429 responses!"),
    ("29 HTTP 429 responses were logged. The sensor timed out at 07:30.", None),
    ("", None),
    (None, None),
])
def test_first_cause_sentence(text, expected):
    assert first_cause_sentence(text) == expected


def test_a_cause_stated_first_and_caveated_later_is_not_kept():
    # The shape of M011's answer in eval/results/20261031T195106Z_answers.
    summary = ("The HTTP 429 responses were caused by a backfill and the scheduled pull "
               "exceeding the rate limit together. This is an inferred cause with medium "
               "confidence, and a provider quota change was not ruled out.")
    medium = [{"claim": "x", "based_on": ["E1"], "confidence": "medium"}]
    assert check(summary, inferences=medium)["qualification_kept"] is False


def test_a_qualified_opening_with_plain_observed_facts_is_kept():
    summary = ("The pull received 29 HTTP 429 responses and finished at 07:24. The report "
               "infers, with medium confidence, that the backfill caused them; a provider "
               "quota change was not ruled out.")
    assert check(summary)["qualification_kept"] is True


def test_observed_facts_alone_are_not_forced_to_hedge():
    # No sentence states a cause: only the overall qualifier and the inference cap apply.
    summary = "The sensor timed out at 07:30 after 14,400 seconds. The cause is not confirmed."
    assert first_cause_sentence("The sensor timed out at 07:30 after 14,400 seconds.") is None
    assert check(summary)["qualification_kept"] is True


# ---- one failure per answer, cited next steps ------------------------------------------------


def test_prompt_keeps_different_failures_apart():
    p = flat_prompt()
    assert "Blocks can describe different failures" in p
    assert "Never give another failure's cause, facts or fix as if they belonged to this one" in p


def test_next_steps_must_cite_the_blocks_that_state_them():
    from incident_copilot.generation.prompt import ModelAnswer

    p = flat_prompt()
    assert 'each with the "evidence_ids" of the blocks that state it' in p
    step = ModelAnswer.model_json_schema()["$defs"]["NextStep"]
    # `kind` (recovery / prevention) is required too, so recovery and prevention stay apart.
    assert set(step["required"]) == {"step", "evidence_ids", "kind"}
    assert set(step["properties"]["kind"]["enum"]) == {"recovery", "prevention"}
