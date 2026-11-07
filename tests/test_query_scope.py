"""Date / pipeline / task extraction and the decision to filter (offline, pure functions)."""

import json

import pytest

from incident_copilot.retrieval.query_scope import LogCatalog, extract_dates, resolve_scope

CATALOG = LogCatalog(frozenset({
    ("orders_etl", "transform_orders", "2026-09-04"),
    ("orders_etl", "transform_orders", "2026-09-21"),
    ("orders_etl", "validate_orders", "2026-08-19"),
    ("payments_sync", "pull_payments_api", "2026-09-04"),
}))


# ---- explicit dates -----------------------------------------------------------------------


@pytest.mark.parametrize(("text", "iso"), [
    ("fail on 2026-09-04?", "2026-09-04"),
    ("fail on 2026/09/04", "2026-09-04"),
    ("fail on 2026-9-4", "2026-09-04"),
    ("fail on September 4, 2026", "2026-09-04"),
    ("fail on Sept 4 2026", "2026-09-04"),
    ("fail on sep. 4th, 2026", "2026-09-04"),
    ("fail on 4 September 2026", "2026-09-04"),
    ("fail on the 4th of September, 2026", "2026-09-04"),
    ("fail on 21/09/2026", "2026-09-21"),     # day > 12, so the order is clear
    ("fail on 04.04.2026", "2026-04-04"),     # day == month, so the order does not matter
])
def test_unambiguous_dates_are_extracted(text, iso):
    assert extract_dates(text) == ([iso], [])


# ---- ambiguous dates ----------------------------------------------------------------------


@pytest.mark.parametrize(("text", "reason"), [
    ("fail on 09/04/2026", "day/month order unclear"),
    ("fail on Sep 4", "no year"),
    ("fail on 4 September", "no year"),
    ("fail yesterday", "relative date"),
    ("fail last night", "relative date"),
    ("fail 3 days ago", "relative date"),
    ("fail on 2026-02-30", "invalid date"),
    ("fail on February 30, 2026", "invalid date"),
])
def test_ambiguous_dates_are_reported_not_guessed(text, reason):
    dates, ambiguities = extract_dates(text)
    assert dates == [] and any(reason in a for a in ambiguities), ambiguities


@pytest.mark.parametrize("question", [
    "Why did transform_orders fail on 09/04/2026?",
    "Why did transform_orders fail on Sep 4?",
    "Why did transform_orders fail yesterday?",
    "Why did transform_orders fail on 2026-09-04 and on 2026-09-21?",
    "Why did transform_orders fail on 2026-09-04, after the run of yesterday?",
])
def test_ambiguous_questions_are_not_filtered(question):
    scope = resolve_scope(question, CATALOG)
    assert scope.applied is False and scope.date is None
    assert "ambiguous" in scope.reason or "multiple dates" in scope.reason


def test_two_different_dates_are_ambiguous_but_a_repeated_date_is_not():
    assert resolve_scope("2026-09-04 vs 2026-09-21 for transform_orders", CATALOG).reason == \
        "multiple dates"
    same = resolve_scope("transform_orders on 2026-09-04 (September 4, 2026)", CATALOG)
    assert same.applied and same.date == "2026-09-04"


# ---- missing dates and identifiers --------------------------------------------------------


@pytest.mark.parametrize("question", [
    "Why does the transform_orders task in the orders_etl pipeline keep failing?",
    "A load task fails with NotNullViolation. How do I fix it?",
    "Which pipeline populates the attribution_fact table?",
    "What is the on-call phone number for the payments team?",
])
def test_questions_without_a_date_are_never_filtered(question):
    scope = resolve_scope(question, CATALOG)
    assert scope.applied is False and scope.reason == "no date in the question"


def test_explicit_date_pipeline_and_task():
    scope = resolve_scope("Why did the transform_orders task in the orders_etl pipeline fail "
                          "on 2026-09-21?", CATALOG)
    assert scope.applied
    assert (scope.date, scope.pipeline, scope.task, scope.matching_logs) == (
        "2026-09-21", "orders_etl", "transform_orders", 1)


def test_date_only_question_filters_logs_by_date():
    scope = resolve_scope("What failed on 2026-09-04?", CATALOG)
    assert scope.applied and (scope.pipeline, scope.task) == (None, None)
    assert scope.matching_logs == 2


def test_unknown_pipeline_makes_the_scope_uncertain():
    scope = resolve_scope("Why did the ml_features pipeline fail on 2026-09-04?", CATALOG)
    assert scope.applied is False and scope.unknown_identifiers == ("ml_features",)


def test_plain_words_next_to_pipeline_are_not_identifiers():
    scope = resolve_scope("Which pipeline failed on 2026-09-04 and which task retried?",
                          CATALOG)
    assert scope.applied and scope.unknown_identifiers == ()


def test_lowercase_may_is_a_verb_not_a_month():
    scope = resolve_scope("Why may 2 tasks in orders_etl fail on 2026-09-04?", CATALOG)
    assert scope.applied and scope.ambiguities == ()


def test_task_from_another_pipeline_is_inconsistent():
    scope = resolve_scope("Why did transform_orders in payments_sync fail on 2026-09-04?",
                          CATALOG)
    assert scope.applied is False and "does not belong" in scope.reason


def test_several_pipelines_keep_the_date_but_drop_the_pipeline_constraint():
    scope = resolve_scope("Did orders_etl or payments_sync fail on 2026-09-04?", CATALOG)
    assert scope.applied and scope.pipeline is None and scope.notes


def test_no_matching_log_means_no_filter():
    scope = resolve_scope("Why did transform_orders fail on 2026-09-05?", CATALOG)
    assert scope.applied is False and scope.matching_logs == 0
    assert scope.reason == "no indexed log matches the date/pipeline/task"


def test_wrong_date_for_an_existing_task_is_not_filtered():
    """validate_orders only ran on 2026-08-19: asking about 2026-09-04 must not hide it."""
    scope = resolve_scope("Why did validate_orders fail on 2026-09-04?", CATALOG)
    assert scope.applied is False and scope.matching_logs == 0


# ---- the seed questions -------------------------------------------------------------------


def test_seed_questions_against_the_real_catalog(dataset):
    manifest = json.loads((dataset["root"] / "manifest.json").read_text())
    catalog = LogCatalog(frozenset((e["pipeline"], e["task"], e["date"])
                                   for e in manifest["files"] if e["source_type"] == "log"))
    questions = [json.loads(line) for line in dataset["eval"].read_text().splitlines()]
    for q in questions:
        scope = resolve_scope(q["question"], catalog)
        if q["category"] == "log_diagnosis":
            log = next(p for p in q["expected_paths"] if p.startswith("logs/"))
            pipeline, rest = log.split("/")[1:]
            day, task = rest.removesuffix(".log").split("__")
            assert scope.applied, (q["external_id"], scope.reason)
            assert (scope.date, scope.pipeline, scope.task) == (day, pipeline, task)
            assert scope.matching_logs == 1
        else:
            assert scope.applied is False, (q["external_id"], scope.reason)
