"""Evaluation metric definitions on hand-made inputs (offline)."""

import pytest

from incident_copilot.evaluation.answers import provider_kind, score_answer, split_by_provider
from incident_copilot.evaluation.metrics import (
    auc,
    classify_failure,
    first_ranks,
    latency_summary,
    leave_one_out_refusal,
    midpoint_threshold,
    refusal_confusion,
    retrieval_metrics,
    roc_points,
    threshold_sweep,
    wilson_interval,
)

LOG = "logs/orders_etl/2026-09-04__transform_orders.log"
RB = "runbooks/column-renamed-upstream.md"


def test_first_ranks_use_the_first_chunk_of_each_document():
    assert first_ranks(["a", "b", "a", "c"]) == {"a": 1, "b": 2, "c": 4}


def test_retrieval_metrics_any_versus_all():
    m = retrieval_metrics([LOG, RB], ["x", LOG, "y", "y", RB, "z"], ks=(1, 3, 5, 8))
    assert m["first_relevant_rank"] == 2 and m["rr"] == 0.5
    assert (m["hit@1"], m["hit@3"], m["hit@5"]) == (0, 1, 1)
    assert (m["recall@1"], m["recall@3"], m["recall@5"]) == (0.0, 0.5, 1.0)
    assert (m["all@3"], m["all@5"], m["all@8"]) == (0, 1, 1)


def test_retrieval_metrics_when_nothing_relevant_is_found():
    m = retrieval_metrics([LOG], ["x", "y"], ks=(1, 3))
    assert m["rr"] == 0.0 and m["first_relevant_rank"] is None
    assert m["hit@3"] == 0 and m["recall@3"] == 0.0 and m["all@3"] == 0


def test_unanswerable_questions_have_no_retrieval_metrics():
    assert retrieval_metrics([], ["x"]) == {}


def test_latency_summary_withholds_unreliable_percentiles():
    small = latency_summary([10, 20, 30])
    assert small["median"] == 20 and small["mean"] == 20
    assert small["p95"] is None and small["p99"] is None and "p95 needs" in small["note"]
    big = latency_summary(range(1, 101))
    assert big["p95"] == pytest.approx(95.05) and big["p99"] == pytest.approx(99.01)
    assert latency_summary([])["n"] == 0


def test_refusal_confusion_positive_class_is_refuse():
    should = [True, True, True, False, False, False, False]
    refused = [True, True, False, True, False, False, False]
    c = refusal_confusion(should, refused)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (2, 1, 1, 3)
    assert c["precision"] == pytest.approx(2 / 3) and c["recall"] == pytest.approx(2 / 3)
    assert c["specificity"] == pytest.approx(3 / 4) and c["f1"] == pytest.approx(2 / 3)
    assert c["false_refusal_rate"] == pytest.approx(1 / 4)
    assert c["missed_refusal_rate"] == pytest.approx(1 / 3)


def test_refusal_confusion_handles_empty_classes():
    c = refusal_confusion([False, False], [False, False])
    assert c["precision"] is None and c["recall"] is None and c["f1"] is None
    assert c["specificity"] == 1.0


def test_wilson_interval_is_wide_for_small_samples():
    lo, hi = wilson_interval(5, 6)
    assert 0.4 < lo < 0.45 and 0.96 < hi < 0.98
    assert wilson_interval(0, 0) is None


def test_threshold_sweep_and_midpoint_rule():
    scores = [0.60, 0.70, 0.76, 0.74, 0.80, 0.90]
    should = [True, True, True, False, False, False]
    assert midpoint_threshold(scores, should) == pytest.approx(0.72)
    rows = {r["threshold"]: r for r in threshold_sweep(scores, should, [0.72, 0.85])}
    assert (rows[0.72]["tp"], rows[0.72]["fn"], rows[0.72]["fp"]) == (2, 1, 0)
    assert (rows[0.85]["tp"], rows[0.85]["fp"]) == (3, 2)


def test_leave_one_out_recomputes_the_threshold_without_the_held_out_question():
    scores = [0.60, 0.70, 0.74, 0.80]
    should = [True, True, False, False]
    loo = leave_one_out_refusal(scores, should)
    # Holding out 0.70 leaves (0.60 + 0.74) / 2 = 0.67, so 0.70 is no longer refused: the
    # in-sample threshold depended on that very question.
    assert loo[1]["threshold"] == pytest.approx(0.67)
    assert [r["refused"] for r in loo] == [True, False, True, False]
    assert loo[2]["threshold"] == pytest.approx(0.75)  # without 0.74 the lowest answerable is 0.80


def test_roc_auc_extremes():
    perfect = roc_points([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
    assert auc(perfect) == pytest.approx(1.0)
    inverted = roc_points([0.8, 0.9, 0.1, 0.2], [True, True, False, False])
    assert auc(inverted) == pytest.approx(0.0)


def classify(expected, retrieved, origins=None, deep=None, k=3, abstain=False, refused=False):
    return classify_failure(expected_paths=expected, retrieved_paths=retrieved,
                            retrieved_origins=origins or ["vector"] * len(retrieved),
                            deep_ranks=deep or {}, k=k, should_abstain=abstain,
                            refused=refused)


def test_failure_wrong_task_date():
    other_day = "logs/orders_etl/2026-09-21__transform_orders.log"
    labels = classify([LOG, RB], [other_day, "incidents/INC-1.md", "x"], deep={LOG: 12})
    assert labels[0] == "wrong_task_date" and "outside_top_k" in labels


def test_failure_missing_runbook_and_incorrect_link():
    labels = classify([LOG, RB], [LOG, "runbooks/oom-killed.md", "x"],
                      origins=["vector", "error_link", "vector"], deep={LOG: 1, RB: 40})
    assert labels == ["missing_runbook", "incorrect_error_link", "outside_top_k"]


def test_failure_not_retrieved_and_gate_errors():
    assert classify([RB], ["a", "b", "c"]) == ["not_retrieved"]
    assert classify([RB], [RB], refused=True) == ["gate_false_refusal"]
    assert classify([], ["a"], abstain=True, refused=False) == ["gate_missed_refusal"]
    assert classify([], ["a"], abstain=True, refused=True) == []
    assert classify([LOG, RB], [RB, LOG, "x"]) == []


# ---- answer scoring (scripted inputs; these test the scorer, not a model) ---------------


def logged(model, abstained=False, cited=("E1",), paths=(LOG, RB)):
    return {"query_id": "q1", "model": model, "abstained": abstained,
            "evidence_ids": ["E1", "E2"], "evidence_paths": list(paths),
            "metadata": {"response": {"citations": list(cited)},
                         "validation": {"dropped_evidence_claims": 1}}}


def test_provider_kinds_keep_fake_answers_apart():
    assert provider_kind("gemini-3.8-flash") == "real_llm"
    assert provider_kind("fake-llm (scripted test double, not a real model)") == "fake_scripted"
    assert provider_kind(None) == "no_llm_call"
    split = split_by_provider([logged("gemini-x"), logged("fake-llm (x)"), logged(None)])
    assert [len(split[k]) for k in ("real_llm", "fake_scripted", "no_llm_call")] == [1, 1, 1]


def test_score_answer_citation_paths_and_abstention():
    s = score_answer(logged("gemini-x", cited=("E1", "E2")), [LOG, RB], False)
    assert s["citation_validity"] is True and s["n_citations"] == 2
    assert s["cited_path_precision"] == 1.0 and s["cited_path_recall"] == 1.0
    assert s["abstention_correct"] is True and s["dropped_claims"] == 1
    assert s["groundedness"] is None
    partial = score_answer(logged("gemini-x", cited=("E1",), paths=(LOG, "runbooks/x.md")),
                           [LOG, RB], False)
    assert partial["cited_path_recall"] == 0.5
    refused = score_answer(logged(None, abstained=True, cited=()), [], True)
    assert refused["abstention_correct"] is True and refused["cited_path_precision"] is None
