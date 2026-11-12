"""Answer evaluation runner and its deterministic metrics. Offline: no DB, no network, no key.

Retrieval is an in-memory stub and generation is the scripted FakeLLMProvider, so these tests
check the bookkeeping (records, files, resume, failures), never model quality.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from incident_copilot.config import get_settings
from incident_copilot.evaluation import answer_eval
from incident_copilot.evaluation.answers import (
    answer_text_metrics,
    error_class,
    mentions,
    parse_ground_truth,
    remediation_actions,
    term_recall,
)
from incident_copilot.evaluation.harness import load_questions
from incident_copilot.generation import FakeLLMProvider, LLMError, LLMNotConfiguredError
from incident_copilot.generation.clarify import NearMiss
from incident_copilot.retrieval import Hit

SEED = Path(__file__).resolve().parents[1] / "eval" / "questions.seed.jsonl"
SECRET = "AIzaTEST-not-a-real-key-123456"
T0 = datetime(2026, 9, 4, 2, 21, tzinfo=UTC)
LOG = "logs/orders_etl/2026-09-04__transform_orders.log"
RUNBOOK = "runbooks/column-renamed-upstream.md"


def hit(rank, path, text, score, source_type="runbook"):
    return Hit(chunk_id=f"{path}#{rank}", document_id=f"doc-{path}", path=path,
               source_type=source_type, section="sec", ts_start=T0, ts_end=T0, text=text,
               chunk_index=rank - 1, content_hash="h", pipeline_name=None, rank=rank,
               scores={"vector_similarity": score, "vector_distance": 1 - score})


@pytest.fixture(scope="module")
def seed():
    records, problems = load_questions(SEED)
    assert not problems
    return {q["external_id"]: q for q in records}


@pytest.fixture
def questions(seed):
    # answerable log question, answerable runbook question, unanswerable question
    return [seed["Q001"], seed["Q019"], seed["Q045"]]


class MapRetriever:
    """Hits per question text; Q045 scores below the 0.72 gate."""

    name = "stub"

    def __init__(self, seed, fail_on: str | None = None):
        self.by_question = {
            seed["Q001"]["question"]: [hit(1, LOG, "KeyError: 'customer_tier'", 0.85, "log"),
                                       hit(2, RUNBOOK, "Update the transform", 0.80)],
            seed["Q019"]["question"]: [hit(1, RUNBOOK, "Update the transform", 0.80)],
            seed["Q045"]["question"]: [hit(1, "schemas/payments_fact.md", "amount", 0.60,
                                           "schema")],
        }
        self.fail_on = fail_on
        self.calls = 0

    def retrieve(self, question, k, filters=None):
        self.calls += 1
        if self.fail_on and question == self.fail_on:
            raise psycopg.OperationalError("connection refused")
        return self.by_question[question][:k]


Q001_ANSWER = {
    "insufficient_evidence": False,
    "summary": "transform_orders failed with KeyError: 'customer_tier' because the column was "
               "renamed to loyalty_tier in orders_raw.",
    "evidence": [{"claim": "The log shows KeyError: 'customer_tier'", "evidence_ids": ["E1"]}],
    "inference": [{"claim": "The storefront team renamed customer_tier to loyalty_tier",
                   "based_on": ["E1", "E2"], "confidence": "high"}],
    "next_steps": ["Update the transform to read loyalty_tier", "Deploy it",
                   "Clear the failed task so it re-runs"],
}
Q019_ANSWER = {
    "insufficient_evidence": False, "summary": "Follow the column-renamed runbook.",
    "evidence": [{"claim": "The runbook says update the transform", "evidence_ids": ["E1"]}],
    "inference": [], "next_steps": [],
}


def run(questions, retriever, provider, run_dir, previous=(), **kw):
    run_dir.mkdir(exist_ok=True)
    meta = {"run_id": "test", "model": provider.model, "retriever": "stub", "ask_top_k": 8,
            "min_similarity": 0.72, "selection": {"question_ids": [q["external_id"]
                                                                   for q in questions]}}
    sleeps = []
    records = answer_eval.run_answer_eval(
        questions, retriever=retriever, provider=provider, run_dir=run_dir, meta=meta, k=8,
        min_similarity=0.72, previous=previous, sleep=sleeps.append, log=lambda s: None, **kw)
    return records, meta, sleeps


def by_id(records):
    return {r["id"]: r for r in records}


# ---- ground-truth term checks -------------------------------------------------------------


def test_ground_truth_template_parts_are_recovered(seed):
    log = parse_ground_truth("log_diagnosis", seed["Q001"]["ground_truth"])
    assert log["error"] == "KeyError: 'customer_tier'"
    assert log["root_cause"].startswith("The storefront team renamed")
    assert log["remediation"].startswith("Update the transform to read loyalty_tier")
    inc = parse_ground_truth("incident", seed["Q037"]["ground_truth"])
    assert inc["incident_id"] == "INC-2043" and "205 minutes" in inc["fact"]
    assert inc["remediation"].startswith("Honor the Retry-After header")
    assert parse_ground_truth("runbook", seed["Q019"]["ground_truth"])["remediation"]
    assert parse_ground_truth("schema", seed["Q039"]["ground_truth"])["fact"]
    assert set(parse_ground_truth("unanswerable", "NOT_ANSWERABLE").values()) == {None}


def test_every_answerable_seed_question_has_something_to_check(seed):
    for q in seed.values():
        if q["should_abstain"]:
            continue
        gt = parse_ground_truth(q["category"], q["ground_truth"])
        assert gt["remediation"] or gt["fact"], q["external_id"]
        if q["category"] == "log_diagnosis":
            assert gt["error"] and gt["root_cause"] and error_class(gt["error"]), \
                q["external_id"]


def test_error_class_uses_the_exception_name_or_a_camelcase_fallback():
    assert error_class('psycopg2.errors.UniqueViolation: duplicate key value') == \
        "UniqueViolation"
    assert error_class("KeyError: 'customer_tier'") == "KeyError"
    assert error_class("Pod etl-27601 terminated: reason=OOMKilled exitCode=137") == "OOMKilled"


def test_term_checks_ignore_question_terms_and_match_whole_tokens():
    assert term_recall("SLA of 6 hours", "about 2026 things") == 0.0   # 6 is not inside 2026
    assert term_recall("SLA of 6 hours", "SLA is 6 hours") == 1.0
    assert term_recall("the orders_fact table", "anything", exclude="orders_fact table") is None
    assert mentions("KeyError", "it raised KeyError: x") and not mentions("KeyError", "MyKeyErrorX")
    assert remediation_actions("Deploy it, then clear the task") == ["Deploy it", "clear the task"]


def test_remediation_in_the_summary_does_not_count_for_next_steps(seed):
    q = seed["Q019"]
    fix = q["ground_truth"]
    m = answer_text_metrics(category="runbook", question=q["question"], ground_truth=fix,
                            summary=fix, claims=[], next_steps=[])
    assert m["remediation_covered_next_steps"] == 0.0
    assert m["remediation_covered_any"] == 1.0
    assert m["error_class"] is None and m["fact_term_recall"] is None


# ---- running ------------------------------------------------------------------------------


def test_a_run_writes_every_output_and_scores_each_question(seed, questions, tmp_path):
    provider = FakeLLMProvider([Q001_ANSWER, Q019_ANSWER])
    records, meta, sleeps = run(questions, MapRetriever(seed), provider, tmp_path / "run")
    out = by_id(records)

    assert provider.call_count == 2                    # Q045 was refused by the gate
    assert sleeps == [4.0, 4.0]                        # a pause after each real LLM call
    assert meta["state"] == "complete"

    q1 = out["Q001"]
    assert q1["status"] == "answered" and q1["model"].startswith("fake-llm")
    assert {c["path"] for c in q1["citations"]} == {LOG, RUNBOOK}
    assert q1["citations"][0]["text"] == "KeyError: 'customer_tier'"
    assert q1["ground_truth"] == seed["Q001"]["ground_truth"]
    m1 = q1["metrics"]
    assert m1["cited_path_recall"] == 1.0 and m1["cited_path_precision"] == 1.0
    assert m1["error_class_mentioned"] is True and m1["remediation_covered_next_steps"] == 1.0
    assert m1["invalid_ids"] == 0 and m1["dropped_claims"] == 0
    assert m1["groundedness_human"] is None

    m19 = out["Q019"]["metrics"]
    assert m19["next_steps_count"] == 0 and m19["remediation_covered_next_steps"] == 0.0

    m45 = out["Q045"]["metrics"]
    assert out["Q045"]["status"] == "insufficient_evidence"
    assert m45["abstained_by"] == "gate" and m45["abstention_correct"] is True
    assert m45["error_class_mentioned"] is None        # no text checks for refusals

    run_dir = tmp_path / "run"
    lines = (run_dir / "answers.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(x)["id"] for x in lines] == ["Q001", "Q019", "Q045"]
    with open(run_dir / "answer_quality.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["id"] for r in rows] == ["Q001", "Q019", "Q045"]
    assert json.loads(rows[0]["cited_paths"]) == sorted([LOG, RUNBOOK])
    summary = (run_dir / "summary.md").read_text("utf-8")
    assert "not model quality" in summary and "NOT MEASURED" in summary
    assert "Q019" in summary  # listed under empty next_steps
    assert json.loads((run_dir / "run_metadata.json").read_text("utf-8"))["state"] == "complete"


def test_an_answerable_question_refused_by_the_model_is_an_unnecessary_abstention(
        seed, questions, tmp_path):
    refusal = {"insufficient_evidence": True, "summary": "missing", "evidence": [],
               "inference": [], "next_steps": []}
    records, _, _ = run(questions[:1], MapRetriever(seed), FakeLLMProvider([refusal]),
                        tmp_path / "run")
    m = records[0]["metrics"]
    assert records[0]["insufficient_reason"] == "model_insufficient_evidence"
    assert m["abstained_by"] == "model" and m["unnecessary_abstention"] is True
    assert m["abstention_correct"] is False and m["cited_path_recall"] is None


def test_generation_failures_are_recorded_safely_and_the_run_continues(seed, questions,
                                                                       tmp_path):
    provider = FakeLLMProvider([LLMError("Gemini API error 503: token=abc123 overloaded"),
                                "not json"])
    records, meta, _ = run(questions, MapRetriever(seed), provider, tmp_path / "run")
    out = by_id(records)
    assert out["Q001"]["error"]["category"] == "llm_upstream_error"
    assert "abc123" not in out["Q001"]["error"]["message"]
    assert "[REDACTED]" in out["Q001"]["error"]["message"]
    assert out["Q001"]["gate"]["passed"] is True       # retrieval kept from the partial outcome
    assert out["Q001"]["metrics"]["expected_paths_retrieved"] == 1.0
    assert out["Q019"]["error"]["category"] == "llm_bad_response"
    assert out["Q045"]["status"] == "insufficient_evidence"
    assert meta["state"] == "complete"
    assert "llm_upstream_error" in (tmp_path / "run" / "summary.md").read_text("utf-8")


def test_the_run_stops_after_consecutive_failures(seed, questions, tmp_path):
    provider = FakeLLMProvider([LLMError("quota"), LLMError("quota")])
    records, meta, _ = run(questions, MapRetriever(seed), provider, tmp_path / "run",
                           max_consecutive_errors=2)
    assert [r["id"] for r in records] == ["Q001", "Q019"]
    assert meta["state"] == "stopped_after_errors"


def test_a_database_failure_aborts_but_keeps_finished_questions(seed, questions, tmp_path):
    retriever = MapRetriever(seed, fail_on=seed["Q019"]["question"])
    run_dir = tmp_path / "run"
    with pytest.raises(psycopg.OperationalError):
        run(questions, retriever, FakeLLMProvider([Q001_ANSWER]), run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    assert meta["state"] == "aborted" and "OperationalError" in meta["abort_reason"]
    assert len((run_dir / "answers.jsonl").read_text("utf-8").splitlines()) == 1


def test_an_interrupted_run_keeps_its_results(seed, questions, tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run(questions, MapRetriever(seed), FakeLLMProvider([Q001_ANSWER, KeyboardInterrupt()]),
            run_dir)
    assert [r["id"] for r in answer_eval.load_records(run_dir)] == ["Q001"]
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    assert meta["state"] == "interrupted"


def test_resume_reruns_only_questions_without_a_successful_record(seed, questions, tmp_path):
    run_dir = tmp_path / "run"
    run(questions, MapRetriever(seed), FakeLLMProvider([Q001_ANSWER, LLMError("503")]), run_dir)
    with open(run_dir / "answers.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"id": "Q0')                          # a write cut off by a crash
    previous = answer_eval.load_records(run_dir)
    assert [r["id"] for r in previous] == ["Q001", "Q019", "Q045"]

    provider = FakeLLMProvider([Q019_ANSWER])
    records, _, _ = run(questions, MapRetriever(seed), provider, run_dir, previous=previous)
    assert provider.call_count == 1                    # only the failed Q019 ran again
    assert {r["id"]: r["status"] for r in records} == {
        "Q001": "answered", "Q019": "answered", "Q045": "insufficient_evidence"}
    with open(run_dir / "answer_quality.csv", encoding="utf-8") as fh:
        assert [r["status"] for r in csv.DictReader(fh)] == [
            "answered", "answered", "insufficient_evidence"]


def test_select_questions(seed):
    records = list(seed.values())
    assert [q["external_id"] for q in answer_eval.select_questions(records, ["q045", "Q001"])] \
        == ["Q001", "Q045"]                            # seed order, ids case-insensitive
    assert len(answer_eval.select_questions(records, None, 3)) == 3
    with pytest.raises(ValueError, match="Q999"):
        answer_eval.select_questions(records, ["Q999"])


# ---- command line -------------------------------------------------------------------------


@pytest.fixture
def cli(seed, monkeypatch, tmp_path):
    """main() with fakes behind every external dependency, and a known secret in settings."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.delenv("MIN_SIMILARITY", raising=False)
    get_settings.cache_clear()
    state = {"provider": None}

    def provider_factory(settings):
        assert settings.gemini_api_key == SECRET
        return state["provider"]

    monkeypatch.setattr(answer_eval, "get_llm_provider", provider_factory)
    monkeypatch.setattr(answer_eval, "get_embedder", lambda *a, **k: object())
    monkeypatch.setattr(answer_eval, "get_retriever", lambda name, emb: MapRetriever(seed))
    monkeypatch.setattr(answer_eval, "index_fingerprint",
                        lambda: {"documents": 3, "chunks": 4, "corpus_sha256": "abc"})

    # The real factories read index metadata from PostgreSQL (clarify.py). Offline doubles:
    # never ask for clarification and never resolve a near miss, so the gate decides alone.
    def no_clarifier(retriever, *, min_similarity, max_candidates):
        return lambda question, hits: []

    def no_near_miss(retriever, *, min_similarity, max_candidates):
        return lambda question, hits: NearMiss()

    monkeypatch.setattr(answer_eval, "make_clarifier", no_clarifier)
    monkeypatch.setattr(answer_eval, "make_near_miss_resolver", no_near_miss)
    root = tmp_path / "results"
    existing = root / "20261022T053248Z"
    existing.mkdir(parents=True)
    (existing / "summary.md").write_text("retrieval results", encoding="utf-8")

    def main(*args, responses=()):
        state["provider"] = FakeLLMProvider(list(responses))
        code = answer_eval.main(["--questions", str(SEED), "--out-root", str(root),
                                 "--pause", "0", *args])
        return code, state["provider"]

    yield main, root, existing
    get_settings.cache_clear()


def _answer_dirs(root):
    return sorted(p for p in root.iterdir() if p.name.endswith("_answers"))


def test_cli_creates_a_new_run_folder_and_never_writes_secrets(cli):
    main, root, existing = cli
    code, provider = main("--ids", "Q001,Q045", responses=[Q001_ANSWER])
    assert code == 0 and provider.call_count == 1
    [run_dir] = _answer_dirs(root)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "answer_quality.csv", "answers.jsonl", "run_metadata.json", "summary.md"]
    assert (existing / "summary.md").read_text("utf-8") == "retrieval results"
    for f in run_dir.iterdir():
        assert SECRET not in f.read_text("utf-8"), f.name
    meta = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    assert meta["kind"] == "answer_eval" and meta["state"] == "complete"
    assert meta["selection"]["question_ids"] == ["Q001", "Q045"]
    assert meta["min_similarity"] == 0.72 and meta["corpus_sha256"] == "abc"
    assert meta["seed_questions_sha256"] and meta["prompt_module_sha256"]


def test_cli_resume_skips_finished_questions_and_refuses_changed_settings(cli, monkeypatch):
    main, root, existing = cli
    main("--ids", "Q001,Q045", responses=[Q001_ANSWER])
    [run_dir] = _answer_dirs(root)

    code, provider = main("--resume", str(run_dir))
    assert code == 0 and provider.call_count == 0
    assert len(_answer_dirs(root)) == 1

    assert main("--resume", str(run_dir), "--limit", "1")[0] == 2
    assert main("--resume", str(existing))[0] == 2     # not an answer-eval folder
    assert (existing / "summary.md").read_text("utf-8") == "retrieval results"

    monkeypatch.setenv("MIN_SIMILARITY", "0.5")
    get_settings.cache_clear()
    assert main("--resume", str(run_dir))[0] == 2


def test_cli_exits_2_without_creating_a_folder_when_unconfigured(cli, monkeypatch, capsys):
    main, root, _ = cli
    assert main("--ids", "Q999")[0] == 2

    def not_configured(settings):
        raise LLMNotConfiguredError("no key")

    monkeypatch.setattr(answer_eval, "get_llm_provider", not_configured)
    assert answer_eval.main(["--questions", str(SEED), "--out-root", str(root),
                             "--ids", "Q001"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().out
    assert _answer_dirs(root) == []


def test_cli_exit_code_reports_failed_questions(cli):
    main, root, _ = cli
    code, _ = main("--ids", "Q001", responses=[LLMError("Gemini API error 500: boom")])
    assert code == 1
    [run_dir] = _answer_dirs(root)
    assert answer_eval.load_records(run_dir)[0]["error"]["category"] == "llm_upstream_error"


# ---- heuristic limits, evidence ceilings, diagnosis fields ---------------------------------


def test_term_heuristics_are_not_correctness(seed):
    """Documents the limits stated in summary.md: a faithful paraphrase scores low, and a wrong
    answer that repeats ground-truth words scores high."""
    q = seed["Q001"]
    kw = dict(category="log_diagnosis", question=q["question"], ground_truth=q["ground_truth"],
              claims=[], next_steps=[])
    paraphrase = answer_text_metrics(
        summary="A field the job expects was given a new name by the team that owns the source "
                "table, so the job looked up a key that is gone.", **kw)
    stuffed = answer_text_metrics(
        summary="Not a rename: the storefront team code reads customer_tier, loyalty_tier and "
                "orders_raw fine since 2026-08-28; the transform still works.", **kw)
    assert paraphrase["root_cause_term_recall"] < 0.2
    assert stuffed["root_cause_term_recall"] > 0.6


def test_incident_ground_truth_fix_is_the_runbook_fix_not_the_incident_resolution(seed):
    """Baseline Q038: the answer reproduced INC-2035's own resolution (roll back, clear the
    failed tasks), but the ground truth uses the runbook's longer fix, so coverage is partial."""
    q = seed["Q038"]
    m = answer_text_metrics(category="incident", question=q["question"],
                            ground_truth=q["ground_truth"], summary="INC-2035.", claims=[],
                            next_steps=["Roll back to the previous worker image.",
                                        "Clear the failed tasks."])
    assert m["incident_id_mentioned"] is True
    assert 0 < m["remediation_covered_next_steps"] < 1


def test_evidence_ceilings_explain_a_refusal(seed, questions, tmp_path):
    refusal = {"insufficient_evidence": True, "summary": "Only the symptoms are shown, no fix.",
               "evidence": [], "inference": [], "next_steps": []}
    records, _, _ = run(questions[1:2], MapRetriever(seed), FakeLLMProvider([refusal]),
                        tmp_path / "run")
    rec = records[0]
    assert rec["refusal_note"] == "Only the symptoms are shown, no fix."
    m = rec["metrics"]
    assert m["unnecessary_abstention"] is True and m["remediation_expected"] is True
    assert m["remediation_covered_next_steps"] is None            # nothing answered
    assert m["remediation_covered_in_evidence"] is not None       # but the evidence is scored
    summary = (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert "Only the symptoms are shown, no fix." in summary and "fix in evidence" in summary


def test_dropped_claims_are_recorded_and_listed(seed, questions, tmp_path):
    bad = dict(Q001_ANSWER, evidence=[*Q001_ANSWER["evidence"],
                                      {"claim": "expired mid-task", "evidence_ids": ["C3"]}])
    records, _, _ = run(questions[:1], MapRetriever(seed), FakeLLMProvider([bad]),
                        tmp_path / "run")
    rec = records[0]
    assert rec["status"] == "answered" and rec["metrics"]["invalid_ids"] == 1
    assert rec["dropped_claims"] == [{"field": "evidence", "claim": "expired mid-task",
                                      "ids": ["C3"], "invalid_ids": ["C3"]}]
    summary = (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert "| Q001 | evidence | C3 | C3 | expired mid-task |" in summary


def test_uncited_and_extra_paths_are_document_level(seed, questions, tmp_path):
    only_log = dict(Q001_ANSWER, inference=[],
                    evidence=[{"claim": "KeyError", "evidence_ids": ["E1"]},
                              {"claim": "same log again", "evidence_ids": ["E1"]}])
    records, _, _ = run(questions[:1], MapRetriever(seed), FakeLLMProvider([only_log]),
                        tmp_path / "run")
    m = records[0]["metrics"]
    assert m["cited_paths"] == [LOG] and m["cited_path_recall"] == 0.5
    assert m["uncited_expected_paths"] == [RUNBOOK] and m["extra_cited_paths"] == []


def test_empty_next_steps_counts_only_questions_whose_ground_truth_has_a_fix(seed, tmp_path):
    retriever = MapRetriever(seed)
    retriever.by_question[seed["Q039"]["question"]] = [
        hit(1, "schemas/orders_fact.md", "owned by data-platform; freshness SLA 6 hours", 0.8,
            "schema")]
    schema_answer = {"insufficient_evidence": False,
                     "summary": "orders_fact is owned by data-platform with a 6 hours SLA.",
                     "evidence": [{"claim": "owner and SLA", "evidence_ids": ["E1"]}],
                     "inference": [], "next_steps": []}
    records, _, _ = run([seed["Q019"], seed["Q039"]], retriever,
                        FakeLLMProvider([Q019_ANSWER, schema_answer]), tmp_path / "run")
    assert by_id(records)["Q039"]["metrics"]["remediation_expected"] is False
    summary = (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert "| Next steps | empty next_steps, answered questions whose ground truth has a fix | " \
           "1/1 (100.0%): Q019 |" in summary
    assert "| Next steps | answered questions with no fix expected (schema) | 1 " in summary


def test_a_clarification_is_counted_apart_from_gate_and_model_refusals(seed, questions,
                                                                      tmp_path):
    from incident_copilot.generation.clarify import Candidate

    offered = [Candidate(id=f"run:p/t/2026-09-1{i}", label=f"p / t on 2026-09-1{i}",
                         description="", incident_id=None, pipeline="p", task="t",
                         date=f"2026-09-1{i}", paths=(f"logs/p/2026-09-1{i}__t.log",),
                         best_similarity=0.69) for i in (6, 7)]
    asked = []

    def clarifier(question, hits):
        asked.append(question)
        return offered

    q45 = [q for q in questions if q["external_id"] == "Q045"]
    records, _, _ = run(q45, MapRetriever(seed), FakeLLMProvider([]), tmp_path / "run",
                        clarifier=clarifier)
    rec = records[0]
    assert asked == [seed["Q045"]["question"]]  # only after the gate refused it
    assert rec["status"] == "insufficient_evidence"
    assert rec["insufficient_reason"] == "ambiguous_query" and rec["generation_ms"] is None
    m = rec["metrics"]
    assert m["abstained_by"] == "clarification" and m["abstention_correct"] is True
    summary = (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert "clarification asked instead (unanswerable / answerable) | Q045 / none" in summary


def test_without_a_clarifier_the_run_is_unchanged(seed, questions, tmp_path):
    q45 = [q for q in questions if q["external_id"] == "Q045"]
    records, _, _ = run(q45, MapRetriever(seed), FakeLLMProvider([]), tmp_path / "run")
    assert records[0]["insufficient_reason"] == "below_min_similarity"
    assert records[0]["metrics"]["abstained_by"] == "gate"
