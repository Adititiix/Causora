"""Retrieval benchmark bookkeeping (evaluation/retrieval_benchmark.py). Offline: stub
retrievers go through the real harness, so no database, model or network is needed."""

import json
from pathlib import Path

import pytest

from incident_copilot.evaluation import retrieval_benchmark as rb
from incident_copilot.evaluation.harness import RetrieverConfig
from incident_copilot.retrieval import Hit

RB = "runbooks/not-null-violation.md"
LOG = "logs/orders_etl/2026-08-31__load_orders_fact.log"
QUESTIONS = [
    {"external_id": "Q015", "question": "log q", "ground_truth": "g", "category": "log_diagnosis",
     "should_abstain": False, "expected_paths": [LOG, RB]},
    {"external_id": "Q033", "question": "rb q", "ground_truth": "g", "category": "runbook",
     "should_abstain": False, "expected_paths": [RB]},
    {"external_id": "Q039", "question": "schema q", "ground_truth": "g", "category": "schema",
     "should_abstain": False, "expected_paths": ["schemas/orders_fact.md"]},
    {"external_id": "Q045", "question": "phone", "ground_truth": "NOT_ANSWERABLE",
     "category": "unanswerable", "should_abstain": True, "expected_paths": []},
]


def h(path, section, score, text="x" * 40, origin="vector"):
    return Hit(chunk_id=f"{path}#{section}", document_id=path, path=path,
               source_type=path.split("/")[0].rstrip("s"), section=section, ts_start=None,
               ts_end=None, text=text, chunk_index=0, content_hash="h", pipeline_name=None,
               rank=0, scores={"vector_similarity": score}, origin=origin)


class Stub:
    def __init__(self, name, table):
        self.name, self.table, self.calls = name, table, []

    def retrieve(self, question, k, filters=None):
        self.calls.append(question)
        return [h.__class__(**{**h.__dict__, "rank": i})
                for i, h in enumerate(self.table[question][:k], start=1)]


BASE = {"log q": [h("incidents/INC-1.md", "Summary", 0.86), h(RB, "Symptoms", 0.5)],
        "rb q": [h(RB, "Symptoms", 0.80), h(LOG, "run summary", 0.7)],
        "schema q": [h("schemas/orders_fact.md", "orders_fact", 0.9)],
        "phone": [h("incidents/INC-1.md", "Summary", 0.63)]}
BETTER = {"log q": [h("incidents/INC-1.md", "Summary", 0.86), h(LOG, "run summary", 0.82),
                    h(RB, "Symptoms", 0.5, origin="error_link"),
                    h(RB, "Resolution", 0.55, origin="resolution_companion")],
          "rb q": [h(RB, "Symptoms", 0.80),
                   h(RB, "Resolution", 0.60, origin="resolution_companion")],
          "schema q": [], "phone": [h("incidents/INC-1.md", "Summary", 0.63)]}


def run(resolution_paths=frozenset({RB}), better=BETTER):
    stubs = {}

    def factory(name, table):
        def make(_embedder):
            stubs[name] = Stub(name, table)
            return stubs[name]
        return make

    configs = [RetrieverConfig("base", factory("base", BASE)),
               RetrieverConfig("cand", factory("cand", better))]
    result = rb.run_benchmark(QUESTIONS, configs, None, baseline="base", k=8,
                              min_similarity=0.72, repeats=2,
                              resolution_paths=set(resolution_paths))
    return result, stubs


def test_every_question_is_evaluated_once_per_configuration():
    result, stubs = run()
    for name in ("base", "cand"):
        assert sorted(r["id"] for r in result["by_config"][name]) == \
            ["Q015", "Q033", "Q039", "Q045"]
        assert len(stubs[name].calls) == 4 * 2  # 2 timed repeats, one result row each


def test_denominators_and_section_coverage():
    result, _ = run()
    base, cand = result["aggregates"]["base"], result["aggregates"]["cand"]
    assert (base["n_answerable"], base["n_unanswerable"]) == (3, 1)
    assert base["resolution_questions"] == 2      # the two questions expecting a runbook
    assert (base["resolution_in_top_k"], cand["resolution_in_top_k"]) == (0, 2)
    assert base["log_questions"] == 1 and (base["logs_in_top_k"], cand["logs_in_top_k"]) == (0, 1)
    assert cand["target_logs_in_top_k"] == 1 and cand["target_log_questions"] == 1
    assert base["unanswerable_gate_pass"] == 0
    assert cand["companions"] == 2
    # MRR and hit@k average over the 3 answerable questions only.
    assert base["MRR"] == pytest.approx((1 / 2 + 1 + 1) / 3)
    # cand: Q015's top hit is an incident (not expected), Q033's is the runbook, Q039 is empty.
    assert cand["hit@1"] == pytest.approx(1 / 3)


def test_a_resolution_counts_only_for_an_expected_runbook():
    other = dict(BETTER, **{"rb q": [h("runbooks/other.md", "Resolution", 0.8)]})
    result, _ = run(better=other)
    q33 = next(r for r in result["by_config"]["cand"] if r["id"] == "Q033")
    assert q33["resolution_in_top_k"] is False


def test_regressions_and_missing_sections_are_reported_not_hidden():
    result, _ = run(resolution_paths=set())       # the index has no Resolution for RB
    assert result["missing_sections"] == [RB]
    changes = {c["id"]: c for c in result["changes"]["cand"]}
    assert changes["Q039"]["direction"] == "regression"   # the schema doc disappeared
    assert changes["Q039"]["expected_ranks_baseline"] == {"schemas/orders_fact.md": 1}
    assert changes["Q039"]["expected_ranks_candidate"] == {"schemas/orders_fact.md": None}
    assert changes["Q015"]["direction"] == "improvement"
    extra = result["extras"]["cand"]
    # Q039 now has no hits, so its gate decision changed: reported, though answerable.
    assert extra["gate_changes"] == ["Q039"] and extra["unanswerable_gate_changes"] == 0
    assert extra["best_similarity_changes"] == ["Q039"]


def test_a_rank_regression_at_any_cut_off_is_reported():
    # Q007 shape: the expected runbook stays in the top 8 but drops from rank 2 to 7.
    filler = [h(f"incidents/INC-{i}.md", "Summary", 0.7) for i in range(5)]
    moved = dict(BASE, **{"rb q": [h(LOG, "run summary", 0.7), *filler, h(RB, "Symptoms", 0.6)]})
    base_rb = dict(BASE, **{"rb q": [h(LOG, "run summary", 0.7), h(RB, "Symptoms", 0.6),
                                     *filler]})
    def factory(table):
        return lambda _embedder: Stub("stub", table)

    configs = [RetrieverConfig("base", factory(base_rb)), RetrieverConfig("cand", factory(moved))]
    result = rb.run_benchmark(QUESTIONS, configs, None, baseline="base", k=8,
                              min_similarity=0.72, repeats=1, resolution_paths={RB})
    change = {c["id"]: c for c in result["changes"]["cand"]}["Q033"]
    assert change["direction"] == "regression"
    assert {"hit@3", "all@3", "all@5", "recall@5", "rr"} <= set(change["changes"])
    assert "all@8" not in change["changes"] and "hit@1" not in change["changes"]


def test_guardrails_are_checked_against_fixed_limits_and_the_baseline():
    baseline = {"all@8": 41 / 44, "hit@5": 38 / 44}       # 93.2% and 86.4% when rounded
    ok = {"resolution_in_top_k": 30, "target_logs_in_top_k": 3, "all@8": 41 / 44,
          "hit@5": 38 / 44, "MRR": 0.68, "unanswerable_gate_changes": 0,
          "evidence_token_growth": 0.25, "median_latency_increase_ms": 50.0}
    # A result identical to the baseline passes, although 41/44 < 0.932 as a literal.
    assert all(rb.check_guardrails(ok, baseline).values())
    bad = {**ok, "resolution_in_top_k": 29, "median_latency_increase_ms": 50.1,
           "hit@5": 37 / 44}
    res = rb.check_guardrails(bad, baseline)
    assert res["resolution_in_top_k"] is False and res["median_latency_increase_ms"] is False
    assert res["hit@5"] is False
    assert rb.check_guardrails({}, baseline)["MRR"] is None  # missing is unknown, not a pass
    assert rb.check_guardrails(ok)["all@8"] is None          # no baseline: cannot judge


def test_reference_differences_are_reported(tmp_path):
    result, _ = run()
    rows = result["by_config"]["base"]
    ref = tmp_path / "per_question_retrieval.csv"
    lines = ["config,id,retrieved_paths,retrieved_sections"]
    for r in rows:
        paths, sections = list(r["retrieved_paths"]), list(r["retrieved_sections"])
        if r["id"] == "Q033":
            paths.reverse()
            sections.reverse()
        lines.append(f'L,{r["id"]},"{json.dumps(paths).replace(chr(34), chr(34) * 2)}",'
                     f'"{json.dumps(sections).replace(chr(34), chr(34) * 2)}"')
    ref.write_text("\n".join(lines), encoding="utf-8")
    assert rb.check_reference(rows, ref, "L") == [{"id": "Q033",
                                                   "problem": "ranked paths or sections differ"}]
    assert rb.check_reference(rows, ref, "missing")[0]["id"] is None


@pytest.fixture
def cli(monkeypatch, tmp_path):
    def factory(name, table):
        return lambda _embedder: Stub(name, table)

    monkeypatch.setattr(rb, "RETRIEVERS", {"base": factory("base", BASE)})
    monkeypatch.setattr(rb, "EXPERIMENTAL_RETRIEVERS", {"cand": factory("cand", BETTER)})
    monkeypatch.setattr(rb, "get_embedder", lambda *a, **k: None)
    monkeypatch.setattr(rb, "load_resolution_sections", lambda: {RB: ()})
    monkeypatch.setattr(rb, "index_fingerprint", lambda: {"documents": 4})
    monkeypatch.setattr(rb, "new_run_id", lambda: "20990101T000000Z")
    questions = tmp_path / "q.jsonl"
    questions.write_text("\n".join(json.dumps(q) for q in QUESTIONS), encoding="utf-8")
    root = tmp_path / "results"

    def main(*args):
        return rb.main(["--questions", str(questions), "--baseline", "base",
                        "--out-root", str(root), "--repeats", "1", *args])
    return main, root


def test_cli_writes_a_new_folder_and_never_reuses_one(cli):
    main, root = cli
    assert main() == 0
    out = root / "20990101T000000Z_retrieval"
    assert sorted(p.name for p in out.iterdir()) == [
        "changes.csv", "comparison.csv", "per_question.csv", "run_metadata.json", "summary.md"]
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        main()                                   # same run id: refuses, writes nothing
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    meta = json.loads((out / "run_metadata.json").read_text("utf-8"))
    assert meta["baseline"] == "base" and meta["candidates"] == ["cand"]
    assert "[REDACTED]" in meta["database"] or "@" not in meta["database"]
    summary = (out / "summary.md").read_text("utf-8")
    assert "No LLM was called" in summary and "Guardrails" in summary


def test_cli_exit_code_flags_a_baseline_that_does_not_reproduce(cli, tmp_path):
    main, _ = cli
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "per_question_retrieval.csv").write_text(
        "config,id,retrieved_paths,retrieved_sections\n", encoding="utf-8")
    assert main("--reference", str(ref), "--reference-label", "nothing") == 1
    assert Path(tmp_path / "results").exists()


def test_traceback_coverage_and_evidence_heuristics_are_measured():
    traced = dict(BETTER, **{"log q": [*BETTER["log q"],
                                       h(LOG, "error context", 0.7, text="Traceback ...",
                                         origin="error_context_companion")]})
    result, _ = run(better=traced)
    base, cand = result["aggregates"]["base"], result["aggregates"]["cand"]
    assert (base["log_error_context_in_top_k"], cand["log_error_context_in_top_k"]) == (0, 1)
    assert cand["log_context_companions"] == 1
    q15 = next(r for r in result["by_config"]["cand"] if r["id"] == "Q015")
    assert "root_cause_term_recall_in_evidence" in q15   # heuristic, from the ground truth
    q45 = next(r for r in result["by_config"]["cand"] if r["id"] == "Q045")
    assert q45["remediation_covered_in_evidence"] is None  # unanswerable: not scored
