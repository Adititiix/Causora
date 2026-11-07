"""compare_runs on two tiny synthetic run folders (offline)."""

import json

import pandas as pd

from incident_copilot.evaluation import compare_runs
from incident_copilot.evaluation.metrics import retrieval_metrics

LOG = "logs/p/2026-09-04__t.log"
OTHER = "logs/p/2026-09-21__t.log"
RB = "runbooks/r.md"


def write_run(folder, run_id, default, rows, refused):
    folder.mkdir()
    records = []
    for config, qid, category, expected, retrieved in rows:
        records.append({"config": config, "id": qid, "category": category,
                        "should_abstain": not expected, "expected_paths": json.dumps(expected),
                        "retrieved_paths": json.dumps(retrieved),
                        **retrieval_metrics(expected, retrieved)})
    pd.DataFrame(records).to_csv(folder / "per_question_retrieval.csv", index=False)
    (folder / "run_metadata.json").write_text(json.dumps(
        {"run_id": run_id, "default_retriever": default, "log_error_context_entries": 8,
         "min_similarity": 0.72}), "utf-8")
    pd.DataFrame([{"id": q, "category": "c", "should_abstain": a, "best_similarity": s,
                   "refused": r} for q, a, s, r in refused]).to_csv(
        folder / "abstention_per_question.csv", index=False)


def test_compare_reports_improvements_regressions_and_gate_changes(tmp_path, capsys):
    a, b = "old | w8 (production index)", "new | w8 (production index)"
    write_run(tmp_path / "base", "BASE", "old", [
        (a, "Q1", "log_diagnosis", [LOG, RB], [OTHER, RB, "x"]),
        (a, "Q2", "runbook", [RB], [RB, "x", "y"]),
        (a, "Q3", "unanswerable", [], ["x"]),
    ], [("Q1", False, 0.8, False), ("Q3", True, 0.75, False)])
    write_run(tmp_path / "new", "NEW", "new", [
        (b, "Q1", "log_diagnosis", [LOG, RB], [LOG, RB, "x"]),
        (b, "Q2", "runbook", [RB], ["x", RB, "y"]),
        (b, "Q3", "unanswerable", [], ["x"]),
    ], [("Q1", False, 0.8, False), ("Q3", True, 0.70, True)])
    assert compare_runs.main([str(tmp_path / "base"), str(tmp_path / "new")]) == 0
    out = capsys.readouterr().out
    assert "1 improved, 1 regressed or mixed" in out
    assert "refusal decisions changed for 1 questions" in out
    md = (tmp_path / "new" / "compare_BASE.md").read_text("utf-8")
    assert "old | w8 (production index)  ->  new | w8 (production index)" in md
    assert (tmp_path / "new" / "compare_BASE.csv").exists()
    assert not (tmp_path / "base" / "compare_BASE.md").exists()  # baseline is read-only
