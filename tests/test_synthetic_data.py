import hashlib
import json
import re
from datetime import date


def load_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def load_questions(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_file_counts(dataset):
    assert dataset["result"]["counts"] == {"runbook": 18, "log": 60, "schema": 12, "incident": 14}
    assert dataset["result"]["questions"] == 50


def test_generation_is_deterministic(dataset, tmp_path):
    gen = dataset["gen"]
    gen.write_all(tmp_path, tmp_path / "q.jsonl", gen.DEFAULT_SEED)
    original = (dataset["root"] / "manifest.json").read_text()
    assert (tmp_path / "manifest.json").read_text() == original
    assert (tmp_path / "q.jsonl").read_text() == dataset["eval"].read_text()


def test_different_seed_changes_output(dataset, tmp_path):
    gen = dataset["gen"]
    gen.write_all(tmp_path, tmp_path / "q.jsonl", 7)
    original = (dataset["root"] / "manifest.json").read_text()
    assert (tmp_path / "manifest.json").read_text() != original


def test_output_uses_unix_line_endings(dataset):
    """CRLF would change file hashes and make Windows output differ from the committed copy."""
    root = dataset["root"]
    for path in [*root.rglob("*.md"), *root.rglob("*.log"), root / "manifest.json",
                 dataset["eval"]]:
        assert b"\r" not in path.read_bytes(), path.name


def test_manifest_hashes_match_files(dataset):
    root = dataset["root"]
    for entry in load_manifest(root)["files"]:
        data = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        assert len(data) == entry["bytes"]


def test_one_log_per_pipeline_per_day(dataset):
    seen = set()
    for e in load_manifest(dataset["root"])["files"]:
        if e["source_type"] == "log":
            key = (e["pipeline"], e["date"])
            assert key not in seen, key
            seen.add(key)


def test_every_runbook_has_required_sections(dataset):
    for path in (dataset["root"] / "runbooks").glob("*.md"):
        text = path.read_text()
        for heading in ("## Symptoms", "## Likely causes", "## Diagnosis", "## Resolution",
                        "## Prevention"):
            assert heading in text, f"{path.name} missing {heading}"


def test_eval_ids_are_unique_and_sequential(dataset):
    ids = [q["external_id"] for q in load_questions(dataset["eval"])]
    assert ids == [f"Q{i:03d}" for i in range(1, 51)]


def test_every_expected_path_exists(dataset):
    for q in load_questions(dataset["eval"]):
        for rel in q["expected_paths"]:
            assert (dataset["root"] / rel).exists(), f"{q['external_id']}: {rel}"


def test_unanswerable_questions_expect_abstention(dataset):
    qs = [q for q in load_questions(dataset["eval"]) if q["category"] == "unanswerable"]
    assert len(qs) == 6
    for q in qs:
        assert q["should_abstain"] is True
        assert q["expected_paths"] == []
    assert all(not q["should_abstain"] for q in load_questions(dataset["eval"])
               if q["category"] != "unanswerable")


def test_log_diagnosis_answers_are_grounded_in_the_log(dataset):
    """The error quoted in each ground truth must literally appear in the expected log."""
    qs = [q for q in load_questions(dataset["eval"]) if q["category"] == "log_diagnosis"]
    assert len(qs) == 18
    for q in qs:
        error_line = q["ground_truth"].split(". Root cause:")[0].removeprefix("It failed with: ")
        first_line = error_line.splitlines()[0]
        log_path = next(p for p in q["expected_paths"] if p.startswith("logs/"))
        assert first_line in (dataset["root"] / log_path).read_text(), q["external_id"]


def test_question_dates_and_tasks_identify_exactly_one_log(dataset):
    manifest = load_manifest(dataset["root"])
    by_key = {(e["pipeline"], e["task"], e["date"]) for e in manifest["files"]
              if e["source_type"] == "log"}
    pat = re.compile(r"Why did the (\w+) task in the (\w+) pipeline fail on (\d{4}-\d{2}-\d{2})\?")
    for q in load_questions(dataset["eval"]):
        m = pat.fullmatch(q["question"])
        if m:
            task, pipeline, day = m.groups()
            assert (pipeline, task, day) in by_key, q["question"]


def test_rename_failures_only_happen_after_the_rename(dataset):
    gen = dataset["gen"]
    renamed = [e for e in load_manifest(dataset["root"])["files"]
               if e.get("scenario") == "column-renamed-upstream" and e["source_type"] == "log"]
    assert renamed
    for e in renamed:
        assert date.fromisoformat(e["date"]) >= gen.RENAME_DATE, e["path"]


def test_fake_secrets_are_present_only_where_flagged(dataset):
    root = dataset["root"]
    flagged = {e["path"] for e in load_manifest(root)["files"] if e.get("has_fake_secret")}
    assert len(flagged) == 8
    for e in load_manifest(root)["files"]:
        has = "FAKE-" in (root / e["path"]).read_text()
        assert has == (e["path"] in flagged), e["path"]


def test_no_realistic_secrets_are_generated(dataset):
    patterns = [r"AKIA[0-9A-Z]{16}", r"BEGIN (RSA )?PRIVATE KEY", r"xox[bp]-\d+"]
    for path in dataset["root"].rglob("*.*"):
        if path.suffix in {".md", ".log"}:
            text = path.read_text()
            for pat in patterns:
                assert not re.search(pat, text), f"{path.name} matches {pat}"


def test_incident_ids_follow_chronological_order(dataset):
    incidents = [e for e in load_manifest(dataset["root"])["files"]
                 if e["source_type"] == "incident"]
    by_id = sorted(incidents, key=lambda e: e["incident_id"])
    dates = []
    for e in by_id:
        text = (dataset["root"] / e["path"]).read_text()
        dates.append(re.search(r"\*\*Date:\*\* (\S+)", text).group(1))
    assert dates == sorted(dates)


def test_freshness_failures_exceed_their_own_sla(dataset):
    pat = re.compile(r"is (\d+)h \d+m old, SLA is (\d+) hours")
    found = 0
    for path in (dataset["root"] / "logs").rglob("*.log"):
        for m in pat.finditer(path.read_text()):
            found += 1
            assert int(m.group(1)) > int(m.group(2)), path.name
    assert found >= 1


def test_schema_docs_exist_for_every_pipeline_table(dataset):
    gen = dataset["gen"]
    for cfg in gen.PIPELINES.values():
        for table in (cfg["fact"], cfg["raw"]):
            assert (dataset["root"] / "schemas" / f"{table}.md").exists(), table
