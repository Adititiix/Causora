from datetime import UTC

from incident_copilot.chunking import CHUNKERS
from incident_copilot.chunking.logs import (
    ENTRY_RE,
    ERROR_CONTEXT_ENTRIES,
    chunk_log,
    describe_log,
    parse_log,
)

KEYERROR_LOG = "logs/orders_etl/2026-09-04__transform_orders.log"


def read(dataset, rel):
    return (dataset["root"] / rel).read_text(encoding="utf-8")


def all_logs(dataset):
    root = dataset["root"]
    for p in sorted((root / "logs").rglob("*.log")):
        yield p.relative_to(root).as_posix(), p.read_text(encoding="utf-8")


def by_section(chunks):
    return {c.section: c for c in chunks}


def test_log_chunker_is_registered():
    assert CHUNKERS["log"] is chunk_log


def test_traceback_lines_belong_to_the_preceding_entry(dataset):
    log = parse_log(read(dataset, KEYERROR_LOG), KEYERROR_LOG)
    for entry in log.entries:
        assert ENTRY_RE.match(entry.lines[0])
    failing = log.entries[log.error_index]
    assert failing.level == "ERROR"
    assert failing.lines[1] == "Traceback (most recent call last):"
    assert failing.lines[-1] == "KeyError: 'customer_tier'"
    assert failing.exception_line == "KeyError: 'customer_tier'"
    assert log.preamble[0].startswith("*** Reading local file:")


def test_every_log_gives_one_to_three_chunks_with_a_context_header(dataset):
    for rel, text in all_logs(dataset):
        chunks = chunk_log(text, rel)
        assert 1 <= len(chunks) <= 3, rel
        assert [c.index for c in chunks] == list(range(len(chunks)))
        pipeline = rel.split("/")[1]
        day, task = rel.split("/")[2].removesuffix(".log").split("__")
        status = "SUCCESS" if "Marking task as SUCCESS" in text else "FAILED"
        for c in chunks:
            assert c.text.splitlines()[0].startswith(
                f"Log: {pipeline} / {task} / {day} / {status}"
            ), rel


def test_a_traceback_is_never_split_across_chunks(dataset):
    for rel, text in all_logs(dataset):
        log = parse_log(text, rel)
        chunks = chunk_log(text, rel)
        for entry in (e for e in log.entries if e.has_traceback):
            whole = "\n".join(entry.lines)
            holders = [c for c in chunks if whole in c.text]
            assert holders, f"{rel}: traceback missing"
            for c in chunks:
                if c not in holders:
                    assert "Traceback (most recent call last):" not in c.text, rel


def test_error_chunk_has_traceback_and_up_to_eight_preceding_entries(dataset):
    text = read(dataset, KEYERROR_LOG)
    log = parse_log(text, KEYERROR_LOG)
    error = by_section(chunk_log(text, KEYERROR_LOG))["error context"]
    assert "KeyError: 'customer_tier'" in error.text
    assert "Loaded 217,392 rows from orders_raw" in error.text  # the clue just before it
    entry_lines = [ln for ln in error.text.splitlines() if ENTRY_RE.match(ln)]
    expected = min(ERROR_CONTEXT_ENTRIES, log.error_index) + 1
    assert len(entry_lines) == expected
    assert entry_lines[-1] == log.entries[log.error_index].lines[0]


def test_run_summary_keeps_start_and_end_of_the_run_together(dataset):
    summary = by_section(chunk_log(read(dataset, KEYERROR_LOG), KEYERROR_LOG))["run summary"]
    assert "Starting attempt 3 of 3" in summary.text
    assert "Marking task as FAILED" in summary.text
    assert "log entries omitted" in summary.text


def test_every_log_entry_appears_in_some_chunk(dataset):
    for rel, text in all_logs(dataset):
        chunks = chunk_log(text, rel)
        combined = "\n".join(c.text for c in chunks)
        for line in text.splitlines():
            if line.strip():
                assert line in combined, f"{rel}: lost line {line[:60]}"


def test_timestamps_are_utc_and_cover_the_chunk(dataset):
    for rel, text in all_logs(dataset):
        for c in chunk_log(text, rel):
            assert c.ts_start.tzinfo is not None and c.ts_start.utcoffset() == UTC.utcoffset(None)
            assert c.ts_start <= c.ts_end
            first_entry = next(ln for ln in c.text.splitlines() if ENTRY_RE.match(ln))
            assert first_entry.startswith("[" + c.ts_start.strftime("%Y-%m-%dT%H:%M:%S"))


def test_log_metadata_has_task_status_and_error(dataset):
    chunks = chunk_log(read(dataset, KEYERROR_LOG), KEYERROR_LOG)
    for c in chunks:
        assert c.metadata["pipeline"] == "orders_etl"
        assert c.metadata["task"] == "transform_orders"
        assert c.metadata["date"] == "2026-09-04"
        assert c.metadata["status"] == "FAILED"
        assert c.metadata["error"] == "KeyError: 'customer_tier'"
    info = describe_log(read(dataset, KEYERROR_LOG), KEYERROR_LOG)
    assert info.pipeline_name == "orders_etl"
    assert info.pipelines == ["orders_etl"]
    assert info.metadata["attempt"] == 3
    assert info.title == "orders_etl / transform_orders / 2026-09-04"


def test_successful_log_has_success_status_and_no_error(dataset):
    rel, text = next((r, t) for r, t in all_logs(dataset) if "Marking task as SUCCESS" in t)
    chunks = chunk_log(text, rel)
    assert all(c.metadata["status"] == "SUCCESS" for c in chunks)
    assert all("error" not in c.metadata for c in chunks)
    assert "error:" not in chunks[0].text.splitlines()[0]


def test_small_log_without_error_is_one_chunk():
    text = "\n".join(
        f"[2026-01-01T00:00:0{i}.000+0000] {{x.py:1}} INFO - step {i}" for i in range(3)
    )
    chunks = chunk_log(text, "logs/p1/2026-01-01__t1.log")
    assert len(chunks) == 1
    assert chunks[0].text.splitlines()[0] == "Log: p1 / t1 / 2026-01-01 / UNKNOWN"


def test_text_not_in_airflow_format_is_kept_whole():
    chunks = chunk_log("just some text\nmore text", "logs/p/2026-01-01__t.log")
    assert len(chunks) == 1
    assert "just some text\nmore text" in chunks[0].text


def test_exception_line_is_not_a_trailing_detail_line(dataset):
    """psycopg prints `DETAIL:` after the exception; the header must carry the exception."""
    rel = "logs/orders_etl/2026-08-31__load_orders_fact.log"
    log = parse_log(read(dataset, rel), rel)
    assert log.entries[log.error_index].lines[-1].startswith("DETAIL:")
    assert log.error.startswith("psycopg2.errors.NotNullViolation: null value in column")
    assert "error: psycopg2.errors.NotNullViolation" in chunk_log(read(dataset, rel), rel)[0].text
    for r, text in all_logs(dataset):
        parsed = parse_log(text, r)
        if parsed.error:
            assert not parsed.error.startswith(("DETAIL:", "CONTEXT:", "HINT:")), r


def test_error_without_traceback_anchors_on_first_error_entry():
    lines = [f"[2026-01-01T00:00:{i:02d}.000+0000] {{x.py:1}} INFO - step {i}" for i in range(12)]
    lines.insert(10, "[2026-01-01T00:00:10.500+0000] {x.py:1} ERROR - Freshness check failed")
    chunks = chunk_log("\n".join(lines), "logs/p/2026-01-01__t.log")
    error = by_section(chunks)["error context"]
    assert "ERROR - Freshness check failed" in error.text
    assert chunks[0].metadata["error"] == "Freshness check failed"
    assert chunks[0].metadata["status"] == "FAILED"
