"""Runbook, incident and schema chunkers, plus stable keys and hashes for all chunks."""

from incident_copilot.chunking import CHUNKERS, DESCRIBERS, normalize_text, sha256_text
from incident_copilot.chunking.schemas import chunk_schema, describe_schema
from incident_copilot.chunking.sections import chunk_incident, chunk_runbook, describe_runbook

RUNBOOK = "runbooks/column-renamed-upstream.md"
INCIDENT = "incidents/INC-2031.md"
SCHEMA = "schemas/orders_raw.md"


def read(dataset, rel):
    return (dataset["root"] / rel).read_text(encoding="utf-8")


# ---- runbooks ---------------------------------------------------------------------------


def test_runbook_is_split_on_level_two_headings(dataset):
    chunks = chunk_runbook(read(dataset, RUNBOOK), RUNBOOK)
    assert [c.section for c in chunks] == [
        "Overview", "Symptoms", "Likely causes", "Diagnosis", "Resolution", "Prevention",
    ]
    for c in chunks:
        assert c.text.startswith("# Runbook: Column renamed upstream breaks a transform\n")
    diagnosis = next(c for c in chunks if c.section == "Diagnosis")
    assert "## Diagnosis" in diagnosis.text
    assert "Check the table's schema doc change history" in diagnosis.text
    assert "## Resolution" not in diagnosis.text


def test_runbook_metadata(dataset):
    info = describe_runbook(read(dataset, RUNBOOK), RUNBOOK)
    assert info.pipelines == ["orders_etl", "customer_360"]
    assert info.pipeline_name is None  # applies to two pipelines
    assert info.metadata["runbook_id"] == "RB-001"
    chunks = chunk_runbook(read(dataset, RUNBOOK), RUNBOOK)
    assert all(c.metadata["pipelines"] == ["orders_etl", "customer_360"] for c in chunks)


def test_every_runbook_keeps_all_required_sections(dataset):
    for p in (dataset["root"] / "runbooks").glob("*.md"):
        rel = f"runbooks/{p.name}"
        sections = {c.section for c in CHUNKERS["runbook"](p.read_text(encoding="utf-8"), rel)}
        assert {"Symptoms", "Likely causes", "Diagnosis", "Resolution", "Prevention"} <= sections


# ---- incidents --------------------------------------------------------------------------


def test_incident_reports_are_split_by_heading_with_title(dataset):
    chunks = chunk_incident(read(dataset, INCIDENT), INCIDENT)
    sections = [c.section for c in chunks]
    assert sections[0] == "Overview"
    assert {"Summary", "Timeline (UTC)", "Root cause", "Resolution"} <= set(sections)
    for c in chunks:
        assert c.text.startswith(
            "# Incident report INC-2031: Row count dropped sharply (partial data) in orders_etl"
        )
    root_cause = next(c for c in chunks if c.section == "Root cause")
    assert "extract was only partial" in root_cause.text


def test_incident_metadata(dataset):
    info = DESCRIBERS["incident"](read(dataset, INCIDENT), INCIDENT)
    assert info.pipeline_name == "orders_etl"
    assert info.metadata["incident_id"] == "INC-2031"
    assert info.metadata["failed_task"] == "validate_orders"
    assert info.metadata["date"] == "2026-08-09"
    assert info.metadata["severity"] == "SEV-2"


def test_every_incident_is_indexed_with_a_pipeline(dataset):
    files = sorted((dataset["root"] / "incidents").glob("*.md"))
    assert len(files) == 14
    for p in files:
        rel = f"incidents/{p.name}"
        text = p.read_text(encoding="utf-8")
        assert len(CHUNKERS["incident"](text, rel)) >= 4
        assert DESCRIBERS["incident"](text, rel).pipeline_name


# ---- schemas ----------------------------------------------------------------------------


def test_schema_is_one_chunk_with_the_whole_file(dataset):
    text = read(dataset, SCHEMA)
    chunks = chunk_schema(text, SCHEMA)
    assert len(chunks) == 1
    assert chunks[0].text == text.strip()
    assert chunks[0].section == "orders_raw"
    assert chunks[0].metadata["table_name"] == "orders_raw"
    assert "Renamed from customer_tier on 2026-08-28" in chunks[0].text
    assert "## Change history" in chunks[0].text


def test_schema_pipeline_comes_from_populated_by(dataset):
    assert describe_schema(read(dataset, SCHEMA), SCHEMA).pipeline_name is None  # CDC
    fact = "schemas/orders_fact.md"
    info = describe_schema(read(dataset, fact), fact)
    assert info.pipeline_name == "orders_etl"
    assert info.metadata["populated_by_task"] == "load_orders_fact"


def test_every_schema_is_exactly_one_chunk(dataset):
    files = list((dataset["root"] / "schemas").glob("*.md"))
    assert len(files) == 12
    for p in files:
        assert len(CHUNKERS["schema"](p.read_text(encoding="utf-8"), f"schemas/{p.name}")) == 1


# ---- stable keys and hashes -------------------------------------------------------------


def test_chunk_keys_and_hashes_are_stable(dataset):
    for rel, source_type in [(RUNBOOK, "runbook"), (INCIDENT, "incident"), (SCHEMA, "schema"),
                             ("logs/orders_etl/2026-09-04__transform_orders.log", "log")]:
        text = read(dataset, rel)
        first = CHUNKERS[source_type](text, rel)
        second = CHUNKERS[source_type](text, rel)
        assert [c.content_hash for c in first] == [c.content_hash for c in second]
        assert [c.key(rel) for c in first] == [f"{rel}#{i}" for i in range(len(first))]
        assert all(len(c.content_hash) == 64 for c in first)


def test_hash_ignores_line_endings_and_trailing_spaces(dataset):
    text = read(dataset, RUNBOOK)
    crlf = text.replace("\n", "  \r\n")
    a = [c.content_hash for c in chunk_runbook(text, RUNBOOK)]
    b = [c.content_hash for c in chunk_runbook(crlf, RUNBOOK)]
    assert a == b
    assert normalize_text("a  \r\nb\r\n\r\n") == "a\nb"
    assert sha256_text("x\n") == sha256_text("x")


def test_editing_one_section_changes_only_that_chunk_hash(dataset):
    text = read(dataset, RUNBOOK)
    edited = text.replace("## Prevention\n", "## Prevention\n\n- Add an alert.\n")
    before = {c.section: c.content_hash for c in chunk_runbook(text, RUNBOOK)}
    after = {c.section: c.content_hash for c in chunk_runbook(edited, RUNBOOK)}
    changed = {s for s in before if before[s] != after[s]}
    assert changed == {"Prevention"}
