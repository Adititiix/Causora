"""Table-schema chunker: one chunk per table containing the whole file (columns, types,
descriptions, notes, change history). Schema docs are small (under ~300 tokens), and
splitting them would separate a column from its change history. Pure functions, no I/O.
"""

import re
from pathlib import PurePosixPath

from incident_copilot.chunking.models import Chunk, DocumentInfo
from incident_copilot.chunking.sections import parse_markdown

TABLE_TITLE_RE = re.compile(r"^Table:\s*(?P<table>\S+)")
POPULATED_BY_RE = re.compile(r"^(?P<pipeline>[a-z0-9_]+) \(task (?P<task>[a-z0-9_]+)\)")


def _table_name(title: str, path: str) -> str:
    if m := TABLE_TITLE_RE.match(title):
        return m["table"]
    return PurePosixPath(path).stem


def chunk_schema(text: str, path: str) -> list[Chunk]:
    doc = parse_markdown(text, path)
    table = _table_name(doc.title, path)
    body = text.replace("\r\n", "\n").strip()
    if not body.startswith("# "):
        body = f"# Table: {table}\n\n{body}"
    info = describe_schema(text, path)
    return [
        Chunk(
            index=0,
            text=body,
            section=table,
            metadata={"table_name": table, "pipelines": info.pipelines},
        )
    ]


def describe_schema(text: str, path: str) -> DocumentInfo:
    doc = parse_markdown(text, path)
    table = _table_name(doc.title, path)
    meta: dict = {**doc.facts, "table_name": table}
    pipelines = []
    if m := POPULATED_BY_RE.match(doc.facts.get("populated_by", "")):
        pipelines = [m["pipeline"]]
        meta["populated_by_task"] = m["task"]
    return DocumentInfo(
        title=doc.title,
        pipeline_name=pipelines[0] if pipelines else None,
        pipelines=pipelines,
        metadata=meta,
    )
