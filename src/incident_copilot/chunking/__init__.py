"""Day 2: one chunker per source type (runbook by heading, log by window, schema per table).

Convention: each chunker is a function `chunk_<type>(text, path) -> list[Chunk]`
registered in a dict keyed by source_type, so ingestion never needs an if/else chain.
Each module also has `describe_<type>(text, path) -> DocumentInfo` for document-level facts.
All of it is pure: no file or database access.
"""

from collections.abc import Callable

from incident_copilot.chunking.logs import chunk_log, describe_log
from incident_copilot.chunking.models import Chunk, DocumentInfo, normalize_text, sha256_text
from incident_copilot.chunking.schemas import chunk_schema, describe_schema
from incident_copilot.chunking.sections import (
    chunk_incident,
    chunk_runbook,
    describe_incident,
    describe_runbook,
)

# Bump when chunking output changes, so unchanged files are re-chunked on the next run.
# v2: log headers use the real exception line, not a trailing `DETAIL:` line.
CHUNKER_VERSION = 2

CHUNKERS: dict[str, Callable[[str, str], list[Chunk]]] = {
    "log": chunk_log,
    "runbook": chunk_runbook,
    "incident": chunk_incident,
    "schema": chunk_schema,
}

DESCRIBERS: dict[str, Callable[[str, str], DocumentInfo]] = {
    "log": describe_log,
    "runbook": describe_runbook,
    "incident": describe_incident,
    "schema": describe_schema,
}

__all__ = [
    "CHUNKERS", "CHUNKER_VERSION", "DESCRIBERS", "Chunk", "DocumentInfo", "normalize_text",
    "sha256_text",
]
