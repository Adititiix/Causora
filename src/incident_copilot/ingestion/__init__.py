"""Day 2-9: loading files, hashing, embedding, upserting.

The core function `ingest_paths(paths, ...)` is plain Python with no FastAPI, CLI, or Airflow
imports, so the same code runs from a CLI (Day 2), an Airflow task (Day 8), and the
API (`POST /ingest`).
"""

from incident_copilot.ingestion.ingest import IndexingLockedError, IngestSummary, ingest_paths
from incident_copilot.ingestion.sources import UnsafePathError

__all__ = ["IndexingLockedError", "IngestSummary", "UnsafePathError", "ingest_paths"]
