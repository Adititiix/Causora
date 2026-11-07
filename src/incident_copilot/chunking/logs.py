"""Airflow task-log chunker. Pure functions, no I/O.

Line format: `[YYYY-MM-DDTHH:MM:SS.mmm+0000] {module} LEVEL - text`. Lines without that
prefix (tracebacks) belong to the preceding entry, so a traceback is never split. Lines
before the first entry (Airflow's `*** Reading local file: ...` banner) are a preamble that
travels with the first entry.

Each log becomes 1 to 3 chunks:
  run summary    the first and last entries of the run together (start + outcome)
  error context  the failing entry with its full traceback and up to 8 preceding entries
  middle         whatever neither of the above covers, if anything

Every chunk starts with a one-line header (pipeline / task / date / status / error), because
chunks are embedded in isolation. The header carries the final exception line so it is
always inside the embedding model's 512-token window even when the chunk body is longer.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from incident_copilot.chunking.models import Chunk, DocumentInfo

ERROR_CONTEXT_ENTRIES = 8
SUMMARY_HEAD = 3
SUMMARY_TAIL = 3
# A log with no error and at most this many entries stays a single chunk.
SINGLE_CHUNK_MAX_ENTRIES = 8

ENTRY_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4})\] "
    r"\{(?P<module>[^}]*)\} (?P<level>[A-Z]+) - (?P<msg>.*)$"
)
PATH_RE = re.compile(
    r"(?:^|/)logs/(?P<pipeline>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2})__(?P<task>[^/]+)\.log$"
)
PREAMBLE_RE = re.compile(r"dag_id=(?P<dag>[^/\s]+)/.*task_id=(?P<task>[^/\s]+)/attempt=(?P<n>\d+)")
STATUS_RE = re.compile(r"Marking task as (?P<status>[A-Z_]+)")
ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}


@dataclass
class LogEntry:
    ts: datetime
    module: str
    level: str
    message: str
    lines: list[str] = field(default_factory=list)  # first line + continuation lines

    @property
    def has_traceback(self) -> bool:
        return any(line.startswith("Traceback") for line in self.lines[1:])

    @property
    def exception_line(self) -> str:
        """The exception line of the (last) traceback, e.g. `KeyError: 'customer_tier'`.

        Python prints indented stack frames, then the unindented exception line, then
        optional extra lines such as psycopg's `DETAIL: ...`, so it is the first unindented
        line after the frames, not necessarily the last line. Falls back to the message.
        """
        if self.has_traceback:
            body = self.lines[1:]
            start = max(i for i, line in enumerate(body) if line.startswith("Traceback"))
            for line in body[start + 1:]:
                if line.strip() and not line[0].isspace():
                    return line.strip()
        return self.message


@dataclass
class ParsedLog:
    preamble: list[str]
    entries: list[LogEntry]
    pipeline: str | None
    task: str | None
    date: str | None
    status: str
    attempt: int | None
    error_index: int | None

    @property
    def error(self) -> str | None:
        if self.error_index is None:
            return None
        return self.entries[self.error_index].exception_line[:200]


def parse_log(text: str, path: str = "") -> ParsedLog:
    preamble: list[str] = []
    entries: list[LogEntry] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = ENTRY_RE.match(line)
        if m:
            entries.append(
                LogEntry(
                    ts=datetime.strptime(m["ts"], "%Y-%m-%dT%H:%M:%S.%f%z"),
                    module=m["module"],
                    level=m["level"],
                    message=m["msg"],
                    lines=[line],
                )
            )
        elif entries:
            if line.strip():
                entries[-1].lines.append(line)
        elif line.strip():
            preamble.append(line)

    pipeline = task = date = None
    attempt = None
    if pm := PATH_RE.search(path):
        pipeline, date, task = pm["pipeline"], pm["date"], pm["task"]
    if am := PREAMBLE_RE.search("\n".join(preamble)):
        pipeline = pipeline or am["dag"]
        task = task or am["task"]
        attempt = int(am["n"])
    if date is None and entries:
        date = entries[0].ts.date().isoformat()

    # Anchor of the error window: the entry carrying a traceback, else the first ERROR.
    error_index = next((i for i, e in enumerate(entries) if e.has_traceback), None)
    if error_index is None:
        error_index = next((i for i, e in enumerate(entries) if e.level in ERROR_LEVELS), None)

    status = "UNKNOWN"
    for entry in entries:
        if sm := STATUS_RE.search(entry.message):
            status = sm["status"]
    if status == "UNKNOWN" and error_index is not None:
        status = "FAILED"

    return ParsedLog(preamble, entries, pipeline, task, date, status, attempt, error_index)


def _header(log: ParsedLog) -> str:
    parts = [log.pipeline or "unknown-pipeline", log.task or "unknown-task",
             log.date or "unknown-date", log.status]
    header = "Log: " + " / ".join(parts)
    if log.error:
        header += f" / error: {log.error}"
    return header


def _render(log: ParsedLog, indexes: list[int], with_preamble: bool) -> str:
    """Entries in order, with a marker line where entries were skipped."""
    out: list[str] = [_header(log)]
    if with_preamble:
        out.extend(log.preamble)
    previous = None
    for i in indexes:
        if previous is not None and i > previous + 1:
            out.append(f"... ({i - previous - 1} log entries omitted) ...")
        out.extend(log.entries[i].lines)
        previous = i
    return "\n".join(out)


def _plan(log: ParsedLog) -> list[tuple[str, list[int]]]:
    """Which entries go into which chunk, as (section, sorted entry indexes)."""
    n = len(log.entries)
    everything = list(range(n))
    head = set(range(min(SUMMARY_HEAD, n)))
    tail = set(range(max(0, n - SUMMARY_TAIL), n))
    summary = head | tail

    if log.error_index is None:
        if n <= SINGLE_CHUNK_MAX_ENTRIES:
            return [("run", everything)]
        summary = set(range(min(4, n))) | set(range(max(0, n - 4), n))
        middle = sorted(set(everything) - summary)
        return [("run summary", sorted(summary))] + ([("middle", middle)] if middle else [])

    a = log.error_index
    error = set(range(max(0, a - ERROR_CONTEXT_ENTRIES), a + 1))
    plan: list[tuple[str, list[int]]] = []
    if not summary <= error:
        plan.append(("run summary", sorted(summary)))
    plan.append(("error context", sorted(error)))
    middle = sorted(set(everything) - summary - error)
    if middle:
        plan.append(("middle", middle))
    return plan


def chunk_log(text: str, path: str) -> list[Chunk]:
    log = parse_log(text, path)
    base_meta = {"pipeline": log.pipeline, "task": log.task, "date": log.date,
                 "status": log.status}
    if log.error:
        base_meta["error"] = log.error

    if not log.entries:  # not in Airflow format: keep it whole rather than drop it
        body = "\n".join([_header(log), text.strip()])
        return [Chunk(index=0, text=body, section="run", metadata={**base_meta, "part": "run"})]

    chunks = []
    preamble_placed = False
    for index, (section, idx) in enumerate(_plan(log)):
        entries = [log.entries[i] for i in idx]
        # The preamble travels with the first chunk that holds entry 0 (only once).
        with_preamble = not preamble_placed and 0 in idx
        preamble_placed = preamble_placed or with_preamble
        chunks.append(
            Chunk(
                index=index,
                text=_render(log, idx, with_preamble),
                section=section,
                ts_start=min(e.ts for e in entries),
                ts_end=max(e.ts for e in entries),
                metadata={**base_meta, "part": section,
                          "levels": sorted({e.level for e in entries})},
            )
        )
    return chunks


def describe_log(text: str, path: str) -> DocumentInfo:
    log = parse_log(text, path)
    meta = {"task": log.task, "date": log.date, "status": log.status}
    if log.attempt is not None:
        meta["attempt"] = log.attempt
    if log.error:
        meta["error"] = log.error
    title = " / ".join(p for p in (log.pipeline, log.task, log.date) if p) or None
    return DocumentInfo(
        title=title,
        pipeline_name=log.pipeline,
        pipelines=[log.pipeline] if log.pipeline else [],
        metadata=meta,
    )
