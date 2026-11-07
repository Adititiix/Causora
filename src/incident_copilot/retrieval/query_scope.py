"""Extract a date, pipeline and task from a question, and decide whether it is safe to use
them as a retrieval filter. Pure functions; the log catalogue is passed in.

The filter only ever narrows LOG chunks (runbooks, schemas and incidents are never excluded),
and it is applied only when all of these hold:
  * exactly one unambiguous calendar date is found (ISO `2026-09-04`, `2026/09/04`,
    `September 4, 2026`, `4 Sep 2026`, or a d/m/y date whose day is > 12);
  * no ambiguous date form is present (`09/04/2026`, `Sep 4` without a year, `yesterday`,
    an impossible date such as `2026-02-30`, two different dates);
  * every identifier-like word (containing `_` or a digit) written next to "pipeline",
    "task", "dag" or "job" is a known pipeline or task (an unknown one, such as a pipeline
    that is not indexed, means we cannot tell what the question is about; plain words such
    as "populates" are ignored);
  * the pipeline and task, if both found, belong together;
  * at least one indexed log matches the resulting (date, pipeline, task) scope.
Otherwise retrieval runs unfiltered and `reason` says why.
"""

import calendar
import re
from dataclasses import dataclass, field
from datetime import date

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS |= {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
MONTHS["sept"] = 9
_MONTH = r"(?P<month>" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"

ISO_RE = re.compile(r"(?<![\d/.-])(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})(?![\d/.-])")
DMY_NUMERIC_RE = re.compile(
    r"(?<![\d/.-])(?P<a>\d{1,2})[/.](?P<b>\d{1,2})[/.](?P<y>\d{4})(?![\d/.-])")
MONTH_FIRST_RE = re.compile(rf"(?i)\b{_MONTH}\s+{_DAY}(?:,?\s+(?P<y>\d{{4}}))?\b")
DAY_FIRST_RE = re.compile(rf"(?i)\b{_DAY}\s+(?:of\s+)?{_MONTH}(?:,?\s+(?P<y>\d{{4}}))?\b")
RELATIVE_RE = re.compile(
    r"(?i)\b(today|yesterday|tonight|last\s+(?:night|week|month|run)|this\s+(?:morning|week)"
    r"|\d+\s+days?\s+ago|day\s+before\s+yesterday)\b")
MENTION_RE = re.compile(r"(?i)\b([a-z][a-z0-9_]*)\s+(?:pipeline|task|dag|job)\b"
                        r"|\b(?:pipeline|task|dag|job)\s+([a-z][a-z0-9_]*)\b")
IDENT_RE = re.compile(r"[a-z][a-z0-9_]*")


def _identifier_like(word: str) -> bool:
    return "_" in word or any(ch.isdigit() for ch in word)


@dataclass(frozen=True)
class LogCatalog:
    """(pipeline, task, date) of every indexed log document."""

    entries: frozenset[tuple[str, str, str]] = frozenset()

    @property
    def pipelines(self) -> set[str]:
        return {p for p, _, _ in self.entries}

    @property
    def tasks(self) -> set[str]:
        return {t for _, t, _ in self.entries}

    def pipelines_of(self, task: str) -> set[str]:
        return {p for p, t, _ in self.entries if t == task}

    def count(self, day: str | None, pipeline: str | None, task: str | None) -> int:
        return sum((day is None or d == day) and (pipeline is None or p == pipeline)
                   and (task is None or t == task) for p, t, d in self.entries)


@dataclass(frozen=True)
class QueryScope:
    applied: bool
    reason: str
    date: str | None = None
    pipeline: str | None = None
    task: str | None = None
    dates_found: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    unknown_identifiers: tuple[str, ...] = ()
    pipelines_found: tuple[str, ...] = ()
    tasks_found: tuple[str, ...] = ()
    matching_logs: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def _valid(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def extract_dates(text: str) -> tuple[list[str], list[str]]:
    """(unambiguous ISO dates in order of appearance, ambiguity reasons)."""
    found: list[tuple[int, str]] = []
    ambiguities: list[str] = []
    taken: list[tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        return not any(s < m.end() and m.start() < e for s, e in taken)

    for m in ISO_RE.finditer(text):
        taken.append(m.span())
        iso = _valid(int(m["y"]), int(m["m"]), int(m["d"]))
        if iso:
            found.append((m.start(), iso))
        else:
            ambiguities.append(f"invalid date {m.group(0)!r}")
    for m in DMY_NUMERIC_RE.finditer(text):
        if not free(m):
            continue
        taken.append(m.span())
        a, b, y = int(m["a"]), int(m["b"]), int(m["y"])
        if a <= 12 and b <= 12 and a != b:
            ambiguities.append(f"day/month order unclear in {m.group(0)!r}")
            continue
        iso = _valid(y, b, a) if a > 12 or a == b else _valid(y, a, b)  # a > 12: day first
        if iso:
            found.append((m.start(), iso))
        else:
            ambiguities.append(f"invalid date {m.group(0)!r}")
    for regex in (MONTH_FIRST_RE, DAY_FIRST_RE):
        for m in regex.finditer(text):
            if not free(m) or m["month"] == "may":  # lowercase "may" is the verb
                continue
            taken.append(m.span())
            if not m["y"]:
                ambiguities.append(f"no year in {m.group(0).strip()!r}")
                continue
            iso = _valid(int(m["y"]), MONTHS[m["month"].lower()], int(m["day"]))
            if iso:
                found.append((m.start(), iso))
            else:
                ambiguities.append(f"invalid date {m.group(0).strip()!r}")
    for m in RELATIVE_RE.finditer(text):
        ambiguities.append(f"relative date {m.group(0)!r}")
    dates = list(dict.fromkeys(iso for _, iso in sorted(found)))
    return dates, ambiguities


def resolve_scope(question: str, catalog: LogCatalog) -> QueryScope:
    dates, ambiguities = extract_dates(question)
    lowered = question.lower()
    words = set(IDENT_RE.findall(lowered))
    pipelines = sorted(words & catalog.pipelines)
    tasks = sorted(words & catalog.tasks)
    mentioned = {g for m in MENTION_RE.finditer(lowered) for g in m.groups() if g}
    unknown = sorted(w for w in mentioned if _identifier_like(w)
                     and w not in catalog.pipelines and w not in catalog.tasks)
    base = {"dates_found": tuple(dates), "ambiguities": tuple(ambiguities),
            "unknown_identifiers": tuple(unknown), "pipelines_found": tuple(pipelines),
            "tasks_found": tuple(tasks)}

    if len(dates) > 1:
        return QueryScope(False, "multiple dates", **base)
    if ambiguities:
        return QueryScope(False, "ambiguous date: " + "; ".join(ambiguities), **base)
    if not dates:
        return QueryScope(False, "no date in the question", **base)
    if unknown:
        return QueryScope(False, f"unknown pipeline/task {unknown}: scope uncertain", **base)

    pipeline = pipelines[0] if len(pipelines) == 1 else None
    task = tasks[0] if len(tasks) == 1 else None
    notes = []
    if len(pipelines) > 1:
        notes.append(f"several pipelines {pipelines}: no pipeline constraint")
    if len(tasks) > 1:
        notes.append(f"several tasks {tasks}: no task constraint")
    if pipeline and task and pipeline not in catalog.pipelines_of(task):
        return QueryScope(False, f"task {task} does not belong to pipeline {pipeline}",
                          **base)
    n = catalog.count(dates[0], pipeline, task)
    if n == 0:
        return QueryScope(False, "no indexed log matches the date/pipeline/task",
                          date=dates[0], pipeline=pipeline, task=task, matching_logs=0,
                          notes=tuple(notes), **base)
    return QueryScope(True, "unambiguous date" + (" and identifiers" if pipeline or task
                                                  else ""),
                      date=dates[0], pipeline=pipeline, task=task, matching_logs=n,
                      notes=tuple(notes), **base)
