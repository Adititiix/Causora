"""Markdown chunker for runbooks and incident reports: one chunk per `##` section.

The text before the first `##` (the bullet list of facts: owner, applies-to, date, severity,
pipeline...) becomes an "Overview" chunk. Every chunk starts with the document title, because
chunks are embedded in isolation; the heading goes in `section`. Pure functions, no I/O.
"""

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from incident_copilot.chunking.models import Chunk, DocumentInfo

FACT_RE = re.compile(r"^- \*\*(?P<key>[^*]+?):\*\* (?P<value>.+)$")
PIPELINE_IN_APPLIES_TO_RE = re.compile(r"([a-z0-9_]+)\s*\(")
INCIDENT_ID_RE = re.compile(r"\bINC-\d+\b")

LABELS = {"runbook": "Runbook", "incident": "Incident report"}


@dataclass
class ParsedMarkdown:
    title: str
    overview: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (heading, body)
    facts: dict[str, str] = field(default_factory=dict)


def _key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def parse_markdown(text: str, path: str = "") -> ParsedMarkdown:
    lines = text.replace("\r\n", "\n").split("\n")
    title = PurePosixPath(path).stem if path else "untitled"
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]

    overview: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            overview.append(line)

    facts = {}
    for line in overview:
        if m := FACT_RE.match(line.strip()):
            facts[_key(m["key"])] = m["value"].strip()

    return ParsedMarkdown(
        title=title,
        overview="\n".join(overview).strip(),
        sections=[(h, "\n".join(b).strip()) for h, b in sections],
        facts=facts,
    )


def _display_title(title: str, source_type: str) -> str:
    label = LABELS.get(source_type)
    if label and not title.lower().startswith(label.lower()):
        return f"{label} {title}" if INCIDENT_ID_RE.match(title) else f"{label}: {title}"
    return title


def _pipelines(doc: ParsedMarkdown, source_type: str) -> list[str]:
    if source_type == "runbook":
        return PIPELINE_IN_APPLIES_TO_RE.findall(doc.facts.get("applies_to", ""))
    pipeline = doc.facts.get("pipeline")
    return [pipeline] if pipeline else []


def chunk_sections(text: str, path: str, source_type: str) -> list[Chunk]:
    doc = parse_markdown(text, path)
    title = f"# {_display_title(doc.title, source_type)}"
    pipelines = _pipelines(doc, source_type)
    base_meta = {"title": doc.title, "pipelines": pipelines}

    parts: list[tuple[str, str]] = []
    if doc.overview:
        parts.append(("Overview", f"{title}\n\n{doc.overview}"))
    for heading, body in doc.sections:
        parts.append((heading, f"{title}\n\n## {heading}\n{body}".rstrip()))
    if not parts:
        parts.append(("Overview", title))

    return [
        Chunk(index=i, text=body, section=heading, metadata={**base_meta, "heading": heading})
        for i, (heading, body) in enumerate(parts)
    ]


def describe_sections(text: str, path: str, source_type: str) -> DocumentInfo:
    doc = parse_markdown(text, path)
    pipelines = _pipelines(doc, source_type)
    meta: dict = {**doc.facts, "sections": [h for h, _ in doc.sections]}
    if source_type == "incident" and (m := INCIDENT_ID_RE.search(doc.title)):
        meta["incident_id"] = m.group(0)
    return DocumentInfo(
        title=doc.title,
        pipeline_name=pipelines[0] if len(pipelines) == 1 else None,
        pipelines=pipelines,
        metadata=meta,
    )


def chunk_runbook(text: str, path: str) -> list[Chunk]:
    return chunk_sections(text, path, "runbook")


def chunk_incident(text: str, path: str) -> list[Chunk]:
    return chunk_sections(text, path, "incident")


def describe_runbook(text: str, path: str) -> DocumentInfo:
    return describe_sections(text, path, "runbook")


def describe_incident(text: str, path: str) -> DocumentInfo:
    return describe_sections(text, path, "incident")
