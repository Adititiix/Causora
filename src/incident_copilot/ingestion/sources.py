"""Find indexable files under the data directory and map them to a source type.

Layout (relative to DATA_DIR):
    runbooks/*.md   logs/<pipeline>/*.log   schemas/*.md   incidents/*.md
    extra/<runbooks|logs|schemas|incidents>/...  treated exactly like the folders above
Anything else (manifest.json, .gitkeep, unknown folders) is ignored.

Paths are always handled as POSIX strings relative to DATA_DIR, which is also what
`documents.path` stores.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

TYPE_DIRS = {"runbooks": "runbook", "logs": "log", "schemas": "schema", "incidents": "incident"}
SUFFIXES = {"runbook": {".md"}, "incident": {".md"}, "schema": {".md"}, "log": {".log"}}


class UnsafePathError(ValueError):
    """A requested path is absolute, contains `..`, or resolves outside the data directory."""


@dataclass(frozen=True)
class SourceFile:
    path: str  # POSIX, relative to the data directory
    abs_path: Path
    source_type: str


def classify(rel_path: str) -> str | None:
    """Source type for a data-relative path, or None if it is not indexable."""
    parts = PurePosixPath(rel_path).parts
    if parts and parts[0] == "extra":
        parts = parts[1:]
    if len(parts) < 2 or parts[0] not in TYPE_DIRS:
        return None
    source_type = TYPE_DIRS[parts[0]]
    if PurePosixPath(rel_path).suffix not in SUFFIXES[source_type]:
        return None
    return source_type


def normalize_root(data_dir: Path, root: str) -> str:
    """Validate a user-supplied root and return it as a clean data-relative POSIX path.

    Rejects absolute paths (POSIX or Windows, including drive letters and UNC), any `..`
    component, and anything that resolves (following symlinks) outside `data_dir`.
    """
    raw = (root or ".").strip()
    if raw in ("", "."):
        return "."
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute() \
            or PureWindowsPath(raw).drive or raw.startswith(("\\", "/")):
        raise UnsafePathError(f"path must be relative to the data directory: {root!r}")
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    if ".." in parts:
        raise UnsafePathError(f"path must not contain '..': {root!r}")
    clean = PurePosixPath(*[p for p in parts if p != "."]).as_posix() if parts else "."
    base = data_dir.resolve()
    target = (base / clean).resolve()
    if target != base and base not in target.parents:
        raise UnsafePathError(f"path resolves outside the data directory: {root!r}")
    return clean or "."


def is_under(path: str, root: str) -> bool:
    return root == "." or path == root or path.startswith(root.rstrip("/") + "/")


def discover(data_dir: Path, roots: list[str]) -> list[SourceFile]:
    """All indexable files under the (already normalized) roots, sorted by path."""
    base = data_dir.resolve()
    found: dict[str, SourceFile] = {}
    for root in roots:
        target = base if root == "." else base / root
        candidates = [target] if target.is_file() else (
            sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else []
        )
        for p in candidates:
            resolved = p.resolve()
            if resolved != base and base not in resolved.parents:
                continue  # symlink escaping the data directory
            rel = p.relative_to(base).as_posix()
            source_type = classify(rel)
            if source_type:
                found[rel] = SourceFile(rel, p, source_type)
    return [found[k] for k in sorted(found)]
