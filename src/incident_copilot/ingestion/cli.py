"""Command line for indexing.

    python -m incident_copilot.ingestion.cli index [--root data] [--path logs/orders_etl]
                                                   [--prune] [--dry-run] [--force] [-v]

Inside Docker:  docker compose exec api python -m incident_copilot.ingestion.cli index
Exit codes: 0 ok, 1 error, 2 another indexing run holds the lock, 3 bad path.
"""

import argparse
import logging
import sys

from incident_copilot.config import get_settings
from incident_copilot.db import mask_dsn
from incident_copilot.embeddings import EmbeddingDimensionError, get_embedder
from incident_copilot.ingestion.ingest import IndexingLockedError, ingest_paths
from incident_copilot.ingestion.sources import UnsafePathError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.ingestion.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="index new and changed files")
    index.add_argument("--root", default=None,
                       help="data directory (default: DATA_DIR setting, 'data')")
    index.add_argument("--path", action="append", dest="paths", default=None,
                       help="file or folder relative to the data directory; repeatable "
                            "(default: everything)")
    index.add_argument("--prune", action="store_true",
                       help="delete documents whose file no longer exists under the paths")
    index.add_argument("--dry-run", action="store_true",
                       help="report what would change; write nothing, embed nothing")
    index.add_argument("--force", action="store_true",
                       help="re-chunk and re-embed every file even if unchanged")
    index.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed",
                       help="'hashing' is a test fake with no semantic quality")
    index.add_argument("-v", "--verbose", action="store_true", help="list every changed file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    root = args.root or settings.data_dir
    try:
        embedder = get_embedder(args.embedder, settings)
    except EmbeddingDimensionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mode = " (dry run: nothing will be written)" if args.dry_run else ""
    print(f"Indexing {root} into {mask_dsn(settings.database_url)}{mode}")
    print(f"embedder: {embedder.name} (dim {embedder.dim})")
    try:
        s = ingest_paths(args.paths, embedder=embedder, data_dir=root, prune=args.prune,
                         dry_run=args.dry_run, force=args.force)
    except IndexingLockedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except UnsafePathError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except EmbeddingDimensionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verb = "would be " if s.dry_run else ""
    print(f"paths:          {', '.join(s.roots)}")
    print(f"files seen:     {s.files_seen}")
    print(f"added:          {len(s.added)}")
    print(f"updated:        {len(s.updated)}")
    print(f"unchanged:      {s.unchanged}")
    print(f"deleted:        {len(s.deleted)}" + ("" if args.prune else " (pruning off)"))
    print(f"chunks {verb}written: {s.chunks_written}")
    print(f"embed calls:    {s.embed_calls} ({s.texts_embedded} texts)")
    print(f"duration:       {s.duration_s:.2f}s")
    if s.run_id:
        print(f"run id:         {s.run_id}")
    if args.verbose or s.dry_run:
        for label, items in (("added", s.added), ("updated", s.updated),
                             ("deleted", s.deleted)):
            for item in items:
                print(f"  {verb}{label}: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
