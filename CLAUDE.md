# CLAUDE.md: rules for working in this repository

Incident Copilot: FastAPI + PostgreSQL/pgvector + Docker Compose. The owner works on
Windows with VS Code and PowerShell.

## Git: hands off

- Never create, switch, rename, or delete Git branches.
- Never commit, push, pull, merge, rebase, cherry-pick, stash, tag, reset, or otherwise
  modify Git history or repository state.
- Read-only Git commands (`git status`, `git diff`, `git log`) are fine.
- Work directly in the current working tree. The owner manages Git.
- Preserve existing changes; never overwrite unrelated user modifications. Check what has
  changed before editing.
- At the end of a task, list every file created or modified.

## Engineering rules

- Extend the existing structure; do not rebuild it. Each layer lives in its own package
  (`chunking/`, `ingestion/`, `retrieval/`, `generation/`, `evaluation/`, `observability/`)
  and talks to the others through plain functions.
- All settings live in `src/incident_copilot/config.py`; nothing else reads `os.environ`.
  Every setting is documented in `.env.example` (a test enforces this).
- All database connections go through `incident_copilot.db.connect()`.
- `chunks.embedding` is `vector(384)`. Never change the dimension.
- Schema changes: add a new idempotent `db/init/NNN_*.sql`; never edit applied files.
  Apply with `python -m incident_copilot.db_migrate`.
- `ingest_paths()` must not import FastAPI or CLI code (Airflow will call it later).
- Never log or store credentials. Redact before embedding or storing; mask passwords in any
  logged connection string.
- Never fabricate an LLM answer. If the LLM is not configured, say so.

## Tests

- Never delete, skip, or weaken an existing test to make it pass. Fix the code.
- Unit tests must not need network, paid APIs, or model downloads. Use the fakes
  (`HashingEmbedder`, `FakeLLMProvider`).
- Database tests are marked `db` and must actually run locally (zero skipped). Tests marked
  `slow` may use the real embedding model and skip only when it is not cached.
- Run `python -m pytest -rs` and `python -m ruff check .` before reporting.
- Report exact counts and every skip reason. Do not claim something works unless it was run.

## Windows

- Document commands as `python -m ...` and `docker compose ...` in PowerShell syntax.
  Do not rely on `make` or bash-only syntax in docs.
- Local Postgres for this machine is on host port 5434 (a native PostgreSQL 18 service
  occupies 5433).
