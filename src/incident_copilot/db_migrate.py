"""Apply every db/init/*.sql file, in name order, to the configured database.

Docker only runs db/init on a brand-new volume. This upgrades an existing database without
wiping it. Every file is idempotent, so re-running is safe; there is deliberately no
migrations-tracking table.

    python -m incident_copilot.db_migrate            (host: uses DATABASE_URL from .env)
    docker compose exec api python -m incident_copilot.db_migrate
"""

import sys
from pathlib import Path

import psycopg

from incident_copilot.config import get_settings
from incident_copilot.db import mask_dsn

INIT_DIR = Path(__file__).resolve().parents[2] / "db" / "init"


def migration_files(init_dir: Path = INIT_DIR) -> list[Path]:
    return sorted(init_dir.glob("*.sql"))


def apply_migrations(conninfo: str | None = None, init_dir: Path = INIT_DIR) -> list[str]:
    files = migration_files(init_dir)
    if not files:
        raise FileNotFoundError(f"no .sql files in {init_dir}")
    with psycopg.connect(conninfo or get_settings().database_url, autocommit=True) as conn:
        for f in files:
            with conn.transaction():
                conn.execute(f.read_text(encoding="utf-8"))
    return [f.name for f in files]


def main() -> int:
    url = get_settings().database_url
    print(f"Migrating {mask_dsn(url)}")
    try:
        applied = apply_migrations(url)
    except (psycopg.Error, FileNotFoundError) as exc:
        print(f"error: {type(exc).__name__}: {str(exc).splitlines()[0]}", file=sys.stderr)
        return 1
    for name in applied:
        print(f"  applied {name}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
