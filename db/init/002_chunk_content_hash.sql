-- Milestone A: per-chunk content hash (SHA-256 of the normalized chunk text).
-- Idempotent: safe to run more than once. Runs automatically on a fresh volume; upgrade an
-- existing database with:  python -m incident_copilot.db_migrate

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
