-- Failed-request observability: every /ask that reaches the API is logged with its HTTP
-- status and, for failures, a fixed error category (never a raw exception or a secret).
-- Idempotent. Rows logged before this migration keep status_code NULL (unknown).

ALTER TABLE queries ADD COLUMN IF NOT EXISTS status_code    INTEGER;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS error_category TEXT;

CREATE INDEX IF NOT EXISTS queries_status_idx ON queries (status_code, created_at DESC);
