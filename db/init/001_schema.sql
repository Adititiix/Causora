-- Incident Copilot schema. Idempotent: safe to run more than once.
-- Runs automatically on first `docker compose up` (see docker-compose.yml).
--
-- Layers that use each table:
--   Day 1-2  documents, chunks            (ingestion + chunking)
--   Day 3-5  queries, retrievals          (logging every answer and every retrieved chunk)
--   Day 6    eval_questions/runs/results  (RAGAS comparison table)
--   Day 8    ingestion_runs               (Airflow runs)

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Ingestion
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger       TEXT NOT NULL DEFAULT 'manual'
                  CHECK (trigger IN ('manual', 'airflow', 'failure_callback', 'ci')),
    dag_run_id    TEXT,
    status        TEXT NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'success', 'failed')),
    docs_seen     INTEGER NOT NULL DEFAULT 0,
    docs_changed  INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Path relative to data/, e.g. runbooks/oom-killed-task.md. Natural key for upserts.
    path          TEXT NOT NULL UNIQUE,
    source_type   TEXT NOT NULL
                  CHECK (source_type IN ('runbook', 'log', 'schema', 'incident')),
    title         TEXT,
    pipeline_name TEXT,
    -- SHA-256 of file bytes. Unchanged hash => skip re-embedding (idempotent ingestion).
    content_hash  TEXT NOT NULL,
    -- True once an engineer has confirmed it. Verified docs can be ranked higher later.
    verified      BOOLEAN NOT NULL DEFAULT TRUE,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_run_id   UUID REFERENCES ingestion_runs(id) ON DELETE SET NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents (source_type);
CREATE INDEX IF NOT EXISTS documents_pipeline_idx    ON documents (pipeline_name);

-- ---------------------------------------------------------------------------
-- Chunks: the unit of retrieval. Holds BOTH the dense vector and the keyword index
-- so hybrid search runs inside one database.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    token_count  INTEGER,
    -- NULL until embedded, so chunking (Day 2a) can be tested before embedding (Day 2b).
    embedding    vector(384),
    -- Keyword half of hybrid search. 'simple' config: no stemming, no stop-word removal, so
    -- error names like KeyError match as written. Identifiers are split on underscores
    -- (orders_fact_pkey -> orders, fact, pkey) and queries are split the same way, so they
    -- still match. Full file paths stay a single token. Ranking uses ts_rank_cd, which is
    -- BM25-like but NOT true BM25 (hence the column name keyword_score below).
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    -- Log chunks: time window covered. Used for recency filters.
    ts_start     TIMESTAMPTZ,
    ts_end       TIMESTAMPTZ,
    -- Section heading for runbooks, table name for schemas, task id for logs, etc.
    section      TEXT,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Approximate nearest-neighbour index for cosine distance (queries use <=>).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
-- Full-text index.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_ts_idx ON chunks (ts_start, ts_end);

-- ---------------------------------------------------------------------------
-- Observability: every question, and which chunks were retrieved for it.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS queries (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question      TEXT NOT NULL,
    answer        TEXT,
    abstained     BOOLEAN NOT NULL DEFAULT FALSE,
    retrieval_config TEXT,            -- naive | hybrid | hybrid_rerank
    latency_ms    INTEGER,
    retrieval_ms  INTEGER,
    generation_ms INTEGER,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    cost_usd      NUMERIC(10, 6),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS queries_created_idx ON queries (created_at DESC);

CREATE TABLE IF NOT EXISTS retrievals (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id       UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    chunk_id       UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    rank           INTEGER NOT NULL,
    keyword_score  REAL,
    vector_score   REAL,
    rrf_score      REAL,
    rerank_score   REAL,
    used_in_answer BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (query_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS retrievals_query_idx ON retrievals (query_id);

-- ---------------------------------------------------------------------------
-- Evaluation
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS eval_questions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Stable human-readable id from the seed file (e.g. Q007). Lets us re-load safely.
    external_id       TEXT NOT NULL UNIQUE,
    question          TEXT NOT NULL,
    ground_truth      TEXT NOT NULL,
    category          TEXT NOT NULL,   -- log_diagnosis | runbook | schema | incident | unanswerable
    should_abstain    BOOLEAN NOT NULL DEFAULT FALSE,
    -- Documents that must be retrieved to answer. Powers a retrieval hit-rate metric
    -- that needs no LLM at all.
    expected_paths    TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config      TEXT NOT NULL,         -- naive | hybrid | hybrid_rerank
    git_sha     TEXT,
    notes       TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id            UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id       UUID NOT NULL REFERENCES eval_questions(id) ON DELETE CASCADE,
    answer            TEXT,
    abstained         BOOLEAN,
    retrieval_hit     BOOLEAN,         -- were all expected_paths in the retrieved set?
    faithfulness      REAL,
    context_precision REAL,
    context_recall    REAL,
    UNIQUE (run_id, question_id)
);
