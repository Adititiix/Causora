-- Milestone B: query logging that survives re-indexing. Idempotent.
--
-- Problem: retrievals.chunk_id was ON DELETE CASCADE, and re-indexing a document deletes and
-- re-inserts its chunks (prune deletes documents, which cascades to chunks). Either way the
-- retrieval history of past questions was silently deleted.
--
-- Fix:
--   * retrievals.chunk_id becomes nullable and ON DELETE SET NULL: the history row stays, and
--     its link is cleared when the chunk it pointed at no longer exists. Replacement chunks
--     always get new ids, so an old row can never resolve to different content.
--   * each retrieval row stores an immutable snapshot of what was shown to the model
--     (path, chunk index, section, timestamps, content hash and the redacted content).
--   * queries gets the model name and a metadata JSON (request, gate, validation outcome).

ALTER TABLE retrievals ALTER COLUMN chunk_id DROP NOT NULL;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint
               WHERE conname = 'retrievals_chunk_id_fkey' AND confdeltype <> 'n') THEN
        ALTER TABLE retrievals DROP CONSTRAINT retrievals_chunk_id_fkey;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'retrievals_chunk_id_fkey') THEN
        ALTER TABLE retrievals ADD CONSTRAINT retrievals_chunk_id_fkey
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE SET NULL;
    END IF;
END $$;

ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS evidence_id        TEXT;   -- E1, E2, ...
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS document_path      TEXT;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS source_type        TEXT;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS chunk_index        INTEGER;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS section            TEXT;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS ts_start           TIMESTAMPTZ;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS ts_end             TIMESTAMPTZ;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS chunk_content_hash TEXT;
ALTER TABLE retrievals ADD COLUMN IF NOT EXISTS content            TEXT;

ALTER TABLE queries ADD COLUMN IF NOT EXISTS model    TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
