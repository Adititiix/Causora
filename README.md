# Incident Copilot

A RAG assistant that answers data-pipeline incident questions ("why did last night's
`transform_orders` task fail?") from logs, runbooks, table schemas, and past incident
reports, with citations, and refuses to guess when the evidence is weak.

Built in layers. Each layer is runnable and tested before the next one starts.

**Status: Layer 1 (foundation), Milestone A (ingestion) and Milestone B (ask) are done.**
Database, synthetic dataset, chunking, redaction, local embeddings, incremental indexing,
vector retrieval, grounded answers with verified citations, refusal, `/ask`, `/ingest`, tests, CI.

---

## 1. Quick start (Windows PowerShell)

**You need:** Docker Desktop and Python 3.11+. `make` is optional; every step below is a plain
`python -m ...` or `docker compose ...` command.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
Copy-Item .env.example .env
# Edit .env: change POSTGRES_PASSWORD and put the same password in DATABASE_URL.

docker compose up -d --build                                    # Postgres + pgvector + API
docker compose exec api python -m incident_copilot.db_migrate   # safe on any existing DB
docker compose exec api python -m incident_copilot.embeddings download   # one time, ~67 MB
docker compose exec api python -m incident_copilot.ingestion.cli index   # index data/
```

Check it works:

```powershell
Invoke-RestMethod http://localhost:8000/health      # status ok, version 0.1.0
Invoke-RestMethod http://localhost:8000/health/db   # status ok, documents 104, chunks 324, per-type counts
python -m pytest -rs                                # see section 11 for the expected counts
```

API docs open at http://localhost:8000/docs.

**`make` targets and their plain equivalents** (the Makefile is for Mac/Linux/WSL):

| make target | plain command |
|---|---|
| `make setup` | `python -m venv .venv`, activate it, `pip install -r requirements-dev.txt`, `pip install -e . --no-deps`, copy `.env.example` to `.env` |
| `make up` | `docker compose up -d --build` |
| `make down` | `docker compose down` |
| `make reset-db` | `docker compose down -v` then `docker compose up -d db` |
| `make test` | `python -m pytest -v` |
| `make test-fast` | `python -m pytest -v -m "not db"` |
| `make lint` | `python -m ruff check .` |
| `make data` | `python scripts/generate_synthetic_data.py --out data --eval-out eval/questions.seed.jsonl` |

Activating the venv: `source .venv/bin/activate` (Mac/Linux/WSL) or `.\.venv\Scripts\Activate.ps1`
(Windows PowerShell).

---

## 2. Folder structure

```
incident-copilot/
├── README.md
├── Makefile                       # shortcuts: make up / test / lint / data
├── docker-compose.yml             # Postgres+pgvector and the API (Airflow joins on Day 8)
├── Dockerfile
├── pyproject.toml                 # package metadata, pytest and ruff config
├── requirements.txt               # runtime dependencies
├── requirements-dev.txt           # + pytest, ruff, httpx
├── requirements-ml.txt            # Day 2+: fastembed (ONNX, no PyTorch), google-genai
├── requirements-eval.txt          # Day 6: ragas, datasets
├── .env.example                   # every setting, documented
├── CLAUDE.md                      # working rules for AI assistants in this repo
├── .github/workflows/ci.yml       # lint + tests + dataset-drift check on every push
│
├── db/init/001_schema.sql         # all tables and indexes; runs on first `docker compose up`
├── db/init/002_chunk_content_hash.sql  # Milestone A: chunks.content_hash
├── db/init/003_query_history.sql       # Milestone B: retrieval history survives re-indexing
├── db/init/004_query_status.sql        # status_code + error_category for failed /ask requests
│
├── src/incident_copilot/
│   ├── config.py                  # all settings, read from environment
│   ├── db.py                      # the only place connections are opened; mask_dsn()
│   ├── db_migrate.py              # applies every db/init/*.sql in order (idempotent)
│   ├── redaction.py               # redact(): credentials out before embed/store/log
│   ├── embeddings.py              # Embedder protocol, FastEmbedEmbedder, HashingEmbedder (fake)
│   ├── api/                       # main.py (/health, /health/db, /ask, /ingest), schemas.py, errors.py
│   ├── chunking/                  # Day 2   logs.py, sections.py (runbooks+incidents), schemas.py
│   ├── ingestion/                 # Day 2,8,9  sources.py, ingest.py (ingest_paths), cli.py
│   ├── retrieval/                 # base.py (Retriever, Hit, Filters), vector.py, error_link.py
│   ├── generation/                # llm.py (Gemini, Fake), prompt.py, answer.py (gate, validation, citations)
│   ├── evaluation/                # retrieval_smoke.py (hit@k, similarity distributions); answer_eval.py (real-LLM answers); RAGAS on Day 6
│   ├── observability/             # query_log.py: queries + retrievals rows with evidence snapshots
│   └── ui/                        # Streamlit chat page (app.py) that calls POST /ask (section 6c)
│
├── airflow/dags/                  # Day 8   ingestion DAG and the "victim" DAG
├── scripts/
│   ├── generate_synthetic_data.py # builds data/ and eval/questions.seed.jsonl
│   └── prompts/                   # optional LLM prompts to add scenarios / harder questions
├── data/                          # the synthetic company (committed; see section 5)
│   ├── runbooks/ logs/ schemas/ incidents/   # generated, do not hand-edit
│   ├── extra/                     # YOUR additions; never overwritten by the generator
│   └── manifest.json              # every file with hash, type, pipeline, scenario
├── eval/questions.seed.jsonl      # 50 questions with ground truth and expected sources
├── tests/                         # 328 tests (85 need Postgres, 2 need the cached model)
└── docs/                          # architecture diagrams, results table (Day 7, 14)
```

**Rule that keeps this restructure-free:** each layer lives in its own package and talks to
the others only through plain functions. `ingest_paths()` has no Airflow imports, so the same
code runs from the command line (Day 2), an Airflow task (Day 8), and an API endpoint (Day 9).

---

## 3. Dependencies

| File | Contents | Needed from |
|---|---|---|
| `requirements.txt` | fastapi, uvicorn, pydantic, pydantic-settings, psycopg 3 (+pool), pgvector | Day 1 |
| `requirements-dev.txt` | the above + pytest, httpx, ruff | Day 1 |
| `requirements-ml.txt` | fastembed (local ONNX embeddings, no PyTorch), google-genai (Gemini) | Day 2 |
| `requirements-eval.txt` | ragas, datasets | Day 6 |
| `requirements-ui.txt` | streamlit, httpx (host venv only, not in the Docker image; Streamlit also pulls in pyarrow and altair) | Chat UI (section 6c) |

`requirements-ml.txt` is baked into the Docker image (it grows from 264 MB to 590 MB; the ML
layer is 248 MB, mostly onnxruntime and numpy). On the host it adds about 174 MB to `.venv`
and is optional: unit tests never need it. Install it on the host only to run the `slow`
test or the indexer outside Docker.

Airflow is **not** pip-installed into the app. It runs as its own container (Day 8), because
Airflow pins many dependencies that conflict with everything else.

---

## 4. Environment variables

Copy `.env.example` to `.env`. A test (`test_env_example_stays_in_sync_with_settings`) fails if
`.env.example` and `config.py` drift apart.

| Variable | Default | Used by | Notes |
|---|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | copilot / *(you set it)* / incident_copilot | Compose | Password is required; Compose refuses to start without it |
| `POSTGRES_PORT` | 5434 | Compose | Host port. 5434 avoids clashing with a native Postgres (often on 5432 or 5433) |
| `DATABASE_URL` | `postgresql://copilot:...@localhost:5434/incident_copilot` | Python on your machine | Password and port **must** match `POSTGRES_PASSWORD` / `POSTGRES_PORT`. Inside Docker, Compose overrides the host to `db` |
| `API_PORT` | 8000 | Compose | |
| `LOG_LEVEL` | INFO | API | |
| `DATA_DIR` | data | Ingestion | Corpus root; `documents.path` is relative to it. Docker sets `/app/data` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | BAAI/bge-small-en-v1.5 / 384 | Day 2 | Free, local, CPU, via fastembed. **384 is baked into the schema** (`vector(384)`); a different size is refused with a clear error |
| `EMBEDDING_CACHE_DIR` | .cache/fastembed | Day 2 | Model download folder. Docker sets `/models`, a named volume, so rebuilds do not re-download |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Day 5 | Free, local |
| `GEMINI_API_KEY` / `LLM_MODEL` | *(blank)* / gemini-3.8-flash | Day 3 | Free key from aistudio.google.com/apikey. See the note on model names below. Blank key: `/ask` answers 503 `llm_not_configured` instead of guessing |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` / `LLM_TOTAL_TIMEOUT_SECONDS` | 30 / 3 / 60 | Day 3 | Per-call timeout (also sent to Gemini as its server deadline; a slower call returns 504 "Deadline expired"); retries with jittered backoff for HTTP 408/429/500/502/503/504 and network timeouts, never for 400/401/403/404; all attempts and waits together stay within the total |
| `ASK_TOP_K` | 8 | Day 3 | Chunks retrieved and shown to the LLM; a request's `top_k` (1-20) overrides it |
| `RETRIEVER` | vector+error_link+evidence+context+scoped+date | Day 3 | Default since 2026-10-29 (owner-approved): `vector+error_link+date` plus evidence selection (the matched runbook's Resolution section, a dated failed log's traceback; `retrieval/evidence.py`). On the 50 seed questions it refused 0/44 answerable questions; `vector+error_link+date` refused 12/44 and 15/44 in two runs (e.g. Q030: the runbook's Resolution was never retrieved). Others: `vector`, `vector+error_link`, `vector+date`, `vector+error_link+date`. Section 6b |
| `MIN_SIMILARITY` | 0.72 | Day 3 | Pre-LLM refusal gate on the best question similarity. Calibrated on the seed questions (section 6b) |
| `UI_API_URL` / `UI_REQUEST_TIMEOUT_SECONDS` | http://127.0.0.1:8000 / 120 | Chat UI | Base URL of the API the Streamlit page calls, and how long it waits for one `/ask`. Section 6c |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_N` | 20 / 5 | Day 4-5 | Candidates fetched / chunks kept after reranking |
| `ABSTAIN_THRESHOLD` | 0.0 | Day 5 | Reserved for a reranker-score gate. `/ask` uses `MIN_SIMILARITY` today |

**Gemini model names change fast.** `gemini-2.0-flash` was shut down on 2026-06-01. Before
Day 3, check https://ai.google.dev/gemini-api/docs/pricing (look for "Free of charge") and the
deprecations page. For bulk eval runs a `flash-lite` model is easier on rate limits.

---

## 5. The synthetic company

Real company logs are not available, so `scripts/generate_synthetic_data.py` builds a fictional
e-commerce company, **ShopStream**, deterministically (same seed gives byte-identical files; a
test checks this and CI fails if the committed copy drifts).

| Folder | Files | What it holds |
|---|---|---|
| `data/runbooks/` | 18 | One runbook per failure type: symptoms, causes, diagnosis, resolution, prevention |
| `data/logs/<pipeline>/` | 60 | Airflow-style task logs: 54 failures (3 per scenario) and 6 successes |
| `data/schemas/` | 12 | One doc per table: columns, owner, freshness SLA, change history |
| `data/incidents/` | 14 | Past incident reports with timelines, in chronological ID order |

6 pipelines (`orders_etl`, `payments_sync`, `inventory_snapshot`, `customer_360`,
`marketing_attribution`, `clickstream_ingest`), 2026-08-03 to 2026-09-30.

**Deliberate traps, so evaluation means something:**

- **Cross-document reasoning.** A `KeyError: 'customer_tier'` log only makes sense with the
  `orders_raw` schema doc, which records the rename to `loyalty_tier` on 2026-08-28. Rename
  failures only occur on or after that date (tested).
- **Look-alike failures.** SIGKILL out-of-memory vs SIGTERM node reclaim; duplicate-key vs
  not-null violations; sensor timeout vs missing file.
- **Fake secrets** (`FAKE-PASSWORD-...`, `FAKE-TOKEN-...`) in 8 logs, to build and test
  redaction on Day 9. They are not real credentials.
- **6 unanswerable questions** (e.g. an on-call phone number), where the correct behaviour is
  "not enough evidence".

`eval/questions.seed.jsonl` has 50 questions: 18 log diagnosis, 18 runbook, 2 incident,
6 schema, 6 unanswerable. Each has `ground_truth`, `expected_paths` (the files that must be
retrieved), and `should_abstain`. A test confirms every expected file exists and that the error
quoted in each log-diagnosis answer literally appears in its log.

**Adding your own:** put files under `data/extra/<runbooks|logs|incidents|schemas>/` and use the
prompts in `scripts/prompts/`. Day 2 ingestion should treat `data/extra/<type>/` the same as
`data/<type>/`. Hand-verify anything an LLM writes.

---

## 6. Database schema

Defined in `db/init/001_schema.sql`, extended by later `db/init/NNN_*.sql` files. Every file is
idempotent (tests apply them twice). Docker runs them automatically only the first time the
database volume is created. To upgrade an existing database **without wiping it**:

```powershell
docker compose exec api python -m incident_copilot.db_migrate   # inside Docker
python -m incident_copilot.db_migrate                           # or from the host venv
```

It applies every `db/init/*.sql` in name order. There is no migrations table on purpose: the
files are idempotent, so re-running them is the upgrade. `docker compose down -v` (wipes all
data) is no longer needed after a schema change.

```
ingestion_runs ──< documents ──< chunks >──< retrievals >── queries
                                                  eval_questions ──< eval_results >── eval_runs
```

| Table | Purpose | Layer |
|---|---|---|
| `documents` | One row per source file. `content_hash` drives incremental re-indexing; `verified` is for the feedback loop | 2, 9 |
| `chunks` | The unit of retrieval. Holds the `embedding vector(384)` **and** a generated `tsv` full-text column, so hybrid search runs in one database. HNSW and GIN indexes | 2, 4 |
| `queries` | Every question: answer, abstained?, latency, tokens, cost | 3, 11 |
| `retrievals` | Which chunks each query retrieved, with `keyword_score`, `vector_score`, `rrf_score`, `rerank_score`, `used_in_answer` | 4, 5, 11 |
| `ingestion_runs` | Each ingestion run (manual, Airflow, failure callback, CI) | 8, 9 |
| `eval_questions`, `eval_runs`, `eval_results` | Eval set and per-config RAGAS scores | 6 |

Decisions worth knowing:

- **Keyword search uses Postgres full-text (`ts_rank_cd`), which is BM25-like but not true
  BM25.** That is why the column is `keyword_score`, and why you should describe this project
  as "hybrid (full-text + vector)" rather than claim BM25. Real BM25 would need an extension
  such as ParadeDB's `pg_search`, or a Python `rank_bm25` index. Tested behaviour: error names
  like `KeyError` match as written; identifiers are split on underscores
  (`orders_fact_pkey` becomes `orders`, `fact`, `pkey`) and the query is split the same way;
  full file paths stay one token.
- `chunks.embedding` is nullable, so you can test chunking (Day 2a) before embedding (Day 2b).
- Deleting a document cascades to its chunks (tested).

---

## 6a. Indexing (Milestone A)

```
data/**  ->  sources.discover  ->  SHA-256 of bytes == stored hash?  --yes-->  skip (0 embed calls)
                                         | no
                                         v
              redact(whole file) -> chunk_<type>() -> embed (fastembed, outside any transaction)
                                         v
              ONE transaction: upsert documents row, delete its chunks, insert new chunks
```

```powershell
# Inside Docker (recommended)
docker compose exec api python -m incident_copilot.ingestion.cli index            # new + changed
docker compose exec api python -m incident_copilot.ingestion.cli index -v         # list each file
docker compose exec api python -m incident_copilot.ingestion.cli index --dry-run  # report only
docker compose exec api python -m incident_copilot.ingestion.cli index --prune    # drop deleted files
docker compose exec api python -m incident_copilot.ingestion.cli index --path logs/orders_etl --prune
docker compose exec api python -m incident_copilot.ingestion.cli index --force    # re-embed everything

# From the host venv (needs requirements-ml.txt and the model cached on the host)
python -m incident_copilot.embeddings download
python -m incident_copilot.ingestion.cli index
```

Exit codes: 0 ok, 1 error, 2 another indexing run holds the lock, 3 path outside the data dir.

**Behaviour**

- **Incremental and idempotent.** A file whose bytes hash the same as last time is skipped with
  zero embedding calls. A file is also re-processed if the embedding model or the chunker
  version changed (both are stored in `documents.metadata`).
- **Atomic replace.** A changed file is embedded first; then one transaction updates the
  document and replaces all of its chunks, so a reader never sees old and new chunks mixed.
  If the write fails, the old chunks stay.
- **Prune** (`--prune`) deletes documents whose file is gone, but only under the paths being
  indexed (`--path runbooks --prune` never touches logs). Chunks cascade. Combine with
  `--dry-run` to preview.
- **One run at a time.** A Postgres advisory lock guards indexing; a second concurrent run exits
  with code 2 and `another indexing run is in progress`.
- **`data/extra/<type>/`** is indexed exactly like `data/<type>/`.
- **Redaction** runs on the whole file before chunking: URL credentials (`://user:pass@`),
  `password=`, `token=`, `api_key=`, `secret=`, JSON `"password": "..."`, bearer tokens and AWS
  key ids become `[REDACTED]`. Connection strings are printed with the password masked.
- `ingest_paths()` (in `ingestion/ingest.py`) has no FastAPI or CLI imports (tested), so
  Airflow can call it directly.

**Chunking** (one module per type in `chunking/`, pure functions):

| Type | Rule | Chunks (this corpus) |
|---|---|---|
| Logs | Parse `[ts] {module} LEVEL - text`; traceback lines belong to the entry before them and are never split. *Run summary*: first 3 + last 3 entries. *Error context*: the failing entry with its whole traceback plus up to 8 preceding entries. *Middle*: anything left. Each chunk starts with `Log: <pipeline> / <task> / <date> / <status> / error: <exception line>` | 120 (2 per log) |
| Runbooks, incidents | Split on `##`; the facts list before the first `##` is an *Overview* chunk; every chunk starts with the document title; heading goes in `section` | 108 + 84 |
| Schemas | One chunk per table: the whole file | 12 |

Every chunk stores: `chunk_key` (`path#index`), `content_hash` (SHA-256 of normalized text),
`source_type`, `section`, pipeline(s), table name for schemas, and `ts_start`/`ts_end`,
`task`, `status`, `error` for logs.

**Measured limitation: long log chunks.** bge-small reads at most 512 tokens; fastembed
silently truncates the rest *for the embedding only* (the stored text is complete). Real token
counts after indexing: 51 of 54 *error context* chunks exceed 512 (median 637, max 754) and
19 of 60 *run summary* chunks do (max 574). Runbook, incident and schema chunks are all under
300. The header line carries the final exception, so it is always embedded; the end of a long
traceback is not. `ERROR_CONTEXT_ENTRIES` in `chunking/logs.py` controls the trade-off.

**Real output on this corpus** (Docker Desktop, CPU):

```
first run:   added 104, chunks written 324, embed calls 104 (324 texts), 82 s
second run:  unchanged 104, chunks written 0, embed calls 0 (0 texts), 2.9 s
one runbook edited: updated 1, unchanged 103, embed calls 1 (6 texts), 5.5 s
```

---

## 6b. Asking questions (Milestone B)

```
POST /ask
  question -> embed (bge, query prefix) -> Retriever.retrieve(k, filters)        retrieval/
                 date filter (only if the question names one unambiguous date): log chunks
                 of other dates/pipelines/tasks are excluded; nothing else is
                 vector: cosine top-k            error_link: + runbook for each retrieved
                                                 failed log, found via its exception line
           -> gate: no hits, or best question-similarity < MIN_SIMILARITY
                 -> "Not enough evidence to determine the cause"  (no LLM call)
           -> prompt: [E1]..[En] <evidence> blocks, escaped, marked untrusted     generation/
           -> LLM (Gemini, JSON schema) -> parse -> drop claims citing unknown ids
                 -> no valid evidence left? -> insufficient evidence
           -> citations built from chunk metadata (path, section, timestamps)
           -> queries row + retrievals rows with evidence snapshots              observability/
```

### Endpoints

| Route | Body | Success | Errors (`{"error": {"code", "message"}}`) |
|---|---|---|---|
| `POST /ask` | `question` (1-1000 chars, trimmed), `top_k` (1-20), `retrieval_only`, `source_types` (runbook/log/schema/incident), `pipeline`, `history` (optional, at most 6 earlier turns of `question`, `answer`, `cited_paths`; see 6c) | 200: `status` (`answered`, `insufficient_evidence`, `retrieval_only`), `answer`, `evidence`, `inference`, `next_steps`, `citations`, `retrieved` (text, scores, origin), `gate`, `validation`, `retrieval_info` (date filter decision; `conversation` when history was sent), `timings` (embedding, retrieval, generation, validation ms; `llm_called`, `llm_attempts`), `model`, latency, tokens, `query_id` | 422 `validation_error`; 503 `db_unavailable` / `database_error` / `llm_not_configured` / `embedding_unavailable`; 502 `llm_upstream_error` / `llm_bad_response`; 500 `internal_error`. Recorded failures include `error.query_id` |
| `POST /ingest` | `paths` (relative to the data dir), `prune`, `dry_run` | 200: the ingest summary (added, updated, unchanged, deleted, chunks, embed calls) | 422 `invalid_path` (absolute, `..`, outside data dir) or `validation_error`; 409 `indexing_in_progress`; 503 |
| `GET /health/db` | | 200: pgvector version, tables, `documents`, `chunks`, `by_source_type` | 503 with `detail` (unchanged, for compatibility) **and** `error` |
| `GET /documents` | | 200: every indexed document: `path`, `source_type`, `title`, `pipeline`, `sections`, `chunk_count`, `content_hash`, `updated_at`, `metadata`. Read-only | 503 |
| `GET /documents/{path}` | | 200: `document`, its indexed `chunks` (redacted, in order), and `original` (the file under DATA_DIR, sensitive values masked) or null with `original_note` saying why | 404 `document_not_found` (also for any path that is not an indexed document); 503 |
| `POST /documents/explain` | `path` | 200: `summary`, `points` (`topic`, `text`, `basis` documented/inferred, `section_ids`), `not_documented`, `sections` (S-id → chunk), `dropped_points`, `truncated`, `cached`, latency, attempts. One LLM call, only on request; cached per content hash | 404; 422; 502/503 as for `/ask` |

`evidence` lists what the documents state; `inference` lists conclusions drawn from them, with
`based_on` ids and a confidence. They are separate fields by design. Every id is checked
against the retrieved set, and citations are built by code, never typed by the model.

There is no authentication yet. Run this API only on your own machine.

### Examples (real output from this repo, Windows PowerShell)

```powershell
$q = "Why did the transform_orders task in the orders_etl pipeline fail on 2026-09-04?"
$body = @{ question = $q; retrieval_only = $true; top_k = 5 } | ConvertTo-Json
$r = Invoke-RestMethod -Method Post http://localhost:8000/ask -ContentType "application/json" -Body $body
$r.retrieved | Format-Table evidence_id, origin, path, section
```
```
E1 rank 1  sim 0.855  vector      incidents/INC-2032.md  [Summary]
E2 rank 2  sim 0.853  vector      logs/orders_etl/2026-09-04__transform_orders.log  [error context]
E3 rank 3  sim 0.754  error_link  runbooks/column-renamed-upstream.md  [Overview]  <- linked from logs/orders_etl/2026-09-04__transform_orders.log (link_sim 0.666)
E4 rank 4  sim 0.849  vector      logs/orders_etl/2026-09-12__transform_orders.log  [run summary]
E5 rank 5  sim 0.843  vector      logs/orders_etl/2026-09-04__transform_orders.log  [run summary]
```

Without a Gemini key, a full answer is refused instead of made up:

```
HTTP 503  {"error":{"code":"llm_not_configured","message":"No LLM is configured. Get a free key at https://aistudio.google.com/apikey, put it in .env as GEMINI_API_KEY=..., and restart the API (docker compose up -d api). Use retrieval_only=true to see the retrieved evidence without an LLM."}}
```

An unanswerable question is refused by the gate, with no LLM call (so it works without a key):

```powershell
$body = @{ question = "What is the on-call phone number for the payments team?" } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8000/ask -ContentType "application/json" -Body $body
```
```
status                : insufficient_evidence
answer                : Not enough evidence to determine the cause
insufficient_reason   : below_min_similarity
gate                  : {"best_similarity":0.631831,"min_similarity":0.72,"passed":false}
```

Indexing through the API:

```powershell
$body = @{ paths = @("runbooks"); prune = $true; dry_run = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8000/ingest -ContentType "application/json" -Body $body
```

### LLM setup (Gemini)

1. Create a free key at https://aistudio.google.com/apikey.
2. In `.env`, set `GEMINI_API_KEY=...`. Leave `LLM_MODEL` as in `.env.example` unless that
   model has been retired (check https://ai.google.dev/gemini-api/docs/deprecations).
3. Run `docker compose up -d api`. Compose passes `.env` to the API container.

The key is read only from settings and is never logged or returned. Upstream errors are
reduced to a status code and a short, redacted message. Calls use `temperature=0.1`, a JSON
response schema, and a timeout. HTTP 429 and 503 are retried with backoff (1 s, 2 s, 4 s).

### Retrieval evaluation (no LLM)

```powershell
docker compose exec api python -m incident_copilot.evaluation.retrieval_smoke             # both retrievers
docker compose exec api python -m incident_copilot.evaluation.retrieval_smoke --window 4  # temp index, 4-entry log window
```

`any@k`: at least one expected document is in the top-k chunks. `all@k`: every expected
document is. Real numbers (bge-small-en-v1.5, 50 seed questions, 324 chunks):

| Retriever, log window | log_diagnosis (18) all@5 | runbook (18) all@5 | schema (6) all@5 | incident (2) all@5 | all 44 any@5 | all 44 all@5 | all 44 all@8 |
|---|---|---|---|---|---|---|---|
| vector, 8 | 0.0% | 83.3% | 100% | 100% | 81.8% | 52.3% | 56.8% |
| **vector+error_link, 8 (default at the time)** | **55.6%** | 88.9% | 100% | 100% | 79.5% | **77.3%** | **81.8%** |
| vector, 4 | 0.0% | 61.1% | 100% | 100% | 70.5% | 43.2% | 54.5% |
| vector+error_link, 4 | 50.0% | 72.2% | 100% | 100% | 72.7% | 68.2% | 81.8% |

Findings, stated as measured:

- **Vector search alone never retrieves log and runbook together for a log question.**
  "Why did X fail on date Y" shares no vocabulary with the failure type, so the right runbook
  ranks 30th-100th or lower. The error link fixes this when the retrieved log is the right
  one. The cost is that a wrongly linked runbook pushes the correct log down: log any@3 falls
  from 55.6% to 38.9%.
- **Dates are the remaining weakness.** All 5 answerable misses in the top 8 are logs of the
  right task on the wrong date; embeddings barely separate `2026-09-04` from `2026-09-21`.
  Hybrid keyword search or date-aware filtering (Day 4) is the planned fix.
- **The 4-entry error window is worse** on almost every metric. It also creates 375 chunks
  instead of 324, and the extra "middle" chunks crowd the top-k. The 8-entry window stays.

### Answer evaluation (real LLM)

Runs seed questions through the same pipeline as `/ask` (`answer_question()` with the
configured retriever, `ASK_TOP_K`, `MIN_SIMILARITY` and `LLM_MODEL`; nothing is overridden)
and scores the answers. It needs `GEMINI_API_KEY`, makes one Gemini call per question that
passes the gate, and writes nothing to the database. Run it from the host venv (it needs
`requirements-ml.txt` and the cached embedding model): the API container mounts `eval/`
read-only, so it cannot save results there.

```powershell
python -m incident_copilot.evaluation.answer_eval --ids Q001,Q019,Q045   # small check, 3 calls max
python -m incident_copilot.evaluation.answer_eval --limit 5               # the first 5 questions
python -m incident_copilot.evaluation.answer_eval --pause 4               # all 50 (about 45 calls)
python -m incident_copilot.evaluation.answer_eval --resume eval\results\<RUN_ID>_answers
```

Each run creates a new folder `eval/results/<RUN_ID>_answers/` (existing results are never
touched) with `answers.jsonl` (question, ground truth, answer, claims, next steps, citations
with the cited chunk text, gate, validation, latency, tokens, error, metrics), one row per
question in `answer_quality.csv`, `summary.md` and `run_metadata.json` (model, settings,
prompt-module and corpus fingerprints; no key). Every question is saved as soon as it finishes.
A failed generation is recorded and the run continues; it stops after 5 failures in a row
(`--max-consecutive-errors`), usually an exhausted quota. `--resume` re-runs only questions
without a successful record and refuses if the model, settings, prompt, questions or corpus
changed. `--pause` (seconds between real calls, default 4) keeps free-tier rate limits happy.

What the metrics are, and are not:

| Metric | Meaning |
|---|---|
| abstention correct / unnecessary / missed | refused == `should_abstain`; split into gate refusals (no LLM call) and model refusals |
| cited-path recall | cited documents in `expected_paths` / `expected_paths` (answered answerable questions) |
| cited-path precision | cited documents in `expected_paths` / cited documents. Citing another relevant document lowers it |
| invalid ids, dropped claims | claims the server dropped because they cited ids that were not retrieved; each dropped claim is listed in `summary.md` |
| `uncited_expected_paths`, `extra_cited_paths` | the documents behind recall < 1 and precision < 1 |
| error class, error/root-cause/fact term recall, remediation coverage | **heuristics**: do ground-truth terms appear in the answer (remediation: in `next_steps`, and anywhere)? |
| `*_in_evidence` | **heuristics**: the same term checks on all retrieved chunk text, a rough ceiling. Low = the model was never shown it |
| empty `next_steps` | counted only for answered questions whose ground truth has a fix (schema questions expect none) |
| `refusal_note` | what the model said was missing when it refused; unverified model text, never returned by `/ask` |
| `groundedness_human` | always null: nobody has judged whether claims are supported by their citations |

Four different things are measured, and none of them stands in for another:

- **Retrieval metrics** (`retrieval_smoke`, the notebook): was the expected *document* among
  the top-k chunks? Document level: a runbook counts as retrieved even when only its Symptoms
  chunk was, not its Resolution.
- **Citation validity** (enforced in code): every cited id was retrieved. It does not show
  that the cited chunk supports the claim.
- **Answer heuristics**: surface term matches against the generated ground truth, not
  correctness. A correct paraphrase can score low and a wrong answer that repeats the right
  words can score high (`tests/test_answer_eval.py` shows both).
- **Groundedness**: not measured until a human labels answers.

What the first full run (`20261022T214719Z_answers`, gemini-3.5-flash-lite) showed, read from
its records: all 12 refusals of answerable questions happened when the needed text was not in
the evidence. In 3 the right log was not retrieved (the date filter applied, but incident
reports outranked the single matching log, as in the retrieval benchmark); in 9 runbook
questions only the runbook's Symptoms/Overview/Diagnosis chunks were retrieved, never its
Resolution. Of the 17 answers with a fix in the ground truth but no `next_steps`, 15 had no
Resolution chunk for that failure in their evidence; Q011's only one was from an unrelated
incident and Q037's question did not ask for a fix. Where only diagnosis steps were retrieved,
the model listed those as `next_steps`. Retrieving the Resolution section is a
retrieval change and is not done yet.

### The refusal gate (`MIN_SIMILARITY`)

Best-hit question similarity, from `retrieval_smoke`:

```
answerable   (44): min 0.736  p25 0.826  median 0.840  p75 0.856  max 0.894
unanswerable ( 6): min 0.585  p25 0.620  median 0.649  p75 0.716  max 0.764
```

`0.72` is the midpoint between the highest blocked unanswerable question (0.700) and the
lowest answerable one (0.736). On these 50 questions it blocks 5 of 6 unanswerable and 0 of
44 answerable. This is a weak signal fitted to a small sample:

- Q049 ("retention period for the clickstream Kafka topic", 0.764) passes the gate. It shares
  words with real clickstream documents, so only the model's `insufficient_evidence` flag
  can refuse it.
- New questions will land inside the overlap. Recalibrate after changing the model or the data.

### Failed requests are recorded too

Every `/ask` that reaches the API is written to `queries`, failures included
(`db/init/004_query_status.sql`):

| Column | Value |
|---|---|
| `status_code` | 200, 422, 500, 502 or 503 (NULL only for rows logged before 004) |
| `error_category` | one fixed code: `validation_error`, `llm_not_configured`, `llm_upstream_error`, `llm_bad_response`, `database_error`, `embedding_unavailable`, `embedding_misconfigured`, `internal_error` |
| `latency_ms` | time until the failure |
| `metadata.error.message` | our own message, passed through `redact()` and truncated; never a stack trace |
| `retrievals` rows | for LLM failures, the evidence that had already been retrieved |

What is stored is safe by construction. Every question, successful or not, is passed through
`redact()` and truncated to 1000 characters. Error messages sent to clients are redacted too:
a test showed an upstream error text could otherwise reach the client unredacted. A
**database outage cannot be recorded in the database**; it is logged as one warning line
with status, category and latency, and the response carries no `query_id`. Validation
failures on other routes (`/ingest`) are not recorded as queries.

```powershell
docker compose exec db psql -U copilot -d incident_copilot -c "SELECT status_code, error_category, count(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS median_ms FROM queries GROUP BY 1, 2 ORDER BY 1, 2"
```

### Query history survives re-indexing

`retrievals.chunk_id` was `ON DELETE CASCADE`, so re-indexing or pruning a document silently
deleted the retrieval history of every past question that used it. `003_query_history.sql`
changes it to `ON DELETE SET NULL` and stores an immutable snapshot on each row: path, chunk
index, section, timestamps, content hash and the redacted text that was shown to the model.

After a re-index, the history row stays. Its `chunk_id` becomes NULL, and
`observability.query_history()` reports `is_current = false`. Replacement chunks always get
new ids, so an old citation can never resolve to new content. Tests cover update, prune, and
"no row links to different content".

## 6c. Chat UI (Streamlit)

`src/incident_copilot/ui/app.py` is a chat page for the API. It is a thin client: every
question is sent to `POST /ask`, so retrieval, the refusal gate, the Gemini call and citation
validation all stay in the API. The page never calls Gemini, never reads the key, and only
renders what the response contains.

It shows, per answer: the status (answered / insufficient evidence / retrieval only), the
summary, **"Stated in the evidence"** (`evidence` claims with the path and section of each cited
id), **"Likely cause and other inferences"** (`inference`, with confidence and `based_on` ids),
next steps, every retrieved chunk with its text, similarity and whether it was cited, the
effective `top_k`, and, if you turn it on, the raw JSON response. The badge says which of
these happened: **answered from the evidence**; **refused: evidence too weak** (similarity
gate, model not called); **refused by the model** (the evidence was similar enough to send,
but does not answer the question: a high similarity score is not proof the answer is there);
**answer withheld** (no valid citation); **clarification needed**. API errors and timeouts
are shown as errors, never as answers. The evidence panel of a refusal says what happened to
the chunks instead of "0 cited". The sidebar has the Causora brand, the Chat and Knowledge
Base links, a small line saying whether the API and database are reachable (`/health/db`;
a problem is shown as a warning) and *Clear conversation*; a collapsed **Advanced** area holds
the retrieval-only switch (no LLM call), the `top_k` override and the raw-JSON toggle. The
brand type is Fredoka, which the browser loads from Google Fonts; without it the brand falls
back to system fonts and nothing else changes (all other text uses the theme font).

**Look and mascot.** Colours come from `.streamlit/config.toml` (start Streamlit from the
repository root so it is read). Answers use a cat avatar; while a request runs, the
thinking cat bobs gently next to "Investigating…" (static under the OS reduced-motion
setting). `/ask` returns the whole answer at once, so there is no separate retrieving or
generating state and no talking animation. The icons are made from the originals in
`ui/assets/originals/` by `python scripts/make_cat_icons.py` (background removed by a
flood fill from the border, so the cats' white fill stays opaque).

**`top_k`** is the maximum number of evidence *chunks* (passages) retrieved for one question,
not the number of documents: several chunks can come from one document. Without the
override the server's `ASK_TOP_K` (8) applies; the override is 1-20, the same range `/ask`
validates (the page refuses other values without sending them), and every answer shows the
`top_k` actually used. Changing it does not change the similarity threshold.

**Chat state.** The conversation lives in `st.session_state` (one per browser session). A
question is recorded once, by the chat box's or example button's callback; every rerun draws
the whole conversation with one loop, in the same place, followed by the pending answer's
placeholder, so earlier messages never move or disappear while Causora works. While a
request runs, the chat box and the options are disabled; if a request is interrupted anyway
(e.g. by switching page), it is sent again when you return, so no question is left without a
reply. Errors and refusals are stored as messages like answers.

**Knowledge Base** (sidebar: *Knowledge Base*). Lists every indexed document from
`GET /documents` by type (troubleshooting guides, task logs, incident reports with their
timelines, table schemas; the UI's names for the `runbook`, `log`, `incident` and `schema`
source types, which are unchanged everywhere else), with search over title, path, pipeline, sections and metadata (try "rate limit",
"deadlock", "INC-2043"). Each document is a card: a click anywhere on it opens a large viewer
(about 90% of the window, its own scrolling; X, Escape or a click on the dimmed background
close it, and the list keeps its filters) with the original file (sensitive values masked,
when the file is under DATA_DIR), the indexed chunks, labelled as such, and the sections and
metadata. Read-only: nothing is written or re-indexed, and the list is cached for a minute.
**Explain with AI** (top left of the viewer) opens a panel beside the document (below it on
narrow screens; the same button hides it) and sends only that document's indexed sections to
the LLM (`POST /documents/explain`, one call per document version, only when clicked; kept for
the session, a failure can be retried and the document stays readable) and shows what it describes, stages,
dependencies, failure modes, diagnosis, remediation, prevention and terms, each point marked
*documented* (cites its sections) or *inferred*, plus what the document does not cover.
Points citing sections that do not exist are dropped and counted. Long documents are cut to
40 chunks / 16,000 characters and the explanation says so.

**Follow-up questions.** The page sends this session's earlier turns (question, answer
summary, cited paths; at most 6) as `history`. The API resolves a follow-up deterministically,
with no extra LLM call (`retrieval/contextualize.py`):

- A question is a follow-up only if it names no incident ID, pipeline, task, error name or
  other snake_case identifier of its own, does not introduce a new subject ("which past
  incident ...", "a load task fails ..."), and points back (it, this, that, again, the root
  cause, what about ...). Anything else is standalone and ignores the history, so a new
  question never inherits an old incident. All 50 seed questions are standalone under this rule.
- The subject comes from the most recent earlier turn that has one: what that question named,
  or the incidents and logs its answer cited (looked up in the index; unknown paths are
  ignored). Retrieval searches with `"<question> (Earlier in this conversation: <subject>)"`;
  the model answers the user's own question and gets the subject as a note marked "not
  evidence". The similarity gate and citation validation apply exactly as before.
- If that turn covered more than one incident or failed run, Causora asks which one is meant
  (`insufficient_reason: ambiguous_reference`; no retrieval, no LLM call). If no earlier turn
  has a subject, the question is searched as it is, and the page shows a hint to name the
  incident, pipeline, task or error.
- The API keeps no conversation state: history exists only in the request, and Streamlit keeps
  it per browser session. Only the number of history turns is logged with a query.

Measured on 16 separate follow-up cases (`eval/followups.jsonl`, run
`eval/results/20261028T155551Z_followups`): the expected document was retrieved for 11/11
cases with one (5/11 with the question alone), and resolution matched for 16/16. Two
unanswerable follow-ups now pass the gate (the incident's words carry the similarity) and
were refused by the model, not the gate. See the run's `summary.md` for the per-case table.
With the current default retriever (run `eval/results/20261029T050213Z_followups`): 15/16
cases ended as expected (question alone: 7/16), 1 LLM call needed a retry, 0 errors. The miss
(F03, "how do we prevent it" after a dependency conflict) is refused by the model because the
runbook's Prevention section is not retrieved: evidence selection adds a runbook's Resolution
only for fail/error/fix wording, and never its Prevention section.
Re-run with `python -m incident_copilot.evaluation.followup_eval` (retrieval only, no LLM) or
add `--llm --pause 4` to generate answers (about 16 Gemini calls).

**Next steps cite evidence ids** (`next_step_evidence`, validated like claims: a step citing an
id that was not retrieved is dropped, one citing none is shown as unsupported). Each step also
has a `next_step_kinds` entry, *recovery* or *prevention* (the cited sections decide when they
are all Resolution/Diagnosis or all Prevention/Follow-ups, otherwise the model's label), and a
`next_step_support` entry decided from the cited chunks only: *incident_report* (an incident
report about this failure states it) or *general_guidance* (only runbooks, schemas or logs:
shown as "General guidance, not documented as this incident's fix; check that it applies").
`unknowns` lists what the evidence leaves open (an alternative not ruled out, a measurement
that was not available); the page shows it under "Not established by the evidence".

**Near misses.** A question the 0.72 gate refuses is checked once more against the index's
own list of failures (every incident report and failed log, matched by pipeline, task and
incident title against the question's topic words; `clarify.resolve_near_miss`), never by
lowering the threshold: several matching failures give a clarification even when retrieval
found only one of them ("why did the shipments fail?" offers INC-2045 and INC-2047); exactly
one, with its own incident report or log retrieved at 0.67 or more, is answered about that
failure with the evidence scoped to it ("why did the loyalty join fail?"), and the response
keeps the real similarity and says `gate.accepted_by`; anything else stays a refusal. A
picked, kept or matched failure also gets its own incident report's resolution and prevention
sections as evidence, added after the gate so they never change it.

### Start Causora locally (PowerShell, from the repository root)

Startup order: database, then API, then UI.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ui.txt      # one time

# 1. Database (pgvector on host port 5434; leaves a native Postgres on 5433 alone)
docker compose up -d db

# 2a. API in Docker. It publishes container port 8000 on API_PORT; a shell value overrides .env
#     (e.g. "127.0.0.1:8010" when another app holds 8000). Use the same value every time, or
#     Compose recreates the container on the new port. Settings such as RETRIEVER come only
#     from .env here, not from the shell.
$env:API_PORT = "127.0.0.1:8010"
docker compose up -d api

# 2b. ... or the API on the host instead (needs requirements-ml.txt and the cached model).
#     Shell variables override .env for this process only, e.g. the previous retriever:
#   $env:RETRIEVER = "vector+error_link+date"
#   python -m uvicorn incident_copilot.api.main:app --host 127.0.0.1 --port 8011

Invoke-RestMethod http://127.0.0.1:8010/health/db  # your API's address; expect status ok

# 3. UI, in another PowerShell window, pointed at that API (default http://127.0.0.1:8000)
$env:UI_API_URL = "http://127.0.0.1:8010"
python -m streamlit run src/incident_copilot/ui/app.py --server.address 127.0.0.1
```

`--server.address 127.0.0.1` keeps the page off your network (Streamlit otherwise listens on
every interface, and the page has no login). After editing files under `ui/`, stop and restart
Streamlit: a browser refresh re-runs `app.py` but can keep older imported modules.
Streamlit prints the page address (http://localhost:8501 by default; add `--server.port 8502`
if that is taken). Without a Gemini key the API answers `/ask` with 503
`llm_not_configured`, and the page says so. Retrieval-only mode and questions the gate refuses
work without a key.

---

## 7. Where each later layer plugs in (no restructuring)

| Layer | Plugs into | What changes |
|---|---|---|
| Hybrid retrieval | `retrieval/` + existing `tsv` and `embedding` columns | New module only; schema already has both indexes |
| Reranking | `retrieval/` + `RERANKER_MODEL`, `rerank_score` column | New module only |
| Citations | `generation/` + `retrievals.used_in_answer` | New module only |
| RAGAS evaluation | `evaluation/` + `eval_*` tables, `questions.seed.jsonl` | New module only; CI gets an `eval` job |
| Observability | `observability/` + `queries`/`retrievals` tables | Writes already have somewhere to go; dashboard is SQL over existing tables |
| Airflow ingestion | `airflow/dags/` + `ingestion/` | Add a Compose service under `profiles: ["airflow"]`; DAG calls the same `ingest_paths()` |
| Live indexing | `api/main.py` + `ingestion/` | New route; `ingestion_runs.trigger` already allows `failure_callback` |
| CI/CD | `.github/workflows/ci.yml` | Add `eval` and `deploy` jobs; the file marks where |

---

## 8. Daily milestones

Every day ends with something you can run. If a "done when" is not true, do not start the next
day. Budget about 4 to 5 focused hours a day.

| Day | Build | Done when |
|---|---|---|
| **1** | Run everything in this README. Read 3 logs, 3 runbooks, 2 schemas, 2 incidents. Write 5 questions of your own | `make test` shows 34 passed; `/health/db` returns ok; you can explain why the `KeyError` log needs the `orders_raw` schema doc |
| **2** | Three chunkers (log by time window, runbook by heading, schema one chunk per table; incident by heading). `ingestion/ingest.py` with the hash check. Embeddings with `BAAI/bge-small-en-v1.5` | Ingesting `data/` fills `documents` and `chunks`; **running it a second time changes 0 documents**; each chunker has unit tests |
| **3** | Vector-only retrieval, Gemini generation, citations, `POST /ask`; write a `queries` row per call | `/ask` returns an answer with chunk ids; every cited id was actually retrieved (enforced in code); latency and tokens are logged |
| **4** | Keyword search and reciprocal rank fusion; log all scores to `retrievals` | A test shows a question using an exact error string that vector-only misses and hybrid finds |
| **5** | Cross-encoder reranker; abstention threshold; strict grounded prompt, low temperature | All 6 unanswerable seed questions abstain; threshold choice is written down |
| **6** | Eval runner. First a **retrieval hit-rate** metric (are all `expected_paths` in the results? no LLM needed). Then RAGAS faithfulness, context precision, context recall | `docs/results.md` has a table: naive vs hybrid vs hybrid+rerank, with real numbers |
| **7** | Buffer. Inspect the worst questions. Document 3 failure cases and fixes. Copy the seed to `eval/questions.jsonl` and extend it with harder questions | Numbers improved or stayed put for a stated reason; failure cases written up |
| **8** | Airflow as a Compose profile. Ingestion DAG calling `ingest_paths()`. A "victim" DAG with tasks you can break on demand | `docker compose --profile airflow up`; changing one file re-embeds exactly one document |
| **9** | `POST /ingest/failure`, `on_failure_callback` on the victim DAG, secret redaction (use the `FAKE-` logs), error fingerprinting, idempotent upsert | Break the victim DAG, ask about it within about a minute, get a cited answer; firing the callback twice creates no duplicates; no `FAKE-` string reaches the database |
| **10** | Streamlit UI with clickable citations and a "mark resolved" form (writes an unverified incident doc) | A non-engineer can use it; resolved incidents appear in later answers, marked unverified |
| **11** | Dashboard: p50/p95 latency, tokens and cost per query, retrieval hit rate, abstain rate | Charts read from `queries`/`retrievals` with no new instrumentation |
| **12** | Terraform: one VM or container service plus managed Postgres. Set a billing alert first | `terraform apply` gives a public URL that answers a question; `terraform destroy` removes everything |
| **13** | CI/CD: add `eval` job (fail if faithfulness drops) and `deploy` job | A pull request that worsens retrieval fails CI |
| **14** | README, architecture diagram, results table, demo GIF, "Future work" section | A stranger can clone, run, and understand it in 15 minutes |

**Where it will run long:** Day 12 (cloud IAM and networking) and Day 6 (judging with a free-tier
LLM is rate-limited; cache every judged result so reruns are free). Day 7 is your buffer.

---

## 9. Cost: everything is free or student-budget

| Piece | Cost | Note |
|---|---|---|
| Postgres + pgvector, Airflow, Streamlit, Terraform | Free | Run locally in Docker |
| Embeddings and reranker | Free | Local models on CPU; first run downloads a few hundred MB |
| Generation and RAGAS judging | Free tier | Gemini free tier has rate limits; the pricing page lists no numbers, so check the current rate-limits page before Day 6 and add caching and backoff |
| CI | Free | GitHub Actions is free for public repositories; private repositories have monthly limits |
| Cloud (Day 12) | Free tier or credits | Set a billing alert on day one; destroy resources when not demoing |

Free-tier terms change. Verify them when you reach the relevant day.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `POSTGRES_PASSWORD is missing a value` | `.env` is missing or empty. `Copy-Item .env.example .env` |
| `/health/db` returns 503 | Run `docker compose ps`; the `db` service must be `healthy`. Check the password in `DATABASE_URL` matches `POSTGRES_PASSWORD` |
| Tests skipped with `password authentication failed for user "copilot"` although Docker's db is healthy | Usually **another Postgres owns the host port**. Check with `Get-NetTCPConnection -State Listen -LocalPort 5433,5434` and `Get-Service *postgres*`. A native Windows PostgreSQL service listening on the same port receives your connections. Pick a free port for `POSTGRES_PORT`, put the same port in `DATABASE_URL`, then `docker compose up -d db`. Also check the two passwords match |
| Password changed in `.env` but the database still rejects it | `POSTGRES_PASSWORD` is only applied when the volume is first created. Use the original password, or `docker compose down -v` (deletes all data) |
| Changed or added a `db/init/*.sql` but nothing changed | Init scripts only run on a new volume. Run `docker compose exec api python -m incident_copilot.db_migrate` |
| Port already in use | Change `POSTGRES_PORT` or `API_PORT` in `.env` |
| Database tests all say SKIPPED | Postgres is not reachable; the skip reason shows the real error. Run `docker compose up -d db`, and make sure `DATABASE_URL` points at it |
| `another indexing run is in progress` | A second indexer was started while one was running. Wait for it to finish |
| `EMBEDDING_DIM=... but chunks.embedding is vector(384)` | Use a 384-dimensional model; the schema dimension is fixed |
| Indexing is slow on the first run | It embeds 324 chunks on CPU (about 80 s in Docker Desktop). Later runs only embed changed files |
| `To support symlinks on Windows...` warning on the host | Harmless; the Hugging Face downloader copies files instead of linking them |
| `/ask` returns 503 `llm_not_configured` | Set `GEMINI_API_KEY` in `.env`, then `docker compose up -d api`. Use `"retrieval_only": true` meanwhile |
| `/ask` returns 502 `llm_upstream_error` | Gemini failed after retries (quota, outage, retired model name). Check `LLM_MODEL` against the deprecations page and the API logs (`docker compose logs api`) |
| `/ask` returns 503 `embedding_unavailable` | The model is not cached and could not be downloaded. Run `docker compose exec api python -m incident_copilot.embeddings download` |
| `/ask` returns 503 `database_error` | Usually an un-migrated schema. Run `docker compose exec api python -m incident_copilot.db_migrate` |
| A real answerable question gets "Not enough evidence" | Check `gate.best_similarity` in the response against `MIN_SIMILARITY`. Rephrase with the task, pipeline or error name |
| `make: command not found` (Windows) | Not needed; use the plain commands in section 1 |
| `pip install` fails on psycopg | Use Python 3.11 or newer; the `[binary]` extra needs no compiler |

---

## 11. Tests

```powershell
python -m pytest -rs                 # everything; -rs prints the reason for any skip
python -m pytest -m "not db"         # no database needed
python -m pytest -m "not slow"       # no embedding model needed
python -m ruff check .
```

Current suite: **508 tests** (run locally in two parts: `-m "not db"` 410 passed and `-m db` 98
passed, 0 skipped; warnings come from third-party packages):

| Marker | Count | Needs | Skips when |
|---|---|---|---|
| *(none)* | 408 | nothing: no network, no model, no API key (uses `HashingEmbedder` and `FakeLLMProvider`) | never |
| `db` | 98 | Postgres from `docker compose up -d db`; each run uses a throwaway database | Postgres unreachable (the reason is printed) |
| `slow` | 2 | `python -m pip install -r requirements-ml.txt` and `python -m incident_copilot.embeddings download` | model not cached; it never downloads |
