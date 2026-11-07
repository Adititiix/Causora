# Prompt: add new failure scenarios (optional, for variety)

The generator already produces 18 scenarios deterministically. Use this prompt with any LLM
(Gemini free tier is fine) to add more, then save the results under `data/extra/` (NOT under
`data/runbooks`, `data/logs`, etc. -- `make data` deletes and regenerates those folders).

Files go to:

- `data/extra/runbooks/<scenario-id>.md`
- `data/extra/logs/<pipeline>/<YYYY-MM-DD>__<task>.log`
- `data/extra/incidents/INC-<number>.md` (continue numbering from INC-2045)

Review every output by hand before committing. LLMs invent plausible-looking nonsense.

---

## Prompt (copy everything below this line)

You are helping build a synthetic dataset for a RAG project about data-pipeline incidents at a
fictional e-commerce company called ShopStream. Produce ONE new failure scenario as three files.

### Company facts you must stay consistent with

Pipelines and their tasks (use only these):

- orders_etl: wait_for_orders_export, extract_orders, validate_orders, transform_orders, load_orders_fact
- payments_sync: pull_payments_api, normalize_payments, load_payments_fact
- inventory_snapshot: wait_for_inventory_export, extract_inventory, compute_stock_levels, load_inventory_snapshot
- customer_360: build_customer_profile, join_loyalty, load_customer_360
- marketing_attribution: wait_for_ad_export, extract_ad_spend, attribute_conversions, load_attribution_fact
- clickstream_ingest: consume_clickstream, sessionize_events, load_sessions_fact

Dates must fall between 2026-08-03 and 2026-09-30. The pipelines run daily at 02:00 UTC.
Tools in use: Airflow 2.x, Postgres, S3, pandas/pyarrow, Kafka (clickstream only), Kubernetes workers.

### Scenarios that already exist (do NOT repeat these)

column-renamed-upstream, source-file-missing, db-connection-timeout, oom-killed,
expired-credentials, duplicate-key-violation, api-rate-limited, type-mismatch-na-values,
disk-full, deadlock-detected, sensor-timeout, division-by-zero, kafka-offset-out-of-range,
freshness-check-failed, not-null-violation, dependency-version-conflict, row-count-anomaly,
zombie-task-sigterm

### What to produce

Pick a realistic failure NOT in the list above (examples: timezone/daylight-saving bug, schema
drift in a JSON field, a stuck Airflow pool, a partition written twice, a bad deploy of shared
code, a permissions change on a bucket). Then write:

1. A runbook in Markdown with exactly these sections: Symptoms, Likely causes, Diagnosis
   (numbered), Resolution (numbered), Prevention. Header lines: Runbook ID (RB-019 upward),
   Applies to, Owner.
2. One Airflow task log of the failed task, 25-40 lines, in this exact line format:
   `[2026-09-14T02:13:45.269+0000] {taskinstance.py:1956} INFO - message`
   Include a Python traceback if the failure raises an exception, the "Marking task as FAILED"
   line, and "Task exited with return code 1" (or the real signal).
   Include 3-5 lines of harmless noise (heartbeats, pool slots, connection ids).
3. A past-incident report: Date, Severity (SEV-2 or SEV-3), Pipeline, Failed task, Duration,
   Impact, Summary, Timeline (UTC) with 5 entries, Root cause, Resolution, Follow-ups.

### Hard rules

- The log, runbook, and incident must agree on the root cause, error text, task, and fix.
- Never output a real-looking secret. If a credential appears, it must start with FAKE-.
- Do not mention anything outside the company facts above (no real company names).
- Make the log's error message specific and searchable (exact exception class and message),
  because the project will test keyword search on it.
- Output the three files in three separate fenced blocks, each preceded by its target path.
