# Runbook: Duplicate key violation on load

- **Runbook ID:** RB-006
- **Applies to:** orders_etl (load_orders_fact), payments_sync (load_payments_fact), clickstream_ingest (load_sessions_fact)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- UniqueViolation on a <table>_pkey constraint.
- The failure follows a retry or a manual re-run.

## Likely causes

- The earlier attempt wrote some rows before failing.
- The load is not idempotent.

## Diagnosis

1. Check whether rows for the failed date already exist in the target table.
2. Look at earlier attempts of the same task for partial writes.

## Resolution

1. Delete the rows for that date from the target table, then clear the task.
2. Or change the load to an upsert (INSERT ... ON CONFLICT DO UPDATE) and re-run.

## Prevention

- Make every load idempotent with upserts keyed on the primary key.
- Load inside a transaction so partial writes roll back.
