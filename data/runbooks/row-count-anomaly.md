# Runbook: Row count dropped sharply (partial data)

- **Runbook ID:** RB-017
- **Applies to:** orders_etl (validate_orders), payments_sync (load_payments_fact), clickstream_ingest (load_sessions_fact)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- DataQualityError: row count check failed ... allowed drop is 40%.
- Row count is a fraction of the 7-day median.

## Likely causes

- Partial extract from the upstream export.
- An extract filter or join accidentally dropped rows.

## Diagnosis

1. Compare the source row count with the loaded row count.
2. Look for missing shards in the raw files.
3. Review recent changes to the extract query.

## Resolution

1. Do not load the partial data.
2. Re-run the extract for the date and confirm row counts match the source.
3. Then re-run the check and the load.

## Prevention

- Compare row counts against the source before loading.
- Keep the 40 percent drop check as a hard gate.
