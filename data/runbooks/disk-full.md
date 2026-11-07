# Runbook: Worker disk full

- **Runbook ID:** RB-009
- **Applies to:** clickstream_ingest (sessionize_events), orders_etl (transform_orders), marketing_attribution (attribute_conversions)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- OSError: [Errno 28] No space left on device under /tmp.
- A warning shows /tmp usage near 100 percent.

## Likely causes

- Leftover intermediate files from earlier failed runs.
- Larger than usual data volume.

## Diagnosis

1. Run df -h /tmp and du -sh /tmp/airflow/* on the worker named in the log.
2. Check the modification times of leftover files.

## Resolution

1. Delete stale files in /tmp/airflow on that worker.
2. Clear the failed task to re-run.
3. Add cleanup in a finally block.

## Prevention

- Schedule a cleanup job for old temporary files.
- Alert when worker disk usage passes 80 percent.
