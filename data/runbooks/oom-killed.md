# Runbook: Task killed for running out of memory

- **Runbook ID:** RB-004
- **Applies to:** orders_etl (transform_orders), clickstream_ingest (sessionize_events), marketing_attribution (attribute_conversions)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- The task ends with return code -9 (Negsignal.SIGKILL) and no Python traceback.
- The log shows reason=OOMKilled and exit code 137.
- Memory usage warnings climb toward the limit just before the task stops.

## Likely causes

- Data volume grew (a sale, a campaign, or a backfill).
- The task loads the entire partition into memory.

## Diagnosis

1. Look for 'Memory usage ... of 4.0 GiB limit' warnings in the log.
2. Compare today's row count with the previous days.
3. Check the pod events for OOMKilled.

## Resolution

1. Raise the memory request and limit for the task and clear it to re-run.
2. Change the task to process the partition in chunks (for example per hour) instead of all at once.

## Prevention

- Alert when memory usage passes 80 percent of the limit.
- Load tests with a 2x volume before major sales events.
