# Runbook: Task killed when the worker node was reclaimed

- **Runbook ID:** RB-018
- **Applies to:** orders_etl (transform_orders), clickstream_ingest (sessionize_events), inventory_snapshot (load_inventory_snapshot)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- AirflowException: Task received SIGTERM signal.
- The log shows a node drain or spot reclaim notice before the failure.

## Likely causes

- Spot instance reclaimed by the cloud provider.
- Node scaled down or drained for maintenance.

## Diagnosis

1. Look for a drain or reclaim notice in the log just before SIGTERM.
2. Check cluster events for node terminations at that time.

## Resolution

1. Clear the task so it retries on a healthy worker.
2. Move critical long-running tasks to on-demand nodes.

## Prevention

- Enable retries on long tasks.
- Run critical pipelines on on-demand capacity.
