# Runbook: Database connection timeout

- **Runbook ID:** RB-003
- **Applies to:** orders_etl (extract_orders), customer_360 (build_customer_profile), inventory_snapshot (extract_inventory)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- psycopg2.OperationalError: connection ... timeout expired.
- Several 'Connection attempt n/3 timed out' warnings before the failure.

## Likely causes

- The replica failed over or restarted.
- All connection slots (max_connections) were in use.

## Diagnosis

1. Note the host in the error message.
2. Run pg_isready against that host from a worker.
3. On the replica, run SELECT state, count(*) FROM pg_stat_activity GROUP BY state and compare with max_connections.
4. Check the database team's channel for a failover or maintenance event.

## Resolution

1. If a failover is in progress, wait until it completes.
2. If connections are exhausted, terminate idle sessions or restart the connection pooler.
3. Clear the failed task so it retries.

## Prevention

- Route pipeline connections through the pooler instead of connecting directly.
- Retry connection errors with exponential backoff.
