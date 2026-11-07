# Runbook: Database deadlock during load

- **Runbook ID:** RB-010
- **Applies to:** orders_etl (load_orders_fact), inventory_snapshot (load_inventory_snapshot), customer_360 (load_customer_360)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- psycopg2 DeadlockDetected with two processes blocking each other.
- The failure happens at about the same time as another job on the same table.

## Likely causes

- Overlapping writers on the same table.
- Rows updated in different orders by different jobs.

## Diagnosis

1. Read the two process ids and the relation name in the error.
2. Check which other jobs write to that table at that time.
3. Check the Postgres log for the queries involved.

## Resolution

1. Clear the failed task; a deadlock victim is safe to retry.
2. Sort rows by primary key before upserting in every writer.
3. Move the other job so schedules do not overlap.

## Prevention

- Always update rows in primary key order.
- Keep transactions short and avoid overlapping schedules on one table.
