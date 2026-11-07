# Runbook: Column renamed upstream breaks a transform

- **Runbook ID:** RB-001
- **Applies to:** orders_etl (transform_orders), customer_360 (join_loyalty)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- The task fails with KeyError: '<column name>' inside a transform step.
- The column list printed in the log does not contain the column the code expects.
- Failures start right after an upstream schema change or release.

## Likely causes

- An upstream team renamed or dropped a column.
- The transform code was not updated for the new schema.

## Diagnosis

1. Read the column name in the KeyError.
2. Find the log line 'Loaded N rows ... (columns: ...)' and compare those columns with what the code expects.
3. Check the table's schema doc change history for recent renames.
4. Ask the owning team whether a migration shipped.

## Resolution

1. Update the transform to use the new column name, or add an alias mapping old to new.
2. Deploy the fix through the normal CI pipeline.
3. Clear the failed task in Airflow so it re-runs for the failed date and confirm downstream tasks succeed.

## Prevention

- Add a schema contract check at the start of the pipeline that fails fast and lists the missing columns.
- Subscribe pipeline owners to upstream schema-change announcements.
