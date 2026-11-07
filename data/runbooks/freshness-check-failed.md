# Runbook: Source data freshness check failed

- **Runbook ID:** RB-014
- **Applies to:** orders_etl (validate_orders), customer_360 (build_customer_profile)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- DataQualityError: freshness check failed ... max(updated_at) is Nh old.
- Latest updated_at in the raw table is from the previous day.

## Likely causes

- The CDC connector stalled or crashed.
- The source system stopped writing.

## Diagnosis

1. Check the CDC connector status and lag.
2. Query max(updated_at) on the raw table.
3. Ask the source team whether they had an outage.

## Resolution

1. Restart the CDC connector.
2. Wait for lag to clear and fresh rows to arrive.
3. Clear the failed check task.

## Prevention

- Alert on CDC connector lag.
- Run the freshness check earlier than the main DAG.
