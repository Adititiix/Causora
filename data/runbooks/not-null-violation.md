# Runbook: Null value in a NOT NULL column

- **Runbook ID:** RB-015
- **Applies to:** orders_etl (load_orders_fact), payments_sync (load_payments_fact), customer_360 (load_customer_360)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- psycopg2 NotNullViolation naming a column and table.
- A warning counts rows with null in that column.

## Likely causes

- A new kind of record (such as guest checkout) has no value for the column.
- The transform does not handle nulls.

## Diagnosis

1. Read the column and relation in the error.
2. Count null values for that column in the input.
3. Check the table schema doc for how nulls should be handled.

## Resolution

1. Update the transform to default or filter nulls as the schema doc specifies.
2. Write rows that cannot be fixed to a quarantine table.
3. Clear the failed load.

## Prevention

- Validate required columns before loading.
- Review new product features with data owners.
