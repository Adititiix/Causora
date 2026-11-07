# Runbook: Numeric column contains text values

- **Runbook ID:** RB-008
- **Applies to:** orders_etl (validate_orders), payments_sync (normalize_payments), inventory_snapshot (compute_stock_levels)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- pyarrow ArrowInvalid: Could not convert 'N/A' ... to int64 or double.
- A warning says the column has dtype object instead of a numeric type.

## Likely causes

- The source began emitting N/A or empty strings for missing numbers.

## Diagnosis

1. Read the column name in the error.
2. Count the non-numeric values in that column.
3. Ask the source team whether the export format changed.

## Resolution

1. Convert with pd.to_numeric(errors='coerce') so bad values become nulls.
2. Write the bad rows to a quarantine table for review.
3. Clear the failed task.

## Prevention

- Validate types and allowed values before conversion.
- Agree on a data contract with the source team.
