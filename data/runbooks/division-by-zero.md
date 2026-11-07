# Runbook: Division by zero in a metric calculation

- **Runbook ID:** RB-012
- **Applies to:** marketing_attribution (attribute_conversions), inventory_snapshot (compute_stock_levels)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- ZeroDivisionError in a metrics function.
- A warning reports entities with a zero denominator.

## Likely causes

- Paused campaigns or empty stock produce zero denominators.

## Diagnosis

1. Read the code line shown in the traceback.
2. Count entities with a zero denominator in the input.

## Resolution

1. Guard the division with a zero check or NULLIF.
2. Deploy and clear the failed task.

## Prevention

- Unit-test metric functions with zero denominators.
