# Runbook: Expired cloud credentials mid-task

- **Runbook ID:** RB-005
- **Applies to:** payments_sync (normalize_payments), inventory_snapshot (compute_stock_levels)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- botocore ClientError with ExpiredToken.
- The task had been running for more than an hour.

## Likely causes

- Temporary role credentials last one hour by default.
- A long task created its client once at the start and never refreshed it.

## Diagnosis

1. Check how long the task had been running when it failed.
2. Confirm the client was created once at the start of the task.

## Resolution

1. Clear the task to re-run with new credentials.
2. Use a refreshable session so credentials renew automatically, or split the work into shorter tasks.

## Prevention

- Use auto-refreshing credentials in the shared io helpers.
- Alert when tasks approach 50 minutes.
