# Runbook: Upstream API rate limit (HTTP 429)

- **Runbook ID:** RB-007
- **Applies to:** payments_sync (pull_payments_api), marketing_attribution (extract_ad_spend)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- requests HTTPError 429 Too Many Requests.
- A Retry-After header is logged just before the failure.

## Likely causes

- Too many requests per minute from this task.
- Another DAG or a backfill is calling the same API at the same time.

## Diagnosis

1. Check whether other DAGs or backfills were using the same API.
2. Read the Retry-After value in the log.

## Resolution

1. Wait for the rate window to reset, then clear the task.
2. Add backoff that honors Retry-After.
3. Put API-calling tasks in an Airflow pool with limited slots.

## Prevention

- Use a shared pool for each external API.
- Do not run backfills during the daily window.
