# Pipeline: payments_sync

- **DAG id:** payments_sync
- **Schedule:** 02:00 UTC daily
- **On-call:** data-platform (#data-platform-oncall); table owner finance-data
- **Reads:** the payment provider API at api.paygate.example (paginated)
- **Writes:** payments_raw, payments_fact
- **Downstream consumers:** finance_reporting (sensor wait_for_payments_sync), payment reconciliation and the refunds dashboard

## Tasks in execution order

1. pull_payments_api: pages through the provider API and writes payments_raw. On HTTP 429 it now waits for the Retry-After interval and retries the page (it did so on 2026-10-12; during INC-2043 a 429 still failed the task). Runs in the Airflow pool paygate_api (1 slot) since 2026-10-13; backfills share the same pool.
2. normalize_payments: converts amount from text to numeric(12,2) and maps provider status strings.
3. load_payments_fact: loads payments_fact.

## Dependencies and sensors

- No sensor. The only upstream is the provider API and its rate limit.
- Backfills of payments_sync may run only between 09:00 and 23:00 UTC.

## Freshness and failure propagation

- payments_raw and payments_fact have a 4 hour freshness SLA. A normal run loads payments_fact by about 02:20 UTC.
- If payments_sync is late, finance_reporting waits up to 4 hours in wait_for_payments_sync and then fails, delaying the daily close report.
- Related incidents: INC-2043 (HTTP 429 rate limit failed pull_payments_api, 2026-09-16), INC-2044 (expired credentials in normalize_payments, 2026-09-29), INC-2046 (rate-limited pull delayed the close report, 2026-10-12).
