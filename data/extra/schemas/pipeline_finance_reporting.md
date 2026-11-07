# Pipeline: finance_reporting

- **DAG id:** finance_reporting
- **Schedule:** 03:30 UTC daily
- **On-call:** data-platform (#data-platform-oncall); report owner finance-data
- **Reads:** orders_fact, payments_fact, shipments_fact
- **Writes:** finance_daily_close
- **Output:** the daily close report for the previous business day, due by 06:00 UTC

## Tasks in execution order

1. wait_for_payments_sync: ExternalTaskSensor that waits for payments_sync.load_payments_fact of the same day. Pokes every 1800 seconds; times out after 14,400 seconds (4 hours).
2. check_source_freshness: fails with DataQualityError if any input is older than its schema doc's freshness SLA: orders_fact 6 hours, payments_fact 4 hours, shipments_fact 24 hours (all on max(loaded_at)).
3. build_daily_close: writes one finance_daily_close row (shipped revenue, captured and refunded payments, shipped orders).
4. publish_close_report: emails the daily close report to finance.

## Dependencies and sensors

- payments_sync (02:00 UTC): hard dependency through the sensor.
- orders_etl (02:00 UTC) and shipments_sync (01:00 UTC): no sensor; only their freshness is checked.

## Failure propagation and impact

- A late payments_sync run makes the sensor time out at about 07:30 UTC, and the report misses its 06:00 deadline (INC-2046, 2026-10-12).
- A failed shipments_sync run leaves shipments_fact stale; check_source_freshness fails and build_daily_close does not run (INC-2045, 2026-10-06).
- build_daily_close never runs on stale inputs, so a late report is preferred to a wrong one.
