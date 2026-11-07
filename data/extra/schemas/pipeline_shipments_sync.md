# Pipeline: shipments_sync

- **DAG id:** shipments_sync
- **Schedule:** 01:00 UTC daily
- **On-call:** data-platform (#data-platform-oncall)
- **Reads:** shipments_raw (CDC replica of the fulfilment database, owned by logistics-eng)
- **Writes:** shipments_fact
- **Downstream consumers:** finance_reporting (check_source_freshness, build_daily_close)

## Tasks in execution order

1. extract_shipments: reads the rows of shipments_raw changed since the previous run.
2. transform_shipments: keeps the latest row per shipment_id, maps carrier_id to the upper-case carrier_code, normalises status. One retry after 5 minutes.
3. load_shipments_fact: upserts into shipments_fact keyed on shipment_id, in one transaction.

## Dependencies and sensors

- No sensor. shipments_raw is replicated continuously; its own freshness SLA is 6 hours on max(updated_at).
- Schema changes to shipments_raw are owned by logistics-eng and recorded in the shipments_raw change history.

## Freshness and failure propagation

- shipments_fact has a 24 hour freshness SLA on max(loaded_at). A normal run loads it by about 01:20 UTC.
- If transform_shipments or load_shipments_fact fails, shipments_fact keeps the previous day's rows. finance_reporting's check_source_freshness at about 03:30 UTC then fails once the table is more than 24 hours old, and the daily close is not built until shipments_fact is reloaded.
- Related incidents: INC-2045 (upstream column rename, 2026-10-06), INC-2047 (duplicate rows from replayed CDC events, 2026-10-15).
