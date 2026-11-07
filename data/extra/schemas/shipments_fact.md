# Table: shipments_fact

- **Layer:** fact
- **Owner:** data-platform
- **Populated by:** shipments_sync (task load_shipments_fact)
- **Freshness SLA:** 24 hours
- **Primary key:** shipment_id

## Columns

| Column | Type | Description |
|---|---|---|
| shipment_id | text | Primary key |
| order_id | bigint | NOT NULL. Joins to orders_fact.order_id |
| carrier_code | text | Upper-case carrier code (UPS, DHL, USPS, FEDEX). Read from shipments_raw.carrier_id since 2026-10-06 |
| ship_date | date | Date the parcel shipped (UTC) |
| delivered_at | timestamptz | Null until delivered |
| status | text | label_created, in_transit, delivered, or returned |
| loaded_at | timestamptz | When the row was loaded |

## Notes

- Freshness is measured on max(loaded_at). finance_reporting checks it before building the daily close.
- Loaded with an upsert keyed on shipment_id (INSERT ... ON CONFLICT DO UPDATE) since 2026-10-15; before that the load was a plain INSERT.
- transform_shipments keeps one row per shipment_id (the latest updated_at) since 2026-10-15.

## Change history

- 2026-10-06: carrier_code is now mapped from the renamed source column shipments_raw.carrier_id (INC-2045). The fact column keeps its name.
- 2026-10-15: Load changed to an upsert and the transform deduplicates on shipment_id (INC-2047).
