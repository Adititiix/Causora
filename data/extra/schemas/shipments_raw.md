# Table: shipments_raw

- **Layer:** raw
- **Owner:** logistics-eng
- **Populated by:** CDC replication from the fulfilment database (not an Airflow task)
- **Freshness SLA:** 6 hours
- **Primary key:** shipment_id (not enforced; see Notes)

## Columns

| Column | Type | Description |
|---|---|---|
| shipment_id | text | Fulfilment shipment id, e.g. SHP-88412907 |
| order_id | bigint | Order the shipment belongs to |
| carrier_id | text | Carrier identifier (ups, dhl, usps, fedex). Renamed from carrier_code on 2026-10-05 |
| shipped_at | timestamptz | When the parcel left the warehouse (UTC) |
| delivered_at | timestamptz | Null until the carrier confirms delivery |
| status | text | label_created, in_transit, delivered, or returned |
| updated_at | timestamptz | Last change in the source system |

## Notes

- Freshness is measured on max(updated_at).
- After a CDC connector restart, events can be replayed, so the same shipment_id can appear more than once. Consumers must keep only the row with the latest updated_at per shipment_id.

## Change history

- 2026-10-05: Column carrier_code renamed to carrier_id by logistics-eng (migration 0311, applied 14:20 UTC). Downstream code that reads carrier_code must be updated.
- 2026-10-15: Note on replayed CDC events added after INC-2047.
