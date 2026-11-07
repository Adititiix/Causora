# Table: inventory_movements

- **Layer:** raw
- **Owner:** supply-chain-data
- **Populated by:** CDC replication from the warehouse system (not an Airflow task)
- **Freshness SLA:** 6 hours
- **Primary key:** movement_id

## Columns

| Column | Type | Description |
|---|---|---|
| movement_id | bigint | Primary key |
| sku_id | text | Product SKU |
| warehouse_id | text | Warehouse code |
| delta_qty | integer | Signed change in stock |
| movement_type | text | receipt, shipment, adjustment, or return |
| moved_at | timestamptz | When the movement happened |

## Notes

- None.
