# Table: inventory_snapshot

- **Layer:** fact
- **Owner:** supply-chain-data
- **Populated by:** inventory_snapshot (task load_inventory_snapshot)
- **Freshness SLA:** 24 hours
- **Primary key:** (sku_id, snapshot_date, warehouse_id)

## Columns

| Column | Type | Description |
|---|---|---|
| sku_id | text | Product SKU |
| snapshot_date | date | Day of the snapshot |
| warehouse_id | text | Warehouse code |
| on_hand_qty | integer | Units physically in stock |
| reserved_qty | integer | Units reserved for open orders |
| avg_stock | numeric(12,2) | Average stock over the trailing 30 days |

## Notes

- avg_stock can be 0 for new SKUs; metrics dividing by it must guard against zero.
