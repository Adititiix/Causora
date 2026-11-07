# Table: orders_raw

- **Layer:** raw
- **Owner:** storefront-eng
- **Populated by:** CDC replication from the storefront database (not an Airflow task)
- **Freshness SLA:** 6 hours
- **Primary key:** order_id

## Columns

| Column | Type | Description |
|---|---|---|
| order_id | bigint | Source order id |
| customer_id | bigint | Null for guest checkouts |
| loyalty_tier | text | Loyalty program tier (bronze, silver, gold). Renamed from customer_tier on 2026-08-28 |
| order_ts | timestamptz | When the order was placed (UTC) |
| quantity | integer | Total items in the order |
| order_total | numeric(12,2) | Order value in USD |
| updated_at | timestamptz | Last change in the source system |

## Notes

- Guest checkouts have a null customer_id.
- Freshness is measured on max(updated_at).

## Change history

- 2026-08-28: Column customer_tier renamed to loyalty_tier by storefront-eng (migration 0147). Downstream code that reads customer_tier must be updated.
