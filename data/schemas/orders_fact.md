# Table: orders_fact

- **Layer:** fact
- **Owner:** data-platform
- **Populated by:** orders_etl (task load_orders_fact)
- **Freshness SLA:** 6 hours
- **Primary key:** order_id

## Columns

| Column | Type | Description |
|---|---|---|
| order_id | bigint | Primary key |
| customer_id | bigint | NOT NULL. Guest orders are mapped to customer_id 0 |
| loyalty_tier | text | Loyalty tier at order time |
| order_date | date | Order date (UTC) |
| quantity | integer | Total items |
| order_total | numeric(12,2) | Order value in USD |
| is_first_order | boolean | True if the customer's first order |
| loaded_at | timestamptz | When the row was loaded |

## Notes

- Loaded with an upsert keyed on order_id for idempotent re-runs.
