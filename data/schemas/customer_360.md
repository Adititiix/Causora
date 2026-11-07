# Table: customer_360

- **Layer:** fact
- **Owner:** growth-analytics
- **Populated by:** customer_360 (task load_customer_360)
- **Freshness SLA:** 12 hours
- **Primary key:** customer_id

## Columns

| Column | Type | Description |
|---|---|---|
| customer_id | bigint | Primary key |
| first_order_date | date | Date of first order |
| lifetime_value | numeric(12,2) | Total spend in USD |
| loyalty_tier | text | Current loyalty tier |
| orders_count | integer | Number of orders |
| last_seen_at | timestamptz | Last activity |

## Notes

- Joins orders_raw and loyalty_members.
