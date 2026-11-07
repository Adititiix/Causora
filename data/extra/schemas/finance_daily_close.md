# Table: finance_daily_close

- **Layer:** mart
- **Owner:** finance-data
- **Populated by:** finance_reporting (task build_daily_close)
- **Freshness SLA:** 24 hours
- **Primary key:** close_date

## Columns

| Column | Type | Description |
|---|---|---|
| close_date | date | Business day being closed (UTC) |
| shipped_revenue | numeric(14,2) | Sum of orders_fact.order_total for orders with a shipment in shipments_fact on close_date |
| captured_payments | numeric(14,2) | Sum of payments_fact.amount with status captured on close_date |
| refunded_payments | numeric(14,2) | Sum of payments_fact.amount with status refunded on close_date |
| shipped_orders | integer | Distinct orders shipped on close_date |
| built_at | timestamptz | When the row was built |

## Notes

- One row per business day, built by finance_reporting after its source freshness check passes.
- Revenue is recognised on the ship date, so a stale shipments_fact understates shipped_revenue. The build therefore never runs on stale inputs.
- The daily close report sent to finance (publish_close_report) is due by 06:00 UTC.
