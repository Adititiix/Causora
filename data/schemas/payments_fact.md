# Table: payments_fact

- **Layer:** fact
- **Owner:** finance-data
- **Populated by:** payments_sync (task load_payments_fact)
- **Freshness SLA:** 4 hours
- **Primary key:** payment_id

## Columns

| Column | Type | Description |
|---|---|---|
| payment_id | text | Primary key |
| order_id | bigint | NOT NULL. Order the payment belongs to |
| amount | numeric(12,2) | Payment amount in USD |
| currency | char(3) | ISO currency code |
| status | text | One of: authorized, captured, refunded, failed |
| paid_at | timestamptz | When the payment completed |
| loaded_at | timestamptz | When the row was loaded |

## Notes

- Refunds are separate rows with status refunded.
