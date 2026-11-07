# Table: payments_raw

- **Layer:** raw
- **Owner:** finance-data
- **Populated by:** payments_sync (task pull_payments_api)
- **Freshness SLA:** 4 hours
- **Primary key:** payment_id

## Columns

| Column | Type | Description |
|---|---|---|
| payment_id | text | Provider payment id |
| order_id | bigint | Order the payment belongs to |
| amount | text | Amount as received from the provider API (string) |
| currency | text | ISO currency code |
| status | text | Provider status string |
| provider_ts | timestamptz | Provider event time |
| ingested_at | timestamptz | When we pulled it |

## Notes

- Amount arrives as text and is converted to numeric in normalize_payments.
