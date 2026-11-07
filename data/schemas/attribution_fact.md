# Table: attribution_fact

- **Layer:** fact
- **Owner:** marketing-analytics
- **Populated by:** marketing_attribution (task load_attribution_fact)
- **Freshness SLA:** 24 hours
- **Primary key:** (campaign_id, conversion_date)

## Columns

| Column | Type | Description |
|---|---|---|
| campaign_id | text | Ad campaign id |
| conversion_date | date | Day of conversion |
| conversions | integer | Attributed conversions |
| conversion_rate | numeric(8,4) | conversions divided by clicks |
| attributed_revenue | numeric(12,2) | Revenue attributed in USD |

## Notes

- None.
