# Table: ad_spend_raw

- **Layer:** raw
- **Owner:** marketing-analytics
- **Populated by:** marketing_attribution (task extract_ad_spend)
- **Freshness SLA:** 24 hours
- **Primary key:** (campaign_id, spend_date)

## Columns

| Column | Type | Description |
|---|---|---|
| campaign_id | text | Ad campaign id |
| spend_date | date | Day of spend |
| spend_usd | numeric(12,2) | Spend in USD |
| clicks | integer | Clicks that day (0 for paused campaigns) |
| impressions | integer | Impressions that day |
| ingested_at | timestamptz | When we pulled it |

## Notes

- Paused campaigns report 0 clicks.
