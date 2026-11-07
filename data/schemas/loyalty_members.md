# Table: loyalty_members

- **Layer:** raw
- **Owner:** growth-analytics
- **Populated by:** CDC replication from the loyalty service (not an Airflow task)
- **Freshness SLA:** 12 hours
- **Primary key:** member_id

## Columns

| Column | Type | Description |
|---|---|---|
| member_id | bigint | Primary key |
| customer_id | bigint | Customer the membership belongs to |
| tier | text | bronze, silver, or gold |
| joined_at | date | Membership start date |
| points_balance | integer | Current points |
| updated_at | timestamptz | Last change in the source system |

## Notes

- None.
