# Table: sessions_fact

- **Layer:** fact
- **Owner:** data-platform
- **Populated by:** clickstream_ingest (task load_sessions_fact)
- **Freshness SLA:** 2 hours
- **Primary key:** session_id

## Columns

| Column | Type | Description |
|---|---|---|
| session_id | text | Primary key |
| customer_id | bigint | Null for anonymous sessions |
| started_at | timestamptz | Session start |
| duration_seconds | integer | Session length |
| page_count | integer | Pages viewed |
| converted | boolean | True if the session ended in a purchase |

## Notes

- None.
