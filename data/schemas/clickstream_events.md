# Table: clickstream_events

- **Layer:** raw
- **Owner:** data-platform
- **Populated by:** clickstream_ingest (task consume_clickstream)
- **Freshness SLA:** 2 hours
- **Primary key:** event_id

## Columns

| Column | Type | Description |
|---|---|---|
| event_id | uuid | Primary key |
| session_key | text | Client-side session key |
| event_type | text | page_view, click, add_to_cart, or purchase |
| page_url | text | URL of the page |
| event_ts | timestamptz | Client event time |
| ingested_at | timestamptz | When the event was consumed |

## Notes

- Consumed from a Kafka topic.
