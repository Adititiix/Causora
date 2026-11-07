# Runbook: Kafka consumer offset out of range

- **Runbook ID:** RB-013
- **Applies to:** clickstream_ingest (consume_clickstream)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- kafka OffsetOutOfRangeError for a topic partition.
- The consumer group was paused or down before the failure.

## Likely causes

- The consumer group was stopped long enough that its committed offset was deleted from the broker.

## Diagnosis

1. Check how long the consumer group was paused.
2. Compare the committed offset with the earliest offset on the partition.

## Resolution

1. Reset the consumer group offsets to the earliest available.
2. Backfill the missing time window from the S3 archive.
3. Clear the failed task.

## Prevention

- Alert on consumer lag.
- Do not pause the consumer group for long maintenance windows.
