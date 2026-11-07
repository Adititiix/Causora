# Runbook: Expected source file is missing

- **Runbook ID:** RB-002
- **Applies to:** orders_etl (extract_orders), inventory_snapshot (extract_inventory), marketing_attribution (extract_ad_spend)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- FileNotFoundError for an s3 path ending in part-0000.parquet.
- The log shows the prefix listing returned 0 objects.

## Likely causes

- The upstream export job ran late or failed.
- The export wrote to a different date partition.

## Diagnosis

1. Open the S3 prefix for the failed date and check whether any files or a _SUCCESS marker exist.
2. Check the upstream export job's last run status.
3. Look for a partition written under the wrong date.

## Resolution

1. If the export is late, wait for it to finish and the _SUCCESS marker to appear.
2. If the export failed, ask its owner to re-run it for the failed date.
3. Clear the failed task so it retries against the new files.

## Prevention

- Put a sensor on the _SUCCESS marker before extracting.
- Alert the export owners when their job misses its expected finish time.
