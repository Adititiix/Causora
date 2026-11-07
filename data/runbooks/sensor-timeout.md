# Runbook: Sensor timed out waiting for upstream

- **Runbook ID:** RB-011
- **Applies to:** orders_etl (wait_for_orders_export), inventory_snapshot (wait_for_inventory_export), marketing_attribution (wait_for_ad_export)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- AirflowSensorTimeout after exactly the configured timeout.
- The log repeats 'Success criteria not met' many times.

## Likely causes

- The upstream export failed or is very late.
- The marker file is written to a different path or date.

## Diagnosis

1. Check the upstream export job's status.
2. List the S3 prefix to see whether any data arrived without a marker.

## Resolution

1. Ask the export owner to fix or re-run the export.
2. Once the marker exists, clear the sensor task.

## Prevention

- Alert the export owners at the expected finish time, before the sensor times out.
- Document export SLAs for every upstream source.
