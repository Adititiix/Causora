# Runbook: Dependency version conflict after image rebuild

- **Runbook ID:** RB-016
- **Applies to:** orders_etl (transform_orders), payments_sync (normalize_payments), marketing_attribution (attribute_conversions)
- **Owner:** data-platform on-call (#data-platform-oncall)

## Symptoms

- ImportError or ModuleNotFoundError at import time.
- Many unrelated DAGs start failing right after a new worker image rollout.

## Likely causes

- An unpinned dependency upgraded during an image rebuild.
- Two libraries now require incompatible versions of a shared package.

## Diagnosis

1. Check the image tag in the log and when it was deployed.
2. Compare pip freeze between the old and new image.

## Resolution

1. Roll back to the previous worker image.
2. Pin versions in a constraints file and rebuild.
3. Clear the failed tasks.

## Prevention

- Use a lockfile or constraints file for every image build.
- Run a smoke test that imports all DAGs in CI before deploying an image.
