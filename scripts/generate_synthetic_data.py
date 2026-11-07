#!/usr/bin/env python3
"""Generate the synthetic ShopStream dataset and the seed evaluation questions.

Why synthetic: a portfolio project cannot use a real company's logs. This generator
builds a realistic fake one, deterministically (same seed -> byte-identical output), so
your eval set never drifts.

Output (under --out, default data/):
    runbooks/*.md     one per failure scenario        (18)
    logs/<pipeline>/  Airflow-style task logs         (54 failures + 6 successes)
    schemas/*.md      one per table                   (12)
    incidents/*.md    past incident reports           (14)
    manifest.json     every file with its hash, type, pipeline, and scenario
Plus --eval-out: 50 seed questions (JSONL) with ground truth and expected source files.

Some logs deliberately contain FAKE credentials (FAKE-PASSWORD-..., FAKE-TOKEN-...) so you
can build and test secret redaction on Day 9. They are not real secrets.

Usage:
    python scripts/generate_synthetic_data.py --out data --eval-out eval/questions.seed.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

DEFAULT_SEED = 42
START = date(2026, 8, 3)
END = date(2026, 9, 30)
RENAME_DATE = date(2026, 8, 29)  # first day after the 2026-08-28 column rename
LOGS_PER_SCENARIO = 3
SUCCESS_LOGS = 6
LOGS_WITH_FAKE_SECRETS = 8
GENERATED_DIRS = ("runbooks", "logs", "schemas", "incidents")

# ---------------------------------------------------------------------------
# The fictional company: pipelines and tables
# ---------------------------------------------------------------------------

PIPELINES: dict[str, dict] = {
    "orders_etl": dict(
        tasks=["wait_for_orders_export", "extract_orders", "validate_orders",
               "transform_orders", "load_orders_fact"],
        fact="orders_fact", raw="orders_raw", schedule="02:00 UTC daily",
        impact="the daily sales dashboard and the finance revenue report",
    ),
    "payments_sync": dict(
        tasks=["pull_payments_api", "normalize_payments", "load_payments_fact"],
        fact="payments_fact", raw="payments_raw", schedule="02:00 UTC daily",
        impact="payment reconciliation and the refunds dashboard",
    ),
    "inventory_snapshot": dict(
        tasks=["wait_for_inventory_export", "extract_inventory", "compute_stock_levels",
               "load_inventory_snapshot"],
        fact="inventory_snapshot", raw="inventory_movements", schedule="02:00 UTC daily",
        impact="warehouse restocking alerts and the stock-level dashboard",
    ),
    "customer_360": dict(
        tasks=["build_customer_profile", "join_loyalty", "load_customer_360"],
        fact="customer_360", raw="loyalty_members", schedule="02:00 UTC daily",
        impact="CRM email segments and the customer lifetime value report",
    ),
    "marketing_attribution": dict(
        tasks=["wait_for_ad_export", "extract_ad_spend", "attribute_conversions",
               "load_attribution_fact"],
        fact="attribution_fact", raw="ad_spend_raw", schedule="02:00 UTC daily",
        impact="the marketing ROI dashboard and weekly budget review",
    ),
    "clickstream_ingest": dict(
        tasks=["consume_clickstream", "sessionize_events", "load_sessions_fact"],
        fact="sessions_fact", raw="clickstream_events", schedule="02:00 UTC daily",
        impact="the web funnel dashboard and product analytics",
    ),
}

TABLES: list[dict] = [
    dict(
        name="orders_raw", layer="raw", owner="storefront-eng", sla=6, pk="order_id",
        populated="CDC replication from the storefront database (not an Airflow task)",
        cols=[
            ("order_id", "bigint", "Source order id"),
            ("customer_id", "bigint", "Null for guest checkouts"),
            ("loyalty_tier", "text", "Loyalty program tier (bronze, silver, gold). "
                                      "Renamed from customer_tier on 2026-08-28"),
            ("order_ts", "timestamptz", "When the order was placed (UTC)"),
            ("quantity", "integer", "Total items in the order"),
            ("order_total", "numeric(12,2)", "Order value in USD"),
            ("updated_at", "timestamptz", "Last change in the source system"),
        ],
        notes=["Guest checkouts have a null customer_id.",
               "Freshness is measured on max(updated_at)."],
        history=[("2026-08-28", "Column customer_tier renamed to loyalty_tier by storefront-eng "
                                "(migration 0147). Downstream code that reads customer_tier must "
                                "be updated.")],
    ),
    dict(
        name="orders_fact", layer="fact", owner="data-platform", sla=6, pk="order_id",
        populated="orders_etl (task load_orders_fact)",
        cols=[
            ("order_id", "bigint", "Primary key"),
            ("customer_id", "bigint", "NOT NULL. Guest orders are mapped to customer_id 0"),
            ("loyalty_tier", "text", "Loyalty tier at order time"),
            ("order_date", "date", "Order date (UTC)"),
            ("quantity", "integer", "Total items"),
            ("order_total", "numeric(12,2)", "Order value in USD"),
            ("is_first_order", "boolean", "True if the customer's first order"),
            ("loaded_at", "timestamptz", "When the row was loaded"),
        ],
        notes=["Loaded with an upsert keyed on order_id for idempotent re-runs."],
        history=[],
    ),
    dict(
        name="payments_raw", layer="raw", owner="finance-data", sla=4, pk="payment_id",
        populated="payments_sync (task pull_payments_api)",
        cols=[
            ("payment_id", "text", "Provider payment id"),
            ("order_id", "bigint", "Order the payment belongs to"),
            ("amount", "text", "Amount as received from the provider API (string)"),
            ("currency", "text", "ISO currency code"),
            ("status", "text", "Provider status string"),
            ("provider_ts", "timestamptz", "Provider event time"),
            ("ingested_at", "timestamptz", "When we pulled it"),
        ],
        notes=["Amount arrives as text and is converted to numeric in normalize_payments."],
        history=[],
    ),
    dict(
        name="payments_fact", layer="fact", owner="finance-data", sla=4, pk="payment_id",
        populated="payments_sync (task load_payments_fact)",
        cols=[
            ("payment_id", "text", "Primary key"),
            ("order_id", "bigint", "NOT NULL. Order the payment belongs to"),
            ("amount", "numeric(12,2)", "Payment amount in USD"),
            ("currency", "char(3)", "ISO currency code"),
            ("status", "text", "One of: authorized, captured, refunded, failed"),
            ("paid_at", "timestamptz", "When the payment completed"),
            ("loaded_at", "timestamptz", "When the row was loaded"),
        ],
        notes=["Refunds are separate rows with status refunded."],
        history=[],
    ),
    dict(
        name="inventory_movements", layer="raw", owner="supply-chain-data", sla=6,
        pk="movement_id",
        populated="CDC replication from the warehouse system (not an Airflow task)",
        cols=[
            ("movement_id", "bigint", "Primary key"),
            ("sku_id", "text", "Product SKU"),
            ("warehouse_id", "text", "Warehouse code"),
            ("delta_qty", "integer", "Signed change in stock"),
            ("movement_type", "text", "receipt, shipment, adjustment, or return"),
            ("moved_at", "timestamptz", "When the movement happened"),
        ],
        notes=[],
        history=[],
    ),
    dict(
        name="inventory_snapshot", layer="fact", owner="supply-chain-data", sla=24,
        pk="(sku_id, snapshot_date, warehouse_id)",
        populated="inventory_snapshot (task load_inventory_snapshot)",
        cols=[
            ("sku_id", "text", "Product SKU"),
            ("snapshot_date", "date", "Day of the snapshot"),
            ("warehouse_id", "text", "Warehouse code"),
            ("on_hand_qty", "integer", "Units physically in stock"),
            ("reserved_qty", "integer", "Units reserved for open orders"),
            ("avg_stock", "numeric(12,2)", "Average stock over the trailing 30 days"),
        ],
        notes=["avg_stock can be 0 for new SKUs; metrics dividing by it must guard against zero."],
        history=[],
    ),
    dict(
        name="loyalty_members", layer="raw", owner="growth-analytics", sla=12,
        pk="member_id",
        populated="CDC replication from the loyalty service (not an Airflow task)",
        cols=[
            ("member_id", "bigint", "Primary key"),
            ("customer_id", "bigint", "Customer the membership belongs to"),
            ("tier", "text", "bronze, silver, or gold"),
            ("joined_at", "date", "Membership start date"),
            ("points_balance", "integer", "Current points"),
            ("updated_at", "timestamptz", "Last change in the source system"),
        ],
        notes=[],
        history=[],
    ),
    dict(
        name="customer_360", layer="fact", owner="growth-analytics", sla=12,
        pk="customer_id",
        populated="customer_360 (task load_customer_360)",
        cols=[
            ("customer_id", "bigint", "Primary key"),
            ("first_order_date", "date", "Date of first order"),
            ("lifetime_value", "numeric(12,2)", "Total spend in USD"),
            ("loyalty_tier", "text", "Current loyalty tier"),
            ("orders_count", "integer", "Number of orders"),
            ("last_seen_at", "timestamptz", "Last activity"),
        ],
        notes=["Joins orders_raw and loyalty_members."],
        history=[],
    ),
    dict(
        name="ad_spend_raw", layer="raw", owner="marketing-analytics", sla=24,
        pk="(campaign_id, spend_date)",
        populated="marketing_attribution (task extract_ad_spend)",
        cols=[
            ("campaign_id", "text", "Ad campaign id"),
            ("spend_date", "date", "Day of spend"),
            ("spend_usd", "numeric(12,2)", "Spend in USD"),
            ("clicks", "integer", "Clicks that day (0 for paused campaigns)"),
            ("impressions", "integer", "Impressions that day"),
            ("ingested_at", "timestamptz", "When we pulled it"),
        ],
        notes=["Paused campaigns report 0 clicks."],
        history=[],
    ),
    dict(
        name="attribution_fact", layer="fact", owner="marketing-analytics", sla=24,
        pk="(campaign_id, conversion_date)",
        populated="marketing_attribution (task load_attribution_fact)",
        cols=[
            ("campaign_id", "text", "Ad campaign id"),
            ("conversion_date", "date", "Day of conversion"),
            ("conversions", "integer", "Attributed conversions"),
            ("conversion_rate", "numeric(8,4)", "conversions divided by clicks"),
            ("attributed_revenue", "numeric(12,2)", "Revenue attributed in USD"),
        ],
        notes=[],
        history=[],
    ),
    dict(
        name="clickstream_events", layer="raw", owner="data-platform", sla=2,
        pk="event_id",
        populated="clickstream_ingest (task consume_clickstream)",
        cols=[
            ("event_id", "uuid", "Primary key"),
            ("session_key", "text", "Client-side session key"),
            ("event_type", "text", "page_view, click, add_to_cart, or purchase"),
            ("page_url", "text", "URL of the page"),
            ("event_ts", "timestamptz", "Client event time"),
            ("ingested_at", "timestamptz", "When the event was consumed"),
        ],
        notes=["Consumed from a Kafka topic."],
        history=[],
    ),
    dict(
        name="sessions_fact", layer="fact", owner="data-platform", sla=2,
        pk="session_id",
        populated="clickstream_ingest (task load_sessions_fact)",
        cols=[
            ("session_id", "text", "Primary key"),
            ("customer_id", "bigint", "Null for anonymous sessions"),
            ("started_at", "timestamptz", "Session start"),
            ("duration_seconds", "integer", "Session length"),
            ("page_count", "integer", "Pages viewed"),
            ("converted", "boolean", "True if the session ended in a purchase"),
        ],
        notes=[],
        history=[],
    ),
]

# ---------------------------------------------------------------------------
# Failure scenarios: the heart of the dataset
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    id: str
    title: str
    combos: list[tuple[str, str]]
    exc: str
    frames: list[tuple[str, str, str]]
    pre: list[tuple[str, str]]
    root_cause: str
    fix_summary: str
    symptoms: list[str]
    causes: list[str]
    diagnosis: list[str]
    resolution: list[str]
    prevention: list[str]
    runbook_question: str
    vars_fn: Callable[[random.Random, dict], dict] | None = None
    exit_line: str = "Task exited with return code 1"
    no_traceback: bool = False
    not_before: date | None = None


DAG_FILE = "/opt/airflow/dags/{pipeline}/tasks.py"
SITE = "/usr/local/lib/python3.11/site-packages"

SRC_FOLDER = {"orders_etl": "orders", "inventory_snapshot": "inventory",
              "marketing_attribution": "ad_spend"}


def _v_column(rng, c):
    return {"old_col": "customer_tier", "new_col": "loyalty_tier"}


def _v_source_file(rng, c):
    src = SRC_FOLDER[c["pipeline"]]
    return {"src": src, "path": f"shopstream-raw/{src}/dt={c['date']}/part-0000.parquet"}


def _v_db_timeout(rng, c):
    return {"host": rng.choice(["pg-orders-replica.internal", "pg-customers-replica.internal"]),
            "ip": f"10.20.{rng.randint(1, 9)}.{rng.randint(10, 99)}"}


def _v_oom(rng, c):
    return {"mem1": f"{rng.uniform(3.4, 3.7):.1f}", "mem2": f"{rng.uniform(3.8, 3.99):.2f}"}


def _v_creds(rng, c):
    return {"op": rng.choice(["PutObject", "GetObject", "UploadPart"]),
            "mins": rng.randint(61, 95)}


def _v_dupkey(rng, c):
    pk = {"orders_fact": "order_id", "payments_fact": "payment_id",
          "sessions_fact": "session_id"}[c["table"]]
    return {"pk": pk, "keyval": rng.randint(1_000_000, 9_999_999)}


def _v_rate_limit(rng, c):
    if c["pipeline"] == "payments_sync":
        host, path = "api.paygate.example", "/v2/transactions"
    else:
        host, path = "ads.adnetwork.example", "/v1/spend"
    page = rng.randint(40, 400)
    return {"api_host": host, "page": page, "url": f"https://{host}{path}?page={page}"}


def _v_type(rng, c):
    col, dtype = {"orders_etl": ("quantity", "int64"), "payments_sync": ("amount", "double"),
                  "inventory_snapshot": ("on_hand_qty", "int64")}[c["pipeline"]]
    return {"col": col, "dtype": dtype, "bad": rng.randint(12, 400)}


def _v_deadlock(rng, c):
    return {"p1": rng.randint(1000, 4000), "p2": rng.randint(4001, 9000),
            "x1": rng.randint(100000, 999999), "x2": rng.randint(100000, 999999),
            "secs": rng.randint(20, 300)}


def _v_sensor(rng, c):
    return {"src": SRC_FOLDER[c["pipeline"]], "dur": f"{21600 + rng.random() * 3:.1f}"}


def _v_zero(rng, c):
    if c["pipeline"] == "marketing_attribution":
        m = ("conversion_rate", "clicks", "conversion_rate = conversions / clicks")
    else:
        m = ("stock_turnover", "average stock", "stock_turnover = units_sold / avg_stock")
    return {"metric": m[0], "denom": m[1], "code": m[2],
            "n": rng.randint(200, 2000), "z": rng.randint(1, 15)}


def _v_kafka(rng, c):
    return {"topic": "clickstream.events.v2", "part": rng.randint(0, 11),
            "off": rng.randint(10_000_000, 99_999_999), "hrs": rng.randint(30, 70)}


def _v_fresh(rng, c):
    # Age always exceeds the table's own SLA, and `latest` is consistent with `age`
    # relative to the ~02:15 UTC check time.
    sla = next(t["sla"] for t in TABLES if t["name"] == c["raw_table"])
    age_h, mins = sla + rng.randint(1, 6), rng.randint(0, 59)
    run = datetime.combine(date.fromisoformat(c["date"]), time(2, 15))
    latest = run - timedelta(hours=age_h, minutes=mins)
    return {"sla": sla, "age": f"{age_h}h {mins}m", "latest": latest.isoformat() + "Z"}


def _v_notnull(rng, c):
    col = {"orders_etl": "customer_id", "payments_sync": "order_id",
           "customer_360": "customer_id"}[c["pipeline"]]
    return {"col": col, "nulls": rng.randint(5, 300), "keyval": rng.randint(1_000_000, 9_999_999)}


def _v_dep(rng, c):
    return {"img": f"{c['date'].replace('-', '.')}-{rng.randint(1, 3)}"}


def _v_rowcount(rng, c):
    chk = c["raw_table"] if c["task"].startswith("validate") else c["table"]
    med = rng.randint(90_000, 160_000)
    n = int(med * rng.uniform(0.28, 0.42))
    return {"chk_table": chk, "n_s": f"{n:,}", "med_s": f"{med:,}",
            "pct": f"{(n - med) / med * 100:.1f}",
            "src_n": f"{int(med * rng.uniform(0.95, 1.05)):,}"}


SCENARIOS: list[Scenario] = [
    Scenario(
        id="column-renamed-upstream",
        title="Column renamed upstream breaks a transform",
        combos=[("orders_etl", "transform_orders"), ("customer_360", "join_loyalty")],
        not_before=RENAME_DATE,
        vars_fn=_v_column,
        exc="KeyError: '{old_col}'",
        frames=[
            (DAG_FILE, "{task}", "df = apply_transforms(df)"),
            ("/opt/airflow/dags/{pipeline}/transforms.py", "apply_transforms",
             "df['{old_col}'] = df['{old_col}'].fillna('none')"),
            (SITE + "/pandas/core/frame.py", "__getitem__",
             "indexer = self.columns.get_loc(key)"),
            (SITE + "/pandas/core/indexes/base.py", "get_loc", "raise KeyError(key) from err"),
        ],
        pre=[("INFO", "Loaded {rows} rows from orders_raw (columns: order_id, customer_id, "
                      "{new_col}, order_ts, quantity, order_total, updated_at)"),
             ("INFO", "Applying transform step 3/5: normalize_loyalty")],
        root_cause="The storefront team renamed the column customer_tier to loyalty_tier in "
                   "orders_raw on 2026-08-28, but the transform code still reads customer_tier.",
        fix_summary="Update the transform to read loyalty_tier (or add an alias from the old "
                    "name), deploy it, then clear the failed task so it re-runs.",
        symptoms=["The task fails with KeyError: '<column name>' inside a transform step.",
                  "The column list printed in the log does not contain the column the code "
                  "expects.",
                  "Failures start right after an upstream schema change or release."],
        causes=["An upstream team renamed or dropped a column.",
                "The transform code was not updated for the new schema."],
        diagnosis=["Read the column name in the KeyError.",
                   "Find the log line 'Loaded N rows ... (columns: ...)' and compare those "
                   "columns with what the code expects.",
                   "Check the table's schema doc change history for recent renames.",
                   "Ask the owning team whether a migration shipped."],
        resolution=["Update the transform to use the new column name, or add an alias mapping "
                    "old to new.",
                    "Deploy the fix through the normal CI pipeline.",
                    "Clear the failed task in Airflow so it re-runs for the failed date and "
                    "confirm downstream tasks succeed."],
        prevention=["Add a schema contract check at the start of the pipeline that fails fast "
                    "and lists the missing columns.",
                    "Subscribe pipeline owners to upstream schema-change announcements."],
        runbook_question="A transform task fails with a KeyError on a column name that used to "
                         "exist. What should I do?",
    ),
    Scenario(
        id="source-file-missing",
        title="Expected source file is missing",
        combos=[("orders_etl", "extract_orders"), ("inventory_snapshot", "extract_inventory"),
                ("marketing_attribution", "extract_ad_spend")],
        vars_fn=_v_source_file,
        exc="FileNotFoundError: {path}",
        frames=[
            (DAG_FILE, "{task}", "df = read_partition(ds)"),
            ("/opt/airflow/dags/common/io.py", "read_partition",
             "return pd.read_parquet(path, filesystem=fs)"),
            (SITE + "/s3fs/core.py", "_info", "raise FileNotFoundError(path)"),
        ],
        pre=[("INFO", "Listing s3://shopstream-raw/{src}/dt={date}/"),
             ("WARNING", "Listing returned 0 objects for prefix {src}/dt={date}/ "
                         "(no _SUCCESS marker found)")],
        root_cause="The upstream export job had not written the day's partition (it finished "
                   "late or failed), so the expected parquet file did not exist when the task "
                   "ran.",
        fix_summary="Check the upstream export job, wait for or re-run it so the partition and "
                    "its _SUCCESS marker exist, then clear the failed task.",
        symptoms=["FileNotFoundError for an s3 path ending in part-0000.parquet.",
                  "The log shows the prefix listing returned 0 objects."],
        causes=["The upstream export job ran late or failed.",
                "The export wrote to a different date partition."],
        diagnosis=["Open the S3 prefix for the failed date and check whether any files or a "
                   "_SUCCESS marker exist.",
                   "Check the upstream export job's last run status.",
                   "Look for a partition written under the wrong date."],
        resolution=["If the export is late, wait for it to finish and the _SUCCESS marker to "
                    "appear.",
                    "If the export failed, ask its owner to re-run it for the failed date.",
                    "Clear the failed task so it retries against the new files."],
        prevention=["Put a sensor on the _SUCCESS marker before extracting.",
                    "Alert the export owners when their job misses its expected finish time."],
        runbook_question="A task fails with FileNotFoundError for a parquet partition in S3. "
                         "What are the steps to fix it?",
    ),
    Scenario(
        id="db-connection-timeout",
        title="Database connection timeout",
        combos=[("orders_etl", "extract_orders"), ("customer_360", "build_customer_profile"),
                ("inventory_snapshot", "extract_inventory")],
        vars_fn=_v_db_timeout,
        exc='psycopg2.OperationalError: connection to server at "{host}" ({ip}), port 5432 '
            'failed: timeout expired',
        frames=[
            (DAG_FILE, "{task}", "conn = get_connection('replica')"),
            ("/opt/airflow/dags/common/db.py", "get_connection",
             "return psycopg2.connect(dsn, connect_timeout=10)"),
            (SITE + "/psycopg2/__init__.py", "connect",
             "conn = _connect(dsn, connection_factory=connection_factory, **kwasync)"),
        ],
        pre=[("WARNING", "Connection attempt 1/3 to {host}:5432 timed out after 10s"),
             ("WARNING", "Connection attempt 2/3 to {host}:5432 timed out after 10s")],
        root_cause="The read replica was unreachable or out of connection slots (a failover was "
                   "in progress or the connection pool was exhausted), so every connection "
                   "attempt timed out.",
        fix_summary="Check replica health and connection counts, wait for the failover to "
                    "finish or free up connections, then clear and re-run the task.",
        symptoms=["psycopg2.OperationalError: connection ... timeout expired.",
                  "Several 'Connection attempt n/3 timed out' warnings before the failure."],
        causes=["The replica failed over or restarted.",
                "All connection slots (max_connections) were in use."],
        diagnosis=["Note the host in the error message.",
                   "Run pg_isready against that host from a worker.",
                   "On the replica, run SELECT state, count(*) FROM pg_stat_activity GROUP BY "
                   "state and compare with max_connections.",
                   "Check the database team's channel for a failover or maintenance event."],
        resolution=["If a failover is in progress, wait until it completes.",
                    "If connections are exhausted, terminate idle sessions or restart the "
                    "connection pooler.",
                    "Clear the failed task so it retries."],
        prevention=["Route pipeline connections through the pooler instead of connecting "
                    "directly.",
                    "Retry connection errors with exponential backoff."],
        runbook_question="A task fails with psycopg2 OperationalError connection timeout "
                         "to a replica. How do I resolve it?",
    ),
    Scenario(
        id="oom-killed",
        title="Task killed for running out of memory",
        combos=[("orders_etl", "transform_orders"), ("clickstream_ingest", "sessionize_events"),
                ("marketing_attribution", "attribute_conversions")],
        vars_fn=_v_oom,
        no_traceback=True,
        exit_line="Task exited with return code Negsignal.SIGKILL (-9)",
        exc="Pod {pod} terminated: reason=OOMKilled exitCode=137 memoryLimit=4Gi",
        frames=[],
        pre=[("INFO", "Loaded {rows} rows into memory"),
             ("WARNING", "Memory usage {mem1} GiB of 4.0 GiB limit"),
             ("WARNING", "Memory usage {mem2} GiB of 4.0 GiB limit")],
        root_cause="The day's data volume grew beyond the 4 GiB pod memory limit because the "
                   "task loads the whole partition into memory at once.",
        fix_summary="Raise the pod memory limit as a short-term fix and process the partition "
                    "in smaller chunks (for example hourly) as the lasting fix.",
        symptoms=["The task ends with return code -9 (Negsignal.SIGKILL) and no Python "
                  "traceback.",
                  "The log shows reason=OOMKilled and exit code 137.",
                  "Memory usage warnings climb toward the limit just before the task stops."],
        causes=["Data volume grew (a sale, a campaign, or a backfill).",
                "The task loads the entire partition into memory."],
        diagnosis=["Look for 'Memory usage ... of 4.0 GiB limit' warnings in the log.",
                   "Compare today's row count with the previous days.",
                   "Check the pod events for OOMKilled."],
        resolution=["Raise the memory request and limit for the task and clear it to re-run.",
                    "Change the task to process the partition in chunks (for example per hour) "
                    "instead of all at once."],
        prevention=["Alert when memory usage passes 80 percent of the limit.",
                    "Load tests with a 2x volume before major sales events."],
        runbook_question="A task disappears with return code -9 and OOMKilled exit code 137. "
                         "What should I do?",
    ),
    Scenario(
        id="expired-credentials",
        title="Expired cloud credentials mid-task",
        combos=[("payments_sync", "normalize_payments"),
                ("inventory_snapshot", "compute_stock_levels")],
        vars_fn=_v_creds,
        exc="botocore.exceptions.ClientError: An error occurred (ExpiredToken) when calling "
            "the {op} operation: The provided token has expired.",
        frames=[
            (DAG_FILE, "{task}", "write_partition(df, ds)"),
            ("/opt/airflow/dags/common/io.py", "write_partition",
             "s3.put_object(Bucket=bucket, Key=key, Body=body)"),
            (SITE + "/botocore/client.py", "_make_api_call",
             "raise error_class(parsed_response, operation_name)"),
        ],
        pre=[("INFO", "Assumed role arn:aws:iam::123456789012:role/shopstream-etl-worker"),
             ("INFO", "Task has been running for {mins} minutes")],
        root_cause="The task ran longer than the one-hour limit of its assumed IAM role "
                   "session, so the temporary credentials expired mid-task.",
        fix_summary="Re-run the task so it gets fresh credentials, and make the task refresh "
                    "its session (or split it into shorter tasks).",
        symptoms=["botocore ClientError with ExpiredToken.",
                  "The task had been running for more than an hour."],
        causes=["Temporary role credentials last one hour by default.",
                "A long task created its client once at the start and never refreshed it."],
        diagnosis=["Check how long the task had been running when it failed.",
                   "Confirm the client was created once at the start of the task."],
        resolution=["Clear the task to re-run with new credentials.",
                    "Use a refreshable session so credentials renew automatically, or split "
                    "the work into shorter tasks."],
        prevention=["Use auto-refreshing credentials in the shared io helpers.",
                    "Alert when tasks approach 50 minutes."],
        runbook_question="A task fails with an ExpiredToken ClientError after running a long "
                         "time. What is the fix?",
    ),
    Scenario(
        id="duplicate-key-violation",
        title="Duplicate key violation on load",
        combos=[("orders_etl", "load_orders_fact"), ("payments_sync", "load_payments_fact"),
                ("clickstream_ingest", "load_sessions_fact")],
        vars_fn=_v_dupkey,
        exc='psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint '
            '"{table}_pkey"\nDETAIL:  Key ({pk})=({keyval}) already exists.',
        frames=[
            (DAG_FILE, "{task}", "bulk_insert(conn, '{table}', df)"),
            ("/opt/airflow/dags/common/db.py", "bulk_insert",
             "cur.executemany(insert_sql, rows)"),
        ],
        pre=[("INFO", "Starting bulk INSERT into {table} (mode=insert) for ds={date}"),
             ("INFO", "Prepared {rows} rows")],
        root_cause="The load task uses a plain INSERT, so re-running it for a date that was "
                   "partly loaded tried to insert rows that already exist.",
        fix_summary="Delete the partial rows for that date (or switch the load to INSERT ... "
                    "ON CONFLICT DO UPDATE) and re-run the task.",
        symptoms=["UniqueViolation on a <table>_pkey constraint.",
                  "The failure follows a retry or a manual re-run."],
        causes=["The earlier attempt wrote some rows before failing.",
                "The load is not idempotent."],
        diagnosis=["Check whether rows for the failed date already exist in the target table.",
                   "Look at earlier attempts of the same task for partial writes."],
        resolution=["Delete the rows for that date from the target table, then clear the task.",
                    "Or change the load to an upsert (INSERT ... ON CONFLICT DO UPDATE) and "
                    "re-run."],
        prevention=["Make every load idempotent with upserts keyed on the primary key.",
                    "Load inside a transaction so partial writes roll back."],
        runbook_question="A load task fails with a duplicate key violation after a retry. "
                         "How do I fix it?",
    ),
    Scenario(
        id="api-rate-limited",
        title="Upstream API rate limit (HTTP 429)",
        combos=[("payments_sync", "pull_payments_api"),
                ("marketing_attribution", "extract_ad_spend")],
        vars_fn=_v_rate_limit,
        exc="requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: {url}",
        frames=[
            (DAG_FILE, "{task}", "records = fetch_all_pages()"),
            ("/opt/airflow/dags/common/http.py", "get_json", "resp.raise_for_status()"),
            (SITE + "/requests/models.py", "raise_for_status",
             "raise HTTPError(http_error_msg, response=self)"),
        ],
        pre=[("INFO", "Fetched {page} pages so far"),
             ("WARNING", "Received 429 from {api_host}; Retry-After header: 30")],
        root_cause="The provider API allows a limited number of requests per minute, and the "
                   "task (often together with a backfill or another DAG) exceeded it.",
        fix_summary="Honor the Retry-After header with exponential backoff, limit concurrency "
                    "with an Airflow pool, and re-run once the rate window has reset.",
        symptoms=["requests HTTPError 429 Too Many Requests.",
                  "A Retry-After header is logged just before the failure."],
        causes=["Too many requests per minute from this task.",
                "Another DAG or a backfill is calling the same API at the same time."],
        diagnosis=["Check whether other DAGs or backfills were using the same API.",
                   "Read the Retry-After value in the log."],
        resolution=["Wait for the rate window to reset, then clear the task.",
                    "Add backoff that honors Retry-After.",
                    "Put API-calling tasks in an Airflow pool with limited slots."],
        prevention=["Use a shared pool for each external API.",
                    "Do not run backfills during the daily window."],
        runbook_question="A task fails with HTTP 429 Too Many Requests from a provider API. "
                         "What should I do?",
    ),
    Scenario(
        id="type-mismatch-na-values",
        title="Numeric column contains text values",
        combos=[("orders_etl", "validate_orders"), ("payments_sync", "normalize_payments"),
                ("inventory_snapshot", "compute_stock_levels")],
        vars_fn=_v_type,
        exc="pyarrow.lib.ArrowInvalid: Could not convert 'N/A' with type str: tried to "
            "convert to {dtype} (column: {col})",
        frames=[
            (DAG_FILE, "{task}", "table = pa.Table.from_pandas(df, schema=SCHEMA)"),
            (SITE + "/pyarrow/table.pxi", "from_pandas",
             "arrays = _dataframe_to_arrays(df, schema, preserve_index)"),
        ],
        pre=[("INFO", "Loaded {rows} rows from {raw_table}"),
             ("WARNING", "Column {col} has dtype object (expected {dtype}); {bad} "
                         "non-numeric values found")],
        root_cause="The upstream system started sending the text value N/A for missing "
                   "numbers, so the column arrived as strings and could not be converted to a "
                   "numeric type.",
        fix_summary="Coerce the column with pd.to_numeric(errors='coerce'), quarantine the bad "
                    "rows for review, and add schema validation before conversion.",
        symptoms=["pyarrow ArrowInvalid: Could not convert 'N/A' ... to int64 or double.",
                  "A warning says the column has dtype object instead of a numeric type."],
        causes=["The source began emitting N/A or empty strings for missing numbers."],
        diagnosis=["Read the column name in the error.",
                   "Count the non-numeric values in that column.",
                   "Ask the source team whether the export format changed."],
        resolution=["Convert with pd.to_numeric(errors='coerce') so bad values become nulls.",
                    "Write the bad rows to a quarantine table for review.",
                    "Clear the failed task."],
        prevention=["Validate types and allowed values before conversion.",
                    "Agree on a data contract with the source team."],
        runbook_question="A task fails with pyarrow ArrowInvalid because of 'N/A' strings in a "
                         "numeric column. How do I fix it?",
    ),
    Scenario(
        id="disk-full",
        title="Worker disk full",
        combos=[("clickstream_ingest", "sessionize_events"), ("orders_etl", "transform_orders"),
                ("marketing_attribution", "attribute_conversions")],
        exc="OSError: [Errno 28] No space left on device: "
            "'/tmp/airflow/{pipeline}/{task}/chunk_{k}.parquet'",
        frames=[
            (DAG_FILE, "{task}", "write_chunks(df)"),
            ("/opt/airflow/dags/common/io.py", "spill_to_disk", "df.to_parquet(path)"),
            (SITE + "/pyarrow/parquet/core.py", "write_table", "writer.write_table(table)"),
        ],
        pre=[("INFO", "Writing intermediate file chunk_{k}.parquet"),
             ("WARNING", "Filesystem /tmp usage at 98% on worker {worker}")],
        root_cause="The worker's /tmp volume was full because intermediate files from earlier "
                   "failed runs were never cleaned up.",
        fix_summary="Delete stale files in /tmp on the worker, re-run the task, and add "
                    "cleanup in a finally block plus a periodic cleanup job.",
        symptoms=["OSError: [Errno 28] No space left on device under /tmp.",
                  "A warning shows /tmp usage near 100 percent."],
        causes=["Leftover intermediate files from earlier failed runs.",
                "Larger than usual data volume."],
        diagnosis=["Run df -h /tmp and du -sh /tmp/airflow/* on the worker named in the log.",
                   "Check the modification times of leftover files."],
        resolution=["Delete stale files in /tmp/airflow on that worker.",
                    "Clear the failed task to re-run.",
                    "Add cleanup in a finally block."],
        prevention=["Schedule a cleanup job for old temporary files.",
                    "Alert when worker disk usage passes 80 percent."],
        runbook_question="A task fails with OSError Errno 28 No space left on device. What "
                         "are the steps?",
    ),
    Scenario(
        id="deadlock-detected",
        title="Database deadlock during load",
        combos=[("orders_etl", "load_orders_fact"),
                ("inventory_snapshot", "load_inventory_snapshot"),
                ("customer_360", "load_customer_360")],
        vars_fn=_v_deadlock,
        exc="psycopg2.errors.DeadlockDetected: deadlock detected\n"
            "DETAIL:  Process {p1} waits for ShareLock on transaction {x1}; blocked by "
            "process {p2}.\n"
            "Process {p2} waits for ShareLock on transaction {x2}; blocked by process {p1}.\n"
            'CONTEXT:  while updating tuple (12,4) in relation "{table}"',
        frames=[
            (DAG_FILE, "{task}", "upsert(conn, '{table}', df)"),
            ("/opt/airflow/dags/common/db.py", "upsert", "cur.executemany(upsert_sql, rows)"),
        ],
        pre=[("INFO", "Upserting {rows} rows into {table} in a single transaction"),
             ("INFO", "Transaction open for {secs}s")],
        root_cause="Two jobs updated the same table in overlapping transactions touching rows in "
                   "different orders (the pipeline load and the nightly reconciliation job), so "
                   "Postgres aborted one as the deadlock victim.",
        fix_summary="Re-run the failed task (deadlock victims are safe to retry), then sort rows "
                    "by primary key in both jobs and stop their schedules from overlapping.",
        symptoms=["psycopg2 DeadlockDetected with two processes blocking each other.",
                  "The failure happens at about the same time as another job on the same table."],
        causes=["Overlapping writers on the same table.",
                "Rows updated in different orders by different jobs."],
        diagnosis=["Read the two process ids and the relation name in the error.",
                   "Check which other jobs write to that table at that time.",
                   "Check the Postgres log for the queries involved."],
        resolution=["Clear the failed task; a deadlock victim is safe to retry.",
                    "Sort rows by primary key before upserting in every writer.",
                    "Move the other job so schedules do not overlap."],
        prevention=["Always update rows in primary key order.",
                    "Keep transactions short and avoid overlapping schedules on one table."],
        runbook_question="A load task fails with DeadlockDetected. How do I resolve it?",
    ),
    Scenario(
        id="sensor-timeout",
        title="Sensor timed out waiting for upstream",
        combos=[("orders_etl", "wait_for_orders_export"),
                ("inventory_snapshot", "wait_for_inventory_export"),
                ("marketing_attribution", "wait_for_ad_export")],
        vars_fn=_v_sensor,
        exc="airflow.exceptions.AirflowSensorTimeout: Sensor has timed out; run duration of "
            "{dur} seconds exceeds the specified timeout of 21600.0.",
        frames=[
            (SITE + "/airflow/sensors/base.py", "execute",
             "raise AirflowSensorTimeout(message)"),
        ],
        pre=[("INFO", "Poking for file s3://shopstream-raw/{src}/dt={date}/_SUCCESS"),
             ("INFO", "Success criteria not met. Sleeping for 300 seconds"),
             ("INFO", "Poking for file s3://shopstream-raw/{src}/dt={date}/_SUCCESS"),
             ("INFO", "Success criteria not met. Sleeping for 300 seconds")],
        root_cause="The upstream export job never wrote the _SUCCESS marker within the "
                   "six-hour sensor timeout, usually because the export failed or ran very late.",
        fix_summary="Find out why the upstream export did not finish, fix or re-run it, then "
                    "clear the sensor task so the pipeline continues.",
        symptoms=["AirflowSensorTimeout after exactly the configured timeout.",
                  "The log repeats 'Success criteria not met' many times."],
        causes=["The upstream export failed or is very late.",
                "The marker file is written to a different path or date."],
        diagnosis=["Check the upstream export job's status.",
                   "List the S3 prefix to see whether any data arrived without a marker."],
        resolution=["Ask the export owner to fix or re-run the export.",
                    "Once the marker exists, clear the sensor task."],
        prevention=["Alert the export owners at the expected finish time, before the sensor "
                    "times out.",
                    "Document export SLAs for every upstream source."],
        runbook_question="A sensor task fails with AirflowSensorTimeout. What should I check?",
    ),
    Scenario(
        id="division-by-zero",
        title="Division by zero in a metric calculation",
        combos=[("marketing_attribution", "attribute_conversions"),
                ("inventory_snapshot", "compute_stock_levels")],
        vars_fn=_v_zero,
        exc="ZeroDivisionError: division by zero",
        frames=[
            (DAG_FILE, "{task}", "metrics = compute_metrics(df)"),
            ("/opt/airflow/dags/{pipeline}/metrics.py", "compute_metrics", "{code}"),
        ],
        pre=[("INFO", "Computing {metric} for {n} entities"),
             ("WARNING", "Found {z} entities with zero {denom} on {date}")],
        root_cause="For some entities the denominator was zero that day (for example a paused "
                   "campaign with no clicks, or a SKU with no stock), and the code divided "
                   "without a guard.",
        fix_summary="Guard the division (return null or zero when the denominator is zero, or "
                    "use NULLIF in SQL), then clear and re-run the task.",
        symptoms=["ZeroDivisionError in a metrics function.",
                  "A warning reports entities with a zero denominator."],
        causes=["Paused campaigns or empty stock produce zero denominators."],
        diagnosis=["Read the code line shown in the traceback.",
                   "Count entities with a zero denominator in the input."],
        resolution=["Guard the division with a zero check or NULLIF.",
                    "Deploy and clear the failed task."],
        prevention=["Unit-test metric functions with zero denominators."],
        runbook_question="A metric task fails with ZeroDivisionError. How should I fix it?",
    ),
    Scenario(
        id="kafka-offset-out-of-range",
        title="Kafka consumer offset out of range",
        combos=[("clickstream_ingest", "consume_clickstream")],
        vars_fn=_v_kafka,
        exc="kafka.errors.OffsetOutOfRangeError: OffsetOutOfRangeError: topic={topic} "
            "partition={part} requested_offset={off}",
        frames=[
            (DAG_FILE, "{task}", "batch = poll_batch()"),
            ("/opt/airflow/dags/clickstream_ingest/consumer.py", "poll_batch",
             "records = consumer.poll(timeout_ms=5000)"),
            (SITE + "/kafka/consumer/fetcher.py", "_reset_offset",
             "raise Errors.OffsetOutOfRangeError(partitions)"),
        ],
        pre=[("WARNING", "Consumer group shopstream-clickstream was paused for {hrs} hours"),
             ("INFO", "Seeking to committed offset {off} on partition {part}")],
        root_cause="The consumer's committed offset pointed to data the broker had already "
                   "deleted, because the consumer group was stopped for longer than the broker "
                   "keeps data.",
        fix_summary="Reset the consumer group offsets to the earliest available offset, backfill "
                    "the missing window from the S3 archive, and alert on consumer lag.",
        symptoms=["kafka OffsetOutOfRangeError for a topic partition.",
                  "The consumer group was paused or down before the failure."],
        causes=["The consumer group was stopped long enough that its committed offset was "
                "deleted from the broker."],
        diagnosis=["Check how long the consumer group was paused.",
                   "Compare the committed offset with the earliest offset on the partition."],
        resolution=["Reset the consumer group offsets to the earliest available.",
                    "Backfill the missing time window from the S3 archive.",
                    "Clear the failed task."],
        prevention=["Alert on consumer lag.",
                    "Do not pause the consumer group for long maintenance windows."],
        runbook_question="The clickstream consumer fails with Kafka OffsetOutOfRangeError. What "
                         "are the steps to recover?",
    ),
    Scenario(
        id="freshness-check-failed",
        title="Source data freshness check failed",
        combos=[("orders_etl", "validate_orders"), ("customer_360", "build_customer_profile")],
        vars_fn=_v_fresh,
        exc="common.quality.DataQualityError: freshness check failed for table {raw_table}: "
            "max(updated_at) is {age} old, SLA is {sla} hours",
        frames=[
            (DAG_FILE, "{task}", "run_checks(conn)"),
            ("/opt/airflow/dags/common/quality.py", "check_freshness",
             "raise DataQualityError(msg)"),
        ],
        pre=[("INFO", "Running freshness check on {raw_table}"),
             ("INFO", "Latest updated_at in {raw_table}: {latest}")],
        root_cause="The CDC replication connector that copies source changes into the raw table "
                   "stalled, so no new rows arrived and the data became older than the "
                   "freshness SLA.",
        fix_summary="Check the CDC connector status, restart it, wait for replication lag to "
                    "clear, then clear and re-run the check.",
        symptoms=["DataQualityError: freshness check failed ... max(updated_at) is Nh old.",
                  "Latest updated_at in the raw table is from the previous day."],
        causes=["The CDC connector stalled or crashed.",
                "The source system stopped writing."],
        diagnosis=["Check the CDC connector status and lag.",
                   "Query max(updated_at) on the raw table.",
                   "Ask the source team whether they had an outage."],
        resolution=["Restart the CDC connector.",
                    "Wait for lag to clear and fresh rows to arrive.",
                    "Clear the failed check task."],
        prevention=["Alert on CDC connector lag.",
                    "Run the freshness check earlier than the main DAG."],
        runbook_question="A validation task fails with a freshness check error because the raw "
                         "table is too old. What should I do?",
    ),
    Scenario(
        id="not-null-violation",
        title="Null value in a NOT NULL column",
        combos=[("orders_etl", "load_orders_fact"), ("payments_sync", "load_payments_fact"),
                ("customer_360", "load_customer_360")],
        vars_fn=_v_notnull,
        exc='psycopg2.errors.NotNullViolation: null value in column "{col}" of relation '
            '"{table}" violates not-null constraint\n'
            "DETAIL:  Failing row contains ({keyval}, null, ...).",
        frames=[
            (DAG_FILE, "{task}", "bulk_insert(conn, '{table}', df)"),
            ("/opt/airflow/dags/common/db.py", "bulk_insert", "cur.executemany(insert_sql, rows)"),
        ],
        pre=[("INFO", "Loading {rows} rows into {table}"),
             ("WARNING", "{nulls} rows have null {col}")],
        root_cause="Some source rows arrived with a null in a column the target table requires "
                   "(for example guest checkouts without a customer_id), and the transform "
                   "passed the nulls through.",
        fix_summary="Default or filter the null values in the transform (for example map guest "
                    "orders to customer_id 0), quarantine bad rows for review, and re-run the "
                    "load.",
        symptoms=["psycopg2 NotNullViolation naming a column and table.",
                  "A warning counts rows with null in that column."],
        causes=["A new kind of record (such as guest checkout) has no value for the column.",
                "The transform does not handle nulls."],
        diagnosis=["Read the column and relation in the error.",
                   "Count null values for that column in the input.",
                   "Check the table schema doc for how nulls should be handled."],
        resolution=["Update the transform to default or filter nulls as the schema doc "
                    "specifies.",
                    "Write rows that cannot be fixed to a quarantine table.",
                    "Clear the failed load."],
        prevention=["Validate required columns before loading.",
                    "Review new product features with data owners."],
        runbook_question="A load task fails with NotNullViolation. How do I fix it?",
    ),
    Scenario(
        id="dependency-version-conflict",
        title="Dependency version conflict after image rebuild",
        combos=[("orders_etl", "transform_orders"), ("payments_sync", "normalize_payments"),
                ("marketing_attribution", "attribute_conversions")],
        vars_fn=_v_dep,
        exc="ImportError: cannot import name 'soft_unicode' from 'markupsafe' "
            "(" + SITE + "/markupsafe/__init__.py)",
        frames=[
            (DAG_FILE, "<module>", "from jinja2 import Template"),
            (SITE + "/jinja2/__init__.py", "<module>", "from .environment import Environment"),
            (SITE + "/jinja2/utils.py", "<module>", "from markupsafe import soft_unicode"),
        ],
        pre=[("INFO", "Worker image: shopstream/airflow-worker:{img}"),
             ("INFO", "Importing task module {pipeline}.tasks")],
        root_cause="The worker image was rebuilt with an unpinned dependency, so a newer "
                   "markupsafe was installed that is incompatible with the installed jinja2.",
        fix_summary="Roll back to the previous worker image, pin the dependency versions with a "
                    "constraints file, rebuild, and re-run the failed tasks.",
        symptoms=["ImportError or ModuleNotFoundError at import time.",
                  "Many unrelated DAGs start failing right after a new worker image rollout."],
        causes=["An unpinned dependency upgraded during an image rebuild.",
                "Two libraries now require incompatible versions of a shared package."],
        diagnosis=["Check the image tag in the log and when it was deployed.",
                   "Compare pip freeze between the old and new image."],
        resolution=["Roll back to the previous worker image.",
                    "Pin versions in a constraints file and rebuild.",
                    "Clear the failed tasks."],
        prevention=["Use a lockfile or constraints file for every image build.",
                    "Run a smoke test that imports all DAGs in CI before deploying an image."],
        runbook_question="Tasks fail with an ImportError right after a new worker image was "
                         "deployed. What should I do?",
    ),
    Scenario(
        id="row-count-anomaly",
        title="Row count dropped sharply (partial data)",
        combos=[("orders_etl", "validate_orders"), ("payments_sync", "load_payments_fact"),
                ("clickstream_ingest", "load_sessions_fact")],
        vars_fn=_v_rowcount,
        exc="common.quality.DataQualityError: row count check failed for {chk_table} "
            "ds={date}: {n_s} rows, 7-day median is {med_s} ({pct}%), allowed drop is 40%",
        frames=[
            (DAG_FILE, "{task}", "run_checks(conn)"),
            ("/opt/airflow/dags/common/quality.py", "check_row_count",
             "raise DataQualityError(msg)"),
        ],
        pre=[("INFO", "Running row count check on {chk_table}"),
             ("INFO", "Source system reported {src_n} rows for ds={date}")],
        root_cause="The extract was only partial (some export shards were missing or the "
                   "extract query filtered too much), so far fewer rows than normal reached the "
                   "check.",
        fix_summary="Do not load the partial data; compare row counts with the source, re-run "
                    "the extract for that date, and only then re-run the check and load.",
        symptoms=["DataQualityError: row count check failed ... allowed drop is 40%.",
                  "Row count is a fraction of the 7-day median."],
        causes=["Partial extract from the upstream export.",
                "An extract filter or join accidentally dropped rows."],
        diagnosis=["Compare the source row count with the loaded row count.",
                   "Look for missing shards in the raw files.",
                   "Review recent changes to the extract query."],
        resolution=["Do not load the partial data.",
                    "Re-run the extract for the date and confirm row counts match the source.",
                    "Then re-run the check and the load."],
        prevention=["Compare row counts against the source before loading.",
                    "Keep the 40 percent drop check as a hard gate."],
        runbook_question="A check task fails because the row count dropped far below the "
                         "7-day median. What should I do?",
    ),
    Scenario(
        id="zombie-task-sigterm",
        title="Task killed when the worker node was reclaimed",
        combos=[("orders_etl", "transform_orders"), ("clickstream_ingest", "sessionize_events"),
                ("inventory_snapshot", "load_inventory_snapshot")],
        exit_line="Task exited with return code Negsignal.SIGTERM (-15)",
        exc="airflow.exceptions.AirflowException: Task received SIGTERM signal",
        frames=[
            (SITE + "/airflow/models/taskinstance.py", "signal_handler",
             "raise AirflowException('Task received SIGTERM signal')"),
        ],
        pre=[("WARNING", "Node of worker {worker} received a drain notice: spot instance will "
                         "be reclaimed in 120 seconds"),
             ("ERROR", "Received SIGTERM. Terminating subprocesses.")],
        root_cause="The worker node (a spot instance) was reclaimed in the middle of the task, "
                   "so the running process was terminated and the scheduler later marked the "
                   "task as failed.",
        fix_summary="Clear the task to retry on a healthy worker, and move critical long-running "
                    "tasks to on-demand nodes with retries enabled.",
        symptoms=["AirflowException: Task received SIGTERM signal.",
                  "The log shows a node drain or spot reclaim notice before the failure."],
        causes=["Spot instance reclaimed by the cloud provider.",
                "Node scaled down or drained for maintenance."],
        diagnosis=["Look for a drain or reclaim notice in the log just before SIGTERM.",
                   "Check cluster events for node terminations at that time."],
        resolution=["Clear the task so it retries on a healthy worker.",
                    "Move critical long-running tasks to on-demand nodes."],
        prevention=["Enable retries on long tasks.",
                    "Run critical pipelines on on-demand capacity."],
        runbook_question="A task fails with 'Task received SIGTERM signal' after a node drain "
                         "notice. What should I do?",
    ),
]

INCIDENT_SCENARIOS = [
    "column-renamed-upstream", "oom-killed", "duplicate-key-violation", "api-rate-limited",
    "source-file-missing", "deadlock-detected", "expired-credentials", "sensor-timeout",
    "kafka-offset-out-of-range", "freshness-check-failed", "not-null-violation",
    "row-count-anomaly", "dependency-version-conflict", "disk-full",
]

UNANSWERABLE = [
    "What is the on-call phone number for the payments team?",
    "How much did our Snowflake warehouse cost last month?",
    "Why did the ml_features pipeline fail on 2026-09-09?",
    "What Kubernetes version do the Airflow workers run?",
    "What is the retention period for the clickstream Kafka topic in production?",
    "Who was the on-call engineer during incident INC-9999?",
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

NOISE = [
    "Using connection ID 'aws_default' for task execution.",
    "Heartbeat ok; worker={worker}",
    "Checkpoint written to /tmp/airflow/{pipeline}/{task}/ckpt_{k}.json",
    "Metrics emitted: task_duration_so_far={secs2}s",
    "Reading partition ds={date}",
    "Pool 'default_pool' has 94 open slots",
]
MODS = ["taskinstance.py:1956", "logging_mixin.py:188", "standard_task_runner.py:60",
        "base.py:77", "connection.py:462"]


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}+0000"


def build_ctx(rng: random.Random, sc: Scenario | None, pipeline: str, task: str, d: date) -> dict:
    p = PIPELINES[pipeline]
    ctx = {
        "pipeline": pipeline, "task": task, "date": d.isoformat(),
        "run_id": f"scheduled__{d.isoformat()}T02:00:00+00:00",
        "table": p["fact"], "raw_table": p["raw"],
        "rows": f"{rng.randint(40_000, 260_000):,}",
        "worker": f"airflow-worker-{rng.randint(1, 6)}",
        "pod": f"{pipeline.replace('_', '-')}-{task.replace('_', '-')}-{rng.randint(10000, 99999)}",
        "k": rng.randint(1, 40), "secs2": rng.randint(5, 900),
    }
    if sc and sc.vars_fn:
        ctx.update(sc.vars_fn(rng, ctx))
    return ctx


class LogWriter:
    def __init__(self, rng: random.Random, start: datetime):
        self.rng, self.t, self.lines = rng, start, []

    def emit(self, level: str, text: str, mod: str | None = None) -> None:
        self.t += timedelta(milliseconds=self.rng.randint(50, 30000))
        mod = mod or self.rng.choice(MODS)
        self.lines.append(f"[{fmt_ts(self.t)}] {{{mod}}} {level} - {text}")


def _start_time(rng: random.Random, ctx: dict) -> datetime:
    idx = PIPELINES[ctx["pipeline"]]["tasks"].index(ctx["task"])
    base = datetime.combine(date.fromisoformat(ctx["date"]), time(2, 0), tzinfo=UTC)
    return base + timedelta(minutes=7 * idx + rng.randint(0, 5), seconds=rng.randint(0, 59))


def _preamble(w: LogWriter, ctx: dict, tries: int) -> None:
    w.lines.append(
        f"*** Reading local file: /opt/airflow/logs/dag_id={ctx['pipeline']}/"
        f"run_id={ctx['run_id']}/task_id={ctx['task']}/attempt={tries}.log")
    w.emit("INFO", f"Dependencies all met for dep_context=non-requeueable deps "
                   f"ti=<TaskInstance: {ctx['pipeline']}.{ctx['task']} {ctx['run_id']} [queued]>")
    w.emit("INFO", f"Starting attempt {tries} of {tries}")
    w.emit("INFO", f"Started process {w.rng.randint(1000, 9999)} to run task",
           "standard_task_runner.py:60")


def render_failure_log(rng, sc: Scenario, ctx: dict, with_secret: bool) -> str:
    tries = rng.choice([1, 3])
    w = LogWriter(rng, _start_time(rng, ctx))
    _preamble(w, ctx, tries)
    if with_secret:
        w.emit("INFO", rng.choice([
            "Connecting with dsn=postgresql://etl_user:FAKE-PASSWORD-123@"
            "pg-orders-replica.internal:5432/orders",
            "Calling upstream API with token=FAKE-TOKEN-9f3a1c and user=svc-etl",
        ]))
    for _ in range(rng.randint(2, 4)):
        w.emit("INFO", rng.choice(NOISE).format(**ctx))
    for level, text in sc.pre:
        w.emit(level, text.format(**ctx))
        if rng.random() < 0.5:
            w.emit("INFO", rng.choice(NOISE).format(**ctx))
    exc = sc.exc.format(**ctx)
    if sc.no_traceback:
        w.emit("ERROR", exc)
    else:
        w.emit("ERROR", "Task failed with exception", "taskinstance.py:2698")
        w.lines.append("Traceback (most recent call last):")
        for file_tpl, func_tpl, code_tpl in sc.frames:
            w.lines.append(f'  File "{file_tpl.format(**ctx)}", line {rng.randint(20, 240)}, '
                           f"in {func_tpl.format(**ctx)}")
            w.lines.append(f"    {code_tpl.format(**ctx)}")
        w.lines.append(exc)
    d = ctx["date"].replace("-", "")
    w.emit("INFO", f"Marking task as FAILED. dag_id={ctx['pipeline']}, task_id={ctx['task']}, "
                   f"run_id={ctx['run_id']}, execution_date={d}T020000, "
                   f"end_date={d}T{w.t:%H%M%S}", "taskinstance.py:1400")
    w.emit("INFO", sc.exit_line, "local_task_job_runner.py:228")
    w.emit("INFO", "0 downstream tasks scheduled from follow-on schedule check",
           "taskinstance.py:3300")
    return "\n".join(w.lines) + "\n"


def render_success_log(rng, ctx: dict) -> str:
    w = LogWriter(rng, _start_time(rng, ctx))
    _preamble(w, ctx, 1)
    for _ in range(rng.randint(4, 7)):
        w.emit("INFO", rng.choice(NOISE).format(**ctx))
    w.emit("INFO", f"Loaded {ctx['rows']} rows from {ctx['raw_table']}")
    w.emit("INFO", f"Task completed successfully in {rng.randint(20, 900)} seconds")
    d = ctx["date"].replace("-", "")
    w.emit("INFO", f"Marking task as SUCCESS. dag_id={ctx['pipeline']}, task_id={ctx['task']}, "
                   f"run_id={ctx['run_id']}, execution_date={d}T020000, "
                   f"end_date={d}T{w.t:%H%M%S}", "taskinstance.py:1400")
    w.emit("INFO", "Task exited with return code 0", "local_task_job_runner.py:228")
    return "\n".join(w.lines) + "\n"


def render_runbook(n: int, sc: Scenario) -> str:
    applies = ", ".join(f"{p} ({t})" for p, t in sc.combos)
    bullets = lambda items: "\n".join(f"- {i}" for i in items)  # noqa: E731
    numbered = lambda items: "\n".join(f"{i}. {s}" for i, s in enumerate(items, 1))  # noqa: E731
    return (f"# Runbook: {sc.title}\n\n"
            f"- **Runbook ID:** RB-{n:03d}\n"
            f"- **Applies to:** {applies}\n"
            f"- **Owner:** data-platform on-call (#data-platform-oncall)\n\n"
            f"## Symptoms\n\n{bullets(sc.symptoms)}\n\n"
            f"## Likely causes\n\n{bullets(sc.causes)}\n\n"
            f"## Diagnosis\n\n{numbered(sc.diagnosis)}\n\n"
            f"## Resolution\n\n{numbered(sc.resolution)}\n\n"
            f"## Prevention\n\n{bullets(sc.prevention)}\n")


def render_schema(t: dict) -> str:
    rows = "\n".join(f"| {n} | {ty} | {desc} |" for n, ty, desc in t["cols"])
    notes = "\n".join(f"- {x}" for x in t["notes"]) or "- None."
    out = (f"# Table: {t['name']}\n\n"
           f"- **Layer:** {t['layer']}\n"
           f"- **Owner:** {t['owner']}\n"
           f"- **Populated by:** {t['populated']}\n"
           f"- **Freshness SLA:** {t['sla']} hours\n"
           f"- **Primary key:** {t['pk']}\n\n"
           f"## Columns\n\n| Column | Type | Description |\n|---|---|---|\n{rows}\n\n"
           f"## Notes\n\n{notes}\n")
    if t["history"]:
        hist = "\n".join(f"- {d}: {txt}" for d, txt in t["history"])
        out += f"\n## Change history\n\n{hist}\n"
    return out


def render_incident(inc: dict, sc: Scenario) -> str:
    return (f"# {inc['id']}: {sc.title} in {inc['pipeline']}\n\n"
            f"- **Date:** {inc['date']}\n"
            f"- **Severity:** SEV-{inc['sev']}\n"
            f"- **Pipeline:** {inc['pipeline']}\n"
            f"- **Failed task:** {inc['task']}\n"
            f"- **Duration:** {inc['minutes']} minutes\n"
            f"- **Impact:** {inc['impact']}\n\n"
            f"## Summary\n\nOn {inc['date']}, the {inc['task']} task failed in the "
            f"{inc['pipeline']} pipeline. Failure type: {sc.title}. "
            f"{PIPELINES[inc['pipeline']]['impact'].capitalize()} showed stale data until the "
            f"fix.\n\n"
            f"## Timeline (UTC)\n\n"
            f"- {inc['t0']} Alert fired: {inc['task']} failed.\n"
            f"- {inc['t1']} On-call engineer acknowledged and opened the task log.\n"
            f"- {inc['t2']} Root cause identified.\n"
            f"- {inc['t3']} Fix applied and the failed task cleared.\n"
            f"- {inc['t4']} Pipeline completed and dashboards refreshed.\n\n"
            f"## Root cause\n\n{sc.root_cause}\n\n"
            f"## Resolution\n\n{sc.resolution[0]} {sc.resolution[-1]}\n\n"
            f"## Follow-ups\n\n- {sc.prevention[0]}\n- {sc.prevention[-1]}\n")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def pick_date(rng, used: set, pipeline: str, not_before: date | None = None) -> date:
    lo = max(START, not_before) if not_before else START
    for _ in range(500):
        d = lo + timedelta(days=rng.randint(0, (END - lo).days))
        if (pipeline, d) not in used:
            used.add((pipeline, d))
            return d
    raise RuntimeError("could not find an unused date")


def generate(seed: int = DEFAULT_SEED) -> tuple[dict[str, dict], list[dict]]:
    """Return ({relative_path: {text, meta}}, eval_questions)."""
    rng = random.Random(seed)
    files: dict[str, dict] = {}
    sc_by_id = {s.id: s for s in SCENARIOS}
    first_log: dict[str, tuple[dict, str]] = {}
    used: set = set()

    for n, sc in enumerate(SCENARIOS, 1):
        files[f"runbooks/{sc.id}.md"] = dict(
            text=render_runbook(n, sc),
            meta=dict(source_type="runbook", title=f"Runbook: {sc.title}", scenario=sc.id))

    secret_idx = set(rng.sample(range(len(SCENARIOS) * LOGS_PER_SCENARIO), LOGS_WITH_FAKE_SECRETS))
    i = 0
    for sc in SCENARIOS:
        for j in range(LOGS_PER_SCENARIO):
            pipeline, task = sc.combos[j % len(sc.combos)]
            d = pick_date(rng, used, pipeline, sc.not_before)
            ctx = build_ctx(rng, sc, pipeline, task, d)
            path = f"logs/{pipeline}/{d.isoformat()}__{task}.log"
            files[path] = dict(
                text=render_failure_log(rng, sc, ctx, i in secret_idx),
                meta=dict(source_type="log", title=f"{pipeline}.{task} {d} (failed)",
                          pipeline=pipeline, task=task, date=d.isoformat(), scenario=sc.id,
                          status="failed", has_fake_secret=i in secret_idx))
            first_log.setdefault(sc.id, (ctx, path))
            i += 1

    for pipeline in rng.sample(sorted(PIPELINES), SUCCESS_LOGS):
        task = rng.choice(PIPELINES[pipeline]["tasks"])
        d = pick_date(rng, used, pipeline)
        ctx = build_ctx(rng, None, pipeline, task, d)
        files[f"logs/{pipeline}/{d.isoformat()}__{task}.log"] = dict(
            text=render_success_log(rng, ctx),
            meta=dict(source_type="log", title=f"{pipeline}.{task} {d} (success)",
                      pipeline=pipeline, task=task, date=d.isoformat(), scenario=None,
                      status="success", has_fake_secret=False))

    for t in TABLES:
        pipeline = next((p for p, v in PIPELINES.items() if t["name"] in (v["fact"], v["raw"])),
                        None)
        files[f"schemas/{t['name']}.md"] = dict(
            text=render_schema(t),
            meta=dict(source_type="schema", title=f"Table: {t['name']}", pipeline=pipeline,
                      scenario=None))

    drafts: list[dict] = []
    for sid in INCIDENT_SCENARIOS:
        sc = sc_by_id[sid]
        pipeline, task = sc.combos[0]
        if sc.not_before:
            d = sc.not_before + timedelta(days=rng.randint(0, 20))
        else:
            d = START + timedelta(days=rng.randint(0, (END - START).days))
        minutes = rng.randint(25, 240)
        t0 = datetime.combine(d, time(2, rng.randint(20, 59)))
        stamp = lambda m, t0=t0: (t0 + timedelta(minutes=m)).strftime("%H:%M")  # noqa: E731
        drafts.append(dict(scenario=sid, date=d.isoformat(), pipeline=pipeline, task=task,
                           sev=rng.choice([2, 3]), minutes=minutes,
                           impact=PIPELINES[pipeline]["impact"],
                           t0=stamp(0), t1=stamp(10), t2=stamp(int(minutes * 0.45)),
                           t3=stamp(int(minutes * 0.8)), t4=stamp(minutes)))

    # Number incidents in chronological order, like a real tracker would.
    incidents: dict[str, dict] = {}
    for k, inc in enumerate(sorted(drafts, key=lambda x: (x["date"], x["scenario"]))):
        inc["id"] = f"INC-{2031 + k}"
        sc = sc_by_id[inc["scenario"]]
        incidents[inc["scenario"]] = inc
        files[f"incidents/{inc['id']}.md"] = dict(
            text=render_incident(inc, sc),
            meta=dict(source_type="incident", title=f"{inc['id']}: {sc.title}",
                      pipeline=inc["pipeline"], scenario=inc["scenario"],
                      incident_id=inc["id"]))

    return files, build_eval(sc_by_id, first_log, incidents)


def build_eval(sc_by_id, first_log, incidents) -> list[dict]:
    qs: list[dict] = []

    def add(category, question, truth, paths, abstain=False):
        qs.append(dict(external_id=f"Q{len(qs) + 1:03d}", question=question,
                       ground_truth=truth, category=category, should_abstain=abstain,
                       expected_paths=paths))

    for sc in SCENARIOS:
        ctx, log_path = first_log[sc.id]
        err = sc.exc.format(**ctx).splitlines()[0]
        add("log_diagnosis",
            f"Why did the {ctx['task']} task in the {ctx['pipeline']} pipeline fail on "
            f"{ctx['date']}?",
            f"It failed with: {err}. Root cause: {sc.root_cause} Fix: {sc.fix_summary}",
            [log_path, f"runbooks/{sc.id}.md"])
    for sc in SCENARIOS:
        add("runbook", sc.runbook_question, sc.fix_summary, [f"runbooks/{sc.id}.md"])

    inc = incidents["api-rate-limited"]
    add("incident",
        "Which past incident was caused by the payments provider API rate limiting our "
        "requests, and how long did it last?",
        f"{inc['id']} on {inc['date']}; it lasted {inc['minutes']} minutes. "
        f"{sc_by_id['api-rate-limited'].fix_summary}",
        [f"incidents/{inc['id']}.md"])
    inc = incidents["dependency-version-conflict"]
    add("incident",
        "Which past incident was caused by an unpinned dependency in the worker image, and how "
        "was it resolved?",
        f"{inc['id']} on {inc['date']}. {sc_by_id['dependency-version-conflict'].fix_summary}",
        [f"incidents/{inc['id']}.md"])

    add("schema", "Who owns the orders_fact table and what is its freshness SLA?",
        "orders_fact is owned by data-platform and has a freshness SLA of 6 hours.",
        ["schemas/orders_fact.md"])
    add("schema", "What is the primary key of the sessions_fact table?",
        "The primary key of sessions_fact is session_id.", ["schemas/sessions_fact.md"])
    add("schema", "Which pipeline populates the attribution_fact table?",
        "attribution_fact is populated by the marketing_attribution pipeline "
        "(task load_attribution_fact).", ["schemas/attribution_fact.md"])
    add("schema", "What was the customer_tier column in orders_raw renamed to, and when?",
        "customer_tier was renamed to loyalty_tier on 2026-08-28.", ["schemas/orders_raw.md"])
    add("schema", "What data type is the amount column in payments_fact?",
        "The amount column in payments_fact is numeric(12,2).", ["schemas/payments_fact.md"])
    add("schema", "What values can the status column in payments_fact take?",
        "status is one of: authorized, captured, refunded, failed.",
        ["schemas/payments_fact.md"])

    for q in UNANSWERABLE:
        add("unanswerable", q, "NOT_ANSWERABLE", [], abstain=True)
    return qs


def write_all(out: Path, eval_out: Path, seed: int) -> dict:
    files, questions = generate(seed)
    for sub in GENERATED_DIRS:
        shutil.rmtree(out / sub, ignore_errors=True)
    manifest_files = []
    for rel in sorted(files):
        f = files[rel]
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = f["text"].encode("utf-8")
        target.write_bytes(data)
        manifest_files.append(dict(path=rel, sha256=hashlib.sha256(data).hexdigest(),
                                   bytes=len(data), **f["meta"]))
    counts: dict[str, int] = {}
    for m in manifest_files:
        counts[m["source_type"]] = counts.get(m["source_type"], 0) + 1
    manifest = dict(seed=seed, company="ShopStream (fictional)", counts=counts,
                    files=manifest_files)
    # write_bytes, not write_text: on Windows write_text turns "\n" into "\r\n", which would
    # make output differ from the committed copy (and break the manifest hashes).
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.write_bytes(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions).encode("utf-8"))
    return dict(counts=counts, questions=len(questions))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--eval-out", type=Path, default=Path("eval/questions.seed.jsonl"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    result = write_all(args.out, args.eval_out, args.seed)
    print(f"Wrote {sum(result['counts'].values())} files to {args.out}/ {result['counts']}")
    print(f"Wrote {result['questions']} eval questions to {args.eval_out}")


if __name__ == "__main__":
    main()
