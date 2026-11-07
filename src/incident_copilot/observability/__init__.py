"""Day 3 onward: log latency, tokens, and cost for every query to the `queries` table.

Day 11 adds the dashboard on top of these tables; no new instrumentation needed then.
"""

from incident_copilot.observability.query_log import (
    ERROR_CATEGORIES,
    log_failed_query,
    log_query,
    query_history,
    safe_question,
)

__all__ = ["ERROR_CATEGORIES", "log_failed_query", "log_query", "query_history",
           "safe_question"]
