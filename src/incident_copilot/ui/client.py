"""HTTP client for the Causora API, used by the Streamlit page.

Failures come back as `ApiFailure` values instead of exceptions, so the page can show them
without ever inventing an answer. Only the request fields the API defines (api/schemas.py
AskRequest) are sent.
"""

from dataclasses import dataclass
from urllib.parse import quote

import httpx

# The API's limits for AskRequest.top_k (api/schemas.py); a test keeps them in sync.
TOP_K_MIN, TOP_K_MAX = 1, 20


@dataclass(frozen=True)
class ApiFailure:
    """What went wrong. `code` is the API's error code, or one of our own transport codes:
    timeout, unreachable, bad_response."""

    code: str
    message: str
    status: int | None = None
    query_id: str | None = None


@dataclass(frozen=True)
class AskResult:
    data: dict | None = None
    failure: ApiFailure | None = None

    @property
    def ok(self) -> bool:
        return self.data is not None


@dataclass(frozen=True)
class HealthResult:
    api_reachable: bool
    db_ok: bool
    message: str
    documents: int | None = None
    chunks: int | None = None


def _failure_from_response(resp: httpx.Response) -> ApiFailure:
    """The API's error body is {"error": {"code", "message", "query_id"?}}."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("code"):
        return ApiFailure(str(err["code"]), str(err.get("message") or ""), resp.status_code,
                          err.get("query_id"))
    return ApiFailure("bad_response", f"HTTP {resp.status_code} without an error body",
                      resp.status_code)


def _transport_failure(exc: httpx.HTTPError, base_url: str, timeout: float) -> ApiFailure:
    if isinstance(exc, httpx.TimeoutException):
        return ApiFailure("timeout", f"no response from the API within {timeout:g} s")
    if isinstance(exc, httpx.ConnectError):
        return ApiFailure("unreachable", f"cannot connect to the API at {base_url}")
    return ApiFailure("unreachable", f"request failed: {type(exc).__name__}")


class CausoraClient:
    def __init__(self, base_url: str, timeout: float = 90.0,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport  # tests pass httpx.MockTransport

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout,
                            transport=self._transport)

    def health(self, timeout: float = 5.0) -> HealthResult:
        """GET /health/db: is the API up, and can it reach the database?"""
        try:
            with self._client(timeout) as client:
                resp = client.get("/health/db")
        except httpx.HTTPError as exc:
            return HealthResult(False, False, _transport_failure(exc, self.base_url,
                                                                 timeout).message)
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                return HealthResult(True, False, "unexpected /health/db response")
            return HealthResult(True, body.get("status") == "ok", "database reachable",
                                body.get("documents"), body.get("chunks"))
        failure = _failure_from_response(resp)
        return HealthResult(True, False, failure.message or failure.code)

    def ask(self, question: str, *, retrieval_only: bool = False,
            top_k: int | None = None, history: list[dict] | None = None,
            subject: dict | None = None) -> AskResult:
        """POST /ask. `top_k=None` leaves the server default (ASK_TOP_K). `history` is this
        session's earlier turns (view.history_for_request); omitted when empty. `subject` is
        a failure picked from a clarification (its `subject` as the API returned it); omitted
        when there is none."""
        payload: dict = {"question": question}
        if retrieval_only:
            payload["retrieval_only"] = True
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) \
                    or not TOP_K_MIN <= top_k <= TOP_K_MAX:
                return AskResult(failure=ApiFailure(
                    "invalid_top_k", f"top_k must be a whole number from {TOP_K_MIN} to "
                                     f"{TOP_K_MAX}; nothing was sent"))
            payload["top_k"] = top_k
        if history:
            payload["history"] = history
        if subject:
            payload["subject"] = {k: v for k, v in subject.items() if v}
        return self._post_json("/ask", payload, self.timeout, expect="status")

    def _post_json(self, route: str, payload: dict, timeout: float, expect: str) -> AskResult:
        try:
            with self._client(timeout) as client:
                resp = client.post(route, json=payload)
        except httpx.HTTPError as exc:
            return AskResult(failure=_transport_failure(exc, self.base_url, timeout))
        return self._json_result(resp, expect)

    @staticmethod
    def _json_result(resp: httpx.Response, expect: str | None) -> AskResult:
        if resp.status_code != 200:
            return AskResult(failure=_failure_from_response(resp))
        try:
            body = resp.json()
        except ValueError:
            return AskResult(failure=ApiFailure("bad_response", "the API did not return JSON",
                                                resp.status_code))
        if expect is not None and (not isinstance(body, dict) or expect not in body):
            return AskResult(failure=ApiFailure("bad_response",
                                                "the response is not the expected shape",
                                                resp.status_code))
        return AskResult(data=body)

    def _get_json(self, route: str, timeout: float = 30.0) -> AskResult:
        try:
            with self._client(timeout) as client:
                resp = client.get(route)
        except httpx.HTTPError as exc:
            return AskResult(failure=_transport_failure(exc, self.base_url, timeout))
        return self._json_result(resp, None)

    # ---- Knowledge Base (read-only) --------------------------------------------------------

    def list_documents(self) -> AskResult:
        """GET /documents: data is a list of document summaries."""
        result = self._get_json("/documents")
        if result.ok and not isinstance(result.data, list):
            return AskResult(failure=ApiFailure("bad_response", "expected a list of documents"))
        return result

    def get_document(self, path: str) -> AskResult:
        """GET /documents/{path}: chunks, and the original file when available."""
        result = self._get_json("/documents/" + quote(path, safe="/"))
        if result.ok and not (isinstance(result.data, dict) and "document" in result.data):
            return AskResult(failure=ApiFailure("bad_response", "expected a document"))
        return result

    def explain(self, path: str) -> AskResult:
        """POST /documents/explain: one LLM call, only when the user asks for it."""
        return self._post_json("/documents/explain", {"path": path}, self.timeout,
                               expect="summary")
