"""Consistent error body for the API: {"error": {"code": "...", "message": "..."}}.

Used by /ask, /ingest and request validation (422). /health/db keeps its original `detail`
field for backward compatibility and adds the same `error` object next to it. When a failed
/ask was recorded in the `queries` table, its id is returned as `error.query_id`.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: list | None = None,
                 query_id: str | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details
        self.query_id = query_id


def error_body(code: str, message: str, details: list | None = None,
               query_id: str | None = None) -> dict:
    err = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    if query_id is not None:
        err["query_id"] = query_id
    return {"error": err}


def _validation_details(exc: RequestValidationError) -> list[dict]:
    # loc/msg/type only: the rejected input value is deliberately not echoed back.
    return [
        {"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", ""),
         "type": e.get("type", "")}
        for e in exc.errors()
    ]


# Called for every validation failure; returns a query id if the failure was recorded.
ValidationHook = Callable[[Request, RequestValidationError, str], Awaitable[str | None]]


def install_error_handlers(app: FastAPI, on_validation_error: ValidationHook | None = None
                           ) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status,
                            content=error_body(exc.code, exc.message, exc.details,
                                               exc.query_id))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = _validation_details(exc)
        first = details[0] if details else {"loc": [], "msg": "invalid request"}
        where = ".".join(p for p in first["loc"] if p != "body") or "body"
        message = f"{where}: {first['msg']}"
        query_id = await on_validation_error(request, exc, message) \
            if on_validation_error else None
        return JSONResponse(status_code=422,
                            content=error_body("validation_error", message, details, query_id))
