"""LLM providers behind one small interface.

GeminiProvider   google-genai SDK; key from GEMINI_API_KEY, model from LLM_MODEL (never
                 hardcoded). Retries transient failures (HTTP 408/429/500/502/503/504 and
                 network timeouts) with jittered exponential backoff inside one overall
                 deadline, and never logs or returns the key.
FakeLLMProvider  scripted responses for tests; counts calls. Not a real model: anything it
                 "answers" was written by the test.
"""

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from incident_copilot.config import Settings, get_settings
from incident_copilot.redaction import redact

logger = logging.getLogger("incident_copilot.generation")

# Transient: request timeout, rate limit, server error, overloaded, deadline expired (Gemini
# returns 504 when a call runs past the X-Server-Timeout the SDK sends from our timeout).
# Anything else (400 bad request, 401/403 key, 404 model) is permanent and never retried.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 20.0
# An attempt is not started with less time than this left in the overall deadline.
MIN_ATTEMPT_SECONDS = 5.0


def _transient_network_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return False
    # RemoteProtocolError: the server closed the connection without a response (seen live).
    return isinstance(exc, httpx.TimeoutException | httpx.NetworkError
                      | httpx.RemoteProtocolError)


def full_jitter(delay: float) -> float:
    """A random wait in [delay/2, delay], so clients retrying together spread out."""
    return delay * (0.5 + random.random() / 2)

SETUP_INSTRUCTIONS = (
    "No LLM is configured. Get a free key at https://aistudio.google.com/apikey, put it in "
    ".env as GEMINI_API_KEY=..., and restart the API (docker compose up -d api). "
    "Use retrieval_only=true to see the retrieved evidence without an LLM."
)


class LLMError(RuntimeError):
    """The LLM call failed upstream (network, quota, server error, timeout). `attempts` is
    how many calls were made, `status` the last HTTP status (None for network errors)."""

    def __init__(self, message: str, *, attempts: int = 1, status: int | None = None,
                 transient: bool = False):
        super().__init__(message)
        self.attempts, self.status, self.transient = attempts, status, transient


class LLMBadResponseError(LLMError):
    """The LLM answered, but not with JSON matching the required schema."""


class LLMNotConfiguredError(RuntimeError):
    """No API key configured; the API must refuse rather than make up an answer."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    attempts: int = 1  # calls made, including retries


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    def generate(self, system: str, user: str,
                 response_schema: type[BaseModel]) -> LLMResponse: ...


class FakeLLMProvider:
    """Returns scripted responses in order (str, dict -> JSON, or an exception to raise)."""

    model = "fake-llm (scripted test double, not a real model)"

    def __init__(self, responses: Sequence | Callable[[str, str], object] = ()):
        self._responses = responses if callable(responses) else list(responses)
        self.calls: list[dict] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, system: str, user: str, response_schema: type[BaseModel]) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "schema": response_schema})
        if callable(self._responses):
            item = self._responses(system, user)
        else:
            if not self._responses:
                raise AssertionError("FakeLLMProvider ran out of scripted responses")
            item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(text=text, model=self.model)  # no real token counts to report


class GeminiProvider:
    """`timeout_s` bounds one call (and is sent to Gemini as its server deadline);
    `total_timeout_s` bounds all attempts and backoff waits together, so a retry chain can
    never outlast what the caller is willing to wait. None means no overall bound."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0, max_retries: int = 3,
                 client=None, sleep: Callable[[float], None] = time.sleep,
                 total_timeout_s: float | None = None,
                 jitter: Callable[[float], float] = full_jitter,
                 clock: Callable[[], float] = time.monotonic):
        if not api_key:
            raise LLMNotConfiguredError(SETUP_INSTRUCTIONS)
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.total_timeout_s = total_timeout_s
        self._sleep, self._jitter, self._clock = sleep, jitter, clock
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
            )
        self._client = client

    def generate(self, system: str, user: str, response_schema: type[BaseModel]) -> LLMResponse:
        from google.genai import errors, types

        start = self._clock()
        attempt = 0
        while True:
            timeout = self.timeout_s
            if self.total_timeout_s is not None:
                timeout = min(timeout, self.total_timeout_s - (self._clock() - start))
            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1,
                # Per attempt, so a retry only gets the time left in the overall deadline.
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=user, config=config
                )
                break
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                message = (f"Gemini API error {code}: "
                           f"{redact(str(getattr(exc, 'message', '') or ''))[:200]}")
                transient = code in RETRYABLE_STATUS
            except Exception as exc:  # noqa: BLE001 - timeouts, DNS, TLS, ...
                code, message = None, f"Gemini request failed: {type(exc).__name__}"
                transient = _transient_network_error(exc)
            delay = self._jitter(min(2.0 ** attempt, MAX_BACKOFF_SECONDS))
            left = (None if self.total_timeout_s is None
                    else self.total_timeout_s - (self._clock() - start) - delay)
            if not transient or attempt >= self.max_retries \
                    or (left is not None and left < MIN_ATTEMPT_SECONDS):
                attempts = attempt + 1
                if transient and attempts > 1:
                    message += f" (after {attempts} attempts)"
                raise LLMError(message, attempts=attempts, status=code,
                               transient=transient) from None
            logger.warning("Gemini call failed (%s); retry %d/%d in %.1fs",
                           code or message.rsplit(": ", 1)[-1], attempt + 1, self.max_retries,
                           delay)
            self._sleep(delay)
            attempt += 1

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=resp.text or "",
            model=self.model,
            tokens_in=getattr(usage, "prompt_token_count", None),
            tokens_out=getattr(usage, "candidates_token_count", None),
            attempts=attempt + 1,
        )


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """The configured provider. Raises LLMNotConfiguredError if GEMINI_API_KEY is empty."""
    settings = settings or get_settings()
    return GeminiProvider(settings.gemini_api_key.strip(), settings.llm_model,
                          settings.llm_timeout_seconds, settings.llm_max_retries,
                          total_timeout_s=settings.llm_total_timeout_seconds)
