"""Worker primitives for provider/account/model routing.

A worker is one concrete provider + account + model route. Health and quota
state belongs to that route, never to the provider globally.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

from model_router_ai.providers import _BaseProvider
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo


class WorkerStatus(str, Enum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_FAILED = "auth_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    FAILED = "failed"


@dataclass
class WorkerResult:
    status: WorkerStatus
    response: ChatResponse | None = None
    error: str = ""
    retry_after: float | None = None
    quota_reset: float | None = None


class ModelWorker:
    """One executable provider/account/model route."""

    def __init__(self, provider: _BaseProvider, model: ModelInfo, api_key: str) -> None:
        self.provider = provider
        self.model = model
        self.account = model.account_name or "default"
        self.api_key = api_key
        self._unavailable_until = 0.0
        self._last_status: WorkerStatus | None = None
        self._last_error = ""

    @property
    def id(self) -> str:
        return f"{self.model.provider}/{self.account}/{self.model.model_id}"

    @property
    def unavailable_until(self) -> float:
        return self._unavailable_until

    def available(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return current >= self._unavailable_until

    def mark_unavailable(self, status: WorkerStatus, *, reset: float | None = None,
                         retry_after: float | None = None, error: str = "") -> None:
        if reset is not None:
            until = reset
        else:
            default_delay = 86400 if status is WorkerStatus.QUOTA_EXHAUSTED else 60
            until = time.time() + max(retry_after or default_delay, 1)
        self._unavailable_until = max(time.time() + 0.001, until)
        self._last_status = status
        self._last_error = error

    def mark_available(self) -> None:
        self._unavailable_until = 0.0
        self._last_status = None
        self._last_error = ""

    async def execute(self, messages: list[ChatMessage], temperature: float = 0.7,
                      max_tokens: int | None = None) -> WorkerResult:
        if not self.available():
            return WorkerResult(
                status=self._last_status or WorkerStatus.QUOTA_EXHAUSTED,
                error=self._last_error or "worker temporarily unavailable",
                quota_reset=self._unavailable_until,
            )
        try:
            response = await self.provider.call(self.model, messages, temperature, max_tokens)
            if not response.content:
                raise RuntimeError("Empty response")
            self.mark_available()
            return WorkerResult(status=WorkerStatus.SUCCESS, response=response)
        except Exception as exc:
            error = str(exc)
            status, retry_after, quota_reset = classify_failure(error)
            if status in (WorkerStatus.RATE_LIMITED, WorkerStatus.QUOTA_EXHAUSTED):
                self.mark_unavailable(status, reset=quota_reset, retry_after=retry_after, error=error)
            return WorkerResult(status=status, error=error, retry_after=retry_after, quota_reset=quota_reset)


def classify_failure(error: str) -> tuple[WorkerStatus, float | None, float | None]:
    """Classify provider exceptions and recover OpenRouter reset metadata."""
    lowered = error.lower()
    status_code = _first_int(r"\bHTTP\s+(\d{3})\b", error)
    retry_after = _first_float(r"Retry-After['\"]?\s*[:=]\s*['\"]?([0-9.]+)", error)
    reset_ms = _first_float(r"X-RateLimit-Reset['\"]?\s*[:=]\s*['\"]?([0-9.]+)", error)
    quota_reset = reset_ms / 1000.0 if reset_ms and reset_ms > 10_000_000_000 else reset_ms

    if status_code == 429:
        if any(token in lowered for token in ("free-models-per-day", "daily", "quota", "limit exceeded")):
            return WorkerStatus.QUOTA_EXHAUSTED, retry_after, quota_reset
        return WorkerStatus.RATE_LIMITED, retry_after, quota_reset
    if status_code in (401, 403):
        return WorkerStatus.AUTH_FAILED, retry_after, quota_reset
    if status_code in (404, 410):
        return WorkerStatus.MODEL_UNAVAILABLE, retry_after, quota_reset
    return WorkerStatus.FAILED, retry_after, quota_reset


def _first_float(pattern: str, value: str) -> float | None:
    match = re.search(pattern, value, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _first_int(pattern: str, value: str) -> int | None:
    match = re.search(pattern, value, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
