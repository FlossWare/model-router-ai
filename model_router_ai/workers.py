"""Provider/account/model workers and arbiter primitives.

Workers are concrete execution routes: provider + account + model.  The
abstractions deliberately do not know about any particular provider so that
routing policy can evolve independently of transport implementations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from model_router_ai.providers import _BaseProvider
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo


class WorkerStatus(str, Enum):
    """Operational state for a worker."""

    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_FAILED = "auth_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    INVALID_REQUEST = "invalid_request"
    FAILED = "failed"


@dataclass
class WorkerResult:
    """Structured outcome of a worker execution."""

    status: WorkerStatus
    response: ChatResponse | None = None
    error: str = ""
    retry_after: float | None = None
    quota_reset: float | None = None
    latency_ms: float = 0.0
    provider: str = ""
    account: str = ""
    model: str = ""


@dataclass
class Worker:
    """A concrete provider/account/model execution route."""

    provider: _BaseProvider
    account_name: str
    model: ModelInfo
    status: WorkerStatus = WorkerStatus.AVAILABLE
    unavailable_until: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def worker_id(self) -> str:
        return f"{self.model.provider}/{self.account_name}/{self.model.model_id}"

    @property
    def available(self) -> bool:
        if self.unavailable_until is not None and time.time() >= self.unavailable_until:
            self.status = WorkerStatus.AVAILABLE
            self.unavailable_until = None
        return self.status == WorkerStatus.AVAILABLE

    def capabilities(self) -> list[str]:
        return list(self.model.capabilities)

    def health(self) -> WorkerStatus:
        _ = self.available
        return self.status

    def mark_unavailable(
        self,
        status: WorkerStatus,
        *,
        retry_after: float | None = None,
        quota_reset: float | None = None,
    ) -> None:
        self.status = status
        reset = quota_reset if quota_reset is not None else retry_after
        if reset is not None:
            self.unavailable_until = reset

    async def execute(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> WorkerResult:
        """Execute through this worker and normalize common HTTP failures."""
        if not self.available:
            return WorkerResult(
                status=self.status,
                error="worker unavailable",
                provider=self.model.provider,
                account=self.account_name,
                model=self.model.model_id,
            )

        started = time.monotonic()
        try:
            response = await self.provider.call(
                self.model, messages, temperature, max_tokens
            )
            latency_ms = (time.monotonic() - started) * 1000
            self.status = WorkerStatus.AVAILABLE
            return WorkerResult(
                status=WorkerStatus.AVAILABLE,
                response=response,
                latency_ms=latency_ms,
                provider=self.model.provider,
                account=self.account_name,
                model=self.model.model_id,
            )
        except Exception as exc:
            status, retry_after, quota_reset = classify_backend_error(exc)
            self.mark_unavailable(
                status, retry_after=retry_after, quota_reset=quota_reset
            )
            return WorkerResult(
                status=status,
                error=str(exc),
                retry_after=retry_after,
                quota_reset=quota_reset,
                latency_ms=(time.monotonic() - started) * 1000,
                provider=self.model.provider,
                account=self.account_name,
                model=self.model.model_id,
            )


class WorkerPool:
    """Registry of independently available workers."""

    def __init__(self, workers: list[Worker] | None = None) -> None:
        self._workers: dict[str, Worker] = {}
        for worker in workers or []:
            self.add(worker)

    def add(self, worker: Worker) -> None:
        self._workers[worker.worker_id] = worker

    def remove(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def all(self) -> list[Worker]:
        return list(self._workers.values())

    def available(
        self,
        *,
        model: str | None = None,
        account: str | None = None,
        provider: str | None = None,
        capabilities: set[str] | None = None,
    ) -> list[Worker]:
        required = capabilities or set()
        result = []
        for worker in self._workers.values():
            if not worker.available:
                continue
            if model and model not in (worker.model.model_id, worker.worker_id):
                continue
            if account and worker.account_name != account:
                continue
            if provider and worker.model.provider != provider:
                continue
            if required and not required.issubset(set(worker.capabilities())):
                continue
            result.append(worker)
        return result


class Arbiter:
    """Selects executable workers without knowing provider-specific details."""

    def __init__(self, pool: WorkerPool, scorer: Callable[[Worker], float] | None = None) -> None:
        self.pool = pool
        self._scorer = scorer or (lambda worker: 0.0)

    def candidates(self, **filters: object) -> list[Worker]:
        return sorted(self.pool.available(**filters), key=self._scorer, reverse=True)

    def select(self, **filters: object) -> Worker | None:
        candidates = self.candidates(**filters)
        return candidates[0] if candidates else None


def _reset_epoch(value: object) -> float | None:
    """Convert common reset representations to a Unix timestamp."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 100_000_000_000:
        return number / 1000.0
    if number > 1_000_000_000:
        return number
    return time.time() + number


def classify_backend_error(exc: Exception) -> tuple[WorkerStatus, float | None, float | None]:
    """Classify provider errors using stable HTTP/error text conventions."""
    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "quota" in text:
        retry_after = _reset_epoch(_extract_number(text, "retry_after"))
        quota_reset = _reset_epoch(_extract_number(text, "x-ratelimit-reset"))
        if "free-models-per-day" in text or "quota exhausted" in text:
            return WorkerStatus.QUOTA_EXHAUSTED, retry_after, quota_reset
        return WorkerStatus.RATE_LIMITED, retry_after, quota_reset
    if "401" in text or "403" in text or "authentication" in text or "unauthorized" in text:
        return WorkerStatus.AUTH_FAILED, None, None
    if "404" in text or "model unavailable" in text:
        return WorkerStatus.MODEL_UNAVAILABLE, None, None
    if "timeout" in text:
        return WorkerStatus.TIMEOUT, None, None
    if "invalid" in text or "400" in text:
        return WorkerStatus.INVALID_REQUEST, None, None
    if "connection" in text or "network" in text:
        return WorkerStatus.NETWORK_ERROR, None, None
    return WorkerStatus.FAILED, None, None


def _extract_number(text: str, name: str) -> float | None:
    import re

    match = re.search(rf"{re.escape(name)}[^0-9]*(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None
