"""Composable decorators for model routing.

Each decorator wraps any ``ModelRouter`` and adds exactly one concern.
Stack them to compose routing behavior::

    router = PolicyGuard(
        BudgetGuard(
            CostAware(
                LatencyOptimizer(
                    ThompsonSamplingSelector(
                        ProviderRouter(...)
                    )
                )
            ),
            max_monthly=300.0,
        ),
        allowed_models=["gemini-*", "claude-*"],
    )

Each decorator satisfies the ``ModelRouter`` protocol, so they're
freely composable in any order.
"""

from __future__ import annotations

import fnmatch
import logging
import random
import time
from typing import Any, Callable

from model_router.types import (
    BudgetStatus,
    ChatMessage,
    ChatResponse,
    ModelCost,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class _RouterDecorator:
    """Base class for router decorators.

    Provides pass-through for ``list_models()`` and ``initialize()``.
    Subclasses override ``chat()`` and optionally ``list_models()``.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    async def initialize(self) -> None:
        await self._wrapped.initialize()

    async def list_models(self) -> list[ModelInfo]:
        return await self._wrapped.list_models()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        return await self._wrapped.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )


class CostAware(_RouterDecorator):
    """Scores models by cost-per-token, preferring cheaper models.

    When no specific model is requested, sorts available models by
    estimated cost (cheapest first) and tries them in order. Models
    without cost data are deprioritized but not excluded.

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    max_cost_per_call:
        Optional hard cap on estimated cost per call (USD).
        Calls estimated to exceed this are skipped.
    prefer_free:
        When True (default), free models (cost=0) are always tried first.
    """

    def __init__(
        self,
        wrapped: Any,
        max_cost_per_call: float | None = None,
        prefer_free: bool = True,
        max_attempts: int = 5,
    ) -> None:
        super().__init__(wrapped)
        self._max_cost_per_call = max_cost_per_call
        self._prefer_free = prefer_free
        self._max_attempts = max_attempts

    async def list_models(self) -> list[ModelInfo]:
        """Return models sorted by cost (cheapest first)."""
        models = await self._wrapped.list_models()
        return self._rank_by_cost(models)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if model is not None:
            return await self._wrapped.chat(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )

        models = await self.list_models()
        candidates = models[: self._max_attempts]

        last_err: Exception | None = None
        for m in candidates:
            if self._max_cost_per_call is not None and m.cost is not None:
                est_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages)
                est_cost = m.cost.estimate(int(est_tokens), int(est_tokens * 0.5))
                if est_cost > self._max_cost_per_call:
                    logger.debug(
                        "Skipping %s/%s: estimated $%.4f > max $%.4f",
                        m.provider, m.model_id, est_cost, self._max_cost_per_call,
                    )
                    continue

            try:
                return await self._wrapped.chat(
                    messages,
                    model=m.model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_err = exc
                continue

        if last_err:
            raise last_err
        raise RuntimeError("No affordable model endpoints available")

    def _rank_by_cost(self, models: list[ModelInfo]) -> list[ModelInfo]:
        def _cost_key(m: ModelInfo) -> tuple[int, float]:
            if m.cost is None:
                return (2, 0.0)
            total = m.cost.input_per_1m + m.cost.output_per_1m
            if total == 0:
                return (0, 0.0) if self._prefer_free else (1, 0.0)
            return (1, total)

        return sorted(models, key=_cost_key)


class BudgetGuard(_RouterDecorator):
    """Tracks spending and enforces a budget cap.

    Accumulates cost from each call's ``cost_usd`` field. When the
    budget is exhausted, raises ``BudgetExhaustedError`` instead of
    making the call.

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    max_monthly:
        Monthly budget cap in USD.
    alert_thresholds:
        Percentages at which to log warnings (default: 50%, 75%, 90%).
    on_alert:
        Optional callback invoked at each threshold crossing.
    """

    def __init__(
        self,
        wrapped: Any,
        max_monthly: float = 300.0,
        alert_thresholds: list[float] | None = None,
        on_alert: Callable[[float, float], None] | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._max = max_monthly
        self._spent = 0.0
        self._calls = 0
        self._thresholds = alert_thresholds or [50.0, 75.0, 90.0]
        self._triggered: set[float] = set()
        self._on_alert = on_alert

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if self._spent >= self._max:
            raise BudgetExhaustedError(
                f"Budget exhausted: ${self._spent:.2f} / ${self._max:.2f}"
            )

        resp = await self._wrapped.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        self._spent += resp.cost_usd
        self._calls += 1
        self._check_thresholds()
        return resp

    def _check_thresholds(self) -> None:
        pct = self._spent / self._max * 100 if self._max > 0 else 0
        for threshold in self._thresholds:
            if pct >= threshold and threshold not in self._triggered:
                self._triggered.add(threshold)
                logger.warning(
                    "Budget alert: %.0f%% used ($%.2f / $%.2f)",
                    pct, self._spent, self._max,
                )
                if self._on_alert:
                    self._on_alert(self._spent, self._max)

    @property
    def status(self) -> BudgetStatus:
        return BudgetStatus(
            spent_usd=self._spent,
            remaining_usd=max(0.0, self._max - self._spent),
            max_usd=self._max,
            calls_made=self._calls,
        )

    def reset(self) -> None:
        self._spent = 0.0
        self._calls = 0
        self._triggered.clear()


class BudgetExhaustedError(RuntimeError):
    """Raised when the budget cap has been reached."""


class PolicyGuard(_RouterDecorator):
    """Filters models by policy (allowlist/blocklist patterns).

    Use glob patterns to match model IDs::

        PolicyGuard(router, allowed=["gemini-*", "claude-*"])
        PolicyGuard(router, blocked=["gpt-4o", "o1-*"])

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    allowed:
        If set, only models matching at least one pattern are used.
    blocked:
        Models matching any pattern are excluded.
    allowed_providers:
        If set, only these providers are used.
    blocked_providers:
        These providers are excluded.
    """

    def __init__(
        self,
        wrapped: Any,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        allowed_providers: list[str] | None = None,
        blocked_providers: list[str] | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._allowed = allowed
        self._blocked = blocked or []
        self._allowed_providers = allowed_providers
        self._blocked_providers = blocked_providers or []

    def _is_allowed(self, model: ModelInfo) -> bool:
        if self._blocked_providers and model.provider in self._blocked_providers:
            return False
        if self._allowed_providers and model.provider not in self._allowed_providers:
            return False

        for pattern in self._blocked:
            if fnmatch.fnmatch(model.model_id, pattern):
                return False

        if self._allowed is not None:
            return any(
                fnmatch.fnmatch(model.model_id, pattern)
                for pattern in self._allowed
            )
        return True

    async def list_models(self) -> list[ModelInfo]:
        all_models = await self._wrapped.list_models()
        return [m for m in all_models if self._is_allowed(m)]

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if model is not None:
            all_models = await self._wrapped.list_models()
            matching = [m for m in all_models if m.model_id == model]
            if matching and not self._is_allowed(matching[0]):
                raise PolicyViolationError(
                    f"Model {model!r} is not allowed by policy"
                )
            return await self._wrapped.chat(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )

        allowed = await self.list_models()
        if not allowed:
            raise PolicyViolationError("No models match the current policy")

        last_err: Exception | None = None
        for m in allowed:
            try:
                return await self._wrapped.chat(
                    messages,
                    model=m.model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_err = exc
                continue

        if last_err:
            raise last_err
        raise PolicyViolationError("All policy-allowed models failed")


class PolicyViolationError(RuntimeError):
    """Raised when a model selection violates policy."""


class LatencyOptimizer(_RouterDecorator):
    """Tracks per-model latency and prefers faster models.

    Maintains a moving average of response times per model and
    adjusts selection to favor lower-latency endpoints for
    interactive use.

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    window_size:
        Number of recent calls to track per model.
    """

    def __init__(self, wrapped: Any, window_size: int = 20) -> None:
        super().__init__(wrapped)
        self._latencies: dict[str, list[float]] = {}
        self._window = window_size

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        resp = await self._wrapped.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        if resp.model:
            key = f"{resp.provider}/{resp.model}"
            samples = self._latencies.setdefault(key, [])
            samples.append(resp.latency_ms)
            if len(samples) > self._window:
                self._latencies[key] = samples[-self._window :]

        return resp

    def avg_latency(self, provider: str, model: str) -> float | None:
        key = f"{provider}/{model}"
        samples = self._latencies.get(key)
        if not samples:
            return None
        return sum(samples) / len(samples)

    def fastest_models(self, top_n: int = 5) -> list[tuple[str, float]]:
        avgs = []
        for key, samples in self._latencies.items():
            avg = sum(samples) / len(samples)
            avgs.append((key, avg))
        return sorted(avgs, key=lambda t: t[1])[:top_n]


class ThompsonSamplingSelector(_RouterDecorator):
    """Bayesian explore/exploit model selection.

    Wraps the router to select models using Thompson Sampling
    when no specific model is requested. Balances exploration
    (trying less-used models) with exploitation (using proven ones).

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    quality_threshold:
        Minimum success rate before a model is considered "proven."
    """

    def __init__(
        self,
        wrapped: Any,
        quality_threshold: float = 0.5,
    ) -> None:
        super().__init__(wrapped)
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}
        self._threshold = quality_threshold

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if model is not None:
            resp = await self._wrapped.chat(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )
            self._record(resp.model or model, True)
            return resp

        models = await self._wrapped.list_models()
        if not models:
            return await self._wrapped.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )

        selected = self._sample_best(models)

        try:
            resp = await self._wrapped.chat(
                messages,
                model=selected.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._record(selected.model_id, bool(resp.content))
            return resp
        except Exception:
            self._record(selected.model_id, False)
            return await self._wrapped.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    def _sample_best(self, models: list[ModelInfo]) -> ModelInfo:
        best_score = -1.0
        best = models[0]
        for m in models:
            a = self._alpha.get(m.model_id, 1.0)
            b = self._beta.get(m.model_id, 1.0)
            score = random.betavariate(a, b)
            if score > best_score:
                best_score = score
                best = m
        return best

    def _record(self, model_id: str, success: bool) -> None:
        if success:
            self._alpha[model_id] = self._alpha.get(model_id, 1.0) + 1.0
        else:
            self._beta[model_id] = self._beta.get(model_id, 1.0) + 1.0

    def performance(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        all_keys = set(self._alpha.keys()) | set(self._beta.keys())
        for key in all_keys:
            a = self._alpha.get(key, 1.0)
            b = self._beta.get(key, 1.0)
            result[key] = {
                "alpha": a,
                "beta": b,
                "mean": a / (a + b),
                "trials": int(a + b - 2),
            }
        return result
