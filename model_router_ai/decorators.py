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

import asyncio
import fnmatch
import logging
import random
import time
from typing import Any, Callable

from model_router_ai.protocol import ModelRouter, ModelSelector, UsageTracker
from model_router_ai.types import (
    BudgetStatus,
    ChatMessage,
    ChatResponse,
    ModelCost,
    ModelInfo,
    UsageInfo,
)

logger = logging.getLogger(__name__)


class _RouterDecorator:
    """Base class for router decorators.

    Provides pass-through for ``list_models()`` and ``initialize()``.
    Subclasses override ``chat()`` and optionally ``list_models()``.
    """

    def __init__(self, wrapped: ModelRouter) -> None:
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
        wrapped: ModelRouter,
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


class _SimpleBudgetTracker:
    """Minimal in-memory budget tracker (lite default).

    Tracks cumulative USD spend and call count. For advanced features
    (per-model rate cards, token-level tracking, persistence), inject
    budget-ai's ``InMemoryBudgetTracker`` instead.
    """

    def __init__(
        self,
        max_monthly: float,
        alert_thresholds: list[float] | None = None,
        on_alert: Callable[[float, float], None] | None = None,
    ) -> None:
        self._max = max_monthly
        self._spent = 0.0
        self._calls = 0
        self._thresholds = alert_thresholds or [50.0, 75.0, 90.0]
        self._triggered: set[float] = set()
        self._on_alert = on_alert

    async def record_usage(
        self, model: str, cost_usd: float, usage: UsageInfo | None = None,
    ) -> None:
        self._spent += cost_usd
        self._calls += 1
        self._check_thresholds()

    async def is_exceeded(self) -> bool:
        return self._spent >= self._max

    async def get_status(self) -> BudgetStatus:
        return BudgetStatus(
            spent_usd=self._spent,
            remaining_usd=max(0.0, self._max - self._spent),
            max_usd=self._max,
            calls_made=self._calls,
        )

    def get_status_sync(self) -> BudgetStatus:
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


class BudgetGuard(_RouterDecorator):
    """Tracks spending and enforces a budget cap.

    Delegates usage tracking to a ``UsageTracker`` instance. When no
    tracker is provided, uses a simple in-memory tracker that counts
    cumulative USD spend.

    For advanced tracking (per-model rate cards, token-level budgets,
    persistence), inject budget-ai's ``InMemoryBudgetTracker``::

        from budget_ai import InMemoryBudgetTracker
        router = BudgetGuard(base, tracker=InMemoryBudgetTracker(...))

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    tracker:
        Optional ``UsageTracker`` to delegate budget tracking to.
        If not provided, a simple internal tracker is used.
    max_monthly:
        Monthly budget cap in USD (used only when no tracker provided).
    alert_thresholds:
        Percentages at which to log warnings (used only when no tracker provided).
    on_alert:
        Optional callback invoked at each threshold crossing (used only when no tracker provided).
    fail_open:
        If True (default), tracker errors are logged and the call proceeds.
        If False, tracker errors propagate and block the call.
    """

    def __init__(
        self,
        wrapped: ModelRouter,
        tracker: UsageTracker | None = None,
        max_monthly: float = 300.0,
        alert_thresholds: list[float] | None = None,
        on_alert: Callable[[float, float], None] | None = None,
        fail_open: bool = True,
    ) -> None:
        super().__init__(wrapped)
        self._tracker: UsageTracker = tracker or _SimpleBudgetTracker(
            max_monthly, alert_thresholds, on_alert
        )
        self._fail_open = fail_open
        self._lock = asyncio.Lock()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        async with self._lock:
            try:
                if await self._tracker.is_exceeded():
                    st = await self._tracker.get_status()
                    raise BudgetExhaustedError(
                        f"Budget exhausted: ${st.spent_usd:.2f} / ${st.max_usd:.2f}"
                    )
            except BudgetExhaustedError:
                raise
            except Exception:
                if not self._fail_open:
                    raise
                logger.warning("Budget tracker check failed, proceeding (fail-open)", exc_info=True)

        resp = await self._wrapped.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        usage_info: UsageInfo | None = None
        if resp.usage:
            usage_info = {
                "prompt_tokens": resp.usage.get("prompt_tokens", 0),
                "completion_tokens": resp.usage.get("completion_tokens", 0),
                "total_tokens": resp.usage.get("total_tokens", 0),
            }

        async with self._lock:
            try:
                await self._tracker.record_usage(resp.model, resp.cost_usd, usage_info)
            except Exception:
                logger.warning("Budget tracker record_usage failed", exc_info=True)
        return resp

    @property
    def status(self) -> BudgetStatus:
        """Synchronous status access.

        Works for the lite default tracker and any tracker that
        implements ``get_status_sync()``. For async-only trackers,
        use ``await async_status()`` instead.
        """
        if hasattr(self._tracker, "get_status_sync"):
            return self._tracker.get_status_sync()
        raise TypeError(
            "Injected tracker requires async access: use 'await guard.async_status()'"
        )

    async def async_status(self) -> BudgetStatus:
        async with self._lock:
            return await self._tracker.get_status()

    def reset(self) -> None:
        self._tracker.reset()


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
        wrapped: ModelRouter,
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

    def __init__(self, wrapped: ModelRouter, window_size: int = 20) -> None:
        super().__init__(wrapped)
        self._latencies: dict[str, list[float]] = {}
        self._window = window_size
        self._lock = asyncio.Lock()

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
            async with self._lock:
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


class _SimpleThompsonSelector:
    """Minimal Beta-Bernoulli bandit selector (lite default).

    Maintains alpha/beta counts per model and samples from
    Beta distributions to balance explore/exploit. For advanced
    features (per-task-type tracking, epsilon-greedy, persistence),
    inject strategy-ai's ``ThompsonSamplingSelector`` instead.
    """

    def __init__(self) -> None:
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}

    async def select(self, candidates: list[str]) -> str:
        best_score = -1.0
        best = candidates[0]
        for model_id in candidates:
            a = self._alpha.get(model_id, 1.0)
            b = self._beta.get(model_id, 1.0)
            score = random.betavariate(a, b)
            if score > best_score:
                best_score = score
                best = model_id
        return best

    async def record(self, model_id: str, success: bool) -> None:
        if success:
            self._alpha[model_id] = self._alpha.get(model_id, 1.0) + 1.0
        else:
            self._beta[model_id] = self._beta.get(model_id, 1.0) + 1.0

    async def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
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


class ThompsonSamplingSelector(_RouterDecorator):
    """Bayesian explore/exploit model selection.

    Delegates selection logic to a ``ModelSelector`` instance. When no
    selector is provided, uses a simple Beta-Bernoulli bandit.

    For advanced selection (per-task-type bandits, epsilon-greedy,
    persistence), inject strategy-ai's selector::

        from strategy_ai import ThompsonSamplingSelector as TSSelector
        router = ThompsonSamplingSelector(base, selector=TSSelector())

    Parameters
    ----------
    wrapped:
        The inner router to delegate to.
    selector:
        Optional ``ModelSelector`` to delegate selection to.
        If not provided, a simple internal Thompson Sampling selector is used.
    quality_threshold:
        Minimum success rate before a model is considered "proven"
        (used only when no selector provided).
    """

    def __init__(
        self,
        wrapped: ModelRouter,
        selector: ModelSelector | None = None,
        quality_threshold: float = 0.5,
    ) -> None:
        super().__init__(wrapped)
        self._selector: ModelSelector = selector or _SimpleThompsonSelector()
        self._threshold = quality_threshold
        self._lock = asyncio.Lock()

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
            async with self._lock:
                try:
                    await self._selector.record(resp.model or model, True)
                except Exception:
                    logger.warning("Selector record failed", exc_info=True)
            return resp

        models = await self._wrapped.list_models()
        if not models:
            return await self._wrapped.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )

        candidates = [m.model_id for m in models]
        async with self._lock:
            try:
                selected_id = await self._selector.select(candidates)
            except Exception:
                logger.warning("Selector failed, using first candidate", exc_info=True)
                selected_id = candidates[0]

        try:
            resp = await self._wrapped.chat(
                messages,
                model=selected_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            async with self._lock:
                try:
                    await self._selector.record(selected_id, bool(resp.content))
                except Exception:
                    logger.warning("Selector record failed", exc_info=True)
            return resp
        except Exception:
            async with self._lock:
                try:
                    await self._selector.record(selected_id, False)
                except Exception:
                    logger.warning("Selector record failed", exc_info=True)
            return await self._wrapped.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    async def performance(self) -> dict[str, Any]:
        return await self._selector.stats()
