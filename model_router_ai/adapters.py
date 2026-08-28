"""Adapters for plugging budget-ai and strategy-ai into model-router-ai.

These adapters bridge the signature differences between model-router-ai's
protocols (``UsageTracker``, ``ModelSelector``) and the corresponding
classes in budget-ai / strategy-ai.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

from model_router_ai.types import BudgetStatus, UsageInfo


class _BudgetRemaining(Protocol):
    cost_used: float
    cost_remaining: float | None
    tokens_remaining: int | None


class _StrategyPerformance(Protocol):
    strategy: str
    alpha: float
    beta: float
    avg_reward: float
    total_trials: int


class BudgetAIAdapter:
    """Wrap a budget-ai tracker to satisfy ``UsageTracker``."""

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker

    async def record_usage(
        self,
        model: str,
        cost_usd: float,
        usage: UsageInfo | None = None,
    ) -> None:
        del cost_usd
        try:
            budget_types = importlib.import_module("budget_ai.types")
            token_usage_type = budget_types.TokenUsage
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "budget-ai must be installed to use BudgetAIAdapter: "
                "pip install budget-ai"
            ) from exc

        token_usage = token_usage_type(
            prompt_tokens=(usage or {}).get("prompt_tokens", 0),
            completion_tokens=(usage or {}).get("completion_tokens", 0),
            total_tokens=(usage or {}).get("total_tokens", 0),
        )
        await self._tracker.record_usage(model, token_usage)

    async def is_exceeded(self) -> bool:
        status = cast(_BudgetRemaining, await self._tracker.remaining())
        if status.cost_remaining is not None:
            return status.cost_remaining <= 0
        if status.tokens_remaining is not None:
            return status.tokens_remaining <= 0
        return False

    async def get_status(self) -> BudgetStatus:
        status = cast(_BudgetRemaining, await self._tracker.remaining())
        return BudgetStatus(
            spent_usd=status.cost_used,
            remaining_usd=status.cost_remaining,
            max_usd=(
                status.cost_used + status.cost_remaining
                if status.cost_remaining is not None
                else None
            ),
            calls_made=0,
        )

    def reset(self) -> None:
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()


class StrategyAIAdapter:
    """Wrap a strategy-ai selector to satisfy ``ModelSelector``."""

    def __init__(self, selector: Any, task_type: str = "model_selection") -> None:
        self._selector = selector
        self._task_type = task_type

    async def select(self, candidates: list[str]) -> str:
        result = await self._selector.select(
            self._task_type,
            candidates=candidates,
        )
        return cast(str, result)

    async def record(self, model_id: str, success: bool) -> None:
        reward = 1.0 if success else 0.0
        await self._selector.update(
            model_id,
            self._task_type,
            reward=reward,
        )

    async def stats(self) -> dict[str, dict[str, float | int]]:
        if not hasattr(self._selector, "performance"):
            return {}
        raw = await self._selector.performance(task_type=self._task_type)
        result = cast(dict[str, _StrategyPerformance], raw)
        return {
            item.strategy: {
                "alpha": item.alpha,
                "beta": item.beta,
                "mean": item.avg_reward,
                "trials": item.total_trials,
            }
            for item in result.values()
        }
