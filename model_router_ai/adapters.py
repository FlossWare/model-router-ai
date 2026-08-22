"""Adapters for plugging budget-ai and strategy-ai into model-router-ai.

These adapters bridge the signature differences between model-router-ai's
protocols (``UsageTracker``, ``ModelSelector``) and the corresponding
classes in budget-ai / strategy-ai.

Usage::

    from model_router_ai.adapters import BudgetAIAdapter, StrategyAIAdapter

    # budget-ai integration
    from budget_ai import InMemoryBudgetTracker
    tracker = BudgetAIAdapter(InMemoryBudgetTracker(max_cost=300.0))
    router = BudgetGuard(base, tracker=tracker)

    # strategy-ai integration
    from strategy_ai import ThompsonSamplingSelector as TSSelector
    selector = StrategyAIAdapter(TSSelector())
    router = ThompsonSamplingSelector(base, selector=selector)

Both adapters are optional — model-router-ai works without budget-ai or
strategy-ai installed. The adapters are imported only when you need them.
"""

from __future__ import annotations

import logging
from typing import Any

from model_router_ai.types import BudgetStatus, UsageInfo

logger = logging.getLogger(__name__)


class BudgetAIAdapter:
    """Wraps a budget-ai ``BudgetTracker`` to satisfy ``UsageTracker``.

    Bridges the signature gap: model-router-ai passes ``(model, cost_usd,
    usage_info)`` while budget-ai expects ``(model, TokenUsage)``.
    """

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker

    async def record_usage(
        self, model: str, cost_usd: float, usage: UsageInfo | None = None,
    ) -> None:
        try:
            from budget_ai.types import TokenUsage
        except ImportError:
            raise ImportError(
                "budget-ai must be installed to use BudgetAIAdapter: "
                "pip install budget-ai"
            )
        token_usage = TokenUsage(
            prompt_tokens=(usage or {}).get("prompt_tokens", 0),
            completion_tokens=(usage or {}).get("completion_tokens", 0),
            total_tokens=(usage or {}).get("total_tokens", 0),
        )
        await self._tracker.record_usage(model, token_usage)

    async def is_exceeded(self) -> bool:
        status = await self._tracker.remaining()
        if status.cost_remaining is not None:
            return status.cost_remaining <= 0
        if status.tokens_remaining is not None:
            return status.tokens_remaining <= 0
        return False

    async def get_status(self) -> BudgetStatus:
        status = await self._tracker.remaining()
        return BudgetStatus(
            spent_usd=status.cost_used,
            remaining_usd=status.cost_remaining,
            max_usd=(
                (status.cost_used + status.cost_remaining)
                if status.cost_remaining is not None
                else None
            ),
            calls_made=0,
        )

    def reset(self) -> None:
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()


class StrategyAIAdapter:
    """Wraps a strategy-ai ``StrategySelector`` to satisfy ``ModelSelector``.

    Bridges the signature gap: model-router-ai calls
    ``select(candidates)`` while strategy-ai uses
    ``select(task_type, candidates=...)``. Both are async,
    so no threading hacks needed.
    """

    def __init__(self, selector: Any, task_type: str = "model_selection") -> None:
        self._selector = selector
        self._task_type = task_type

    async def select(self, candidates: list[str]) -> str:
        return await self._selector.select(self._task_type, candidates=candidates)

    async def record(self, model_id: str, success: bool) -> None:
        reward = 1.0 if success else 0.0
        await self._selector.update(model_id, self._task_type, reward=reward)

    async def stats(self) -> dict[str, Any]:
        if hasattr(self._selector, "performance"):
            result = await self._selector.performance(task_type=self._task_type)
            return {
                s.strategy: {
                    "alpha": s.alpha,
                    "beta": s.beta,
                    "mean": s.avg_reward,
                    "trials": s.total_trials,
                }
                for s in result.values()
            }
        return {}
