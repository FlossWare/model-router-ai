"""ModelRouter protocol and injectable protocols for delegation.

Uses ``typing.Protocol`` for structural subtyping. Any class that
implements the required methods is a valid implementation —
no inheritance required.

Delegation protocols (``UsageTracker``, ``ModelSelector``) enable
budget-ai and strategy-ai to be injected into model-router-ai
decorators without hard imports between packages.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from model_router_ai.types import BudgetStatus, ChatMessage, ChatResponse, ModelInfo, UsageInfo


@runtime_checkable
class ModelRouter(Protocol):
    """Async LLM router protocol.

    Every decorator and base router satisfies this interface,
    making them freely composable.
    """

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send messages to an LLM and return the response."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Return all available model endpoints."""
        ...

    async def initialize(self) -> None:
        """Initialize the router (discover models, load config, etc.)."""
        ...


@runtime_checkable
class UsageTracker(Protocol):
    """Protocol for external budget/usage tracking.

    Defines the contract that ``BudgetGuard`` delegates to. The lite
    default (``_SimpleBudgetTracker``) satisfies this directly.

    budget-ai's ``InMemoryBudgetTracker`` has a different signature
    (``record_usage(model, usage: TokenUsage)``), so use the provided
    ``BudgetAIAdapter`` to bridge the two::

        from model_router_ai.adapters import BudgetAIAdapter
        from budget_ai import InMemoryBudgetTracker
        router = BudgetGuard(base, tracker=BudgetAIAdapter(InMemoryBudgetTracker(...)))
    """

    async def record_usage(
        self, model: str, cost_usd: float, usage: UsageInfo | None = None,
    ) -> None:
        """Record spending and token consumption for one call."""
        ...

    async def is_exceeded(self) -> bool:
        """Return True if the budget has been exhausted."""
        ...

    async def get_status(self) -> BudgetStatus:
        """Return current budget status."""
        ...

    def reset(self) -> None:
        """Reset tracked state. Implementations may no-op."""
        ...


@runtime_checkable
class ModelSelector(Protocol):
    """Async protocol for external model selection strategy.

    Defines the contract that ``ThompsonSamplingSelector`` delegates to.
    The lite default (``_SimpleThompsonSelector``) satisfies this directly.

    strategy-ai's selectors also use async methods, so the
    ``StrategyAIAdapter`` can bridge without threading hacks::

        from model_router_ai.adapters import StrategyAIAdapter
        from strategy_ai import ThompsonSamplingSelector as TSSelector
        router = ThompsonSamplingSelector(base, selector=StrategyAIAdapter(TSSelector()))
    """

    async def select(self, candidates: list[str]) -> str:
        """Choose the best candidate model ID."""
        ...

    async def record(self, model_id: str, success: bool) -> None:
        """Record a success/failure observation for a model."""
        ...

    async def stats(self) -> dict[str, Any]:
        """Return per-model performance statistics."""
        ...
