"""Standalone, decorator-based model router for LLM orchestration.

Zero external dependencies (stdlib only). Designed to be pluggable
into any platform: loom-ai, Claude Code, Crush, Codex, etc.

Stack decorators to compose routing behavior::

    router = PolicyGuard(
        BudgetGuard(
            CostAware(
                ProviderRouter(providers=[...])
            ),
            max_monthly=300.0,
        ),
        allowed_models=["gemini-*", "claude-*"],
    )
    response = await router.chat(messages)

Each decorator adds exactly one concern without changing the interface.
"""

from model_router_ai.types import (
    BudgetStatus,
    ChatMessage,
    ChatResponse,
    ModelCost,
    ModelInfo,
)
from model_router_ai.protocol import ModelRouter
from model_router_ai.router import ProviderRouter
from model_router_ai.providers import (
    OpenAICompatProvider,
    GeminiProvider,
    CohereProvider,
    VertexAIProvider,
)
from model_router_ai.decorators import (
    CostAware,
    BudgetGuard,
    PolicyGuard,
    LatencyOptimizer,
    ThompsonSamplingSelector,
)
from model_router_ai.strategies import (
    ThompsonSamplingStrategy,
    RoundRobinStrategy,
    LatencyWeightedStrategy,
    CascadeStrategy,
)

__all__ = [
    "BudgetGuard",
    "BudgetStatus",
    "CascadeStrategy",
    "ChatMessage",
    "ChatResponse",
    "CohereProvider",
    "CostAware",
    "GeminiProvider",
    "LatencyOptimizer",
    "LatencyWeightedStrategy",
    "ModelCost",
    "ModelInfo",
    "ModelRouter",
    "OpenAICompatProvider",
    "PolicyGuard",
    "ProviderRouter",
    "RoundRobinStrategy",
    "ThompsonSamplingSelector",
    "ThompsonSamplingStrategy",
    "VertexAIProvider",
]
