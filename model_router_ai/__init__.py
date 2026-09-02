"""Standalone, decorator-based model router for LLM orchestration.

Zero external dependencies (stdlib only). Designed to be pluggable
into any platform: loom-ai, Claude Code, Crush, Codex, etc.
"""

from model_router_ai.adapters import BudgetAIAdapter, StrategyAIAdapter
from model_router_ai.decorators import (
    BudgetGuard,
    CostAware,
    LatencyOptimizer,
    PolicyGuard,
    ThompsonSamplingSelector,
)
from model_router_ai.discovery import (
    Account,
    ProviderDefinition,
    discover_accounts,
    discover_all_models,
    discover_identities,
    discover_identity,
    discover_models,
    provider_definitions,
)
from model_router_ai.protocol import ModelRouter, ModelSelector, UsageTracker
from model_router_ai.providers import (
    CohereProvider,
    GeminiProvider,
    OpenAICompatProvider,
    VertexAIProvider,
)
from model_router_ai.router import ProviderRouter
from model_router_ai.strategies import (
    CascadeStrategy,
    LatencyWeightedStrategy,
    RoundRobinStrategy,
    ThompsonSamplingStrategy,
)
from model_router_ai.types import (
    BudgetStatus,
    ChatMessage,
    ChatResponse,
    ModelCost,
    ModelInfo,
    UsageInfo,
)
from model_router_ai.workers import (
    ModelWorker,
    WorkerResult,
    WorkerStatus,
    classify_failure,
)

__version__ = "0.1"

__all__ = [
    "Account",
    "BudgetAIAdapter",
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
    "ModelSelector",
    "ModelWorker",
    "OpenAICompatProvider",
    "PolicyGuard",
    "ProviderDefinition",
    "ProviderRouter",
    "RoundRobinStrategy",
    "StrategyAIAdapter",
    "ThompsonSamplingSelector",
    "ThompsonSamplingStrategy",
    "UsageInfo",
    "UsageTracker",
    "VertexAIProvider",
    "WorkerResult",
    "WorkerStatus",
    "classify_failure",
    "discover_accounts",
    "discover_all_models",
    "discover_identities",
    "discover_identity",
    "discover_models",
    "provider_definitions",
]
