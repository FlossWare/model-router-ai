"""Standalone, decorator-based model router for LLM orchestration.

Zero external dependencies (stdlib only). Designed to be pluggable
into any platform: loom-ai, Claude Code, Crush, Codex, etc.
"""

from model_router_ai.types import (
    BudgetStatus, ChatMessage, ChatResponse, ModelCost, ModelInfo, UsageInfo,
)
from model_router_ai.protocol import ModelRouter, ModelSelector, UsageTracker
from model_router_ai.router import ProviderRouter
from model_router_ai.providers import OpenAICompatProvider, GeminiProvider, CohereProvider, VertexAIProvider
from model_router_ai.decorators import CostAware, BudgetGuard, PolicyGuard, LatencyOptimizer, ThompsonSamplingSelector
from model_router_ai.strategies import ThompsonSamplingStrategy, RoundRobinStrategy, LatencyWeightedStrategy, CascadeStrategy
from model_router_ai.adapters import BudgetAIAdapter, StrategyAIAdapter
from model_router_ai.discovery import Account, ProviderDefinition, discover_accounts, discover_all_models, discover_models, provider_definitions
from model_router_ai.workers import Arbiter, Worker, WorkerPool, WorkerResult, WorkerStatus

__version__ = "0.2"

__all__ = [
    "Account", "Arbiter", "BudgetAIAdapter", "BudgetGuard", "BudgetStatus", "CascadeStrategy", "ChatMessage",
    "ChatResponse", "CohereProvider", "CostAware", "GeminiProvider", "LatencyOptimizer",
    "LatencyWeightedStrategy", "ModelCost", "ModelInfo", "ModelRouter", "ModelSelector",
    "OpenAICompatProvider", "PolicyGuard", "ProviderDefinition", "ProviderRouter", "RoundRobinStrategy",
    "StrategyAIAdapter", "ThompsonSamplingSelector", "ThompsonSamplingStrategy", "UsageInfo",
    "UsageTracker", "VertexAIProvider", "Worker", "WorkerPool", "WorkerResult", "WorkerStatus",
    "discover_accounts", "discover_all_models", "discover_models", "provider_definitions",
]
