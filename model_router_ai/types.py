"""Standalone data types for the model router.

Zero external dependencies. Compatible with loom-ai's types but
independent — can be used without loom-ai installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class UsageInfo(TypedDict, total=False):
    """Token usage from a single LLM invocation.

    Aligns with budget-ai's ``TokenUsage`` fields so that objects
    satisfying one type also satisfy the other via structural subtyping.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatMessage:
    """A single message in an LLM conversation."""

    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM completion request."""

    content: str
    model: str = ""
    provider: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ModelInfo:
    """Metadata about a model endpoint."""

    model_id: str
    provider: str
    api_key: str = field(default="", repr=False)
    account_name: str = ""
    capabilities: list[str] = field(default_factory=list)
    context_window: int = 0
    cost: ModelCost | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelCost:
    """Cost per token for a model (USD per 1M tokens)."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cached_input_per_1m: float = 0.0

    @property
    def input_per_1k(self) -> float:
        return self.input_per_1m / 1000.0

    @property
    def output_per_1k(self) -> float:
        return self.output_per_1m / 1000.0

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_1m / 1_000_000
            + output_tokens * self.output_per_1m / 1_000_000
        )


@dataclass
class BudgetStatus:
    """Current budget status."""

    spent_usd: float = 0.0
    remaining_usd: float | None = None
    max_usd: float | None = None
    calls_made: int = 0

    @property
    def exhausted(self) -> bool:
        if self.max_usd is None:
            return False
        return self.spent_usd >= self.max_usd

    @property
    def percent_used(self) -> float:
        if self.max_usd is None or self.max_usd == 0:
            return 0.0
        return min(100.0, self.spent_usd / self.max_usd * 100.0)
