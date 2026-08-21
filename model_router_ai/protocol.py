"""ModelRouter protocol — the interface all routers and decorators satisfy.

Uses ``typing.Protocol`` for structural subtyping. Any class that
implements ``chat()`` and ``list_models()`` is a valid ModelRouter,
no inheritance required.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo


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
