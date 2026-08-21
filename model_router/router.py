"""Base ProviderRouter — the foundation that talks to LLM providers.

Handles provider registration, model discovery, endpoint selection,
and raw chat() calls. Decorators wrap this to add cost awareness,
budget tracking, policy enforcement, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from model_router.providers import _BaseProvider
from model_router.strategies import SelectionStrategy, ThompsonSamplingStrategy
from model_router.types import ChatMessage, ChatResponse, ModelInfo

logger = logging.getLogger(__name__)


class _EndpointStats:
    __slots__ = ("successes", "failures")

    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0


class ProviderRouter:
    """Base model router that manages providers and routes calls.

    Satisfies the ``ModelRouter`` protocol. This is the innermost
    layer in a decorator stack — it actually talks to LLM APIs.

    Usage::

        router = ProviderRouter()
        router.add_provider(OpenAICompatProvider("groq"), api_key="gsk_...")
        router.add_provider(GeminiProvider(), api_key="AIza...")
        await router.initialize()
        response = await router.chat(messages)
    """

    def __init__(
        self,
        strategy: SelectionStrategy | None = None,
    ) -> None:
        self._strategy = strategy or ThompsonSamplingStrategy()
        self._providers: list[tuple[_BaseProvider, str]] = []
        self._models: list[ModelInfo] = []
        self._stats: dict[str, _EndpointStats] = {}
        self._initialized = False

    def add_provider(
        self,
        provider: _BaseProvider,
        api_key: str,
        account_name: str = "default",
    ) -> None:
        self._providers.append((provider, api_key))

    async def initialize(self) -> None:
        import asyncio

        self._models.clear()

        tasks = []
        for provider, api_key in self._providers:
            tasks.append(self._discover(provider, api_key))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for provider_tuple, result in zip(self._providers, results):
            provider, _ = provider_tuple
            if isinstance(result, BaseException):
                logger.warning("Discovery failed for %s: %s", provider.name, result)
            elif isinstance(result, list):
                self._models.extend(result)

        self._initialized = True
        providers = {m.provider for m in self._models}
        logger.info(
            "Discovered %d models across %d providers",
            len(self._models),
            len(providers),
        )

    async def _discover(
        self, provider: _BaseProvider, api_key: str
    ) -> list[ModelInfo]:
        return await provider.discover_models(api_key)

    def _get_stats(self, model: ModelInfo) -> _EndpointStats:
        key = f"{model.provider}/{model.model_id}"
        if key not in self._stats:
            self._stats[key] = _EndpointStats()
        return self._stats[key]

    def _select_models(
        self, model_filter: str | None = None
    ) -> list[ModelInfo]:
        candidates = self._models
        if model_filter:
            exact = [m for m in candidates if m.model_id == model_filter]
            if exact:
                candidates = exact
            else:
                partial = [m for m in candidates if model_filter in m.model_id]
                if partial:
                    candidates = partial

        def _score(m: ModelInfo) -> float:
            stats = self._get_stats(m)
            key = f"{m.provider}/{m.model_id}"
            return self._strategy.score(
                successes=stats.successes,
                failures=stats.failures,
                model_id=m.model_id,
                provider=m.provider,
                endpoint_key=key,
            )

        return sorted(candidates, key=_score, reverse=True)

    def _record(self, model: ModelInfo, success: bool, latency_s: float = 0.0) -> None:
        stats = self._get_stats(model)
        if success:
            stats.successes += 1
        else:
            stats.failures += 1
        key = f"{model.provider}/{model.model_id}"
        self._strategy.record(
            success=success,
            endpoint_key=key,
            latency_s=latency_s,
        )

    def _find_provider(self, provider_name: str) -> _BaseProvider | None:
        for provider, _ in self._providers:
            if provider.name == provider_name:
                return provider
        return None

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if not self._initialized:
            await self.initialize()

        candidates = self._select_models(model)
        if not candidates:
            raise RuntimeError("No model endpoints available")

        last_err: Exception | None = None
        for model_info in candidates:
            provider = self._find_provider(model_info.provider)
            if provider is None:
                continue

            t0 = time.monotonic()
            try:
                resp = await provider.call(
                    model_info, messages, temperature, max_tokens
                )
                if not resp.content:
                    raise RuntimeError("Empty response")
                elapsed = time.monotonic() - t0
                self._record(model_info, True, elapsed)
                resp.latency_ms = elapsed * 1000

                if model_info.cost:
                    input_tokens = resp.usage.get("prompt_tokens", 0)
                    output_tokens = resp.usage.get("completion_tokens", 0)
                    resp.cost_usd = model_info.cost.estimate(input_tokens, output_tokens)

                return resp
            except Exception as exc:
                self._record(model_info, False, time.monotonic() - t0)
                last_err = exc
                logger.debug(
                    "Failed %s/%s: %s", model_info.provider, model_info.model_id, exc
                )

        raise RuntimeError(
            f"All {len(candidates)} endpoints failed. Last: {last_err}"
        )

    async def list_models(self) -> list[ModelInfo]:
        if not self._initialized:
            await self.initialize()
        return list(self._models)

    def stats(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for key, s in self._stats.items():
            total = s.successes + s.failures
            result[key] = {
                "successes": s.successes,
                "failures": s.failures,
                "success_rate": s.successes / total if total > 0 else 0.0,
            }
        return result
