"""Base ProviderRouter — the foundation that talks to LLM providers.

Handles provider registration, account-aware model discovery, endpoint selection,
and raw chat() calls. Decorators wrap this to add cost awareness, budget tracking,
policy enforcement, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from model_router_ai.providers import _BaseProvider
from model_router_ai.strategies import SelectionStrategy, ThompsonSamplingStrategy
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo

logger = logging.getLogger(__name__)


class _EndpointStats:
    __slots__ = ("successes", "failures")

    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0


class ProviderRouter:
    """Base model router with first-class multi-account endpoints.

    A provider may have multiple independent accounts. ``account_name`` is
    therefore part of endpoint identity and is preserved on every discovered
    ``ModelInfo``. Credentials remain in the caller's credential source and
    are never persisted by the router.
    """

    def __init__(
        self,
        strategy: SelectionStrategy | None = None,
    ) -> None:
        self._strategy = strategy or ThompsonSamplingStrategy()
        self._providers: list[tuple[_BaseProvider, str, str]] = []
        self._models: list[ModelInfo] = []
        self._stats: dict[str, _EndpointStats] = {}
        self._initialized = False

    def add_provider(
        self,
        provider: _BaseProvider,
        api_key: str,
        account_name: str = "default",
    ) -> None:
        """Register a provider credential as a distinct routing account."""
        self._providers.append((provider, api_key, account_name))
        self._initialized = False

    async def initialize(self) -> None:
        import asyncio

        self._models.clear()
        tasks = [self._discover(provider, api_key, account_name)
                 for provider, api_key, account_name in self._providers]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for provider_tuple, result in zip(self._providers, results):
            provider, _, account_name = provider_tuple
            if isinstance(result, BaseException):
                logger.warning("Discovery failed for %s/%s: %s", provider.name, account_name, result)
            elif isinstance(result, list):
                self._models.extend(result)

        if not self._models:
            logger.warning("No models discovered from any provider/account")
            return

        self._initialized = True
        endpoints = {self._endpoint_key(m) for m in self._models}
        logger.info("Discovered %d models across %d provider accounts", len(self._models), len(endpoints))

    async def _discover(
        self, provider: _BaseProvider, api_key: str, account_name: str
    ) -> list[ModelInfo]:
        models = await provider.discover_models(api_key)
        for model in models:
            model.account_name = account_name
        return models

    @staticmethod
    def _endpoint_key(model: ModelInfo) -> str:
        return f"{model.provider}/{model.account_name}/{model.model_id}"

    def _get_stats(self, model: ModelInfo) -> _EndpointStats:
        key = self._endpoint_key(model)
        if key not in self._stats:
            self._stats[key] = _EndpointStats()
        return self._stats[key]

    def _select_models(
        self, model_filter: str | None = None
    ) -> list[ModelInfo]:
        candidates = self._models
        if model_filter:
            # Accept provider/account/model addressing as well as the legacy
            # model-id-only form. This makes duplicate models deterministic.
            exact_address = [m for m in candidates if self._endpoint_key(m) == model_filter]
            if exact_address:
                candidates = exact_address
            else:
                exact = [m for m in candidates if m.model_id == model_filter]
                if exact:
                    candidates = exact
                else:
                    partial = [m for m in candidates if model_filter in m.model_id]
                    if partial:
                        candidates = partial

        def _score(m: ModelInfo) -> float:
            stats = self._get_stats(m)
            key = self._endpoint_key(m)
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
        key = self._endpoint_key(model)
        self._strategy.record(success=success, endpoint_key=key, latency_s=latency_s)

    def _find_provider(self, provider_name: str, account_name: str = "") -> _BaseProvider | None:
        for provider, _, account in self._providers:
            if provider.name == provider_name and (not account_name or account == account_name):
                return provider
        return None

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        account: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Route a request, optionally constrained to a named account."""
        if not self._initialized:
            await self.initialize()

        candidates = self._select_models(model)
        if account:
            candidates = [m for m in candidates if m.account_name == account]
        if not candidates:
            raise RuntimeError("No model endpoints available for the requested account/model")

        last_err: Exception | None = None
        for model_info in candidates:
            provider = self._find_provider(model_info.provider, model_info.account_name)
            if provider is None:
                continue

            t0 = time.monotonic()
            try:
                resp = await provider.call(model_info, messages, temperature, max_tokens)
                if not resp.content:
                    raise RuntimeError("Empty response")
                elapsed = time.monotonic() - t0
                self._record(model_info, True, elapsed)
                resp.latency_ms = elapsed * 1000

                if model_info.cost:
                    input_tokens = resp.usage.get("prompt_tokens", 0)
                    output_tokens = resp.usage.get("completion_tokens", 0)
                    resp.cost_usd = model_info.cost.estimate(input_tokens, output_tokens)
                resp.provider = model_info.provider
                resp.model = model_info.model_id
                return resp
            except Exception as exc:
                self._record(model_info, False, time.monotonic() - t0)
                last_err = exc
                logger.debug("Failed %s: %s", self._endpoint_key(model_info), exc)

        raise RuntimeError(f"All {len(candidates)} endpoints failed. Last: {last_err}")

    async def list_models(self, account: str | None = None) -> list[ModelInfo]:
        if not self._initialized:
            await self.initialize()
        if account:
            return [m for m in self._models if m.account_name == account]
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
