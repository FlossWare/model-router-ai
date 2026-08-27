"""Base ProviderRouter — the foundation that talks to LLM providers.

Provider, account, model, and worker are distinct routing concepts. A worker
is one concrete provider/account/model route and owns its health/quota state.
Decorators wrap the router to add cost awareness, budgets, policy, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from model_router_ai.providers import _BaseProvider
from model_router_ai.strategies import SelectionStrategy, ThompsonSamplingStrategy
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo
from model_router_ai.workers import ModelWorker, WorkerStatus

logger = logging.getLogger(__name__)


class _EndpointStats:
    __slots__ = ("successes", "failures")

    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0


class ProviderRouter:
    """Provider-neutral router backed by independent model workers."""

    def __init__(self, strategy: SelectionStrategy | None = None) -> None:
        self._strategy = strategy or ThompsonSamplingStrategy()
        self._providers: list[tuple[_BaseProvider, str, str]] = []
        self._models: list[ModelInfo] = []
        self._workers: dict[str, ModelWorker] = {}
        self._stats: dict[str, _EndpointStats] = {}
        self._initialized = False

    def add_provider(
        self,
        provider: _BaseProvider,
        api_key: str,
        account_name: str = "default",
    ) -> None:
        """Register a provider credential as a distinct worker account."""
        self._providers.append((provider, api_key, account_name))
        self._initialized = False

    async def initialize(self) -> None:
        import asyncio

        self._models.clear()
        self._workers.clear()
        tasks = [
            self._discover(provider, api_key, account_name)
            for provider, api_key, account_name in self._providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for provider_tuple, result in zip(self._providers, results):
            provider, _, account_name = provider_tuple
            if isinstance(result, BaseException):
                logger.warning(
                    "Discovery failed for %s/%s: %s",
                    provider.name,
                    account_name,
                    result,
                )
            elif isinstance(result, list):
                self._models.extend(result)

        for model in self._models:
            provider = self._find_provider(model.provider, model.account_name)
            if provider is not None:
                self._workers[self._endpoint_key(model)] = ModelWorker(
                    provider, model, model.api_key
                )

        if not self._models:
            logger.warning("No models discovered from any provider/account")
            return

        self._initialized = True
        logger.info(
            "Discovered %d models across %d worker endpoints",
            len(self._models),
            len(self._workers),
        )

    async def _discover(
        self,
        provider: _BaseProvider,
        api_key: str,
        account_name: str,
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

    def _select_models(self, model_filter: str | None = None) -> list[ModelInfo]:
        candidates = [
            m
            for m in self._models
            if self._workers.get(self._endpoint_key(m)) is not None
            and self._workers[self._endpoint_key(m)].available()
        ]
        if model_filter:
            exact_address = [
                m for m in candidates if self._endpoint_key(m) == model_filter
            ]
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
            return self._strategy.score(
                successes=stats.successes,
                failures=stats.failures,
                model_id=m.model_id,
                provider=m.provider,
                endpoint_key=self._endpoint_key(m),
            )

        return sorted(candidates, key=_score, reverse=True)

    def _record(
        self,
        model: ModelInfo,
        success: bool,
        latency_s: float = 0.0,
    ) -> None:
        stats = self._get_stats(model)
        if success:
            stats.successes += 1
        else:
            stats.failures += 1
        self._strategy.record(
            success=success,
            endpoint_key=self._endpoint_key(model),
            latency_s=latency_s,
        )

    def _find_provider(
        self,
        provider_name: str,
        account_name: str = "",
    ) -> _BaseProvider | None:
        for provider, _, account in self._providers:
            if provider.name == provider_name and (
                not account_name or account == account_name
            ):
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
        """Route through available workers, failing over per account/model."""
        if not self._initialized:
            await self.initialize()

        candidates = self._select_models(model)
        if account:
            candidates = [m for m in candidates if m.account_name == account]
        if not candidates:
            raise RuntimeError("No model endpoints available for the requested account/model")

        last_err: Exception | None = None
        for model_info in candidates:
            worker = self._workers.get(self._endpoint_key(model_info))
            if worker is None or not worker.available():
                continue
            t0 = time.monotonic()
            result = await worker.execute(messages, temperature, max_tokens)
            elapsed = time.monotonic() - t0
            if result.status is WorkerStatus.SUCCESS and result.response is not None:
                self._record(model_info, True, elapsed)
                resp = result.response
                resp.latency_ms = elapsed * 1000
                if model_info.cost:
                    input_tokens = resp.usage.get("prompt_tokens", 0)
                    output_tokens = resp.usage.get("completion_tokens", 0)
                    resp.cost_usd = model_info.cost.estimate(input_tokens, output_tokens)
                resp.provider = model_info.provider
                resp.model = model_info.model_id
                return resp
            self._record(model_info, False, elapsed)
            last_err = RuntimeError(result.error or result.status.value)
            logger.debug("Worker %s failed: %s", worker.id, last_err)

        raise RuntimeError(f"All {len(candidates)} workers failed. Last: {last_err}")

    async def list_models(self, account: str | None = None) -> list[ModelInfo]:
        if not self._initialized:
            await self.initialize()
        if account:
            return [m for m in self._models if m.account_name == account]
        return list(self._models)

    def worker_status(self) -> dict[str, dict[str, Any]]:
        """Return health/quota state for each concrete worker."""
        return {
            key: {
                "available": worker.available(),
                "unavailable_until": worker.unavailable_until,
            }
            for key, worker in self._workers.items()
        }

    def stats(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for key, stats in self._stats.items():
            total = stats.successes + stats.failures
            result[key] = {
                "successes": stats.successes,
                "failures": stats.failures,
                "success_rate": stats.successes / total if total > 0 else 0.0,
            }
        return result
