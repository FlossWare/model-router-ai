"""Base ProviderRouter — the foundation that talks to LLM providers.

The router keeps the public API stable while representing every executable
provider/account/model combination as an independent worker.  This lets an
exhausted account fail independently and gives higher-level policies a clean
worker pool to select from.
"""

from __future__ import annotations

import logging
from typing import Any

from model_router_ai.providers import _BaseProvider
from model_router_ai.strategies import SelectionStrategy, ThompsonSamplingStrategy
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo
from model_router_ai.workers import Arbiter, Worker, WorkerPool

logger = logging.getLogger(__name__)


class _EndpointStats:
    __slots__ = ("successes", "failures")

    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0


class ProviderRouter:
    """Base model router with first-class provider/account/model workers."""

    def __init__(self, strategy: SelectionStrategy | None = None) -> None:
        self._strategy = strategy or ThompsonSamplingStrategy()
        self._providers: list[tuple[_BaseProvider, str, str]] = []
        self._models: list[ModelInfo] = []
        self._stats: dict[str, _EndpointStats] = {}
        self._workers = WorkerPool()
        self._arbiter = Arbiter(self._workers, scorer=self._worker_score)
        self._initialized = False

    @property
    def worker_pool(self) -> WorkerPool:
        """Return the live worker pool for inspection and advanced policies."""
        return self._workers

    @property
    def arbiter(self) -> Arbiter:
        """Return the worker arbiter used by this router."""
        return self._arbiter

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
        self._workers = WorkerPool()
        self._arbiter = Arbiter(self._workers, scorer=self._worker_score)
        tasks = [
            self._discover(provider, api_key, account_name)
            for provider, api_key, account_name in self._providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for provider_tuple, result in zip(self._providers, results):
            provider, _, account_name = provider_tuple
            if isinstance(result, BaseException):
                logger.warning(
                    "Discovery failed for %s/%s: %s", provider.name, account_name, result
                )
            elif isinstance(result, list):
                self._models.extend(result)

        for model in self._models:
            provider = self._find_provider(model.provider, model.account_name)
            if provider is not None:
                self._workers.add(Worker(provider, model.account_name, model))

        self._initialized = True
        if not self._models:
            logger.warning("No models discovered from any provider/account")
            return

        logger.info(
            "Discovered %d models across %d workers",
            len(self._models), len(self._workers.all()),
        )

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

    def _select_models(self, model_filter: str | None = None) -> list[ModelInfo]:
        candidates = self._models
        if model_filter:
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
        return sorted(candidates, key=self._model_score, reverse=True)

    def _model_score(self, model: ModelInfo) -> float:
        stats = self._get_stats(model)
        return self._strategy.score(
            successes=stats.successes,
            failures=stats.failures,
            model_id=model.model_id,
            provider=model.provider,
            endpoint_key=self._endpoint_key(model),
        )

    def _worker_score(self, worker: Worker) -> float:
        return self._model_score(worker.model)

    def _record(self, model: ModelInfo, success: bool, latency_s: float = 0.0) -> None:
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
        """Route a request through independently managed workers."""
        if not self._initialized:
            await self.initialize()

        model_candidates = self._select_models(model)
        allowed = {self._endpoint_key(m) for m in model_candidates}
        candidates = [
            worker for worker in self._workers.available(account=account)
            if worker.worker_id in allowed
        ]
        candidates.sort(key=self._worker_score, reverse=True)

        if not candidates:
            raise RuntimeError("No model endpoints available for the requested account/model")

        last_err: Exception | None = None
        for worker in candidates:
            result = await worker.execute(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            if result.response is not None:
                self._record(worker.model, True, result.latency_ms / 1000.0)
                response = result.response
                if worker.model.cost:
                    input_tokens = response.usage.get("prompt_tokens", 0)
                    output_tokens = response.usage.get("completion_tokens", 0)
                    response.cost_usd = worker.model.cost.estimate(input_tokens, output_tokens)
                response.provider = worker.model.provider
                response.model = worker.model.model_id
                return response

            self._record(worker.model, False, result.latency_ms / 1000.0)
            last_err = RuntimeError(
                f"{worker.worker_id}: {result.status.value}: {result.error}"
            )
            logger.debug("Worker failed %s: %s", worker.worker_id, result.error)

        raise RuntimeError(f"All {len(candidates)} workers failed. Last: {last_err}")

    async def list_models(self, account: str | None = None) -> list[ModelInfo]:
        if not self._initialized:
            await self.initialize()
        if account:
            return [m for m in self._models if m.account_name == account]
        return list(self._models)

    def stats(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for key, s in self._stats.items():
            total = s.successes + s.failures
            result[key] = {
                "successes": s.successes,
                "failures": s.failures,
                "success_rate": s.successes / total if total > 0 else 0.0,
            }
        return result
