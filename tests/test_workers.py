"""Worker fabric regression tests."""

import asyncio

import pytest

from model_router_ai.providers import OpenAICompatProvider, _BaseProvider
from model_router_ai.router import ProviderRouter
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo
from model_router_ai.workers import ModelWorker, WorkerStatus, classify_failure


def test_classifies_openrouter_daily_quota_and_reset():
    error = (
        'HTTP 429: {"error":{"message":"Rate limit exceeded: free-models-per-day",'
        '"metadata":{"headers":{"X-RateLimit-Reset":"1787875200000"}}}}'
    )
    status, retry_after, reset = classify_failure(error)
    assert status is WorkerStatus.QUOTA_EXHAUSTED
    assert retry_after is None
    assert reset == 1787875200.0


def test_classifies_auth_failure():
    status, _, _ = classify_failure("HTTP 401: unauthorized")
    assert status is WorkerStatus.AUTH_FAILED


def test_classifies_timeout_and_network_failures():
    assert classify_failure("timeout while calling model")[0] is WorkerStatus.TIMEOUT
    assert classify_failure("network connection reset")[0] is WorkerStatus.NETWORK_ERROR


def test_worker_identity_is_account_specific():
    provider = OpenAICompatProvider("openrouter")
    model = ModelInfo(
        model_id="qwen/test",
        provider="openrouter",
        account_name="ncrr",
    )
    worker = ModelWorker(provider, model, "secret")
    assert worker.id == "openrouter/ncrr/qwen/test"
    assert worker.available()


def test_quota_worker_becomes_unavailable_until_reset():
    provider = OpenAICompatProvider("openrouter")
    model = ModelInfo(
        model_id="qwen/test",
        provider="openrouter",
        account_name="ncrr",
    )
    worker = ModelWorker(provider, model, "secret")
    worker.mark_unavailable(WorkerStatus.QUOTA_EXHAUSTED, reset=1787875200.0)
    assert not worker.available(now=1787875199.0)
    assert worker.available(now=1787875200.1)


def test_auth_worker_is_quarantined():
    provider = OpenAICompatProvider("openrouter")
    model = ModelInfo(
        model_id="qwen/test",
        provider="openrouter",
        account_name="ncrr",
    )
    worker = ModelWorker(provider, model, "secret")
    worker.mark_unavailable(WorkerStatus.AUTH_FAILED)
    assert not worker.available()
    assert worker.last_status is WorkerStatus.AUTH_FAILED


class FakeProvider(_BaseProvider):
    def __init__(self, name, models, failures=None):
        self.name = name
        self.models = models
        self.failures = list(failures or [])
        self.calls = 0

    async def discover_models(self, api_key):
        return [
            ModelInfo(model_id=model, provider=self.name, api_key=api_key)
            for model in self.models
        ]

    async def call(
        self,
        model_info,
        messages,
        temperature=0.7,
        max_tokens=None,
    ):
        self.calls += 1
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        return ChatResponse(
            content="BASELINE_OK",
            model=model_info.model_id,
            provider=self.name,
        )


def test_router_fails_over_between_accounts():
    exhausted = FakeProvider(
        "openrouter",
        ["qwen/test"],
        ["HTTP 429 quota exhausted"],
    )
    healthy = FakeProvider("openrouter", ["qwen/test"])
    router = ProviderRouter()
    router.add_provider(exhausted, "key-a", "flossware")
    router.add_provider(healthy, "key-b", "ncrr")
    asyncio.run(router.initialize())

    response = asyncio.run(
        router.chat([ChatMessage("user", "test")], model="qwen/test")
    )
    assert response.content == "BASELINE_OK"
    assert exhausted.calls == 1
    assert healthy.calls == 1
    assert not router.worker_status()["openrouter/flossware/qwen/test"]["available"]
    assert router.worker_status()["openrouter/ncrr/qwen/test"]["available"]


def test_router_all_workers_exhausted():
    first = FakeProvider(
        "openrouter",
        ["qwen/test"],
        ["HTTP 429 quota exhausted"],
    )
    second = FakeProvider(
        "openrouter",
        ["qwen/test"],
        ["HTTP 429 quota exhausted"],
    )
    router = ProviderRouter()
    router.add_provider(first, "key-a", "flossware")
    router.add_provider(second, "key-b", "ncrr")
    asyncio.run(router.initialize())

    with pytest.raises(RuntimeError, match="All 2 workers failed"):
        asyncio.run(router.chat([ChatMessage("user", "test")], model="qwen/test"))
    assert first.calls == 1
    assert second.calls == 1
    assert all(not item["available"] for item in router.worker_status().values())
