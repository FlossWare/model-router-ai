import time

import pytest

from model_router_ai import ChatMessage, ChatResponse, ModelInfo, ProviderRouter
from model_router_ai.providers import _BaseProvider
from model_router_ai.workers import (
    Arbiter,
    Worker,
    WorkerPool,
    WorkerStatus,
    classify_backend_error,
)


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

    async def call(self, model_info, messages, temperature=0.7, max_tokens=None):
        self.calls += 1
        if self.failures:
            failure = self.failures.pop(0)
            raise RuntimeError(failure)
        return ChatResponse(content="ok", model=model_info.model_id, provider=self.name)


def worker(provider, account, model):
    return Worker(
        provider=provider,
        account_name=account,
        model=ModelInfo(model_id=model, provider=provider.name, api_key="secret"),
    )


def test_worker_identity_is_provider_account_model():
    provider = FakeProvider("openrouter", ["qwen"])
    item = worker(provider, "flossware", "qwen")
    assert item.worker_id == "openrouter/flossware/qwen"


def test_worker_pool_filters_account_provider_and_model():
    p1 = FakeProvider("openrouter", ["qwen"])
    p2 = FakeProvider("groq", ["qwen"])
    pool = WorkerPool([worker(p1, "a", "qwen"), worker(p2, "b", "qwen")])
    assert [w.worker_id for w in pool.available(account="a")] == ["openrouter/a/qwen"]
    assert [w.worker_id for w in pool.available(provider="groq")] == ["groq/b/qwen"]
    assert [w.worker_id for w in pool.available(model="qwen")] == [
        "openrouter/a/qwen",
        "groq/b/qwen",
    ]


def test_arbiter_selects_available_worker():
    provider = FakeProvider("local", ["qwen"])
    first = worker(provider, "a", "qwen")
    second = worker(provider, "b", "qwen")
    arbiter = Arbiter(WorkerPool([first, second]), scorer=lambda w: 1.0 if w.account_name == "b" else 0.0)
    assert arbiter.select().account_name == "b"


def test_quota_error_is_classified_and_reset_is_honored():
    reset = int((time.time() + 60) * 1000)
    status, retry_after, quota_reset = classify_backend_error(
        RuntimeError(f"HTTP 429 free-models-per-day x-ratelimit-reset={reset}")
    )
    assert status is WorkerStatus.QUOTA_EXHAUSTED
    assert retry_after is None
    assert quota_reset is not None
    assert quota_reset >= time.time() + 50


@pytest.mark.asyncio
async def test_worker_marks_quota_exhausted_and_does_not_retry():
    provider = FakeProvider("openrouter", ["qwen"], ["HTTP 429 quota exhausted"])
    item = worker(provider, "a", "qwen")
    result = await item.execute([ChatMessage("user", "test")])
    assert result.status is WorkerStatus.QUOTA_EXHAUSTED
    assert item.available is False
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_router_fails_over_between_accounts():
    exhausted = FakeProvider("openrouter", ["qwen"], ["HTTP 429 quota exhausted"])
    healthy = FakeProvider("openrouter", ["qwen"])
    router = ProviderRouter()
    router.add_provider(exhausted, "key-a", "flossware")
    router.add_provider(healthy, "key-b", "ncrr")
    await router.initialize()

    response = await router.chat([ChatMessage("user", "test")], model="qwen")
    assert response.content == "ok"
    assert exhausted.calls == 1
    assert healthy.calls == 1
    assert not router.worker_pool.available(account="flossware")
    assert router.worker_pool.available(account="ncrr")


@pytest.mark.asyncio
async def test_router_all_workers_unavailable():
    p1 = FakeProvider("openrouter", ["qwen"], ["HTTP 429 quota exhausted"])
    p2 = FakeProvider("openrouter", ["qwen"], ["HTTP 429 quota exhausted"])
    router = ProviderRouter()
    router.add_provider(p1, "key-a", "flossware")
    router.add_provider(p2, "key-b", "ncrr")
    await router.initialize()

    with pytest.raises(RuntimeError, match="No model endpoints available"):
        await router.chat([ChatMessage("user", "test")], model="qwen")
    assert p1.calls == 1
    assert p2.calls == 1


@pytest.mark.asyncio
async def test_router_with_zero_workers_is_valid():
    router = ProviderRouter()
    await router.initialize()
    assert router.worker_pool.all() == []
    with pytest.raises(RuntimeError, match="No model endpoints available"):
        await router.chat([ChatMessage("user", "test")])
