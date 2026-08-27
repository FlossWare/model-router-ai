"""Integration tests for ProviderRouter worker failover."""

import pytest

from model_router_ai.router import ProviderRouter
from model_router_ai.types import ChatMessage, ChatResponse, ModelInfo
from model_router_ai.workers import WorkerStatus
from model_router_ai.providers import _BaseProvider


class FakeProvider(_BaseProvider):
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)

    async def discover_models(self, api_key):
        return [ModelInfo(model_id="test-model", provider=self.name, api_key=api_key)]

    async def call(self, model_info, messages, temperature=0.7, max_tokens=None):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ChatResponse(content=response, model=model_info.model_id, provider=self.name)


@pytest.mark.asyncio
async def test_router_uses_second_account_after_quota_failure():
    first = FakeProvider("test", ['HTTP 429: {"message":"free-models-per-day"}'])
    second = FakeProvider("test", ["SECOND_OK"])
    router = ProviderRouter()
    router.add_provider(first, "key-a", account_name="account-a")
    router.add_provider(second, "key-b", account_name="account-b")

    result = await router.chat([ChatMessage("user", "hello")])

    assert result.content == "SECOND_OK"
    assert router.worker_status()["test/account-a/test-model"]["available"] is False
    assert router.worker_status()["test/account-b/test-model"]["available"] is True
