"""Worker fabric regression tests."""

from model_router_ai.workers import ModelWorker, WorkerStatus, classify_failure
from model_router_ai.types import ModelInfo
from model_router_ai.providers import OpenAICompatProvider


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


def test_worker_identity_is_account_specific():
    provider = OpenAICompatProvider("openrouter")
    model = ModelInfo(model_id="qwen/test", provider="openrouter", account_name="ncrr")
    worker = ModelWorker(provider, model, "secret")
    assert worker.id == "openrouter/ncrr/qwen/test"
    assert worker.available()


def test_quota_worker_becomes_unavailable_until_reset():
    provider = OpenAICompatProvider("openrouter")
    model = ModelInfo(model_id="qwen/test", provider="openrouter", account_name="ncrr")
    worker = ModelWorker(provider, model, "secret")
    worker.mark_unavailable(WorkerStatus.QUOTA_EXHAUSTED, reset=1787875200.0)
    assert not worker.available(now=1787875199.0)
    assert worker.available(now=1787875200.1)
