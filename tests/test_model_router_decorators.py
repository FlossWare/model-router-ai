"""Tests for the standalone model_router_ai decorator pattern.

Tests each decorator independently and composed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_router_ai.types import ChatMessage, ChatResponse, ModelCost, ModelInfo
from model_router_ai.decorators import (
    BudgetExhaustedError,
    BudgetGuard,
    CostAware,
    LatencyOptimizer,
    PolicyGuard,
    PolicyViolationError,
    ThompsonSamplingSelector,
)


class FakeRouter:
    """Minimal router for testing decorators in isolation."""

    def __init__(
        self,
        models: list[ModelInfo] | None = None,
        response: ChatResponse | None = None,
        fail_models: set[str] | None = None,
    ) -> None:
        self._models = models or []
        self._response = response or ChatResponse(content="ok", model="test", provider="fake")
        self._fail_models = fail_models or set()
        self.call_log: list[dict] = []

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.call_log.append({"model": model, "temperature": temperature})
        if model and model in self._fail_models:
            raise RuntimeError(f"Model {model} failed")
        resp = ChatResponse(
            content=self._response.content,
            model=model or self._response.model,
            provider=self._response.provider,
            usage=self._response.usage,
            latency_ms=self._response.latency_ms,
            cost_usd=self._response.cost_usd,
        )
        return resp


# -- CostAware tests ----------------------------------------------------------


class TestCostAware:
    def test_prefers_cheaper_models(self):
        models = [
            ModelInfo(model_id="expensive", provider="a", cost=ModelCost(input_per_1m=30.0, output_per_1m=60.0)),
            ModelInfo(model_id="cheap", provider="b", cost=ModelCost(input_per_1m=0.5, output_per_1m=1.0)),
            ModelInfo(model_id="free", provider="c", cost=ModelCost(input_per_1m=0.0, output_per_1m=0.0)),
        ]
        fake = FakeRouter(models=models)
        router = CostAware(fake)

        asyncio.run(router.chat([ChatMessage(role="user", content="hello")]))
        assert fake.call_log[0]["model"] == "free"

    def test_skips_too_expensive(self):
        models = [
            ModelInfo(model_id="expensive", provider="a", cost=ModelCost(input_per_1m=1000.0, output_per_1m=2000.0)),
            ModelInfo(model_id="cheap", provider="b", cost=ModelCost(input_per_1m=0.1, output_per_1m=0.2)),
        ]
        fake = FakeRouter(models=models)
        router = CostAware(fake, max_cost_per_call=0.001)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        assert fake.call_log[0]["model"] == "cheap"

    def test_passthrough_when_model_specified(self):
        fake = FakeRouter()
        router = CostAware(fake)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")], model="specific"))
        assert fake.call_log[0]["model"] == "specific"


# -- BudgetGuard tests --------------------------------------------------------


class TestBudgetGuard:
    def test_tracks_spending(self):
        fake = FakeRouter(response=ChatResponse(content="ok", cost_usd=10.0))
        router = BudgetGuard(fake, max_monthly=100.0)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        assert router.status.spent_usd == 10.0
        assert router.status.remaining_usd == 90.0

    def test_blocks_when_exhausted(self):
        fake = FakeRouter(response=ChatResponse(content="ok", cost_usd=50.0))
        router = BudgetGuard(fake, max_monthly=100.0)

        asyncio.run(router.chat([ChatMessage(role="user", content="1")]))
        asyncio.run(router.chat([ChatMessage(role="user", content="2")]))

        with pytest.raises(BudgetExhaustedError):
            asyncio.run(router.chat([ChatMessage(role="user", content="3")]))

    def test_alerts_at_thresholds(self):
        alerts: list[tuple[float, float]] = []
        fake = FakeRouter(response=ChatResponse(content="ok", cost_usd=30.0))
        router = BudgetGuard(
            fake,
            max_monthly=100.0,
            alert_thresholds=[25.0, 50.0, 75.0],
            on_alert=lambda spent, max_: alerts.append((spent, max_)),
        )

        asyncio.run(router.chat([ChatMessage(role="user", content="1")]))
        assert len(alerts) == 1  # 30% > 25%

        asyncio.run(router.chat([ChatMessage(role="user", content="2")]))
        assert len(alerts) == 2  # 60% > 50%

    def test_reset(self):
        fake = FakeRouter(response=ChatResponse(content="ok", cost_usd=10.0))
        router = BudgetGuard(fake, max_monthly=100.0)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        assert router.status.spent_usd == 10.0

        router.reset()
        assert router.status.spent_usd == 0.0

    def test_percent_used(self):
        fake = FakeRouter(response=ChatResponse(content="ok", cost_usd=75.0))
        router = BudgetGuard(fake, max_monthly=300.0)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        assert router.status.percent_used == 25.0


# -- PolicyGuard tests --------------------------------------------------------


class TestPolicyGuard:
    def test_allowlist_filters_models(self):
        models = [
            ModelInfo(model_id="gemini-2.5-flash", provider="google"),
            ModelInfo(model_id="gpt-4o", provider="openai"),
            ModelInfo(model_id="claude-sonnet-5", provider="anthropic"),
        ]
        fake = FakeRouter(models=models)
        router = PolicyGuard(fake, allowed=["gemini-*", "claude-*"])

        filtered = asyncio.run(router.list_models())
        ids = [m.model_id for m in filtered]
        assert "gemini-2.5-flash" in ids
        assert "claude-sonnet-5" in ids
        assert "gpt-4o" not in ids

    def test_blocklist_excludes_models(self):
        models = [
            ModelInfo(model_id="gemini-2.5-flash", provider="google"),
            ModelInfo(model_id="gpt-4o", provider="openai"),
        ]
        fake = FakeRouter(models=models)
        router = PolicyGuard(fake, blocked=["gpt-*"])

        filtered = asyncio.run(router.list_models())
        ids = [m.model_id for m in filtered]
        assert "gemini-2.5-flash" in ids
        assert "gpt-4o" not in ids

    def test_provider_allowlist(self):
        models = [
            ModelInfo(model_id="model-a", provider="google"),
            ModelInfo(model_id="model-b", provider="openai"),
        ]
        fake = FakeRouter(models=models)
        router = PolicyGuard(fake, allowed_providers=["google"])

        filtered = asyncio.run(router.list_models())
        assert len(filtered) == 1
        assert filtered[0].provider == "google"

    def test_rejects_blocked_model_by_name(self):
        models = [ModelInfo(model_id="gpt-4o", provider="openai")]
        fake = FakeRouter(models=models)
        router = PolicyGuard(fake, blocked=["gpt-*"])

        with pytest.raises(PolicyViolationError):
            asyncio.run(router.chat([ChatMessage(role="user", content="hi")], model="gpt-4o"))


# -- LatencyOptimizer tests ---------------------------------------------------


class TestLatencyOptimizer:
    def test_tracks_latency(self):
        fake = FakeRouter(
            response=ChatResponse(content="ok", model="fast", provider="p", latency_ms=50.0)
        )
        router = LatencyOptimizer(fake)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        avg = router.avg_latency("p", "fast")
        assert avg == 50.0

    def test_fastest_models(self):
        router = LatencyOptimizer(FakeRouter())
        router._latencies = {
            "a/slow": [200.0, 180.0],
            "b/fast": [30.0, 40.0],
            "c/mid": [100.0, 110.0],
        }

        fastest = router.fastest_models(top_n=2)
        assert fastest[0][0] == "b/fast"
        assert len(fastest) == 2


# -- ThompsonSamplingSelector tests -------------------------------------------


class TestThompsonSampling:
    def test_selects_from_available(self):
        models = [
            ModelInfo(model_id="a", provider="p1"),
            ModelInfo(model_id="b", provider="p2"),
        ]
        fake = FakeRouter(models=models)
        router = ThompsonSamplingSelector(fake)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        assert fake.call_log[0]["model"] in ("a", "b")

    def test_passthrough_when_model_specified(self):
        fake = FakeRouter()
        router = ThompsonSamplingSelector(fake)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")], model="specific"))
        assert fake.call_log[0]["model"] == "specific"

    def test_records_success_failure(self):
        models = [ModelInfo(model_id="m1", provider="p")]
        fake = FakeRouter(models=models)
        router = ThompsonSamplingSelector(fake)

        asyncio.run(router.chat([ChatMessage(role="user", content="hi")]))
        perf = router.performance()
        assert "m1" in perf
        assert perf["m1"]["trials"] == 1


# -- Composition tests --------------------------------------------------------


class TestComposition:
    def test_full_stack(self):
        """PolicyGuard -> BudgetGuard -> CostAware -> base."""
        models = [
            ModelInfo(
                model_id="gemini-flash", provider="google",
                cost=ModelCost(input_per_1m=0.0, output_per_1m=0.0),
            ),
            ModelInfo(
                model_id="gpt-4o", provider="openai",
                cost=ModelCost(input_per_1m=5.0, output_per_1m=15.0),
            ),
        ]
        fake = FakeRouter(models=models, response=ChatResponse(content="ok", cost_usd=0.0))

        router = PolicyGuard(
            BudgetGuard(
                CostAware(fake),
                max_monthly=300.0,
            ),
            allowed=["gemini-*"],
        )

        asyncio.run(router.initialize())
        filtered = asyncio.run(router.list_models())
        assert len(filtered) == 1
        assert filtered[0].model_id == "gemini-flash"

    def test_decorator_order_matters(self):
        """Budget check happens before cost sorting."""
        models = [
            ModelInfo(model_id="m1", provider="p", cost=ModelCost(input_per_1m=1.0, output_per_1m=1.0)),
        ]
        fake = FakeRouter(models=models, response=ChatResponse(content="ok", cost_usd=200.0))

        router = BudgetGuard(CostAware(fake), max_monthly=300.0)

        asyncio.run(router.chat([ChatMessage(role="user", content="1")]))
        asyncio.run(router.chat([ChatMessage(role="user", content="2")]))

        with pytest.raises(BudgetExhaustedError):
            asyncio.run(router.chat([ChatMessage(role="user", content="3")]))
