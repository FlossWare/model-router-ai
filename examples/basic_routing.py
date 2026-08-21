"""Basic model routing with cost-aware selection and budget tracking."""

import asyncio
import os

from model_router_ai import (
    BudgetGuard,
    ChatMessage,
    CostAware,
    GeminiProvider,
    OpenAICompatProvider,
    PolicyGuard,
    ProviderRouter,
    ThompsonSamplingSelector,
)


async def main() -> None:
    base = ProviderRouter()

    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if groq_key:
        base.add_provider(OpenAICompatProvider("groq"), api_key=groq_key)
    if gemini_key:
        base.add_provider(GeminiProvider(), api_key=gemini_key)

    router = PolicyGuard(
        BudgetGuard(
            CostAware(
                ThompsonSamplingSelector(base),
                prefer_free=True,
            ),
            max_monthly=300.0,
        ),
        allowed=["gemini-*", "llama-*"],
    )

    await router.initialize()

    models = await router.list_models()
    print(f"Available models: {len(models)}")
    for m in models[:5]:
        print(f"  {m.provider}/{m.model_id}")

    response = await router.chat([
        ChatMessage(role="user", content="What is the decorator pattern?"),
    ])

    print(f"\nModel: {response.provider}/{response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Cost: ${response.cost_usd:.6f}")
    print(f"\n{response.content[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
