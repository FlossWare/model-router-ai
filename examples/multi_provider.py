"""Multi-provider routing with OpenRouter free models."""

import asyncio
import os

from model_router_ai import (
    ChatMessage,
    CohereProvider,
    CostAware,
    GeminiProvider,
    LatencyOptimizer,
    OpenAICompatProvider,
    ProviderRouter,
)


async def main() -> None:
    base = ProviderRouter()

    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    cohere_key = os.environ.get("COHERE_API_KEY", "")

    if or_key:
        base.add_provider(
            OpenAICompatProvider("openrouter", free_only=True),
            api_key=or_key,
        )
    if gemini_key:
        base.add_provider(GeminiProvider(), api_key=gemini_key)
    if cohere_key:
        base.add_provider(CohereProvider(), api_key=cohere_key)

    router = LatencyOptimizer(CostAware(base, prefer_free=True))

    await router.initialize()

    messages = [ChatMessage(role="user", content="Hello, who are you?")]

    for i in range(3):
        resp = await router.chat(messages)
        print(f"Call {i + 1}: {resp.provider}/{resp.model} ({resp.latency_ms:.0f}ms)")

    print("\nFastest models:")
    for key, avg in router.fastest_models(top_n=3):
        print(f"  {key}: {avg:.0f}ms avg")


if __name__ == "__main__":
    asyncio.run(main())
