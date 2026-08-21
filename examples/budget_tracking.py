"""Budget tracking with alert callbacks."""

import asyncio
import os

from model_router_ai import (
    BudgetGuard,
    ChatMessage,
    CostAware,
    GeminiProvider,
    OpenAICompatProvider,
    ProviderRouter,
)


def on_budget_alert(spent: float, max_budget: float) -> None:
    pct = spent / max_budget * 100
    print(f"ALERT: Budget at {pct:.0f}% (${spent:.2f} / ${max_budget:.2f})")


async def main() -> None:
    base = ProviderRouter()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        base.add_provider(GeminiProvider(), api_key=gemini_key)

    router = BudgetGuard(
        CostAware(base, prefer_free=True),
        max_monthly=50.0,
        alert_thresholds=[25, 50, 75, 90],
        on_alert=on_budget_alert,
    )

    await router.initialize()

    for i in range(5):
        resp = await router.chat([
            ChatMessage(role="user", content=f"Count to {i + 1}"),
        ])
        status = router.status
        print(
            f"Call {i + 1}: ${status.spent_usd:.4f} spent "
            f"({status.percent_used:.1f}% of ${status.max_usd})"
        )


if __name__ == "__main__":
    asyncio.run(main())
