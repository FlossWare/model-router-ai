# model-router-ai

Decorator-based LLM model router with cost-aware, budget-tracking, and policy-enforcing composable layers.

**Zero external dependencies** (stdlib only). Pluggable into any platform: loom-ai, Claude Code, Crush, Codex, or your own orchestration.

## Install

```bash
pip install git+https://github.com/FlossWare/model-router-ai.git
```

## Quick Start

```python
import asyncio
import os
from model_router_ai import (
    ProviderRouter,
    OpenAICompatProvider,
    GeminiProvider,
    CostAware,
    BudgetGuard,
    PolicyGuard,
    LatencyOptimizer,
    ThompsonSamplingSelector,
    ChatMessage,
)

async def main():
    # 1. Create the base router with providers
    base = ProviderRouter()
    base.add_provider(OpenAICompatProvider("groq"), api_key=os.environ["GROQ_API_KEY"])
    base.add_provider(GeminiProvider(), api_key=os.environ["GEMINI_API_KEY"])

    # 2. Stack decorators — each adds one concern
    router = PolicyGuard(
        BudgetGuard(
            CostAware(
                LatencyOptimizer(
                    ThompsonSamplingSelector(base)
                ),
                prefer_free=True,
            ),
            max_monthly=300.0,
        ),
        allowed=["gemini-*", "llama-*"],
    )

    await router.initialize()

    # 3. Use it
    response = await router.chat([
        ChatMessage(role="user", content="Explain the decorator pattern")
    ])
    print(response.content)

asyncio.run(main())
```

## Architecture: The Decorator Pattern

Each decorator wraps a `ModelRouter` and adds exactly one concern. Stack them to compose routing behavior:

```
PolicyGuard          → filters models by allowlist/blocklist
  └─ BudgetGuard     → tracks spending against $300/month cap
      └─ CostAware   → sorts models by cost, prefers cheaper
          └─ ThompsonSamplingSelector  → Bayesian explore/exploit
              └─ LatencyOptimizer      → tracks per-model latency
                  └─ ProviderRouter    → talks to LLM APIs
```

**Order matters.** PolicyGuard on the outside means "only consider allowed models, then optimize cost within those." Put CostAware outside ThompsonSampling to get "cheapest among proven models."

## Decorators

### CostAware

Sorts models by cost-per-token (cheapest first). Free models get priority by default.

```python
router = CostAware(base, max_cost_per_call=0.01, prefer_free=True)
```

### BudgetGuard

Tracks cumulative spending and enforces a monthly cap. Alerts at configurable thresholds.

```python
router = BudgetGuard(base, max_monthly=300.0, alert_thresholds=[50, 75, 90])

# Check status anytime
print(router.status)  # BudgetStatus(spent_usd=42.50, remaining_usd=257.50, ...)
print(router.status.percent_used)  # 14.2
```

Raises `BudgetExhaustedError` when the cap is reached.

### PolicyGuard

Filters models using glob patterns. Enforced on both explicit and auto-selected models.

```python
# Only allow specific model families
router = PolicyGuard(base, allowed=["gemini-*", "claude-*"])

# Block expensive models
router = PolicyGuard(base, blocked=["gpt-4o", "o1-*"])

# Filter by provider
router = PolicyGuard(base, allowed_providers=["google", "groq"])
```

Raises `PolicyViolationError` on blocked model access.

### LatencyOptimizer

Tracks per-model response times with a sliding window.

```python
router = LatencyOptimizer(base, window_size=20)

# After some calls:
print(router.fastest_models(top_n=5))
print(router.avg_latency("groq", "llama-3.3-70b"))
```

### ThompsonSamplingSelector

Bayesian explore/exploit for model selection. Balances trying new models with using proven ones.

```python
router = ThompsonSamplingSelector(base)

# After some calls:
print(await router.performance())
# {'gemini-2.5-flash': {'alpha': 15.0, 'beta': 2.0, 'mean': 0.88, 'trials': 15}, ...}
```

## Providers

| Provider | Class | Free Tier |
|----------|-------|-----------|
| Groq | `OpenAICompatProvider("groq")` | Yes |
| OpenRouter | `OpenAICompatProvider("openrouter")` | Yes (with `free_only=True`) |
| Cerebras | `OpenAICompatProvider("cerebras")` | Yes |
| DeepInfra | `OpenAICompatProvider("deepinfra")` | Yes |
| NVIDIA | `OpenAICompatProvider("nvidia")` | Yes |
| Google Gemini | `GeminiProvider()` | Yes |
| Cohere | `CohereProvider()` | Yes |
| OpenAI | `OpenAICompatProvider("openai")` | No |
| Vertex AI | `VertexAIProvider(project_id="...")` | No |

### Adding a custom provider

Any OpenAI-compatible API works:

```python
base.add_provider(
    OpenAICompatProvider("my-provider", base_url="https://my-api.com/v1"),
    api_key="sk-..."
)
```

## Cost Data

Built-in cost metadata for popular models (USD per 1M tokens):

| Model | Input | Output |
|-------|-------|--------|
| Gemini 2.5 Flash | $0.15 | $0.60 |
| GPT-4o Mini | $0.15 | $0.60 |
| Claude Haiku 4.5 | $0.80 | $4.00 |
| O3-mini | $1.10 | $4.40 |
| Gemini 2.5 Pro | $1.25 | $10.00 |
| GPT-4o | $2.50 | $10.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Opus 4.6 | $15.00 | $75.00 |

Custom costs:

```python
from model_router_ai import ModelCost

base.add_provider(
    OpenAICompatProvider("openai", cost_map={
        "gpt-4o": ModelCost(input_per_1m=2.50, output_per_1m=10.0),
    }),
    api_key="sk-..."
)
```

## Selection Strategies

The base `ProviderRouter` accepts a pluggable selection strategy:

```python
from model_router_ai import ProviderRouter, CascadeStrategy

router = ProviderRouter(strategy=CascadeStrategy(preferred=["gemini-2.5-flash"]))
```

| Strategy | Behavior |
|----------|----------|
| `ThompsonSamplingStrategy` | Bayesian explore/exploit (default) |
| `RoundRobinStrategy` | Even spread across endpoints |
| `LatencyWeightedStrategy` | Prefer faster endpoints |
| `CascadeStrategy` | Try preferred models first |

## Protocol Delegation

BudgetGuard and ThompsonSamplingSelector can delegate to external implementations via injectable protocols. This lets packages like [budget-ai](https://github.com/FlossWare/budget-ai) and [strategy-ai](https://github.com/FlossWare/strategy-ai) provide rich implementations without model-router-ai depending on them.

### UsageTracker Protocol

Inject a custom budget tracker into BudgetGuard:

```python
from model_router_ai import BudgetGuard, UsageTracker

class MyTracker:  # satisfies UsageTracker protocol via structural subtyping
    async def record_usage(self, model, cost_usd, usage=None): ...
    async def is_exceeded(self): ...
    async def get_status(self): ...
    def reset(self): ...

router = BudgetGuard(base, tracker=MyTracker())
```

Without an injected tracker, BudgetGuard uses a built-in `_SimpleBudgetTracker`.

### ModelSelector Protocol

Inject a custom model selection strategy into ThompsonSamplingSelector:

```python
from model_router_ai import ThompsonSamplingSelector, ModelSelector

class MySelector:  # satisfies ModelSelector protocol
    async def select(self, candidates: list[str]) -> str: ...
    async def record(self, model_id: str, success: bool) -> None: ...
    async def stats(self) -> dict: ...

router = ThompsonSamplingSelector(base, selector=MySelector())
```

### Adapters for budget-ai / strategy-ai

Pre-built adapters bridge the signature differences between model-router-ai's protocols and sibling FlossWare packages:

```python
from model_router_ai.adapters import BudgetAIAdapter, StrategyAIAdapter

# budget-ai integration
from budget_ai import InMemoryBudgetTracker
router = BudgetGuard(base, tracker=BudgetAIAdapter(InMemoryBudgetTracker(max_cost=300.0)))

# strategy-ai integration
from strategy_ai import ThompsonSamplingSelector as TSSelector
router = ThompsonSamplingSelector(base, selector=StrategyAIAdapter(TSSelector()))
```

Both adapters are optional — model-router-ai works without budget-ai or strategy-ai installed. Error handling is fail-open by default (`fail_open=True` on BudgetGuard).

## Integrations

Ready-to-use integration code lives in `integrations/`:

| Integration | Location | Description |
|---|---|---|
| **MCP Server** | `integrations/mcp/` | Exposes `chat`, `multi_model_chat`, `list_models`, `budget_status`, `model_performance` as MCP tools |
| **CLI** | `integrations/cli/` | `mr-chat` command with `--free-only`, `--json`, `--list-models`, `--status` |
| **Skills** | `integrations/skills/` | Claude Code `/multi-model-query`, `/budget-check`, `/model-stats` |
| **CLAUDE.md** | `integrations/claude-code/` | Paste-ready snippet for project integration |

See `integrations/README.md` and `INTEGRATIONS.md` for setup details.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT
