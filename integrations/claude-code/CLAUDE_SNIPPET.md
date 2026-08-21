## Model Routing (model-router-ai)

Zero-dependency Python package (stdlib only) for cost-aware, policy-controlled LLM routing with decorator-based composition.

### Install

```bash
pip install "git+https://github.com/FlossWare/model-router-ai.git"
```

### Quick Start

```python
import os
from model_router_ai import (
    # Core
    ProviderRouter, ChatMessage, ChatResponse, ModelRouter,
    # Providers
    OpenAICompatProvider, GeminiProvider, CohereProvider, VertexAIProvider,
    # Decorators (stack these to compose behavior)
    CostAware, BudgetGuard, PolicyGuard, LatencyOptimizer, ThompsonSamplingSelector,
    # Strategies
    ThompsonSamplingStrategy, RoundRobinStrategy, LatencyWeightedStrategy, CascadeStrategy,
    # Types
    BudgetStatus, ModelInfo, ModelCost,
)

# 1. Create base router and add providers
router = ProviderRouter()
router.add_provider(OpenAICompatProvider("openrouter", free_only=True), api_key=os.environ["OPENROUTER_API_KEY"])
router.add_provider(GeminiProvider(), api_key=os.environ["GEMINI_API_KEY"])
router.add_provider(OpenAICompatProvider("groq"), api_key=os.environ["GROQ_API_KEY"])

# 2. Stack decorators (outermost applied first)
routed = PolicyGuard(
    BudgetGuard(
        LatencyOptimizer(
            CostAware(
                ThompsonSamplingSelector(router),
                prefer_free=True,
            ),
        ),
        max_monthly=300.0,
    ),
    allowed=["gemini-*", "llama-*", "command-*"],
)

# 3. Chat (auto-initializes on first call)
resp = await routed.chat([ChatMessage(role="user", content="Hello")])
# resp.content, resp.model, resp.provider, resp.latency_ms, resp.cost_usd
```

### Decorator Reference

| Decorator                  | Purpose                                      | Key params                                   |
|----------------------------|----------------------------------------------|----------------------------------------------|
| `CostAware`                | Sorts models by cost, prefers cheapest        | `max_cost_per_call`, `prefer_free`           |
| `BudgetGuard`              | Enforces monthly spending cap                 | `max_monthly`, `alert_thresholds`, `on_alert`|
| `PolicyGuard`              | Allowlist/blocklist models by glob pattern    | `allowed`, `blocked`, `allowed_providers`    |
| `LatencyOptimizer`         | Tracks latency, exposes fastest_models()      | `window_size`                                |
| `ThompsonSamplingSelector` | Bayesian explore/exploit model selection       | `quality_threshold`                          |

Decorators satisfy the `ModelRouter` protocol and are freely composable in any order.

### Provider Reference

| Provider              | API style          | Constructor                                      |
|-----------------------|--------------------|--------------------------------------------------|
| `OpenAICompatProvider`| OpenAI-compatible   | `OpenAICompatProvider("groq")` (known bases: groq, openrouter, cerebras, deepinfra, nvidia, openai) |
| `GeminiProvider`      | Google Gemini       | `GeminiProvider()`                               |
| `CohereProvider`      | Cohere v2           | `CohereProvider()`                               |
| `VertexAIProvider`    | Google Vertex AI    | `VertexAIProvider(project_id="...", region="...")` |

### Available Skills

If you have the model-router-ai skills installed (from `integrations/skills/`):

- `/multi-model-query` -- Send a query to 3-5 models and get a consensus answer with latency/cost comparison
- `/budget-check` -- Check current spending against monthly budget cap
- `/model-stats` -- Show Thompson Sampling bandit statistics and latency rankings for all routed models

### MCP Server

An MCP server is available at `integrations/mcp/` for tool-based access to model-router-ai from any MCP-compatible client.
