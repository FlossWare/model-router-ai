# model-router-ai Integration Guide

Install:
```bash
pip install "git+https://github.com/FlossWare/model-router-ai.git"
```

---

## Claude Code

### CLAUDE.md snippet

```markdown
## Model Routing (model-router-ai)

This project uses `model-router-ai` for cost-aware, policy-controlled LLM routing.
Stack decorators to compose routing behavior (budget limits, model allowlists, latency optimization).

```python
from model_router import (
    CostAware, BudgetGuard, PolicyGuard, LatencyOptimizer,
    ProviderRouter, OpenAICompatProvider,
)

provider = OpenAICompatProvider(
    name="openrouter", base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    models=["meta-llama/llama-4-maverick", "google/gemini-2.5-flash"],
)

router = PolicyGuard(
    BudgetGuard(
        CostAware(ProviderRouter(providers=[provider])),
        max_monthly=300.0,
    ),
    allowed_models=["meta-llama/*", "google/*"],
)

response = await router.chat(messages)
```
```

### Skills

Create `.claude/skills/budget-check.md`:
```markdown
---
name: budget-check
description: Check current LLM spending against monthly budget
---

Use model-router-ai's BudgetGuard to report current spending status.
Show: spent so far, remaining budget, projected end-of-month total.
```

---

## OpenAI Codex

### AGENTS.md snippet
```markdown
## Model Routing

Use `model-router-ai` for routing LLM calls across providers:

```python
from model_router import (
    CostAware, BudgetGuard, PolicyGuard,
    ProviderRouter, OpenAICompatProvider,
)

router = PolicyGuard(
    BudgetGuard(CostAware(ProviderRouter(providers=[...])), max_monthly=300.0),
    allowed_models=["claude-*", "gemini-*"],
)
response = await router.chat(messages)
```

Decorators stack: each adds one concern (cost tracking, budget limits, policy).
```

---

## Cursor

### .cursorrules snippet
```
Use model-router-ai for LLM routing:
- ProviderRouter: base router with provider backends
- CostAware: wraps router to track per-call costs
- BudgetGuard: enforces monthly spending limits
- PolicyGuard: restricts which models can be used (glob patterns)
- LatencyOptimizer: prefers lowest-latency models
- ThompsonSamplingSelector: bandit-based model selection

Decorator stacking order (outermost first): PolicyGuard > BudgetGuard > CostAware > base
Import from: from model_router import CostAware, BudgetGuard, PolicyGuard, ProviderRouter
Package: pip install "git+https://github.com/FlossWare/model-router-ai.git"
Zero dependencies — stdlib only.
```

---

## Crush

### Configuration
```python
from crush import Agent
from model_router import (
    CostAware, BudgetGuard, PolicyGuard,
    ProviderRouter, OpenAICompatProvider,
)

class RoutedAgent(Agent):
    def __init__(self):
        provider = OpenAICompatProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            models=["meta-llama/llama-4-maverick"],
        )
        self.router = PolicyGuard(
            BudgetGuard(
                CostAware(ProviderRouter(providers=[provider])),
                max_monthly=300.0,
            ),
            allowed_models=["meta-llama/*"],
        )

    async def call_model(self, prompt: str) -> str:
        resp = await self.router.chat([{"role": "user", "content": prompt}])
        return resp.content
```

---

## Generic Python Agent

### Full usage patterns
```python
import os
from model_router import (
    CostAware, BudgetGuard, PolicyGuard, LatencyOptimizer,
    ThompsonSamplingSelector,
    ProviderRouter, OpenAICompatProvider, GeminiProvider,
    ThompsonSamplingStrategy, CascadeStrategy,
    ChatMessage, ChatResponse, BudgetStatus,
)

# 1. Set up providers
openrouter = OpenAICompatProvider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    models=["meta-llama/llama-4-maverick", "nvidia/nemotron-super-49b"],
)
gemini = GeminiProvider(
    api_key=os.environ["GOOGLE_API_KEY"],
    models=["gemini-2.5-flash", "gemini-2.5-pro"],
)

# 2. Stack decorators
router = PolicyGuard(
    BudgetGuard(
        LatencyOptimizer(
            CostAware(ProviderRouter(providers=[openrouter, gemini])),
        ),
        max_monthly=300.0,
    ),
    allowed_models=["meta-llama/*", "gemini-*"],
)

# 3. Chat
messages = [ChatMessage(role="user", content="Hello")]
response: ChatResponse = await router.chat(messages)
print(response.content)

# 4. Check budget
status: BudgetStatus = await router.budget_status()
print(f"Spent: ${status.spent:.2f} / ${status.limit:.2f}")

# 5. Bandit-based selection
bandit_router = ThompsonSamplingSelector(
    ProviderRouter(providers=[openrouter, gemini]),
    strategy=ThompsonSamplingStrategy(),
)
response = await bandit_router.chat(messages)

# 6. Cascade (try cheap model first, fall back to expensive)
cascade = CascadeStrategy(
    models=["gemini-2.5-flash", "gemini-2.5-pro"],
    quality_threshold=0.7,
)
```

---

## Cross-Package Integration

### With resilience-ai + security-ai + observability-ai
```python
from resilience_ai import with_retry, with_circuit_breaker
from observability_ai import track_execution, ExecutionTelemetry
from security_ai import mask_secrets
from model_router import CostAware, BudgetGuard, ProviderRouter

telemetry = ExecutionTelemetry()

@with_retry(max_attempts=3)
@with_circuit_breaker(provider="openrouter", max_failures=5)
@track_execution(telemetry=telemetry, provider="openrouter")
@mask_secrets(patterns=[r"(sk-)[a-zA-Z0-9]+"])
async def routed_call(prompt: str) -> str:
    router = BudgetGuard(
        CostAware(ProviderRouter(providers=[...])),
        max_monthly=300.0,
    )
    resp = await router.chat([{"role": "user", "content": prompt}])
    return resp.content
```

### With consensus-ai
```python
from consensus_ai import with_consensus
from model_router import ProviderRouter, OpenAICompatProvider

provider = OpenAICompatProvider(...)
router = ProviderRouter(providers=[provider])

@with_consensus(models=["llama-4-maverick", "gemini-flash", "command-a"])
async def consensus_call(prompt: str, model: str = "", **kwargs) -> str:
    resp = await router.chat(
        [{"role": "user", "content": prompt}],
        model=model,
    )
    return resp.content
```
