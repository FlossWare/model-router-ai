"""MCP server for model-router-ai.

Exposes LLM routing, multi-model consensus, budget tracking, and
model performance as MCP tools for Claude Code and other MCP clients.

Usage:
    python3 integrations/mcp/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Allow running without pip install -- add package root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastmcp import FastMCP

from model_router_ai import (
    BudgetGuard,
    ChatMessage,
    CohereProvider,
    CostAware,
    GeminiProvider,
    LatencyOptimizer,
    OpenAICompatProvider,
    PolicyGuard,
    ProviderRouter,
    ThompsonSamplingSelector,
)
from model_router_ai.types import ModelInfo

logger = logging.getLogger(__name__)

mcp = FastMCP("model-router")

# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

_SECRET_ENDPOINT = os.environ.get("SECRET_ORCHESTRATOR_URL", "http://aio-01:5000/secrets")
_SECRET_TIMEOUT = 2

_KEY_ENV_MAP = {
    "GROQ_API_KEY": "GROQ_API_KEY",
    "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "COHERE_API_KEY": "COHERE_API_KEY",
    "CEREBRAS_API_KEY": "CEREBRAS_API_KEY",
}


def _fetch_secret(name: str) -> str | None:
    """Try fetching a secret from the orchestrator endpoint."""
    try:
        url = f"{_SECRET_ENDPOINT}/{name}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_SECRET_TIMEOUT) as resp:  # noqa: S310
            data = json.loads(resp.read())
            return data.get("value") or data.get("secret") or None
    except Exception:
        return None


def _resolve_key(env_var: str) -> str:
    """Resolve an API key from env, falling back to the orchestrator."""
    value = os.environ.get(env_var, "")
    if value:
        return value
    fetched = _fetch_secret(env_var)
    return fetched or ""


# ---------------------------------------------------------------------------
# Singleton router (lazy init)
# ---------------------------------------------------------------------------

_router: PolicyGuard | None = None
_budget_guard: BudgetGuard | None = None
_thompson: ThompsonSamplingSelector | None = None
_latency_opt: LatencyOptimizer | None = None
_router_lock = asyncio.Lock()


async def _get_router() -> PolicyGuard:
    """Build and initialize the decorator-stacked router on first use."""
    global _router, _budget_guard, _thompson, _latency_opt

    if _router is not None:
        return _router

    async with _router_lock:
        if _router is not None:
            return _router

        base = ProviderRouter()

        groq_key = _resolve_key("GROQ_API_KEY")
        if groq_key:
            base.add_provider(OpenAICompatProvider("groq"), api_key=groq_key)

        openrouter_key = _resolve_key("OPENROUTER_API_KEY")
        if openrouter_key:
            base.add_provider(
                OpenAICompatProvider("openrouter", free_only=True),
                api_key=openrouter_key,
            )

        gemini_key = _resolve_key("GEMINI_API_KEY")
        if gemini_key:
            base.add_provider(GeminiProvider(), api_key=gemini_key)

        cohere_key = _resolve_key("COHERE_API_KEY")
        if cohere_key:
            base.add_provider(CohereProvider(), api_key=cohere_key)

        cerebras_key = _resolve_key("CEREBRAS_API_KEY")
        if cerebras_key:
            base.add_provider(OpenAICompatProvider("cerebras"), api_key=cerebras_key)

        _thompson = ThompsonSamplingSelector(base)
        _latency_opt = LatencyOptimizer(_thompson)
        cost_aware = CostAware(_latency_opt)

        max_monthly = float(os.environ.get("BUDGET_MAX_MONTHLY", "300.0"))
        _budget_guard = BudgetGuard(cost_aware, max_monthly=max_monthly)

        _router = PolicyGuard(_budget_guard)

        await _router.initialize()
        return _router


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def chat(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> str:
    """Send a prompt to an LLM through the model router.

    Routes the request through the full decorator stack: policy guard,
    budget guard, cost-aware selection, latency optimization, and
    Thompson Sampling exploration/exploitation.

    Args:
        prompt: The user prompt to send.
        model: Optional model ID to target (e.g. "gemini-2.0-flash").
        temperature: Sampling temperature (0.0-2.0, default 0.7).
        max_tokens: Maximum tokens in the response.
        system_prompt: Optional system instruction prepended to the conversation.
    """
    router = await _get_router()

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=prompt))

    resp = await router.chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return json.dumps(
        {
            "content": resp.content,
            "model": resp.model,
            "provider": resp.provider,
            "latency_ms": round(resp.latency_ms, 1),
            "cost_usd": round(resp.cost_usd, 6),
            "usage": resp.usage,
        },
        indent=2,
    )


@mcp.tool()
async def multi_model_chat(
    prompt: str,
    num_models: int = 3,
    system_prompt: str | None = None,
) -> str:
    """Send the same prompt to multiple models for comparison or consensus.

    Picks N distinct models from the available pool and runs them
    concurrently, returning all responses for comparison.

    Args:
        prompt: The user prompt to send to all models.
        num_models: Number of distinct models to query (default 3).
        system_prompt: Optional system instruction for all models.
    """
    router = await _get_router()

    all_models = await router.list_models()
    if not all_models:
        return json.dumps({"error": "No models available"})

    # Pick N distinct models (or fewer if not enough available)
    count = min(num_models, len(all_models))
    selected = random.sample(all_models, count)

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=prompt))

    async def _call_model(model_info: ModelInfo) -> dict:
        try:
            resp = await router.chat(
                messages,
                model=model_info.model_id,
                temperature=0.7,
            )
            return {
                "content": resp.content,
                "model": resp.model,
                "provider": resp.provider,
                "latency_ms": round(resp.latency_ms, 1),
                "cost_usd": round(resp.cost_usd, 6),
                "usage": resp.usage,
            }
        except Exception as exc:
            return {
                "model": model_info.model_id,
                "provider": model_info.provider,
                "error": str(exc),
            }

    results = await asyncio.gather(*[_call_model(m) for m in selected])

    return json.dumps(list(results), indent=2)


@mcp.tool()
async def list_models(provider: str | None = None) -> str:
    """List all available model endpoints.

    Returns metadata for each discovered model including provider,
    context window size, and cost information.

    Args:
        provider: Optional provider name to filter by (e.g. "groq", "gemini").
    """
    router = await _get_router()
    models = await router.list_models()

    if provider:
        models = [m for m in models if m.provider == provider]

    result = []
    for m in models:
        entry: dict[str, Any] = {
            "model_id": m.model_id,
            "provider": m.provider,
            "context_window": m.context_window,
        }
        if m.cost is not None:
            entry["cost"] = {
                "input_per_1m": m.cost.input_per_1m,
                "output_per_1m": m.cost.output_per_1m,
            }
        else:
            entry["cost"] = None
        result.append(entry)

    return json.dumps(result, indent=2)


@mcp.tool()
async def budget_status() -> str:
    """Show current spending and remaining budget.

    Returns the accumulated spend, remaining budget, call count,
    and percentage used for the current session.
    """
    # Ensure router is initialized so _budget_guard exists
    await _get_router()

    if _budget_guard is None:
        return json.dumps({"error": "Budget guard not initialized"})

    status = _budget_guard.status
    return json.dumps(
        {
            "spent_usd": round(status.spent_usd, 6),
            "remaining_usd": round(status.remaining_usd, 6) if status.remaining_usd is not None else None,
            "max_usd": status.max_usd,
            "calls_made": status.calls_made,
            "percent_used": round(status.percent_used, 2),
        },
        indent=2,
    )


@mcp.tool()
async def model_performance() -> str:
    """Show Thompson Sampling bandit statistics per model.

    Returns alpha/beta parameters, mean success rate, and trial
    count for each model that has been used in this session.
    """
    # Ensure router is initialized so _thompson exists
    await _get_router()

    if _thompson is None:
        return json.dumps({"error": "Thompson Sampling selector not initialized"})

    perf = _thompson.performance()

    # Round floats for readability
    result: dict[str, dict] = {}
    for model_id, stats in perf.items():
        result[model_id] = {
            "alpha": round(stats["alpha"], 2),
            "beta": round(stats["beta"], 2),
            "mean": round(stats["mean"], 4),
            "trials": stats["trials"],
        }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
