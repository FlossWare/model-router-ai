"""Provider, account, and model discovery without persisting credentials.

Discovery is intentionally stdlib-only. Credential values are read from the
process environment and are never returned in discovery results.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    name: str
    kind: str
    environment: str | None
    base_url: str
    models_url: str | None = None
    free_capable: bool = False


@dataclass(frozen=True)
class Account:
    id: str
    provider: str
    credential_source: str
    configured: bool


PROVIDERS: tuple[ProviderDefinition, ...] = (
    ProviderDefinition("anthropic", "Anthropic", "anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com", None),
    ProviderDefinition("openai", "OpenAI", "openai-compatible", "OPENAI_API_KEY", "https://api.openai.com/v1", "https://api.openai.com/v1/models"),
    ProviderDefinition("openrouter", "OpenRouter", "openai-compatible", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models", True),
    ProviderDefinition("groq", "Groq", "openai-compatible", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/models", True),
    ProviderDefinition("cerebras", "Cerebras", "openai-compatible", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/models", True),
    ProviderDefinition("deepinfra", "DeepInfra", "openai-compatible", "DEEPINFRA_API_TOKEN", "https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/models", True),
    ProviderDefinition("nvidia", "NVIDIA", "openai-compatible", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/models", True),
    ProviderDefinition("gemini", "Google Gemini", "gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta", "https://generativelanguage.googleapis.com/v1beta/models", True),
    ProviderDefinition("cohere", "Cohere", "cohere", "COHERE_API_KEY", "https://api.cohere.com", None, True),
    ProviderDefinition("huggingface", "Hugging Face", "openai-compatible", "HUGGINGFACE_API_KEY", "https://router.huggingface.co/v1", "https://router.huggingface.co/v1/models", True),
)


def provider_definitions() -> list[dict[str, Any]]:
    """Return public provider metadata only."""
    return [asdict(p) for p in PROVIDERS]


def discover_accounts() -> list[dict[str, Any]]:
    """Discover configured accounts from environment presence, never values."""
    accounts: list[dict[str, Any]] = []
    for p in PROVIDERS:
        if p.environment and os.environ.get(p.environment):
            accounts.append(asdict(Account(p.id, p.id, f"environment:{p.environment}", True)))
    return accounts


def _request_json(url: str, headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_models(provider_id: str, timeout: float = 8.0) -> list[dict[str, Any]]:
    """Discover models for one configured provider.

    Results contain provider/model metadata only. Authentication headers are
    constructed locally and are never included in the result.
    """
    provider = next((p for p in PROVIDERS if p.id == provider_id), None)
    if provider is None:
        raise ValueError(f"unknown provider: {provider_id}")
    if not provider.environment or not os.environ.get(provider.environment):
        return []
    if not provider.models_url:
        return []

    key = os.environ[provider.environment]
    headers: dict[str, str]
    if provider.id == "gemini":
        url = f"{provider.models_url}?key={key}"
        headers = {}
    elif provider.id == "anthropic":
        url = provider.models_url
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url = provider.models_url
        headers = {"Authorization": f"Bearer {key}"}

    try:
        payload = _request_json(url, headers, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []

    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    if provider.id == "gemini" and isinstance(payload, dict):
        raw_models = payload.get("models", [])

    models: list[dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name")
        if not model_id:
            continue
        models.append({
            "provider": provider.id,
            "id": model_id,
            "name": item.get("name") or model_id,
            "free_capable": provider.free_capable,
        })
    return models


def discover_all_models(timeout: float = 8.0) -> list[dict[str, Any]]:
    """Discover models from every configured provider."""
    models: list[dict[str, Any]] = []
    for account in discover_accounts():
        models.extend(discover_models(account["provider"], timeout=timeout))
    return models
