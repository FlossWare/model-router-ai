"""Provider, account, identity, and model discovery without persisting credentials."""
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
    identity_url: str | None = None
    free_capable: bool = False

@dataclass(frozen=True)
class Account:
    id: str
    provider: str
    credential_source: str
    configured: bool

PROVIDERS: tuple[ProviderDefinition, ...] = (
    ProviderDefinition("anthropic", "Anthropic", "anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com", "https://api.anthropic.com/v1/models", None),
    ProviderDefinition("openai", "OpenAI", "openai-compatible", "OPENAI_API_KEY", "https://api.openai.com/v1", "https://api.openai.com/v1/models", "https://api.openai.com/v1/me"),
    ProviderDefinition("openrouter", "OpenRouter", "openai-compatible", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models", "https://openrouter.ai/api/v1/auth/key", True),
    ProviderDefinition("groq", "Groq", "openai-compatible", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/models", None, True),
    ProviderDefinition("cerebras", "Cerebras", "openai-compatible", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/models", None, True),
    ProviderDefinition("deepinfra", "DeepInfra", "openai-compatible", "DEEPINFRA_API_TOKEN", "https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/models", None, True),
    ProviderDefinition("nvidia", "NVIDIA", "openai-compatible", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/models", None, True),
    ProviderDefinition("gemini", "Google Gemini", "gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta", "https://generativelanguage.googleapis.com/v1beta/models", None, True),
    ProviderDefinition("cohere", "Cohere", "cohere", "COHERE_API_KEY", "https://api.cohere.com", None, None, True),
    ProviderDefinition("huggingface", "Hugging Face", "openai-compatible", "HUGGINGFACE_API_KEY", "https://router.huggingface.co/v1", "https://router.huggingface.co/v1/models", None, True),
)

def provider_definitions() -> list[dict[str, Any]]:
    return [asdict(p) for p in PROVIDERS]

def discover_accounts() -> list[dict[str, Any]]:
    return [asdict(Account(p.id, p.id, f"environment:{p.environment}", True)) for p in PROVIDERS if p.environment and os.environ.get(p.environment)]

def _headers(provider: ProviderDefinition, key: str) -> dict[str, str]:
    if provider.id == "gemini":
        return {"x-goog-api-key": key}
    if provider.id == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}

def _request_json(url: str, headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def discover_identity(provider_id: str, timeout: float = 8.0) -> dict[str, Any]:
    provider = next((p for p in PROVIDERS if p.id == provider_id), None)
    if provider is None:
        raise ValueError(f"unknown provider: {provider_id}")
    if not provider.environment or not os.environ.get(provider.environment):
        return {"provider": provider_id, "status": "not_configured", "identity": None}
    if not provider.identity_url:
        return {"provider": provider_id, "status": "configured", "identity": None, "identity_status": "unsupported"}
    try:
        payload = _request_json(provider.identity_url, _headers(provider, os.environ[provider.environment]), timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"provider": provider_id, "status": "configured", "identity": None, "identity_status": "unverified", "error": type(exc).__name__}
    identity = payload.get("data", payload)
    safe: dict[str, Any] = {}
    if isinstance(identity, dict):
        for key in ("email", "name", "label", "id", "organization", "plan"):
            if key in identity and not any(secret in key.lower() for secret in ("key", "token", "secret")):
                safe[key] = identity[key]
    return {"provider": provider_id, "status": "configured", "identity": safe, "identity_status": "verified"}

def discover_identities(timeout: float = 8.0) -> list[dict[str, Any]]:
    return [discover_identity(account["provider"], timeout) for account in discover_accounts()]

def discover_models(provider_id: str, timeout: float = 8.0) -> list[dict[str, Any]]:
    provider = next((p for p in PROVIDERS if p.id == provider_id), None)
    if provider is None:
        raise ValueError(f"unknown provider: {provider_id}")
    if not provider.environment or not os.environ.get(provider.environment) or not provider.models_url:
        return []
    try:
        payload = _request_json(provider.models_url, _headers(provider, os.environ[provider.environment]), timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    raw_models = payload.get("models", []) if provider.id == "gemini" else payload.get("data", [])
    models = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name")
        if model_id:
            models.append({"provider": provider.id, "id": model_id, "name": item.get("name") or model_id, "free_capable": provider.free_capable})
    return models

def discover_all_models(timeout: float = 8.0) -> list[dict[str, Any]]:
    models = []
    for account in discover_accounts():
        models.extend(discover_models(account["provider"], timeout=timeout))
    return models
