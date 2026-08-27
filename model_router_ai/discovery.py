"""Provider, account, identity, and model discovery without persisting credentials."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
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
    label: str
    credential_source: str
    configured: bool

PROVIDERS: tuple[ProviderDefinition, ...] = (
    ProviderDefinition(
        "anthropic", "Anthropic", "anthropic", "ANTHROPIC_API_KEY",
        "https://api.anthropic.com", "https://api.anthropic.com/v1/models",
    ),
    ProviderDefinition(
        "openai", "OpenAI", "openai-compatible", "OPENAI_API_KEY",
        "https://api.openai.com/v1", "https://api.openai.com/v1/models",
        "https://api.openai.com/v1/me",
    ),
    ProviderDefinition(
        "openrouter", "OpenRouter", "openai-compatible", "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models",
        "https://openrouter.ai/api/v1/auth/key", True,
    ),
    ProviderDefinition(
        "groq", "Groq", "openai-compatible", "GROQ_API_KEY",
        "https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/models",
        free_capable=True,
    ),
    ProviderDefinition(
        "cerebras", "Cerebras", "openai-compatible", "CEREBRAS_API_KEY",
        "https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/models",
        free_capable=True,
    ),
    ProviderDefinition(
        "deepinfra", "DeepInfra", "openai-compatible", "DEEPINFRA_API_TOKEN",
        "https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/models",
        free_capable=True,
    ),
    ProviderDefinition(
        "nvidia", "NVIDIA", "openai-compatible", "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/models",
        free_capable=True,
    ),
    ProviderDefinition(
        "gemini", "Google Gemini", "gemini", "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta",
        "https://generativelanguage.googleapis.com/v1beta/models",
        free_capable=True,
    ),
    ProviderDefinition(
        "cohere", "Cohere", "cohere", "COHERE_API_KEY",
        "https://api.cohere.com", free_capable=True,
    ),
    ProviderDefinition(
        "huggingface", "Hugging Face", "openai-compatible", "HUGGINGFACE_API_KEY",
        "https://router.huggingface.co/v1", "https://router.huggingface.co/v1/models",
        free_capable=True,
    ),
)

ROOT = Path(os.environ.get("FLOSSWARE_AI_ROOT", Path.home() / ".flossware" / "ai"))
ACCOUNTS_FILE = Path(
    os.environ.get("FLOSSWARE_ACCOUNTS_FILE", ROOT / "config" / "accounts.toml")
)

def provider_definitions() -> list[dict[str, Any]]:
    return [asdict(provider) for provider in PROVIDERS]

def _provider(provider_id: str) -> ProviderDefinition:
    provider = next((item for item in PROVIDERS if item.id == provider_id), None)
    if provider is None:
        raise ValueError(f"unknown provider: {provider_id}")
    return provider

def _configured(value: str | None) -> bool:
    return bool(value and os.environ.get(value))

def _configured_accounts() -> list[Account]:
    accounts: list[Account] = []
    if ACCOUNTS_FILE.exists():
        try:
            data = tomllib.loads(ACCOUNTS_FILE.read_text())
            raw_accounts = data.get("accounts", {})
            if isinstance(raw_accounts, dict):
                for account_id, raw in raw_accounts.items():
                    if not isinstance(raw, dict):
                        continue
                    provider_id = str(raw.get("provider", ""))
                    try:
                        provider = _provider(provider_id)
                    except ValueError:
                        continue
                    env = raw.get("credential_env")
                    if isinstance(env, str) and env:
                        accounts.append(
                            Account(
                                str(account_id),
                                provider.id,
                                str(raw.get("label", account_id)),
                                f"environment:{env}",
                                _configured(env),
                            )
                        )
        except (OSError, tomllib.TOMLDecodeError):
            pass
    if accounts:
        return accounts
    for provider in PROVIDERS:
        if _configured(provider.environment):
            accounts.append(
                Account(
                    f"{provider.id}-default",
                    provider.id,
                    "default",
                    f"environment:{provider.environment}",
                    True,
                )
            )
    return accounts

def discover_accounts() -> list[dict[str, Any]]:
    return [asdict(account) for account in _configured_accounts() if account.configured]

def _account(account: str | dict[str, Any]) -> Account:
    accounts = _configured_accounts()
    if isinstance(account, dict):
        return Account(
            str(account["id"]),
            str(account["provider"]),
            str(account.get("label", account["id"])),
            str(account["credential_source"]),
            bool(account.get("configured")),
        )
    for item in accounts:
        if item.id == account or item.provider == account:
            return item
    raise ValueError(f"unknown account: {account}")

def _credential(account: Account) -> str | None:
    prefix = "environment:"
    if account.credential_source.startswith(prefix):
        return os.environ.get(account.credential_source[len(prefix):])
    return None

def _headers(provider: ProviderDefinition, key: str) -> dict[str, str]:
    if provider.id == "gemini":
        return {"x-goog-api-key": key}
    if provider.id == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}

def _request_json(url: str, headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(url, headers=headers)
    # URLs originate from the static PROVIDERS table, not user input.
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.load(response)

def discover_identity(account: str | dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
    item = _account(account)
    provider = _provider(item.provider)
    key = _credential(item)
    result: dict[str, Any] = {
        "account": item.id,
        "label": item.label,
        "provider": item.provider,
        "status": "configured",
        "identity": None,
    }
    if not key:
        result.update(status="not_configured", identity_status="unverified")
        return result
    if not provider.identity_url:
        result["identity_status"] = "unsupported"
        return result
    try:
        payload = _request_json(provider.identity_url, _headers(provider, key), timeout)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        result.update(identity_status="unverified", error=type(exc).__name__)
        return result
    identity = payload.get("data", payload)
    safe: dict[str, Any] = {}
    if isinstance(identity, dict):
        for key_name in ("email", "name", "label", "id", "organization", "plan"):
            if key_name in identity:
                safe[key_name] = identity[key_name]
    result.update(identity=safe, identity_status="verified")
    return result

def discover_identities(timeout: float = 8.0) -> list[dict[str, Any]]:
    return [discover_identity(account, timeout) for account in discover_accounts()]

def discover_models(account: str | dict[str, Any], timeout: float = 8.0) -> list[dict[str, Any]]:
    item = _account(account)
    provider = _provider(item.provider)
    key = _credential(item)
    if not key or not provider.models_url:
        return []
    try:
        payload = _request_json(provider.models_url, _headers(provider, key), timeout)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ):
        return []
    raw_models = payload.get("models", []) if provider.id == "gemini" else payload.get("data", [])
    models: list[dict[str, Any]] = []
    for model in raw_models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id") or model.get("name")
        if model_id:
            models.append(
                {
                    "account": item.id,
                    "account_label": item.label,
                    "provider": provider.id,
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "free_capable": provider.free_capable,
                }
            )
    return models

def discover_all_models(timeout: float = 8.0) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for account in discover_accounts():
        models.extend(discover_models(account, timeout=timeout))
    return models
