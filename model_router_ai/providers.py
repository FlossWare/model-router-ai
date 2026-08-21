"""Provider implementations for different LLM APIs.

Each provider knows how to:
1. Discover available models
2. Format requests for its API
3. Parse responses into ChatResponse

Zero external dependencies — uses stdlib urllib/json only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from model_router_ai.types import ChatMessage, ChatResponse, ModelCost, ModelInfo

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT = 10
_CHAT_TIMEOUT = 120


def _http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict | None = None,
    timeout: int = _CHAT_TIMEOUT,
    retries: int = 2,
) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    last_status, last_body = 0, {"error": "no attempts"}
    for attempt in range(1 + retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                last_status, last_body = exc.code, json.loads(exc.read(8192))
            except Exception:
                last_status, last_body = exc.code, {"error": str(exc)}
            if exc.code < 500:
                return last_status, last_body
        except Exception as exc:
            last_status, last_body = 0, {"error": str(exc)}
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    return last_status, last_body


async def _async_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict | None = None,
    timeout: int = _CHAT_TIMEOUT,
) -> tuple[int, dict]:
    async with asyncio.timeout(timeout):
        return await asyncio.to_thread(
            _http_request, method, url, headers, body, timeout=timeout
        )


def _openai_chat_body(
    messages: list[ChatMessage],
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> dict:
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    payload: dict = {"model": model, "messages": msgs, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _normalize_usage(raw: dict) -> dict:
    """Normalize provider-specific usage dicts to canonical keys."""
    return {
        "prompt_tokens": (
            raw.get("prompt_tokens")
            or raw.get("promptTokenCount")
            or raw.get("input_tokens")
            or 0
        ),
        "completion_tokens": (
            raw.get("completion_tokens")
            or raw.get("candidatesTokenCount")
            or raw.get("output_tokens")
            or 0
        ),
        "total_tokens": (
            raw.get("total_tokens")
            or raw.get("totalTokenCount")
            or (
                (raw.get("prompt_tokens") or raw.get("promptTokenCount") or raw.get("input_tokens") or 0)
                + (raw.get("completion_tokens") or raw.get("candidatesTokenCount") or raw.get("output_tokens") or 0)
            )
        ),
    }


def _parse_openai_response(body: dict, provider: str) -> ChatResponse:
    choice = body.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    return ChatResponse(
        content=content,
        model=body.get("model", ""),
        provider=provider,
        usage=_normalize_usage(body.get("usage", {})),
    )


class _BaseProvider:
    """Common interface for LLM providers."""

    name: str = ""

    async def discover_models(self, api_key: str) -> list[ModelInfo]:
        raise NotImplementedError

    async def call(
        self,
        model_info: ModelInfo,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        raise NotImplementedError


class OpenAICompatProvider(_BaseProvider):
    """Provider for any OpenAI-compatible API (Groq, Cerebras, DeepInfra, etc.)."""

    KNOWN_BASES: dict[str, str] = {
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "deepinfra": "https://api.deepinfra.com/v1/openai",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "openai": "https://api.openai.com/v1",
        "azure": "",  # set via base_url
    }

    def __init__(
        self,
        name: str,
        base_url: str = "",
        free_only: bool = False,
        cost_map: dict[str, ModelCost] | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url or self.KNOWN_BASES.get(name, "")
        self._free_only = free_only
        self._cost_map = cost_map or {}

    async def discover_models(self, api_key: str) -> list[ModelInfo]:
        if not self._base_url:
            return []
        url = f"{self._base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        status, body = await _async_request("GET", url, headers, timeout=_PROBE_TIMEOUT)
        if status != 200:
            return []

        models = []
        for m in body.get("data", []):
            model_id = m.get("id", "")
            if not model_id:
                continue

            if self.name == "openrouter" and self._free_only:
                pricing = m.get("pricing", {})
                if pricing.get("prompt") != "0" or pricing.get("completion") != "0":
                    continue

            cost = self._cost_map.get(model_id)
            if cost is None and self.name == "openrouter":
                pricing = m.get("pricing", {})
                try:
                    input_price = float(pricing.get("prompt", "0"))
                    output_price = float(pricing.get("completion", "0"))
                    cost = ModelCost(
                        input_per_1m=input_price * 1_000_000,
                        output_per_1m=output_price * 1_000_000,
                    )
                except (ValueError, TypeError):
                    pass

            models.append(
                ModelInfo(
                    model_id=model_id,
                    provider=self.name,
                    api_key=api_key,
                    cost=cost,
                    context_window=m.get("context_length", 0),
                )
            )
        return models

    async def call(
        self,
        model_info: ModelInfo,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {model_info.api_key}"}
        body = _openai_chat_body(messages, model_info.model_id, temperature, max_tokens)
        t0 = time.monotonic()
        status, resp = await _async_request("POST", url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {resp}")
        result = _parse_openai_response(resp, self.name)
        result.latency_ms = latency
        return result


class GeminiProvider(_BaseProvider):
    """Google Gemini (direct API, not Vertex)."""

    name = "gemini"

    def __init__(self, cost_map: dict[str, ModelCost] | None = None) -> None:
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._cost_map = cost_map or {}

    async def discover_models(self, api_key: str) -> list[ModelInfo]:
        url = f"{self._base_url}/models"
        headers = {"x-goog-api-key": api_key}
        status, body = await _async_request("GET", url, headers, timeout=_PROBE_TIMEOUT)
        if status != 200:
            return []
        models = []
        for m in body.get("models", []):
            methods = str(m.get("supportedGenerationMethods", []))
            if "generateContent" not in methods:
                continue
            model_id = m["name"].removeprefix("models/")
            models.append(
                ModelInfo(
                    model_id=model_id,
                    provider="gemini",
                    api_key=api_key,
                    cost=self._cost_map.get(model_id),
                )
            )
        return models

    async def call(
        self,
        model_info: ModelInfo,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        url = f"{self._base_url}/models/{model_info.model_id}:generateContent"
        headers = {"x-goog-api-key": model_info.api_key}

        contents: list[dict] = []
        system_text: str | None = None
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})
        body: dict = {"contents": contents}
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}
        gen_config: dict = {"temperature": temperature}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        body["generationConfig"] = gen_config

        t0 = time.monotonic()
        status, resp = await _async_request("POST", url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        if status != 200:
            raise RuntimeError(f"Gemini HTTP {status}: {resp}")

        text = ""
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            text = parts[-1].get("text", "")
        except (KeyError, IndexError):
            pass

        return ChatResponse(
            content=text,
            model=model_info.model_id,
            provider="gemini",
            usage=_normalize_usage(resp.get("usageMetadata", {})),
            latency_ms=latency,
        )


class CohereProvider(_BaseProvider):
    """Cohere API provider."""

    name = "cohere"
    DEFAULT_MODELS = ["command-a-03-2025", "command-r-plus", "command-r"]

    def __init__(self, cost_map: dict[str, ModelCost] | None = None) -> None:
        self._base_url = "https://api.cohere.com/v2"
        self._cost_map = cost_map or {}

    async def discover_models(self, api_key: str) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=mid,
                provider="cohere",
                api_key=api_key,
                cost=self._cost_map.get(mid),
            )
            for mid in self.DEFAULT_MODELS
        ]

    async def call(
        self,
        model_info: ModelInfo,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        url = f"{self._base_url}/chat"
        headers = {"Authorization": f"Bearer {model_info.api_key}"}
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        body: dict = {
            "model": model_info.model_id,
            "messages": msgs,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        t0 = time.monotonic()
        status, resp = await _async_request("POST", url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        if status != 200:
            raise RuntimeError(f"Cohere HTTP {status}: {resp}")

        text = ""
        try:
            text = resp["message"]["content"][0]["text"]
        except (KeyError, IndexError):
            pass

        cohere_usage = resp.get("usage", {})
        billed = cohere_usage.get("billed_units", {})
        normalized = _normalize_usage({
            "input_tokens": billed.get("input_tokens", cohere_usage.get("input_tokens", 0)),
            "output_tokens": billed.get("output_tokens", cohere_usage.get("output_tokens", 0)),
        })

        return ChatResponse(
            content=text,
            model=body.get("model", ""),
            provider="cohere",
            usage=normalized,
            latency_ms=latency,
        )


class VertexAIProvider(_BaseProvider):
    """Google Vertex AI provider (supports Gemini and Anthropic models).

    Requires a GCP project ID and either a service account key or
    default credentials. Supports the $300/month enterprise quota.
    """

    name = "vertex"

    def __init__(
        self,
        project_id: str = "",
        region: str = "us-central1",
        cost_map: dict[str, ModelCost] | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._cost_map = cost_map or {}
        self._available_models = available_models or []

    async def discover_models(self, api_key: str) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=mid,
                provider="vertex",
                api_key=api_key,
                cost=self._cost_map.get(mid),
            )
            for mid in self._available_models
        ]

    async def call(
        self,
        model_info: ModelInfo,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        publisher = "anthropic" if model_info.model_id.startswith("claude") else "google"
        base = (
            f"https://{self._region}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project_id}/locations/{self._region}/"
            f"publishers/{publisher}/models/{model_info.model_id}"
        )
        endpoint = "rawPredict" if publisher == "anthropic" else "generateContent"
        url = f"{base}:{endpoint}"
        headers = {
            "Authorization": f"Bearer {model_info.api_key}",
            "Content-Type": "application/json",
        }

        contents: list[dict] = []
        system_text: str | None = None
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        body: dict = {"contents": contents}
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}
        gen_config: dict = {"temperature": temperature}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        body["generationConfig"] = gen_config

        t0 = time.monotonic()
        status, resp = await _async_request("POST", url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        if status != 200:
            raise RuntimeError(f"Vertex HTTP {status}: {resp}")

        text = ""
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            text = parts[-1].get("text", "")
        except (KeyError, IndexError):
            pass

        return ChatResponse(
            content=text,
            model=model_info.model_id,
            provider="vertex",
            usage=_normalize_usage(resp.get("usageMetadata", {})),
            latency_ms=latency,
        )
