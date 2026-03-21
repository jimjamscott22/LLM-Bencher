from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OpenAICompatAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible chat/completions APIs (LM Studio, etc.)."""

    provider_slug: str = "openai-compat"

    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def health_check(self) -> ProviderHealth:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/models")
                resp.raise_for_status()
            return ProviderHealth(is_available=True, checked_at=_utc_now())
        except Exception as exc:
            return ProviderHealth(
                is_available=False,
                detail=str(exc),
                checked_at=_utc_now(),
            )

    async def list_models(self) -> list[DiscoveredModel]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/models")
            resp.raise_for_status()
            data = resp.json()
        return [
            DiscoveredModel(
                id=m["id"],
                name=m["id"],
                provider_slug=self.provider_slug,
                metadata=m,
            )
            for m in data.get("data", [])
        ]

    async def run_chat(self, request: RunRequest) -> RunResult:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        payload: dict = {"model": request.model_id, "messages": messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.monotonic() - start) * 1000)
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return RunResult(
            output_text=choice["message"]["content"],
            response_metadata={
                "finish_reason": choice.get("finish_reason"),
                "model": data.get("model"),
            },
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_payload=data,
        )
