from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama using its native API."""

    provider_slug: str = "ollama"

    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def health_check(self) -> ProviderHealth:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
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
            resp = await client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [
            DiscoveredModel(
                id=m["name"],
                name=m["name"],
                provider_slug=self.provider_slug,
                metadata=m,
            )
            for m in data.get("models", [])
        ]

    async def run_chat(self, request: RunRequest) -> RunResult:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        payload: dict = {
            "model": request.model_id,
            "messages": messages,
            "stream": False,
        }
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = request.max_tokens

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.monotonic() - start) * 1000)
        output_text = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None or completion_tokens is not None
            else None
        )

        return RunResult(
            output_text=output_text,
            response_metadata={
                "done": data.get("done"),
                "done_reason": data.get("done_reason"),
                "model": data.get("model"),
            },
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_payload=data,
        )
