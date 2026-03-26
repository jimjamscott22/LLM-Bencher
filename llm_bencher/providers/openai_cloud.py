from __future__ import annotations

import httpx

from llm_bencher.providers.openai_compat import OpenAICompatAdapter
from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


class OpenAICloudAdapter(OpenAICompatAdapter):
    """Adapter for the OpenAI cloud API (adds Authorization header)."""

    provider_slug: str = "openai"

    def __init__(self, base_url: str, timeout: float, api_key: str = "") -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def health_check(self) -> ProviderHealth:
        from datetime import datetime, timezone
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                resp.raise_for_status()
            return ProviderHealth(
                is_available=True,
                checked_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            return ProviderHealth(
                is_available=False,
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
            )

    async def list_models(self) -> list[DiscoveredModel]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
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
        import time

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
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
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
