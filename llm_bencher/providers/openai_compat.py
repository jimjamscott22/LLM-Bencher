from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OpenAICompatAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible chat/completions APIs (LM Studio, etc.)."""

    provider_slug: str = "openai-compat"

    async def health_check(self) -> ProviderHealth:
        try:
            resp = await self._get_client().get(f"{self._base_url}/models")
            resp.raise_for_status()
            return ProviderHealth(is_available=True, checked_at=_utc_now())
        except Exception as exc:
            return ProviderHealth(
                is_available=False,
                detail=str(exc),
                checked_at=_utc_now(),
            )

    async def list_models(self) -> list[DiscoveredModel]:
        resp = await self._get_client().get(f"{self._base_url}/models")
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
        resp = await self._get_client().post(
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

    async def run_chat_stream(
        self, request: RunRequest
    ) -> AsyncGenerator[dict, None]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        payload: dict = {
            "model": request.model_id,
            "messages": messages,
            "stream": True,
            # Ask the provider to include token usage in the final SSE chunk.
            # Supported by OpenAI and some compatible servers; ignored by others.
            "stream_options": {"include_usage": True},
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        async with self._get_client().stream(
            "POST", f"{self._base_url}/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.rstrip("\r")
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Usage object — present in the final chunk when
                # stream_options.include_usage is supported.
                if data.get("usage"):
                    usage = data["usage"]
                    yield {
                        "type": "usage",
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }

                choices = data.get("choices") or []
                if choices:
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        yield {"type": "chunk", "text": content}
