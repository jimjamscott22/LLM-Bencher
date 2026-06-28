from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama using its native API."""

    provider_slug: str = "ollama"

    async def health_check(self) -> ProviderHealth:
        try:
            resp = await self._get_client().get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            return ProviderHealth(is_available=True, checked_at=_utc_now())
        except Exception as exc:
            return ProviderHealth(
                is_available=False,
                detail=str(exc),
                checked_at=_utc_now(),
            )

    async def list_models(self) -> list[DiscoveredModel]:
        resp = await self._get_client().get(f"{self._base_url}/api/tags")
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
        resp = await self._get_client().post(f"{self._base_url}/api/chat", json=payload)
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
        }
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = request.max_tokens

        async with self._get_client().stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = (data.get("message") or {}).get("content", "")
                if content:
                    yield {"type": "chunk", "text": content}

                if data.get("done"):
                    pt = data.get("prompt_eval_count")
                    ct = data.get("eval_count")
                    total = (
                        (pt or 0) + (ct or 0)
                        if pt is not None or ct is not None
                        else None
                    )
                    yield {
                        "type": "usage",
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": total,
                    }
                    break
