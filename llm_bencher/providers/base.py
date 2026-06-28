from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

import httpx

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


class ProviderAdapter(ABC):
    provider_slug: str

    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_headers = default_headers or {}
        self._client: httpx.AsyncClient | None = None

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def _get_client(self) -> httpx.AsyncClient:
        """Return a lazily-created, reusable client with keep-alive pooling.

        Reusing one client across calls avoids a fresh TCP (and TLS, for cloud
        endpoints) handshake per request, which is the dominant overhead when a
        batch fires many requests at the same provider.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._default_headers or None,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> "ProviderAdapter":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check whether the provider endpoint is reachable."""

    @abstractmethod
    async def list_models(self) -> list[DiscoveredModel]:
        """Return the currently available models for the provider."""

    @abstractmethod
    async def run_chat(self, request: RunRequest) -> RunResult:
        """Execute one chat-style inference request."""

    async def run_chat_stream(
        self, request: RunRequest
    ) -> AsyncGenerator[dict, None]:
        """
        Stream a chat-style inference request, yielding structured event dicts.

        Each yielded dict has a ``type`` key:
          ``{"type": "chunk", "text": str}``          – incremental output text
          ``{"type": "usage", "prompt_tokens": int|None,
              "completion_tokens": int|None, "total_tokens": int|None}``

        The default implementation falls back to the blocking ``run_chat`` and
        yields the full result as a single chunk so callers always get a
        consistent event stream regardless of whether the adapter overrides this.
        """
        result = await self.run_chat(request)
        yield {"type": "chunk", "text": result.output_text}
        yield {
            "type": "usage",
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
