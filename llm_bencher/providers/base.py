from __future__ import annotations

from abc import ABC, abstractmethod

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
