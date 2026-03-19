from __future__ import annotations

from abc import ABC, abstractmethod

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest, RunResult


class ProviderAdapter(ABC):
    provider_slug: str

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check whether the provider endpoint is reachable."""

    @abstractmethod
    async def list_models(self) -> list[DiscoveredModel]:
        """Return the currently available models for the provider."""

    @abstractmethod
    async def run_chat(self, request: RunRequest) -> RunResult:
        """Execute one chat-style inference request."""
