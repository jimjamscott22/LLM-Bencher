from __future__ import annotations

from llm_bencher.config import Settings
from llm_bencher.models import Provider, ProviderKind
from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.providers.lm_studio import LMStudioAdapter
from llm_bencher.providers.ollama import OllamaAdapter
from llm_bencher.providers.openai_cloud import OpenAICloudAdapter
from llm_bencher.providers.openai_compat import OpenAICompatAdapter


def get_adapter(provider: Provider, settings: Settings) -> ProviderAdapter:
    """Return the correct adapter instance for a Provider ORM row."""
    timeout = settings.provider_timeout_seconds
    api_key = provider.api_key or ""
    match provider.kind:
        case ProviderKind.LM_STUDIO:
            return LMStudioAdapter(base_url=provider.base_url, timeout=timeout)
        case ProviderKind.OLLAMA:
            return OllamaAdapter(base_url=provider.base_url, timeout=timeout)
        case ProviderKind.OPENAI:
            key = api_key or settings.openai_api_key
            return OpenAICloudAdapter(base_url=provider.base_url, timeout=timeout, api_key=key)
        case ProviderKind.OPENAI_COMPAT:
            if api_key:
                return OpenAICloudAdapter(base_url=provider.base_url, timeout=timeout, api_key=api_key)
            return OpenAICompatAdapter(base_url=provider.base_url, timeout=timeout)
        case _:
            raise ValueError(f"Unknown provider kind: {provider.kind!r}")
