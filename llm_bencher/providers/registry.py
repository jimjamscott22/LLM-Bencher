from __future__ import annotations

from threading import Lock

from llm_bencher.config import Settings
from llm_bencher.models import Provider, ProviderKind
from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.providers.lm_studio import LMStudioAdapter
from llm_bencher.providers.ollama import OllamaAdapter
from llm_bencher.providers.openai_cloud import OpenAICloudAdapter
from llm_bencher.providers.openai_compat import OpenAICompatAdapter


_ADAPTER_CACHE: dict[tuple[str, str, float, str], ProviderAdapter] = {}
_ADAPTER_CACHE_LOCK = Lock()


def get_adapter(provider: Provider, settings: Settings) -> ProviderAdapter:
    """Return the correct adapter instance for a Provider ORM row."""
    timeout = settings.provider_timeout_seconds
    api_key = provider.api_key or ""
    resolved_key = api_key or (settings.openai_api_key if provider.kind == ProviderKind.OPENAI else "")
    cache_key = (provider.kind.value, provider.base_url, timeout, resolved_key)

    with _ADAPTER_CACHE_LOCK:
        cached = _ADAPTER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    match provider.kind:
        case ProviderKind.LM_STUDIO:
            adapter: ProviderAdapter = LMStudioAdapter(base_url=provider.base_url, timeout=timeout)
        case ProviderKind.OLLAMA:
            adapter = OllamaAdapter(base_url=provider.base_url, timeout=timeout)
        case ProviderKind.OPENAI:
            adapter = OpenAICloudAdapter(base_url=provider.base_url, timeout=timeout, api_key=resolved_key)
        case ProviderKind.OPENAI_COMPAT:
            if api_key:
                adapter = OpenAICloudAdapter(base_url=provider.base_url, timeout=timeout, api_key=api_key)
            else:
                adapter = OpenAICompatAdapter(base_url=provider.base_url, timeout=timeout)
        case _:
            raise ValueError(f"Unknown provider kind: {provider.kind!r}")

    with _ADAPTER_CACHE_LOCK:
        _ADAPTER_CACHE[cache_key] = adapter
    return adapter
