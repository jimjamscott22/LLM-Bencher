from __future__ import annotations

from llm_bencher.providers.openai_compat import OpenAICompatAdapter


class OpenAICloudAdapter(OpenAICompatAdapter):
    """Adapter for the OpenAI cloud API (adds Authorization header)."""

    provider_slug: str = "openai"

    def __init__(self, base_url: str, timeout: float, api_key: str = "") -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        super().__init__(base_url=base_url, timeout=timeout, default_headers=headers)
        self._api_key = api_key
