from __future__ import annotations

from llm_bencher.providers.openai_compat import OpenAICompatAdapter


class LMStudioAdapter(OpenAICompatAdapter):
    """Adapter for LM Studio (fully OpenAI-compatible API)."""

    provider_slug: str = "lm-studio"
