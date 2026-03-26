from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_bencher.config import Settings
from llm_bencher.models import Provider, ProviderKind


DEFAULT_PROVIDERS = (
    {
        "slug": "lm-studio",
        "name": "LM Studio",
        "kind": ProviderKind.LM_STUDIO,
        "base_url": "lm_studio_base_url",
    },
    {
        "slug": "ollama",
        "name": "Ollama",
        "kind": ProviderKind.OLLAMA,
        "base_url": "ollama_base_url",
    },
)


def seed_default_providers(session: Session, settings: Settings) -> None:
    for provider_data in DEFAULT_PROVIDERS:
        provider = session.scalar(
            select(Provider).where(Provider.slug == provider_data["slug"])
        )
        base_url = getattr(settings, provider_data["base_url"])

        if provider is None:
            session.add(
                Provider(
                    slug=provider_data["slug"],
                    name=provider_data["name"],
                    kind=provider_data["kind"],
                    base_url=base_url,
                    is_default=True,
                )
            )
            continue

        provider.name = provider_data["name"]
        provider.kind = provider_data["kind"]
        provider.base_url = base_url
        provider.is_enabled = True
        provider.is_default = True
