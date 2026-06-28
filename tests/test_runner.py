from __future__ import annotations

import httpx
import pytest

from llm_bencher.models import Provider, ProviderKind
from llm_bencher.providers.lm_studio import LMStudioAdapter
from llm_bencher.providers.openai_cloud import OpenAICloudAdapter
from llm_bencher.providers.registry import provider_timeout_seconds
from llm_bencher.runner import format_adapter_error


def test_format_adapter_error_timeout_with_seconds():
    exc = httpx.ReadTimeout("The read operation timed out")
    assert format_adapter_error(exc, timeout_seconds=30.0) == (
        "Provider request timed out after 30s"
    )


def test_format_adapter_error_timeout_without_seconds():
    exc = httpx.ReadTimeout("")
    assert format_adapter_error(exc) == "Provider request timed out"


def test_format_adapter_error_preserves_message():
    assert format_adapter_error(RuntimeError("model not found")) == "model not found"


def test_format_adapter_error_falls_back_to_exception_name():
    class SilentError(Exception):
        def __str__(self) -> str:
            return ""

    assert format_adapter_error(SilentError()) == "SilentError"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ProviderKind.LM_STUDIO, 300.0),
        (ProviderKind.OLLAMA, 300.0),
        (ProviderKind.OPENAI, 30.0),
        (ProviderKind.OPENAI_COMPAT, 30.0),
    ],
)
def test_provider_timeout_seconds_by_kind(kind, expected):
    from llm_bencher.config import Settings

    settings = Settings()
    provider = Provider(
        slug="test",
        name="Test",
        kind=kind,
        base_url="http://localhost:1234/v1",
    )
    assert provider_timeout_seconds(provider, settings) == expected


def test_local_adapter_uses_local_timeout():
    from llm_bencher.config import Settings
    from llm_bencher.providers.registry import get_adapter

    settings = Settings(local_provider_timeout_seconds=600.0)
    provider = Provider(
        slug="lm",
        name="LM Studio",
        kind=ProviderKind.LM_STUDIO,
        base_url="http://127.0.0.1:1234/v1",
    )
    adapter = get_adapter(provider, settings)
    assert isinstance(adapter, LMStudioAdapter)
    assert adapter.timeout_seconds == 600.0


def test_cloud_adapter_uses_cloud_timeout():
    from llm_bencher.config import Settings
    from llm_bencher.providers.registry import get_adapter

    settings = Settings(provider_timeout_seconds=45.0)
    provider = Provider(
        slug="openai",
        name="OpenAI",
        kind=ProviderKind.OPENAI,
        base_url="https://api.openai.com/v1",
    )
    adapter = get_adapter(provider, settings)
    assert isinstance(adapter, OpenAICloudAdapter)
    assert adapter.timeout_seconds == 45.0
