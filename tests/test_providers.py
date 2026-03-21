from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_bencher.providers.lm_studio import LMStudioAdapter
from llm_bencher.providers.ollama import OllamaAdapter
from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LMS_PATCH = "llm_bencher.providers.openai_compat.httpx.AsyncClient"
_OLLAMA_PATCH = "llm_bencher.providers.ollama.httpx.AsyncClient"


def _make_httpx_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _mock_client(get_resp=None, post_resp=None, side_effect=None):
    """Return a mock AsyncClient context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    if side_effect:
        client.get = AsyncMock(side_effect=side_effect)
        client.post = AsyncMock(side_effect=side_effect)
    else:
        if get_resp is not None:
            client.get = AsyncMock(return_value=get_resp)
        if post_resp is not None:
            client.post = AsyncMock(return_value=post_resp)
    return client


# ---------------------------------------------------------------------------
# LMStudioAdapter unit tests
# ---------------------------------------------------------------------------

class TestLMStudioAdapter:
    def setup_method(self):
        self.adapter = LMStudioAdapter(base_url="http://localhost:1234/v1", timeout=5.0)

    def test_health_check_available(self):
        resp = _make_httpx_response(200, {"object": "list", "data": []})
        with patch(_LMS_PATCH, return_value=_mock_client(get_resp=resp)):
            health = asyncio.run(self.adapter.health_check())
        assert health.is_available is True
        assert health.detail is None
        assert health.checked_at is not None

    def test_health_check_unavailable_on_connection_error(self):
        with patch(_LMS_PATCH, return_value=_mock_client(side_effect=Exception("Connection refused"))):
            health = asyncio.run(self.adapter.health_check())
        assert health.is_available is False
        assert "Connection refused" in health.detail

    def test_list_models(self):
        resp = _make_httpx_response(200, {
            "object": "list",
            "data": [
                {"id": "llama-3.1-8b", "object": "model"},
                {"id": "mistral-7b", "object": "model"},
            ],
        })
        with patch(_LMS_PATCH, return_value=_mock_client(get_resp=resp)):
            models = asyncio.run(self.adapter.list_models())
        assert len(models) == 2
        assert models[0].id == "llama-3.1-8b"
        assert models[0].provider_slug == "lm-studio"
        assert models[1].id == "mistral-7b"

    def test_run_chat(self):
        resp = _make_httpx_response(200, {
            "id": "chatcmpl-123",
            "model": "llama-3.1-8b",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        with patch(_LMS_PATCH, return_value=_mock_client(post_resp=resp)):
            req = RunRequest(
                provider_id=1,
                model_id="llama-3.1-8b",
                user_prompt="Say hi",
                system_prompt="You are helpful.",
            )
            result = asyncio.run(self.adapter.run_chat(req))
        assert result.output_text == "Hello!"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.latency_ms is not None


# ---------------------------------------------------------------------------
# OllamaAdapter unit tests
# ---------------------------------------------------------------------------

class TestOllamaAdapter:
    def setup_method(self):
        self.adapter = OllamaAdapter(base_url="http://localhost:11434", timeout=5.0)

    def test_health_check_available(self):
        resp = _make_httpx_response(200, {"models": []})
        with patch(_OLLAMA_PATCH, return_value=_mock_client(get_resp=resp)):
            health = asyncio.run(self.adapter.health_check())
        assert health.is_available is True

    def test_health_check_unavailable(self):
        with patch(_OLLAMA_PATCH, return_value=_mock_client(side_effect=Exception("timeout"))):
            health = asyncio.run(self.adapter.health_check())
        assert health.is_available is False
        assert "timeout" in health.detail

    def test_list_models(self):
        resp = _make_httpx_response(200, {
            "models": [
                {"name": "llama3:latest", "size": 4700000000},
                {"name": "mistral:7b", "size": 4100000000},
            ]
        })
        with patch(_OLLAMA_PATCH, return_value=_mock_client(get_resp=resp)):
            models = asyncio.run(self.adapter.list_models())
        assert len(models) == 2
        assert models[0].id == "llama3:latest"
        assert models[0].provider_slug == "ollama"

    def test_run_chat(self):
        resp = _make_httpx_response(200, {
            "model": "llama3:latest",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 4,
        })
        with patch(_OLLAMA_PATCH, return_value=_mock_client(post_resp=resp)):
            req = RunRequest(provider_id=1, model_id="llama3:latest", user_prompt="Say hi")
            result = asyncio.run(self.adapter.run_chat(req))
        assert result.output_text == "Hi there!"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4
        assert result.total_tokens == 12


# ---------------------------------------------------------------------------
# API endpoint tests (via TestClient)
# ---------------------------------------------------------------------------

def test_check_provider_not_found(client) -> None:
    response = client.post("/api/providers/9999/check")
    assert response.status_code == 404


def test_get_provider_models_not_found(client) -> None:
    response = client.get("/api/providers/9999/models")
    assert response.status_code == 404


def test_get_provider_models_empty(client) -> None:
    """Seeded providers have no models yet — endpoint returns empty list."""
    assert client.get("/providers").status_code == 200
    response = client.get("/api/providers/1/models")
    assert response.status_code == 200
    assert response.json() == []


def _mock_adapter(is_connected: bool, models: list[DiscoveredModel] | None = None):
    adapter = AsyncMock()
    adapter.health_check = AsyncMock(
        return_value=ProviderHealth(
            is_available=is_connected,
            detail=None if is_connected else "Connection refused",
            checked_at=datetime.now(timezone.utc),
        )
    )
    adapter.list_models = AsyncMock(return_value=models or [])
    return adapter


def test_check_provider_updates_connection_state(client) -> None:
    """When the provider is unreachable, is_connected becomes False."""
    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter(is_connected=False)):
        response = client.post("/api/providers/1/check")

    assert response.status_code == 200
    data = response.json()
    assert data["is_connected"] is False
    assert "Connection refused" in data["detail"]
    assert data["models"] == []


def test_check_provider_discovers_models(client) -> None:
    """When healthy, discovered models are persisted and returned."""
    discovered = [
        DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama"),
        DiscoveredModel(id="mistral:7b", name="mistral:7b", provider_slug="ollama"),
    ]
    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter(True, discovered)):
        response = client.post("/api/providers/1/check")

    assert response.status_code == 200
    data = response.json()
    assert data["is_connected"] is True
    assert len(data["models"]) == 2
    model_ids = {m["external_id"] for m in data["models"]}
    assert {"llama3:latest", "mistral:7b"} == model_ids

    # Models persist and are accessible via GET
    get_resp = client.get("/api/providers/1/models")
    assert get_resp.status_code == 200
    assert len(get_resp.json()) == 2


def test_check_provider_marks_stale_models_unavailable(client) -> None:
    """Models not returned in a subsequent check are marked is_available=False."""
    two_models = [
        DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama"),
        DiscoveredModel(id="mistral:7b", name="mistral:7b", provider_slug="ollama"),
    ]
    # First check: two models discovered
    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter(True, two_models)):
        client.post("/api/providers/1/check")

    # Second check: only one model returned — mistral:7b goes stale
    one_model = [DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama")]
    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter(True, one_model)):
        response = client.post("/api/providers/1/check")

    data = response.json()
    assert data["is_connected"] is True
    by_id = {m["external_id"]: m for m in data["models"]}
    assert by_id["llama3:latest"]["is_available"] is True
    assert by_id["mistral:7b"]["is_available"] is False
