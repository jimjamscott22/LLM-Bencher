from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUITE_JSON = json.dumps({
    "slug": "run-test-suite",
    "name": "Run Test Suite",
    "prompts": [
        {
            "slug": "hello",
            "title": "Hello",
            "system_prompt": "You are helpful.",
            "user_prompt_template": "Say hi.",
            "default_temperature": 0.5,
            "default_max_tokens": 128,
        }
    ],
})


def _seed_models(client):
    """Seed the first provider with one available model and return its external_id."""
    with patch("llm_bencher.web.api.get_adapter") as mock_get_adapter:
        adapter = AsyncMock()
        adapter.health_check = AsyncMock(
            return_value=ProviderHealth(
                is_available=True, checked_at=datetime.now(timezone.utc)
            )
        )
        adapter.list_models = AsyncMock(
            return_value=[
                DiscoveredModel(
                    id="llama3:latest",
                    name="llama3:latest",
                    provider_slug="ollama",
                )
            ]
        )
        mock_get_adapter.return_value = adapter
        client.post("/api/providers/1/check")
    return "llama3:latest"


def _mock_successful_adapter(output_text="Hello!"):
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(
            output_text=output_text,
            latency_ms=420,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
    )
    return adapter


def _mock_failing_adapter(error="Connection refused"):
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(side_effect=Exception(error))
    return adapter


# ---------------------------------------------------------------------------
# runner.py unit tests
# ---------------------------------------------------------------------------

def test_build_run_request_maps_fields(client):
    """build_run_request maps Run ORM fields to RunRequest correctly."""
    from llm_bencher.models import Run, RunStatus
    from llm_bencher.runner import build_run_request

    run = Run(
        provider_id=1,
        model_identifier="llama3:latest",
        model_name="LLaMA 3",
        system_prompt="Be concise.",
        user_prompt="Hello",
        temperature=0.7,
        max_tokens=256,
        template_inputs={},
        status=RunStatus.PENDING,
    )
    req = build_run_request(run)
    assert req.provider_id == 1
    assert req.model_id == "llama3:latest"
    assert req.model_name == "LLaMA 3"
    assert req.system_prompt == "Be concise."
    assert req.user_prompt == "Hello"
    assert req.temperature == 0.7
    assert req.max_tokens == 256


# ---------------------------------------------------------------------------
# POST /api/runs — success path
# ---------------------------------------------------------------------------

def test_create_run_succeeds(client):
    _seed_models(client)

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_successful_adapter()):
        resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": "llama3:latest",
                "user_prompt": "Say hi",
            },
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "succeeded"
    assert data["output_text"] == "Hello!"
    assert data["latency_ms"] == 420
    assert data["total_tokens"] == 15
    assert data["run_id"] is not None


def test_create_run_with_all_fields(client):
    _seed_models(client)

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_successful_adapter("Done.")):
        resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": "llama3:latest",
                "model_name": "LLaMA 3",
                "system_prompt": "Be brief.",
                "user_prompt": "Explain gravity.",
                "temperature": 0.3,
                "max_tokens": 512,
            },
        )

    assert resp.status_code == 201
    assert resp.json()["status"] == "succeeded"


# ---------------------------------------------------------------------------
# POST /api/runs — failure paths
# ---------------------------------------------------------------------------

def test_create_run_adapter_failure_returns_failed_status(client):
    _seed_models(client)

    with patch(
        "llm_bencher.web.api.get_adapter",
        return_value=_mock_failing_adapter("Model not loaded"),
    ):
        resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": "llama3:latest",
                "user_prompt": "Hi",
            },
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "failed"
    assert "Model not loaded" in data["failure_message"]
    assert data["output_text"] is None


def test_create_run_unknown_provider_returns_404(client):
    resp = client.post(
        "/api/runs",
        json={
            "provider_id": 9999,
            "model_external_id": "whatever",
            "user_prompt": "Hi",
        },
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run persisted to DB
# ---------------------------------------------------------------------------

def test_run_is_persisted(client):
    """After a run, it should appear in history."""
    _seed_models(client)

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_successful_adapter()):
        run_resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": "llama3:latest",
                "user_prompt": "Persist me",
            },
        )

    run_id = run_resp.json()["run_id"]

    # History page should show the run
    history = client.get("/history")
    assert history.status_code == 200
    assert "succeeded" in history.text.lower() or str(run_id) in history.text


def test_failed_run_is_persisted_without_result(client):
    _seed_models(client)

    with patch(
        "llm_bencher.web.api.get_adapter",
        return_value=_mock_failing_adapter("timeout"),
    ):
        resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": "llama3:latest",
                "user_prompt": "Will fail",
            },
        )

    assert resp.json()["status"] == "failed"
    run_id = resp.json()["run_id"]
    assert run_id is not None


# ---------------------------------------------------------------------------
# GET /api/prompts/{id}
# ---------------------------------------------------------------------------

def test_get_prompt_not_found(client):
    assert client.get("/api/prompts/9999").status_code == 404


def test_get_prompt_returns_fields(client):
    # Import a suite to create a prompt
    client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SUITE_JSON.encode(), "application/json")},
    )
    # Get the suite to find the prompt id
    suites = client.get("/api/suites").json()
    suite_id = suites[0]["id"]
    suite_detail = client.get(f"/api/suites/{suite_id}").json()
    prompt_id = suite_detail["prompts"][0]["id"]

    resp = client.get(f"/api/prompts/{prompt_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "hello"
    assert data["system_prompt"] == "You are helpful."
    assert data["user_prompt_template"] == "Say hi."
    assert data["default_temperature"] == 0.5
    assert data["default_max_tokens"] == 128


# ---------------------------------------------------------------------------
# Run with prompt_id linked
# ---------------------------------------------------------------------------

def test_create_run_with_prompt_id(client):
    client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SUITE_JSON.encode(), "application/json")},
    )
    suites = client.get("/api/suites").json()
    suite_id = suites[0]["id"]
    suite_detail = client.get(f"/api/suites/{suite_id}").json()
    prompt_id = suite_detail["prompts"][0]["id"]

    _seed_models(client)

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_successful_adapter("Linked!")):
        resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": "llama3:latest",
                "user_prompt": "Say hi.",
                "prompt_id": prompt_id,
            },
        )

    assert resp.status_code == 201
    assert resp.json()["output_text"] == "Linked!"


# ---------------------------------------------------------------------------
# Run page renders
# ---------------------------------------------------------------------------

def test_run_page_renders(client):
    resp = client.get("/runs/new")
    assert resp.status_code == 200
    assert "Run a prompt" in resp.text
    assert "LM Studio" in resp.text or "Ollama" in resp.text
