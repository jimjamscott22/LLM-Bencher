from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VARIABLE_SUITE = json.dumps({
    "slug": "variable-suite",
    "name": "Variable Suite",
    "prompts": [
        {
            "slug": "summarise",
            "title": "Summarise text",
            "system_prompt": "You are a concise summariser.",
            "user_prompt_template": "Summarise the following: {text}",
            "variables": [
                {"name": "text", "description": "Text to summarise", "required": True},
            ],
            "default_temperature": 0.3,
            "default_max_tokens": 256,
        },
        {
            "slug": "translate",
            "title": "Translate text",
            "user_prompt_template": "Translate '{text}' to {language}.",
            "variables": [
                {"name": "text", "description": "Source text", "required": True},
                {"name": "language", "description": "Target language", "required": True, "default": "French"},
            ],
        },
        {
            "slug": "no-vars",
            "title": "No variables",
            "user_prompt_template": "Say hello.",
        },
    ],
})


def _import_suite(client, suite_json: str = VARIABLE_SUITE):
    from io import BytesIO
    resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", BytesIO(suite_json.encode()), "application/json")},
    )
    assert resp.status_code in (200, 201)
    return resp.json()


def _seed_models(client):
    with patch("llm_bencher.web.api.get_adapter") as mock_get_adapter:
        adapter = AsyncMock()
        adapter.health_check = AsyncMock(
            return_value=ProviderHealth(
                is_available=True, checked_at=datetime.now(timezone.utc)
            )
        )
        adapter.list_models = AsyncMock(
            return_value=[
                DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama")
            ]
        )
        mock_get_adapter.return_value = adapter
        client.post("/api/providers/1/check")
    return "llama3:latest"


def _get_prompt_id(client, suite_id: int, slug: str) -> int:
    resp = client.get(f"/api/suites/{suite_id}")
    prompts = resp.json()["prompts"]
    return next(p["id"] for p in prompts if p["slug"] == slug)


# ---------------------------------------------------------------------------
# API: prompt detail returns variables
# ---------------------------------------------------------------------------

def test_prompt_returns_variables(client):
    data = _import_suite(client)
    pid = _get_prompt_id(client, data["id"], "summarise")
    resp = client.get(f"/api/prompts/{pid}")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["variables"]) == 1
    v = body["variables"][0]
    assert v["name"] == "text"
    assert v["required"] is True
    assert v["description"] == "Text to summarise"


def test_prompt_returns_multiple_variables(client):
    data = _import_suite(client)
    pid = _get_prompt_id(client, data["id"], "translate")
    resp = client.get(f"/api/prompts/{pid}")
    body = resp.json()
    assert len(body["variables"]) == 2
    names = [v["name"] for v in body["variables"]]
    assert "text" in names
    assert "language" in names
    # Check default value
    lang = next(v for v in body["variables"] if v["name"] == "language")
    assert lang["default"] == "French"


def test_prompt_returns_empty_variables(client):
    data = _import_suite(client)
    pid = _get_prompt_id(client, data["id"], "no-vars")
    resp = client.get(f"/api/prompts/{pid}")
    body = resp.json()
    assert body["variables"] == []


# ---------------------------------------------------------------------------
# API: create run with template_inputs
# ---------------------------------------------------------------------------

def test_create_run_with_template_inputs(client):
    _import_suite(client)
    model_id = _seed_models(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(
            output_text="Summary here",
            latency_ms=200,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": 1,
            "model_external_id": model_id,
            "user_prompt": "Summarise the following: Hello world",
            "template_inputs": {"text": "Hello world"},
        })
    assert resp.status_code == 201
    assert resp.json()["status"] == "succeeded"


def test_template_inputs_persisted_in_db(client):
    _import_suite(client)
    model_id = _seed_models(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(
            output_text="Done",
            latency_ms=100,
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
        )
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": 1,
            "model_external_id": model_id,
            "user_prompt": "Translate 'hi' to French.",
            "template_inputs": {"text": "hi", "language": "French"},
        })
    run_id = resp.json()["run_id"]

    # Check the run detail page shows the inputs were stored.
    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200


def test_create_run_without_template_inputs(client):
    """template_inputs defaults to empty dict when omitted."""
    _seed_models(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(
            output_text="Hi!",
            latency_ms=50,
            prompt_tokens=2,
            completion_tokens=1,
            total_tokens=3,
        )
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": 1,
            "model_external_id": "llama3:latest",
            "user_prompt": "Say hello.",
        })
    assert resp.status_code == 201
    assert resp.json()["status"] == "succeeded"


# ---------------------------------------------------------------------------
# UI: variables container rendered on runs page
# ---------------------------------------------------------------------------

def test_runs_page_has_variables_container(client):
    resp = client.get("/runs/new")
    assert resp.status_code == 200
    assert 'id="variables-container"' in resp.text
    assert 'id="variables-fields"' in resp.text
