from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TAGGED_SUITE = json.dumps({
    "slug": "tagged-suite",
    "name": "Tagged Suite",
    "prompts": [
        {
            "slug": "writing-task",
            "title": "Writing Task",
            "user_prompt_template": "Write a story.",
            "tags": ["writing", "creative"],
        },
        {
            "slug": "math-task",
            "title": "Math Task",
            "user_prompt_template": "Solve 2+2.",
            "tags": ["math", "reasoning"],
        },
        {
            "slug": "untagged-task",
            "title": "Untagged Task",
            "user_prompt_template": "Do something.",
            "tags": [],
        },
    ],
})

SECOND_SUITE = json.dumps({
    "slug": "second-suite",
    "name": "Second Suite",
    "prompts": [
        {
            "slug": "creative-writing",
            "title": "Creative Writing",
            "user_prompt_template": "Write a poem.",
            "tags": ["writing", "poetry"],
        },
    ],
})


def _import_suite(client, suite_json: str):
    """Import a suite JSON string via the API."""
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


def _make_run(client, prompt_id=None, user_prompt="Hello"):
    """Create a succeeded run, optionally linked to a prompt."""
    model_id = _seed_models(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(
            output_text="Response",
            latency_ms=100,
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
        )
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        body = {
            "provider_id": 1,
            "model_external_id": model_id,
            "user_prompt": user_prompt,
        }
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        resp = client.post("/api/runs", json=body)
    assert resp.status_code == 201
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------

def test_tags_empty(client):
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert resp.json() == []


def test_tags_returns_unique_sorted(client):
    _import_suite(client, TAGGED_SUITE)
    _import_suite(client, SECOND_SUITE)
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    tags = resp.json()
    assert tags == ["creative", "math", "poetry", "reasoning", "writing"]


def test_tags_excludes_deleted_suites(client):
    data = _import_suite(client, TAGGED_SUITE)
    client.delete(f"/api/suites/{data['id']}")
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Prompts page tag filtering
# ---------------------------------------------------------------------------

def test_prompts_page_shows_tag_pills(client):
    _import_suite(client, TAGGED_SUITE)
    resp = client.get("/prompts")
    assert resp.status_code == 200
    assert "tag-pill" in resp.text
    assert "writing" in resp.text
    assert "math" in resp.text


def test_prompts_page_filter_by_tag(client):
    _import_suite(client, TAGGED_SUITE)
    resp = client.get("/prompts?tag=math")
    assert resp.status_code == 200
    assert "math" in resp.text
    assert "Showing prompts tagged" in resp.text


def test_prompts_page_filter_no_match(client):
    _import_suite(client, TAGGED_SUITE)
    resp = client.get("/prompts?tag=nonexistent")
    assert resp.status_code == 200
    assert "No suites imported yet" in resp.text or "Showing prompts tagged" in resp.text


def test_prompts_page_no_tags_no_pill_bar(client):
    """Without imported suites, no tag bar is rendered."""
    resp = client.get("/prompts")
    assert resp.status_code == 200
    assert "tag-pill" not in resp.text


def test_prompts_page_clear_link(client):
    _import_suite(client, TAGGED_SUITE)
    resp = client.get("/prompts?tag=writing")
    assert resp.status_code == 200
    assert "Clear" in resp.text


# ---------------------------------------------------------------------------
# History page tag filtering
# ---------------------------------------------------------------------------

def test_history_filter_by_tag(client):
    data = _import_suite(client, TAGGED_SUITE)
    # Get the prompt IDs from the suite.
    suite_resp = client.get(f"/api/suites/{data['id']}")
    prompts = suite_resp.json()["prompts"]
    writing_prompt = next(p for p in prompts if p["slug"] == "writing-task")
    math_prompt = next(p for p in prompts if p["slug"] == "math-task")

    # Create runs linked to prompts.
    _make_run(client, prompt_id=writing_prompt["id"], user_prompt="Write a story.")
    _make_run(client, prompt_id=math_prompt["id"], user_prompt="Solve 2+2.")

    # Filter by "writing" tag — should only show the writing run.
    resp = client.get("/history?tag=writing")
    assert resp.status_code == 200
    text = resp.text
    assert "Writing Task" in text
    # The math run should not appear.
    assert "Math Task" not in text


def test_history_filter_by_tag_no_match(client):
    data = _import_suite(client, TAGGED_SUITE)
    suite_resp = client.get(f"/api/suites/{data['id']}")
    prompts = suite_resp.json()["prompts"]
    writing_prompt = next(p for p in prompts if p["slug"] == "writing-task")
    _make_run(client, prompt_id=writing_prompt["id"], user_prompt="Write.")

    resp = client.get("/history?tag=nonexistent")
    assert resp.status_code == 200
    assert "No runs match" in resp.text or "0 run" in resp.text


def test_history_tag_dropdown_rendered(client):
    _import_suite(client, TAGGED_SUITE)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "All tags" in resp.text
    assert "writing" in resp.text


def test_history_no_tags_no_dropdown(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "All tags" not in resp.text
