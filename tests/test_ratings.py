from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_models(client):
    """Seed provider 1 with one available model."""
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


def _make_run(client, user_prompt="Hello", output_text="Hi!"):
    """Create a succeeded run and return the run_id."""
    model_id = _seed_models(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(
            output_text=output_text,
            latency_ms=420,
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
        )
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post(
            "/api/runs",
            json={
                "provider_id": 1,
                "model_external_id": model_id,
                "user_prompt": user_prompt,
            },
        )
    assert resp.status_code == 201
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# POST /api/runs/{run_id}/rating — create / update
# ---------------------------------------------------------------------------

def test_create_rating(client):
    run_id = _make_run(client)
    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 4})
    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == run_id
    assert data["score"] == 4
    assert data["notes"] is None
    assert data["action"] == "created"


def test_create_rating_with_notes(client):
    run_id = _make_run(client)
    resp = client.post(
        f"/api/runs/{run_id}/rating",
        json={"score": 5, "notes": "Great answer!"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["score"] == 5
    assert data["notes"] == "Great answer!"


def test_update_existing_rating(client):
    run_id = _make_run(client)
    client.post(f"/api/runs/{run_id}/rating", json={"score": 3})
    resp = client.post(
        f"/api/runs/{run_id}/rating",
        json={"score": 5, "notes": "Changed my mind"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] == 5
    assert data["notes"] == "Changed my mind"
    assert data["action"] == "updated"


def test_create_rating_invalid_score_zero(client):
    run_id = _make_run(client)
    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 0})
    assert resp.status_code == 422


def test_create_rating_invalid_score_six(client):
    run_id = _make_run(client)
    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 6})
    assert resp.status_code == 422


def test_create_rating_nonexistent_run(client):
    resp = client.post("/api/runs/9999/rating", json={"score": 3})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/rating
# ---------------------------------------------------------------------------

def test_get_rating(client):
    run_id = _make_run(client)
    client.post(f"/api/runs/{run_id}/rating", json={"score": 4, "notes": "Good"})
    resp = client.get(f"/api/runs/{run_id}/rating")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert data["score"] == 4
    assert data["notes"] == "Good"
    assert "created_at" in data


def test_get_rating_not_found(client):
    run_id = _make_run(client)
    resp = client.get(f"/api/runs/{run_id}/rating")
    assert resp.status_code == 404


def test_get_rating_nonexistent_run(client):
    resp = client.get("/api/runs/9999/rating")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/runs/{run_id}/rating
# ---------------------------------------------------------------------------

def test_delete_rating(client):
    run_id = _make_run(client)
    client.post(f"/api/runs/{run_id}/rating", json={"score": 2})
    resp = client.delete(f"/api/runs/{run_id}/rating")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Confirm it's gone.
    resp = client.get(f"/api/runs/{run_id}/rating")
    assert resp.status_code == 404


def test_delete_rating_not_found(client):
    run_id = _make_run(client)
    resp = client.delete(f"/api/runs/{run_id}/rating")
    assert resp.status_code == 404


def test_delete_rating_nonexistent_run(client):
    resp = client.delete("/api/runs/9999/rating")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Template integration
# ---------------------------------------------------------------------------

def test_run_detail_shows_rating_widget(client):
    run_id = _make_run(client)
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "rating-panel" in resp.text
    assert "star-btn" in resp.text


def test_run_detail_shows_existing_rating(client):
    run_id = _make_run(client)
    client.post(f"/api/runs/{run_id}/rating", json={"score": 4, "notes": "Solid"})
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "4/5" in resp.text
    assert "Solid" in resp.text


def test_history_shows_rating_indicator(client):
    run_id = _make_run(client)
    client.post(f"/api/runs/{run_id}/rating", json={"score": 3})
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "rating-indicator" in resp.text
    assert "Rated 3/5" in resp.text


def test_history_no_rating_no_indicator(client):
    _make_run(client)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "rating-indicator" not in resp.text
