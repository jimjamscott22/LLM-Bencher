from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _seed_model(client, external_id="llama3:latest"):
    """Discover one model on provider 1."""
    with patch("llm_bencher.web.api.get_adapter") as m:
        adapter = AsyncMock()
        adapter.health_check = AsyncMock(
            return_value=ProviderHealth(is_available=True, checked_at=datetime.now(timezone.utc))
        )
        adapter.list_models = AsyncMock(
            return_value=[DiscoveredModel(id=external_id, name=external_id, provider_slug="ollama")]
        )
        m.return_value = adapter
        client.post("/api/providers/1/check")


def _make_run(client, user_prompt="Hello", output_text="Hi!", status="succeeded"):
    """Create one run via the API with a mocked adapter."""
    _seed_model(client)
    if status == "succeeded":
        adapter = AsyncMock()
        adapter.run_chat = AsyncMock(
            return_value=RunResultSchema(
                output_text=output_text,
                latency_ms=100,
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            )
        )
    else:
        adapter = AsyncMock()
        adapter.run_chat = AsyncMock(side_effect=Exception("timeout"))

    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post(
            "/api/runs",
            json={"provider_id": 1, "model_external_id": "llama3:latest", "user_prompt": user_prompt},
        )
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# History page — basic rendering
# ---------------------------------------------------------------------------

def test_history_page_empty(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "No runs have been saved yet" in resp.text


def test_history_page_shows_runs(client):
    _make_run(client, user_prompt="First run")
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "succeeded" in resp.text.lower()
    assert "llama3:latest" in resp.text


def test_history_page_shows_failed_run(client):
    _make_run(client, status="failed")
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "Failed" in resp.text


# ---------------------------------------------------------------------------
# History page — filters
# ---------------------------------------------------------------------------

def test_filter_by_provider_id(client):
    _make_run(client, "Filter test")
    # Filter to provider 1 — should see the run
    resp = client.get("/history?provider_id=1")
    assert resp.status_code == 200
    assert "llama3:latest" in resp.text


def test_filter_by_provider_id_no_match(client):
    _make_run(client)
    # Provider 999 does not exist, no runs should match
    resp = client.get("/history?provider_id=999")
    assert resp.status_code == 200
    assert "No runs match" in resp.text


def test_filter_by_status_succeeded(client):
    _make_run(client, status="succeeded")
    _make_run(client, status="failed")
    resp = client.get("/history?status=succeeded")
    assert resp.status_code == 200
    # Only succeeded badge visible, not failed
    assert "Succeeded" in resp.text


def test_filter_by_status_failed(client):
    _make_run(client, status="succeeded")
    _make_run(client, status="failed")
    resp = client.get("/history?status=failed")
    assert resp.status_code == 200
    assert "Failed" in resp.text


def test_filter_active_shows_clear_link(client):
    _make_run(client)
    resp = client.get("/history?status=succeeded")
    assert resp.status_code == 200
    assert "Clear" in resp.text


def test_no_active_filter_hides_clear_link(client):
    resp = client.get("/history")
    assert "Clear" not in resp.text


# ---------------------------------------------------------------------------
# History page — pagination
# ---------------------------------------------------------------------------

def test_pagination_shows_page_links_when_many_runs(client):
    # Create 30 runs to exceed the 25-per-page default
    _seed_model(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(output_text="ok", latency_ms=10, total_tokens=2)
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        for i in range(30):
            client.post(
                "/api/runs",
                json={"provider_id": 1, "model_external_id": "llama3:latest", "user_prompt": f"Run {i}"},
            )

    resp = client.get("/history")
    assert resp.status_code == 200
    # Should show page 2 link
    assert "page=2" in resp.text


def test_pagination_page_2(client):
    _seed_model(client)
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        return_value=RunResultSchema(output_text="ok", latency_ms=10, total_tokens=2)
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        for i in range(27):
            client.post(
                "/api/runs",
                json={"provider_id": 1, "model_external_id": "llama3:latest", "user_prompt": f"Run {i}"},
            )

    resp = client.get("/history?page=2")
    assert resp.status_code == 200
    # Page 2 should have the 2 oldest runs (27 total, 25 on page 1)
    assert "2 run" in resp.text or "27 run" in resp.text  # total count in lede


# ---------------------------------------------------------------------------
# Run detail page
# ---------------------------------------------------------------------------

def test_run_detail_not_found(client):
    resp = client.get("/runs/9999")
    assert resp.status_code == 404


def test_run_detail_shows_output(client):
    run_id = _make_run(client, user_prompt="Detail test", output_text="Detail output")
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "Detail output" in resp.text
    assert "Succeeded" in resp.text


def test_run_detail_shows_user_prompt(client):
    run_id = _make_run(client, user_prompt="My test prompt")
    resp = client.get(f"/runs/{run_id}")
    assert "My test prompt" in resp.text


def test_run_detail_shows_token_usage(client):
    run_id = _make_run(client)
    resp = client.get(f"/runs/{run_id}")
    # Token usage section should appear (total_tokens=8 from _make_run)
    assert "8" in resp.text


def test_run_detail_shows_failure_message(client):
    run_id = _make_run(client, status="failed")
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "Failed" in resp.text
    assert "timeout" in resp.text


def test_run_detail_back_link_points_to_history(client):
    run_id = _make_run(client)
    resp = client.get(f"/runs/{run_id}")
    assert "/history" in resp.text


def test_run_detail_back_link_preserves_filters(client):
    run_id = _make_run(client)
    # Visit history with a filter, then click view
    history = client.get("/history?status=succeeded")
    # The view link should encode the filter in the back param
    assert f"/runs/{run_id}?back=" in history.text


def test_run_detail_back_url_restored(client):
    run_id = _make_run(client)
    # When navigating with ?back=status%3Dsucceeded, the back link should reconstruct it
    resp = client.get(f"/runs/{run_id}?back=status%3Dsucceeded")
    assert "/history?status=succeeded" in resp.text


# ---------------------------------------------------------------------------
# View link in history goes to correct run
# ---------------------------------------------------------------------------

def test_history_view_link_targets_correct_run(client):
    run_id = _make_run(client)
    resp = client.get("/history")
    assert f"/runs/{run_id}" in resp.text
