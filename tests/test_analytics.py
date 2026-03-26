from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_models(client, provider_id=1):
    with patch("llm_bencher.web.api.get_adapter") as mock:
        adapter = AsyncMock()
        adapter.health_check = AsyncMock(
            return_value=ProviderHealth(is_available=True, checked_at=datetime.now(timezone.utc))
        )
        adapter.list_models = AsyncMock(return_value=[
            DiscoveredModel(id="model-a", name="model-a", provider_slug="ollama"),
            DiscoveredModel(id="model-b", name="model-b", provider_slug="ollama"),
        ])
        mock.return_value = adapter
        client.post(f"/api/providers/{provider_id}/check")


def _create_run(client, model_id="model-a", succeed=True, latency_ms=100, tokens=50, provider_id=None):
    """Create a run with a mocked adapter."""
    if provider_id is None:
        providers = client.get("/api/providers").json()
        provider_id = providers[0]["id"]

    if succeed:
        adapter = AsyncMock()
        adapter.run_chat = AsyncMock(return_value=RunResultSchema(
            output_text="test output",
            latency_ms=latency_ms,
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
        ))
    else:
        adapter = AsyncMock()
        adapter.run_chat = AsyncMock(side_effect=Exception("fail"))

    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": provider_id,
            "model_external_id": model_id,
            "user_prompt": "Hello",
        })
    return resp.json()


# ---------------------------------------------------------------------------
# Summary endpoint
# ---------------------------------------------------------------------------

def test_summary_empty(client):
    resp = client.get("/api/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_runs"] == 0
    assert data["success_rate"] == 0
    assert data["avg_latency_ms"] == 0
    assert data["total_tokens"] == 0


def test_summary_with_runs(client):
    _seed_models(client)
    _create_run(client, "model-a", succeed=True, latency_ms=200, tokens=100)
    _create_run(client, "model-b", succeed=True, latency_ms=400, tokens=200)
    _create_run(client, "model-a", succeed=False)

    resp = client.get("/api/analytics/summary")
    data = resp.json()
    assert data["total_runs"] == 3
    assert data["succeeded"] == 2
    assert data["failed"] == 1
    assert data["success_rate"] == 66.7
    assert data["avg_latency_ms"] == 300.0
    assert data["total_tokens"] == 300


# ---------------------------------------------------------------------------
# Latency endpoint
# ---------------------------------------------------------------------------

def test_latency_empty(client):
    resp = client.get("/api/analytics/latency")
    assert resp.status_code == 200
    assert resp.json() == []


def test_latency_grouped_by_model(client):
    _seed_models(client)
    _create_run(client, "model-a", latency_ms=100)
    _create_run(client, "model-a", latency_ms=300)
    _create_run(client, "model-b", latency_ms=200)

    resp = client.get("/api/analytics/latency")
    data = resp.json()
    models = {d["model"]: d for d in data}
    assert "model-a" in models
    assert "model-b" in models
    assert models["model-a"]["avg_ms"] == 200.0
    assert models["model-a"]["min_ms"] == 100
    assert models["model-a"]["max_ms"] == 300
    assert models["model-a"]["count"] == 2
    assert models["model-b"]["count"] == 1


# ---------------------------------------------------------------------------
# Tokens endpoint
# ---------------------------------------------------------------------------

def test_tokens_empty(client):
    resp = client.get("/api/analytics/tokens")
    assert resp.status_code == 200
    assert resp.json() == []


def test_tokens_grouped_by_model(client):
    _seed_models(client)
    _create_run(client, "model-a", tokens=100)
    _create_run(client, "model-a", tokens=200)
    _create_run(client, "model-b", tokens=50)

    resp = client.get("/api/analytics/tokens")
    data = resp.json()
    models = {d["model"]: d for d in data}
    assert models["model-a"]["total_tokens"] == 300
    assert models["model-b"]["total_tokens"] == 50


# ---------------------------------------------------------------------------
# Success rate endpoint
# ---------------------------------------------------------------------------

def test_success_rate_empty(client):
    resp = client.get("/api/analytics/success-rate")
    assert resp.status_code == 200
    assert resp.json() == []


def test_success_rate_calculation(client):
    _seed_models(client)
    _create_run(client, "model-a", succeed=True)
    _create_run(client, "model-a", succeed=True)
    _create_run(client, "model-a", succeed=False)

    resp = client.get("/api/analytics/success-rate")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["model"] == "model-a"
    assert data[0]["succeeded"] == 2
    assert data[0]["failed"] == 1
    assert data[0]["rate"] == 66.7


# ---------------------------------------------------------------------------
# Timeline endpoint
# ---------------------------------------------------------------------------

def test_timeline_empty(client):
    resp = client.get("/api/analytics/timeline?period=30d")
    assert resp.status_code == 200
    assert resp.json() == []


def test_timeline_with_data(client):
    _seed_models(client)
    _create_run(client, "model-a", latency_ms=100)
    _create_run(client, "model-a", latency_ms=200)

    resp = client.get("/api/analytics/timeline?period=30d")
    data = resp.json()
    assert len(data) >= 1
    assert "day" in data[0]
    assert "avg_latency_ms" in data[0]
    assert data[0]["count"] == 2


# ---------------------------------------------------------------------------
# Analytics page
# ---------------------------------------------------------------------------

def test_analytics_page_renders(client):
    resp = client.get("/analytics")
    assert resp.status_code == 200
    assert "Analytics dashboard" in resp.text
    assert "stat-total-runs" in resp.text


def test_analytics_nav_link(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Analytics" in resp.text
