"""Edge case and boundary condition tests."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, patch

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(output="ok", latency=100, tokens=50, succeed=True):
    adapter = AsyncMock()
    if succeed:
        adapter.run_chat = AsyncMock(return_value=RunResultSchema(
            output_text=output,
            latency_ms=latency,
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
        ))
    else:
        adapter.run_chat = AsyncMock(side_effect=Exception("error"))

    adapter.health_check = AsyncMock(
        return_value=ProviderHealth(is_available=True, checked_at=datetime.now(timezone.utc))
    )
    adapter.list_models = AsyncMock(return_value=[
        DiscoveredModel(id="edge-model", name="edge-model", provider_slug="test"),
    ])
    return adapter


def _seed(client):
    providers = client.get("/api/providers").json()
    pid = providers[0]["id"]
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        client.post(f"/api/providers/{pid}/check")
    return pid


# ---------------------------------------------------------------------------
# Unicode handling
# ---------------------------------------------------------------------------

def test_unicode_in_prompt(client):
    """Unicode characters in prompts are preserved."""
    pid = _seed(client)
    adapter = _make_adapter(output="Bonjour!")
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "Translate: cafe\u0301 \u2603 \U0001f600",
        })
    assert resp.status_code == 201
    assert resp.json()["status"] == "succeeded"


def test_unicode_in_output(client):
    """Unicode in model output is handled correctly."""
    pid = _seed(client)
    adapter = _make_adapter(output="\U0001f680 Unicode output \u2764\ufe0f with CJK: \u4f60\u597d")
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "Hello",
        })
    assert resp.status_code == 201
    assert resp.json()["output_text"] == "\U0001f680 Unicode output \u2764\ufe0f with CJK: \u4f60\u597d"


# ---------------------------------------------------------------------------
# Long content
# ---------------------------------------------------------------------------

def test_very_long_prompt(client):
    """Very long prompts are accepted and stored."""
    pid = _seed(client)
    long_prompt = "x" * 50000
    adapter = _make_adapter(output="ok")
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": long_prompt,
        })
    assert resp.status_code == 201


def test_very_long_output(client):
    """Very long model outputs are stored and exported."""
    pid = _seed(client)
    long_output = "word " * 10000
    adapter = _make_adapter(output=long_output)
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "Write a lot",
        })
    assert resp.status_code == 201

    # Export should contain the full output.
    csv_resp = client.get("/api/export/history")
    assert len(csv_resp.text) > 40000


# ---------------------------------------------------------------------------
# Empty / boundary conditions
# ---------------------------------------------------------------------------

def test_empty_batch_validation(client):
    """Batch with no models or no prompts is rejected."""
    resp = client.post("/api/batches", json={
        "models": [],
        "prompt_ids": [1],
    })
    assert resp.status_code == 422

    resp = client.post("/api/batches", json={
        "models": [{"provider_id": 1, "model_external_id": "x"}],
        "prompt_ids": [],
    })
    assert resp.status_code == 422


def test_comparison_requires_two_runs(client):
    """Comparison with fewer than 2 runs is rejected."""
    pid = _seed(client)
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        r = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "solo",
        })
    run_id = r.json()["run_id"]

    resp = client.post("/api/comparisons", json={"run_ids": [run_id]})
    assert resp.status_code == 422


def test_rating_boundary_values(client):
    """Rating at boundaries (1 and 5) should be accepted."""
    pid = _seed(client)
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        r = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "test",
        })
    run_id = r.json()["run_id"]

    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 1})
    assert resp.status_code == 201

    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 5})
    assert resp.status_code == 200  # Update


def test_rating_out_of_bounds(client):
    """Rating outside 1-5 range is rejected."""
    pid = _seed(client)
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        r = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "test",
        })
    run_id = r.json()["run_id"]

    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 0})
    assert resp.status_code == 422

    resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 6})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# JSON column edge cases
# ---------------------------------------------------------------------------

def test_empty_tags_in_suite(client):
    """Suite with empty tag arrays should work fine."""
    suite_json = '{"slug":"empty-tags","name":"Empty Tags","prompts":[{"slug":"p1","title":"P1","user_prompt_template":"hi","tags":[]}]}'
    resp = client.post("/api/suites/import", files={
        "file": ("t.json", BytesIO(suite_json.encode()), "application/json")
    })
    assert resp.status_code == 201


def test_template_inputs_with_empty_dict(client):
    """Run with empty template_inputs dict."""
    pid = _seed(client)
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "test",
            "template_inputs": {},
        })
    assert resp.status_code == 201


def test_template_inputs_with_special_chars(client):
    """Template inputs with special characters."""
    pid = _seed(client)
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "edge-model",
            "user_prompt": "test with {var}",
            "template_inputs": {"var": "hello 'world' \"quoted\" <tag>&amp;"},
        })
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Provider edge cases
# ---------------------------------------------------------------------------

def test_provider_slug_uniqueness(client):
    """Cannot create two providers with the same slug."""
    client.post("/api/providers", json={
        "slug": "unique-test",
        "name": "Unique Test",
        "kind": "openai_compat",
        "base_url": "http://localhost:1111/v1",
    })
    resp = client.post("/api/providers", json={
        "slug": "unique-test",
        "name": "Duplicate",
        "kind": "openai_compat",
        "base_url": "http://localhost:2222/v1",
    })
    assert resp.status_code == 409


def test_update_nonexistent_provider(client):
    resp = client.put("/api/providers/99999", json={"name": "Ghost"})
    assert resp.status_code == 404


def test_delete_nonexistent_provider(client):
    resp = client.delete("/api/providers/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Pages render without errors
# ---------------------------------------------------------------------------

def test_all_pages_render_empty_state(client):
    """All major pages should render without errors even with no data."""
    pages = ["/", "/providers", "/prompts", "/runs/new", "/runs/batch", "/history", "/analytics"]
    for page in pages:
        resp = client.get(page)
        assert resp.status_code == 200, f"{page} returned {resp.status_code}"


def test_history_pagination_edge(client):
    """Page 0 or negative should default to page 1."""
    resp = client.get("/history?page=0")
    assert resp.status_code == 200

    resp = client.get("/history?page=-1")
    assert resp.status_code == 200

    resp = client.get("/history?page=abc")
    assert resp.status_code == 200


def test_nonexistent_run_detail_returns_404(client):
    resp = client.get("/runs/99999")
    assert resp.status_code == 404


def test_nonexistent_batch_detail_returns_404(client):
    resp = client.get("/batches/99999")
    assert resp.status_code == 404


def test_nonexistent_comparison_returns_404(client):
    resp = client.get("/compare/99999")
    assert resp.status_code == 404
