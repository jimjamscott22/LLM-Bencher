from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUITE = json.dumps({
    "slug": "compare-suite",
    "name": "Compare Suite",
    "prompts": [
        {"slug": "hello", "title": "Hello", "user_prompt_template": "Say hello."},
        {"slug": "goodbye", "title": "Goodbye", "user_prompt_template": "Say goodbye."},
    ],
})


def _import_suite(client):
    from io import BytesIO
    resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", BytesIO(SUITE.encode()), "application/json")},
    )
    assert resp.status_code in (200, 201)
    return resp.json()


def _seed_models(client, models=None):
    if models is None:
        models = [
            DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama"),
            DiscoveredModel(id="mistral:latest", name="mistral:latest", provider_slug="ollama"),
        ]
    with patch("llm_bencher.web.api.get_adapter") as mock:
        adapter = AsyncMock()
        adapter.health_check = AsyncMock(
            return_value=ProviderHealth(is_available=True, checked_at=datetime.now(timezone.utc))
        )
        adapter.list_models = AsyncMock(return_value=models)
        mock.return_value = adapter
        client.post("/api/providers/1/check")


def _get_prompt_ids(client, suite_id: int) -> list[int]:
    resp = client.get(f"/api/suites/{suite_id}")
    return [p["id"] for p in resp.json()["prompts"]]


def _mock_adapter(output_text="Response"):
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
    return adapter


def _make_run(client, user_prompt="Hello", model_id="llama3:latest"):
    adapter = _mock_adapter(output_text=f"Reply to: {user_prompt}")
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": 1,
            "model_external_id": model_id,
            "user_prompt": user_prompt,
        })
    assert resp.status_code == 201
    return resp.json()["run_id"]


def _make_batch(client, prompt_ids):
    adapter = _mock_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=adapter):
            resp = client.post("/api/batches", json={
                "models": [
                    {"provider_id": 1, "model_external_id": "llama3:latest"},
                    {"provider_id": 1, "model_external_id": "mistral:latest"},
                ],
                "prompt_ids": prompt_ids,
            })
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# POST /api/comparisons
# ---------------------------------------------------------------------------

def test_create_comparison(client):
    _seed_models(client)
    run1 = _make_run(client, "Hello")
    run2 = _make_run(client, "World")

    resp = client.post("/api/comparisons", json={"run_ids": [run1, run2]})
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data


def test_create_comparison_with_name(client):
    _seed_models(client)
    run1 = _make_run(client)
    run2 = _make_run(client)

    resp = client.post("/api/comparisons", json={
        "run_ids": [run1, run2],
        "name": "My comparison",
    })
    assert resp.status_code == 201


def test_create_comparison_too_few_runs(client):
    _seed_models(client)
    run1 = _make_run(client)
    resp = client.post("/api/comparisons", json={"run_ids": [run1]})
    assert resp.status_code == 422


def test_create_comparison_invalid_run(client):
    _seed_models(client)
    run1 = _make_run(client)
    resp = client.post("/api/comparisons", json={"run_ids": [run1, 9999]})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/comparisons/from-batch/{batch_id}
# ---------------------------------------------------------------------------

def test_create_comparisons_from_batch(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])
    batch = _make_batch(client, prompt_ids)

    resp = client.post(f"/api/comparisons/from-batch/{batch['id']}")
    assert resp.status_code == 201
    data = resp.json()
    # 2 prompts × 2 models = 2 comparisons (one per prompt, each with 2 runs).
    assert len(data["comparison_ids"]) == 2


def test_create_comparisons_from_batch_not_found(client):
    resp = client.post("/api/comparisons/from-batch/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/comparisons
# ---------------------------------------------------------------------------

def test_list_comparisons_empty(client):
    resp = client.get("/api/comparisons")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_comparisons_after_create(client):
    _seed_models(client)
    run1 = _make_run(client)
    run2 = _make_run(client)
    client.post("/api/comparisons", json={"run_ids": [run1, run2]})

    resp = client.get("/api/comparisons")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["run_count"] == 2


# ---------------------------------------------------------------------------
# GET /api/comparisons/{id}
# ---------------------------------------------------------------------------

def test_get_comparison_detail(client):
    _seed_models(client)
    run1 = _make_run(client, "Hello", "llama3:latest")
    run2 = _make_run(client, "Hello", "mistral:latest")
    create_resp = client.post("/api/comparisons", json={"run_ids": [run1, run2]})
    comp_id = create_resp.json()["id"]

    resp = client.get(f"/api/comparisons/{comp_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 2
    assert data["runs"][0]["output_text"] is not None
    assert data["runs"][1]["output_text"] is not None


def test_get_comparison_not_found(client):
    resp = client.get("/api/comparisons/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def test_comparison_page_renders(client):
    _seed_models(client)
    run1 = _make_run(client, "Hello", "llama3:latest")
    run2 = _make_run(client, "Hello", "mistral:latest")
    create_resp = client.post("/api/comparisons", json={"run_ids": [run1, run2]})
    comp_id = create_resp.json()["id"]

    resp = client.get(f"/compare/{comp_id}")
    assert resp.status_code == 200
    assert "Side-by-Side Comparison" in resp.text
    assert "llama3:latest" in resp.text
    assert "mistral:latest" in resp.text


def test_comparison_page_not_found(client):
    resp = client.get("/compare/9999")
    assert resp.status_code == 404


def test_history_page_has_compare_checkboxes(client):
    _seed_models(client)
    _make_run(client)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert 'class="run-cb"' in resp.text
    assert "Compare selected" in resp.text


def test_batch_detail_has_compare_button(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])
    batch = _make_batch(client, prompt_ids)

    resp = client.get(f"/batches/{batch['id']}")
    assert resp.status_code == 200
    assert "Compare outputs" in resp.text
