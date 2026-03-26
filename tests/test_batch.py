from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUITE = json.dumps({
    "slug": "batch-suite",
    "name": "Batch Suite",
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


def _seed_models(client, provider_id=1, models=None):
    if models is None:
        models = [
            DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama"),
        ]
    with patch("llm_bencher.web.api.get_adapter") as mock:
        adapter = AsyncMock()
        adapter.health_check = AsyncMock(
            return_value=ProviderHealth(is_available=True, checked_at=datetime.now(timezone.utc))
        )
        adapter.list_models = AsyncMock(return_value=models)
        mock.return_value = adapter
        client.post(f"/api/providers/{provider_id}/check")


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


def _mock_failing_adapter(error="Connection refused"):
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(side_effect=Exception(error))
    return adapter


# ---------------------------------------------------------------------------
# Batch creation + execution
# ---------------------------------------------------------------------------

def test_create_batch_succeeds(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_adapter()):
            resp = client.post("/api/batches", json={
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "completed"
    assert data["total_runs"] == 2
    assert data["completed_runs"] == 2
    assert data["failed_runs"] == 0


def test_create_batch_2_models_x_2_prompts(client):
    suite_data = _import_suite(client)
    # Seed 2 models on provider 1.
    _seed_models(client, provider_id=1, models=[
        DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama"),
        DiscoveredModel(id="mistral:latest", name="mistral:latest", provider_slug="ollama"),
    ])
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_adapter()):
            resp = client.post("/api/batches", json={
                "models": [
                    {"provider_id": 1, "model_external_id": "llama3:latest"},
                    {"provider_id": 1, "model_external_id": "mistral:latest"},
                ],
                "prompt_ids": prompt_ids,
            })

    assert resp.status_code == 201
    data = resp.json()
    assert data["total_runs"] == 4
    assert data["completed_runs"] == 4


def test_batch_partial_failure(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    # Alternate success/failure by using a side_effect list.
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(
        side_effect=[
            RunResultSchema(output_text="OK", latency_ms=50, prompt_tokens=2, completion_tokens=1, total_tokens=3),
            Exception("timeout"),
        ]
    )
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=adapter):
            resp = client.post("/api/batches", json={
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "partial"
    assert data["completed_runs"] == 1
    assert data["failed_runs"] == 1


def test_batch_all_failed(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_failing_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_failing_adapter()):
            resp = client.post("/api/batches", json={
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "failed"
    assert data["completed_runs"] == 0
    assert data["failed_runs"] == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_batch_no_models_returns_422(client):
    suite_data = _import_suite(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])
    resp = client.post("/api/batches", json={
        "models": [],
        "prompt_ids": prompt_ids,
    })
    assert resp.status_code == 422


def test_batch_no_prompts_returns_422(client):
    _seed_models(client)
    resp = client.post("/api/batches", json={
        "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
        "prompt_ids": [],
    })
    assert resp.status_code == 422


def test_batch_unknown_prompt_returns_404(client):
    _seed_models(client)
    resp = client.post("/api/batches", json={
        "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
        "prompt_ids": [9999],
    })
    assert resp.status_code == 404


def test_batch_unknown_provider_returns_404(client):
    suite_data = _import_suite(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])
    resp = client.post("/api/batches", json={
        "models": [{"provider_id": 9999, "model_external_id": "foo"}],
        "prompt_ids": prompt_ids,
    })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Batch list + detail
# ---------------------------------------------------------------------------

def test_list_batches_empty(client):
    resp = client.get("/api/batches")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_batches_after_create(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_adapter()):
            client.post("/api/batches", json={
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    resp = client.get("/api/batches")
    assert resp.status_code == 200
    batches = resp.json()
    assert len(batches) == 1
    assert batches[0]["total_runs"] == 2


def test_get_batch_detail(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_adapter()):
            create_resp = client.post("/api/batches", json={
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    batch_id = create_resp.json()["id"]
    resp = client.get(f"/api/batches/{batch_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) == 2
    assert all(r["status"] == "succeeded" for r in data["runs"])


def test_get_batch_not_found(client):
    resp = client.get("/api/batches/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def test_batch_page_renders(client):
    resp = client.get("/runs/batch")
    assert resp.status_code == 200
    assert "Run prompts across models" in resp.text


def test_batch_detail_page_renders(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_adapter()):
            create_resp = client.post("/api/batches", json={
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    batch_id = create_resp.json()["id"]
    resp = client.get(f"/batches/{batch_id}")
    assert resp.status_code == 200
    assert "Batch details" in resp.text
    assert "Completed" in resp.text


def test_batch_detail_page_not_found(client):
    resp = client.get("/batches/9999")
    assert resp.status_code == 404


def test_batch_with_custom_name(client):
    suite_data = _import_suite(client)
    _seed_models(client)
    prompt_ids = _get_prompt_ids(client, suite_data["id"])

    with patch("llm_bencher.web.api.get_adapter", return_value=_mock_adapter()):
        with patch("llm_bencher.batch_runner.get_adapter", return_value=_mock_adapter()):
            resp = client.post("/api/batches", json={
                "name": "My Test Batch",
                "models": [{"provider_id": 1, "model_external_id": "llama3:latest"}],
                "prompt_ids": prompt_ids,
            })

    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Test Batch"
