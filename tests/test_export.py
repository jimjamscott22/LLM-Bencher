from __future__ import annotations

import csv
import io
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
        ])
        mock.return_value = adapter
        client.post(f"/api/providers/{provider_id}/check")


def _create_run(client, model_id="model-a", provider_id=None):
    if provider_id is None:
        providers = client.get("/api/providers").json()
        provider_id = providers[0]["id"]

    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(return_value=RunResultSchema(
        output_text="hello world",
        latency_ms=150,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    ))
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        resp = client.post("/api/runs", json={
            "provider_id": provider_id,
            "model_external_id": model_id,
            "user_prompt": "test prompt",
        })
    return resp.json()


def _parse_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


# ---------------------------------------------------------------------------
# History export
# ---------------------------------------------------------------------------

def test_export_history_empty(client):
    resp = client.get("/api/export/history")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.text == ""


def test_export_history_with_runs(client):
    _seed_models(client)
    _create_run(client)
    _create_run(client)

    resp = client.get("/api/export/history")
    assert resp.status_code == 200
    rows = _parse_csv(resp.text)
    assert len(rows) == 2
    assert "run_id" in rows[0]
    assert "status" in rows[0]
    assert rows[0]["status"] == "succeeded"
    assert rows[0]["latency_ms"] == "150"
    assert rows[0]["output"] == "hello world"


def test_export_history_with_filter(client):
    _seed_models(client)
    _create_run(client)

    providers = client.get("/api/providers").json()
    pid = providers[0]["id"]

    resp = client.get(f"/api/export/history?provider_id={pid}")
    rows = _parse_csv(resp.text)
    assert len(rows) == 1

    # Filter by non-existent provider should return empty.
    resp = client.get("/api/export/history?provider_id=9999")
    assert resp.text == ""


def test_export_history_csv_headers(client):
    _seed_models(client)
    _create_run(client)

    resp = client.get("/api/export/history")
    first_line = resp.text.split("\n")[0]
    assert "run_id" in first_line
    assert "provider" in first_line
    assert "model" in first_line
    assert "latency_ms" in first_line


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def test_export_batch_not_found(client):
    resp = client.get("/api/export/batch/9999")
    assert resp.status_code == 404


def test_export_batch_with_runs(client):
    _seed_models(client)

    # Import a suite so we have a prompt.
    suite_json = '{"slug":"export-suite","name":"Export Suite","prompts":[{"slug":"p1","title":"Prompt 1","user_prompt_template":"hello"}]}'
    from io import BytesIO
    client.post("/api/suites/import", files={"file": ("test.json", BytesIO(suite_json.encode()), "application/json")})

    suites = client.get("/api/suites").json()
    suite = suites[0]
    suite_detail = client.get(f"/api/suites/{suite['id']}").json()
    prompt_id = suite_detail["prompts"][0]["id"]

    providers = client.get("/api/providers").json()
    pid = providers[0]["id"]

    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(return_value=RunResultSchema(
        output_text="batch output",
        latency_ms=100,
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
    ))
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter), \
         patch("llm_bencher.batch_runner.get_adapter", return_value=adapter):
        batch_resp = client.post("/api/batches", json={
            "models": [{"provider_id": pid, "model_external_id": "model-a"}],
            "prompt_ids": [prompt_id],
        })
    assert batch_resp.status_code == 201
    batch_id = batch_resp.json()["id"]

    resp = client.get(f"/api/export/batch/{batch_id}")
    assert resp.status_code == 200
    rows = _parse_csv(resp.text)
    assert len(rows) == 1
    assert rows[0]["output"] == "batch output"


# ---------------------------------------------------------------------------
# Comparison export
# ---------------------------------------------------------------------------

def test_export_comparison_not_found(client):
    resp = client.get("/api/export/comparison/9999")
    assert resp.status_code == 404


def test_export_comparison_with_runs(client):
    _seed_models(client)
    r1 = _create_run(client)
    r2 = _create_run(client)

    comp_resp = client.post("/api/comparisons", json={"run_ids": [r1["run_id"], r2["run_id"]]})
    comp_id = comp_resp.json()["id"]

    resp = client.get(f"/api/export/comparison/{comp_id}")
    assert resp.status_code == 200
    rows = _parse_csv(resp.text)
    assert len(rows) == 2
    assert "position" in rows[0]
    assert "output" in rows[0]


# ---------------------------------------------------------------------------
# Template buttons
# ---------------------------------------------------------------------------

def test_history_page_has_export_button(client):
    _seed_models(client)
    _create_run(client)
    resp = client.get("/history")
    assert "Export CSV" in resp.text


def test_batch_detail_has_export_button(client):
    _seed_models(client)
    suite_json = '{"slug":"exp-suite2","name":"Exp Suite2","prompts":[{"slug":"p1","title":"P1","user_prompt_template":"hi"}]}'
    from io import BytesIO
    client.post("/api/suites/import", files={"file": ("t.json", BytesIO(suite_json.encode()), "application/json")})
    suites = client.get("/api/suites").json()
    suite_detail = client.get(f"/api/suites/{suites[0]['id']}").json()
    prompt_id = suite_detail["prompts"][0]["id"]
    providers = client.get("/api/providers").json()
    pid = providers[0]["id"]

    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(return_value=RunResultSchema(
        output_text="out", latency_ms=50, prompt_tokens=5, completion_tokens=5, total_tokens=10,
    ))
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter), \
         patch("llm_bencher.batch_runner.get_adapter", return_value=adapter):
        batch_resp = client.post("/api/batches", json={
            "models": [{"provider_id": pid, "model_external_id": "model-a"}],
            "prompt_ids": [prompt_id],
        })
    batch_id = batch_resp.json()["id"]

    resp = client.get(f"/batches/{batch_id}")
    assert "Export CSV" in resp.text
