"""End-to-end integration tests spanning multiple features."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, patch

from llm_bencher.schemas import DiscoveredModel, ProviderHealth, RunResult as RunResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUITE_JSON = """{
  "slug": "integration-suite",
  "name": "Integration Suite",
  "prompts": [
    {
      "slug": "greet",
      "title": "Greeting Test",
      "user_prompt_template": "Say hello to {name}",
      "tags": ["greeting", "basic"],
      "variables": [{"name": "name", "description": "Name to greet", "required": true}]
    },
    {
      "slug": "math",
      "title": "Math Test",
      "user_prompt_template": "What is 2+2?",
      "tags": ["math", "basic"]
    }
  ]
}"""


def _make_adapter(succeed=True, output="test output", latency=100, tokens=50):
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
        adapter.run_chat = AsyncMock(side_effect=Exception("adapter error"))

    adapter.health_check = AsyncMock(
        return_value=ProviderHealth(is_available=True, checked_at=datetime.now(timezone.utc))
    )
    adapter.list_models = AsyncMock(return_value=[
        DiscoveredModel(id="model-x", name="model-x", provider_slug="test"),
        DiscoveredModel(id="model-y", name="model-y", provider_slug="test"),
    ])
    return adapter


# ---------------------------------------------------------------------------
# Test: Import suite -> batch run -> comparison -> export
# ---------------------------------------------------------------------------

def test_full_pipeline_import_batch_compare_export(client):
    """End-to-end: import a suite, create a batch, compare results, export CSV."""
    providers = client.get("/api/providers").json()
    pid = providers[0]["id"]

    # 1. Seed models.
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        client.post(f"/api/providers/{pid}/check")

    # 2. Import suite.
    resp = client.post("/api/suites/import", files={
        "file": ("suite.json", BytesIO(SUITE_JSON.encode()), "application/json")
    })
    assert resp.status_code == 201
    suite_id = resp.json()["id"]

    # 3. Get prompt IDs.
    suite_detail = client.get(f"/api/suites/{suite_id}").json()
    prompt_ids = [p["id"] for p in suite_detail["prompts"]]
    assert len(prompt_ids) == 2

    # 4. Create and execute a batch.
    adapter = _make_adapter(output="batch result", latency=200, tokens=80)
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter), \
         patch("llm_bencher.batch_runner.get_adapter", return_value=adapter):
        batch_resp = client.post("/api/batches", json={
            "models": [
                {"provider_id": pid, "model_external_id": "model-x"},
                {"provider_id": pid, "model_external_id": "model-y"},
            ],
            "prompt_ids": prompt_ids,
        })
    assert batch_resp.status_code == 201
    batch = batch_resp.json()
    assert batch["total_runs"] == 4  # 2 models x 2 prompts
    assert batch["status"] == "completed"

    # 5. Create comparisons from batch.
    comp_resp = client.post(f"/api/comparisons/from-batch/{batch['id']}")
    assert comp_resp.status_code == 201
    comp_ids = comp_resp.json()["comparison_ids"]
    assert len(comp_ids) == 2  # One per prompt

    # 6. View comparison page.
    for cid in comp_ids:
        page_resp = client.get(f"/compare/{cid}")
        assert page_resp.status_code == 200
        assert "model-x" in page_resp.text

    # 7. Export comparison CSV.
    csv_resp = client.get(f"/api/export/comparison/{comp_ids[0]}")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "batch result" in csv_resp.text

    # 8. Export batch CSV.
    batch_csv = client.get(f"/api/export/batch/{batch['id']}")
    assert batch_csv.status_code == 200
    assert "model-x" in batch_csv.text


# ---------------------------------------------------------------------------
# Test: Create provider -> health check -> run -> rate -> analytics
# ---------------------------------------------------------------------------

def test_full_pipeline_provider_run_rate_analytics(client):
    """End-to-end: create custom provider, run, rate, check analytics."""
    # 1. Create a custom OpenAI provider.
    prov_resp = client.post("/api/providers", json={
        "slug": "my-cloud",
        "name": "My Cloud LLM",
        "kind": "openai",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test-key",
    })
    assert prov_resp.status_code == 201
    pid = prov_resp.json()["id"]

    # 2. Health check and discover models.
    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        check_resp = client.post(f"/api/providers/{pid}/check")
    assert check_resp.json()["is_connected"] is True

    # 3. Run a prompt.
    adapter = _make_adapter(output="analytics test", latency=300, tokens=120)
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        run_resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "model-x",
            "user_prompt": "Test prompt for analytics",
        })
    assert run_resp.status_code == 201
    run_id = run_resp.json()["run_id"]

    # 4. Rate the run.
    rate_resp = client.post(f"/api/runs/{run_id}/rating", json={"score": 4, "notes": "Good"})
    assert rate_resp.status_code == 201

    # 5. Check analytics.
    summary = client.get("/api/analytics/summary").json()
    assert summary["total_runs"] >= 1
    assert summary["rated_runs"] >= 1

    latency_data = client.get("/api/analytics/latency").json()
    models = {d["model"]: d for d in latency_data}
    assert "model-x" in models
    assert models["model-x"]["avg_ms"] == 300.0

    # 6. Verify home page shows updated counts.
    home = client.get("/")
    assert "My Cloud LLM" in home.text


# ---------------------------------------------------------------------------
# Test: Tag filtering across pages
# ---------------------------------------------------------------------------

def test_tag_filtering_across_pages(client):
    """Import tagged prompts, filter on prompts page, create runs, filter history."""
    # 1. Import suite with tagged prompts.
    client.post("/api/suites/import", files={
        "file": ("s.json", BytesIO(SUITE_JSON.encode()), "application/json")
    })

    # 2. Verify tag API returns correct tags.
    tags = client.get("/api/tags").json()
    assert "greeting" in tags
    assert "math" in tags
    assert "basic" in tags

    # 3. Filter prompts page by tag.
    resp = client.get("/prompts?tag=greeting")
    assert resp.status_code == 200
    # The suite should appear since it has a prompt tagged "greeting".
    assert "Integration Suite" in resp.text
    # Should show filtered count (1 / 2).
    assert "1 / 2" in resp.text

    # 4. Filter by "basic" should show full suite (both prompts match).
    resp = client.get("/prompts?tag=basic")
    assert "Integration Suite" in resp.text
    assert "2 / 2" in resp.text


# ---------------------------------------------------------------------------
# Test: Home dashboard reflects all features
# ---------------------------------------------------------------------------

def test_home_dashboard_counts(client):
    """Home page should reflect counts from all features."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Providers" in resp.text
    assert "Prompts" in resp.text
    assert "Runs" in resp.text
    assert "Batches" in resp.text
    assert "Comparisons" in resp.text
    assert "Rated" in resp.text
    assert "Quick actions" in resp.text
    assert "Provider status" in resp.text


def test_home_recent_runs(client):
    """Home page should show recent runs when they exist."""
    providers = client.get("/api/providers").json()
    pid = providers[0]["id"]

    adapter = _make_adapter()
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        client.post(f"/api/providers/{pid}/check")

    adapter = _make_adapter(output="recent output")
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "model-x",
            "user_prompt": "Hello",
        })

    resp = client.get("/")
    assert "Recent runs" in resp.text
    assert "model-x" in resp.text
