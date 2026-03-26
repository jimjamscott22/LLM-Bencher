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
            DiscoveredModel(id="llama3:latest", name="llama3:latest", provider_slug="ollama"),
        ])
        mock.return_value = adapter
        client.post(f"/api/providers/{provider_id}/check")


# ---------------------------------------------------------------------------
# GET /api/providers
# ---------------------------------------------------------------------------

def test_list_providers(client):
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    data = resp.json()
    # Default seeded providers: LM Studio + Ollama
    assert len(data) == 2
    slugs = {p["slug"] for p in data}
    assert "lm-studio" in slugs
    assert "ollama" in slugs
    # Default providers should be marked as such
    for p in data:
        assert p["is_default"] is True


# ---------------------------------------------------------------------------
# POST /api/providers
# ---------------------------------------------------------------------------

def test_create_provider(client):
    resp = client.post("/api/providers", json={
        "slug": "my-openai",
        "name": "My OpenAI",
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test-key",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "my-openai"
    assert data["kind"] == "openai"
    assert data["has_api_key"] is True
    assert data["is_default"] is False


def test_create_provider_openai_compat(client):
    resp = client.post("/api/providers", json={
        "slug": "local-llm",
        "name": "Local LLM",
        "kind": "openai_compat",
        "base_url": "http://localhost:5000/v1",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["kind"] == "openai_compat"
    assert data["has_api_key"] is False


def test_create_provider_invalid_kind(client):
    resp = client.post("/api/providers", json={
        "slug": "bad",
        "name": "Bad",
        "kind": "invalid_kind",
        "base_url": "http://localhost",
    })
    assert resp.status_code == 422


def test_create_provider_duplicate_slug(client):
    resp = client.post("/api/providers", json={
        "slug": "ollama",
        "name": "Duplicate",
        "kind": "ollama",
        "base_url": "http://localhost:11434",
    })
    assert resp.status_code == 409


def test_create_provider_shows_in_list(client):
    client.post("/api/providers", json={
        "slug": "custom-one",
        "name": "Custom One",
        "kind": "openai_compat",
        "base_url": "http://localhost:9000/v1",
    })
    resp = client.get("/api/providers")
    slugs = {p["slug"] for p in resp.json()}
    assert "custom-one" in slugs


# ---------------------------------------------------------------------------
# PUT /api/providers/{id}
# ---------------------------------------------------------------------------

def test_update_provider_name(client):
    # Get the Ollama provider ID.
    providers = client.get("/api/providers").json()
    ollama = [p for p in providers if p["slug"] == "ollama"][0]

    resp = client.put(f"/api/providers/{ollama['id']}", json={
        "name": "Ollama (Updated)",
    })
    assert resp.status_code == 200
    assert resp.json()["name"] == "Ollama (Updated)"


def test_update_provider_base_url(client):
    providers = client.get("/api/providers").json()
    ollama = [p for p in providers if p["slug"] == "ollama"][0]

    resp = client.put(f"/api/providers/{ollama['id']}", json={
        "base_url": "http://localhost:99999",
    })
    assert resp.status_code == 200
    assert resp.json()["base_url"] == "http://localhost:99999"


def test_update_provider_api_key(client):
    # Create a custom provider first.
    create_resp = client.post("/api/providers", json={
        "slug": "keyed",
        "name": "Keyed",
        "kind": "openai",
        "base_url": "https://api.example.com/v1",
    })
    pid = create_resp.json()["id"]
    assert create_resp.json()["has_api_key"] is False

    resp = client.put(f"/api/providers/{pid}", json={"api_key": "sk-new"})
    assert resp.status_code == 200
    assert resp.json()["has_api_key"] is True

    # Clear the key by sending empty string.
    resp = client.put(f"/api/providers/{pid}", json={"api_key": ""})
    assert resp.status_code == 200
    assert resp.json()["has_api_key"] is False


def test_update_provider_not_found(client):
    resp = client.put("/api/providers/9999", json={"name": "Nope"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/providers/{id}
# ---------------------------------------------------------------------------

def test_delete_custom_provider(client):
    create_resp = client.post("/api/providers", json={
        "slug": "to-delete",
        "name": "To Delete",
        "kind": "openai_compat",
        "base_url": "http://localhost:1111/v1",
    })
    pid = create_resp.json()["id"]

    resp = client.delete(f"/api/providers/{pid}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Verify it's gone.
    providers = client.get("/api/providers").json()
    assert pid not in {p["id"] for p in providers}


def test_delete_default_provider_forbidden(client):
    providers = client.get("/api/providers").json()
    ollama = [p for p in providers if p["slug"] == "ollama"][0]

    resp = client.delete(f"/api/providers/{ollama['id']}")
    assert resp.status_code == 403


def test_delete_provider_with_runs_blocked(client):
    """Cannot delete a provider that has runs associated with it."""
    # Create a custom provider and make a run against it.
    create_resp = client.post("/api/providers", json={
        "slug": "has-runs",
        "name": "Has Runs",
        "kind": "openai_compat",
        "base_url": "http://localhost:2222/v1",
    })
    pid = create_resp.json()["id"]

    # Seed models for the custom provider.
    _seed_models(client, provider_id=pid)

    # Create a run.
    adapter = AsyncMock()
    adapter.run_chat = AsyncMock(return_value=RunResultSchema(
        output_text="Hello",
        latency_ms=50,
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
    ))
    with patch("llm_bencher.web.api.get_adapter", return_value=adapter):
        run_resp = client.post("/api/runs", json={
            "provider_id": pid,
            "model_external_id": "llama3:latest",
            "user_prompt": "Hello",
        })
    assert run_resp.status_code == 201

    resp = client.delete(f"/api/providers/{pid}")
    assert resp.status_code == 409
    assert "existing run" in resp.json()["detail"].lower()


def test_delete_provider_not_found(client):
    resp = client.delete("/api/providers/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def test_providers_page_renders_add_form(client):
    resp = client.get("/providers")
    assert resp.status_code == 200
    assert "Add custom provider" in resp.text
    assert "add-provider-form" in resp.text


def test_providers_page_shows_default_badge(client):
    resp = client.get("/providers")
    assert resp.status_code == 200
    assert "Default" in resp.text


def test_providers_page_shows_edit_button(client):
    resp = client.get("/providers")
    assert resp.status_code == 200
    assert "Edit" in resp.text


def test_providers_page_no_delete_for_defaults(client):
    """Default providers should not show a delete button."""
    resp = client.get("/providers")
    # The delete button has class="delete-btn" — should NOT appear for default providers.
    # The string appears once in the JS querySelectorAll, but zero times as actual buttons.
    assert resp.text.count('class="delete-btn"') == 0


def test_providers_page_shows_delete_for_custom(client):
    """Custom providers should show a delete button."""
    client.post("/api/providers", json={
        "slug": "custom-prov",
        "name": "Custom Prov",
        "kind": "openai_compat",
        "base_url": "http://localhost:3333/v1",
    })
    resp = client.get("/providers")
    assert 'class="delete-btn"' in resp.text


# ---------------------------------------------------------------------------
# Registry / adapter wiring
# ---------------------------------------------------------------------------

def test_registry_openai_adapter(client):
    """Verify the registry returns OpenAICloudAdapter for OPENAI kind."""
    from llm_bencher.providers.registry import get_adapter
    from llm_bencher.providers.openai_cloud import OpenAICloudAdapter

    # Create a real provider in the DB.
    create_resp = client.post("/api/providers", json={
        "slug": "reg-openai",
        "name": "Reg OpenAI",
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
    })
    pid = create_resp.json()["id"]

    session_factory = client.app.state.session_factory
    settings = client.app.state.settings
    with session_factory() as session:
        from llm_bencher.models import Provider
        provider = session.get(Provider, pid)
        adapter = get_adapter(provider, settings)
        assert isinstance(adapter, OpenAICloudAdapter)


def test_registry_openai_compat_adapter_no_key(client):
    """OPENAI_COMPAT without api_key returns plain OpenAICompatAdapter."""
    from llm_bencher.providers.registry import get_adapter
    from llm_bencher.providers.openai_compat import OpenAICompatAdapter
    from llm_bencher.providers.openai_cloud import OpenAICloudAdapter

    create_resp = client.post("/api/providers", json={
        "slug": "reg-compat",
        "name": "Reg Compat",
        "kind": "openai_compat",
        "base_url": "http://localhost:5000/v1",
    })
    pid = create_resp.json()["id"]

    session_factory = client.app.state.session_factory
    settings = client.app.state.settings
    with session_factory() as session:
        from llm_bencher.models import Provider
        provider = session.get(Provider, pid)
        adapter = get_adapter(provider, settings)
        assert isinstance(adapter, OpenAICompatAdapter)
        assert not isinstance(adapter, OpenAICloudAdapter)


def test_registry_openai_compat_adapter_with_key(client):
    """OPENAI_COMPAT with api_key returns OpenAICloudAdapter."""
    from llm_bencher.providers.registry import get_adapter
    from llm_bencher.providers.openai_cloud import OpenAICloudAdapter

    create_resp = client.post("/api/providers", json={
        "slug": "reg-compat-key",
        "name": "Reg Compat Key",
        "kind": "openai_compat",
        "base_url": "http://localhost:5000/v1",
        "api_key": "sk-compat",
    })
    pid = create_resp.json()["id"]

    session_factory = client.app.state.session_factory
    settings = client.app.state.settings
    with session_factory() as session:
        from llm_bencher.models import Provider
        provider = session.get(Provider, pid)
        adapter = get_adapter(provider, settings)
        assert isinstance(adapter, OpenAICloudAdapter)
