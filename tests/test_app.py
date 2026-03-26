from __future__ import annotations


def test_home_page_renders(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "LLM Bencher" in response.text
    assert "Test, compare, and benchmark your local LLMs." in response.text


def test_health_endpoint(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_provider_page_shows_seeded_providers(client) -> None:
    response = client.get("/providers")

    assert response.status_code == 200
    assert "LM Studio" in response.text
    assert "Ollama" in response.text
