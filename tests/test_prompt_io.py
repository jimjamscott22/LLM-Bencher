from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_bencher.prompt_io import (
    compute_checksum,
    export_suite,
    export_suite_to_json,
    import_suite,
    load_suite_file,
    load_suite_from_string,
)
from llm_bencher.schemas import PromptRecord, PromptSuiteFile, PromptVariable


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_SUITE = {
    "slug": "test-suite",
    "name": "Test Suite",
    "prompts": [
        {
            "slug": "hello",
            "title": "Hello prompt",
            "user_prompt_template": "Say hello.",
        }
    ],
}

FULL_SUITE = {
    "slug": "full-suite",
    "name": "Full Suite",
    "description": "A suite with all fields",
    "version": "1.0.0",
    "prompts": [
        {
            "slug": "summarise",
            "title": "Summarise text",
            "category": "writing",
            "description": "Summarise the given text",
            "system_prompt": "You are a concise summariser.",
            "user_prompt_template": "Summarise: {text}",
            "tags": ["writing", "summarisation"],
            "variables": [{"name": "text", "required": True}],
            "default_temperature": 0.3,
            "default_max_tokens": 256,
        }
    ],
}


def _db_session(test_settings):
    """Return a live session for the test database."""
    from llm_bencher.database import get_session_factory, initialize_database

    test_settings.ensure_directories()
    initialize_database(test_settings.database_url, echo=False)
    factory = get_session_factory(test_settings.database_url, echo=False)
    return factory()


# ---------------------------------------------------------------------------
# Unit: parsing
# ---------------------------------------------------------------------------

def test_load_suite_from_string_minimal():
    suite = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    assert suite.slug == "test-suite"
    assert len(suite.prompts) == 1
    assert suite.prompts[0].slug == "hello"


def test_load_suite_from_string_full():
    suite = load_suite_from_string(json.dumps(FULL_SUITE))
    p = suite.prompts[0]
    assert p.system_prompt == "You are a concise summariser."
    assert p.default_temperature == 0.3
    assert p.variables[0].name == "text"
    assert p.variables[0].required is True


def test_load_suite_from_string_bad_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_suite_from_string("not json {")


def test_load_suite_from_string_missing_required_field():
    with pytest.raises(ValueError, match="Invalid suite schema"):
        load_suite_from_string(json.dumps({"name": "no slug here"}))


def test_load_suite_file(tmp_path: Path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps(MINIMAL_SUITE), encoding="utf-8")
    suite = load_suite_file(p)
    assert suite.slug == "test-suite"


def test_load_suite_file_missing(tmp_path: Path):
    with pytest.raises(ValueError, match="Cannot read"):
        load_suite_file(tmp_path / "nonexistent.json")


def test_compute_checksum_is_deterministic():
    content = json.dumps(MINIMAL_SUITE)
    assert compute_checksum(content) == compute_checksum(content)
    assert compute_checksum(content) != compute_checksum(content + " ")


# ---------------------------------------------------------------------------
# Unit: import (DB)
# ---------------------------------------------------------------------------

def test_import_creates_suite(test_settings):
    suite_file = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    with _db_session(test_settings) as session:
        suite, action = import_suite(session, suite_file, "test.json", "abc123")
        session.commit()

    assert action == "created"
    assert suite.slug == "test-suite"
    assert suite.imported_at is not None


def test_import_upserts_suite_on_second_import(test_settings):
    suite_file = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    with _db_session(test_settings) as session:
        import_suite(session, suite_file, "test.json", "abc")
        session.commit()

    updated = {**MINIMAL_SUITE, "name": "Updated Name"}
    suite_file2 = load_suite_from_string(json.dumps(updated))
    with _db_session(test_settings) as session:
        suite, action = import_suite(session, suite_file2, "test.json", "def")
        session.commit()

    assert action == "updated"
    assert suite.name == "Updated Name"


def test_import_upserts_existing_prompts(test_settings):
    suite_file = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    with _db_session(test_settings) as session:
        import_suite(session, suite_file, "test.json", "v1")
        session.commit()

    modified = {
        **MINIMAL_SUITE,
        "prompts": [{"slug": "hello", "title": "Updated Hello", "user_prompt_template": "Hi!"}],
    }
    with _db_session(test_settings) as session:
        suite, _ = import_suite(session, load_suite_from_string(json.dumps(modified)), "test.json", "v2")
        session.commit()
        assert suite.prompts[0].title == "Updated Hello"


def test_import_adds_new_prompt_to_existing_suite(test_settings):
    suite_file = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    with _db_session(test_settings) as session:
        import_suite(session, suite_file, "test.json", "v1")
        session.commit()

    expanded = {
        **MINIMAL_SUITE,
        "prompts": [
            MINIMAL_SUITE["prompts"][0],
            {"slug": "goodbye", "title": "Goodbye", "user_prompt_template": "Say bye."},
        ],
    }
    with _db_session(test_settings) as session:
        suite, _ = import_suite(session, load_suite_from_string(json.dumps(expanded)), "test.json", "v2")
        session.flush()
        assert len(suite.prompts) == 2


def test_import_writes_import_record(test_settings):
    from sqlalchemy import select
    from llm_bencher.models import PromptImportRecord

    suite_file = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    with _db_session(test_settings) as session:
        import_suite(session, suite_file, "my-file.json", "checksum-xyz")
        session.commit()

    with _db_session(test_settings) as session:
        record = session.scalar(select(PromptImportRecord))
        assert record is not None
        assert record.source_path == "my-file.json"
        assert record.status == "success"


# ---------------------------------------------------------------------------
# Unit: export (round-trip)
# ---------------------------------------------------------------------------

def test_export_round_trips(test_settings):
    suite_file = load_suite_from_string(json.dumps(FULL_SUITE))
    with _db_session(test_settings) as session:
        suite, _ = import_suite(session, suite_file, "full.json", "chk")
        session.commit()
        exported = export_suite(suite)

    assert exported.slug == suite_file.slug
    assert exported.name == suite_file.name
    assert len(exported.prompts) == len(suite_file.prompts)
    p = exported.prompts[0]
    assert p.system_prompt == "You are a concise summariser."
    assert p.variables[0].name == "text"
    assert p.default_temperature == 0.3


def test_export_suite_to_json_is_valid(test_settings):
    suite_file = load_suite_from_string(json.dumps(MINIMAL_SUITE))
    with _db_session(test_settings) as session:
        suite, _ = import_suite(session, suite_file, "t.json", "chk")
        session.commit()
        json_str = export_suite_to_json(export_suite(suite))

    data = json.loads(json_str)
    assert data["slug"] == "test-suite"
    assert len(data["prompts"]) == 1


# ---------------------------------------------------------------------------
# API endpoint tests (via TestClient)
# ---------------------------------------------------------------------------

SAMPLE_JSON = json.dumps(MINIMAL_SUITE)


def test_list_suites_empty(client):
    resp = client.get("/api/suites")
    assert resp.status_code == 200
    assert resp.json() == []


def test_import_suite_via_api(client):
    resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "test-suite"
    assert data["action"] == "created"
    assert data["prompt_count"] == 1


def test_import_suite_second_time_returns_200(client):
    client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    assert resp.status_code == 200
    assert resp.json()["action"] == "updated"


def test_import_suite_bad_json_returns_422(client):
    resp = client.post(
        "/api/suites/import",
        files={"file": ("bad.json", b"not json", "application/json")},
    )
    assert resp.status_code == 422


def test_list_suites_after_import(client):
    client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    resp = client.get("/api/suites")
    assert resp.status_code == 200
    suites = resp.json()
    assert len(suites) == 1
    assert suites[0]["slug"] == "test-suite"
    assert suites[0]["prompt_count"] == 1


def test_get_suite_not_found(client):
    assert client.get("/api/suites/9999").status_code == 404


def test_get_suite_returns_prompts(client):
    import_resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    suite_id = import_resp.json()["id"]
    resp = client.get(f"/api/suites/{suite_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "test-suite"
    assert len(data["prompts"]) == 1
    assert data["prompts"][0]["slug"] == "hello"


def test_export_suite_endpoint(client):
    import_resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    suite_id = import_resp.json()["id"]
    resp = client.get(f"/api/suites/{suite_id}/export")
    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    data = json.loads(resp.content)
    assert data["slug"] == "test-suite"


def test_delete_suite(client):
    import_resp = client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    suite_id = import_resp.json()["id"]

    del_resp = client.delete(f"/api/suites/{suite_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # No longer appears in list
    suites = client.get("/api/suites").json()
    assert all(s["id"] != suite_id for s in suites)


def test_delete_suite_not_found(client):
    assert client.delete("/api/suites/9999").status_code == 404


def test_prompts_page_renders(client):
    resp = client.get("/prompts")
    assert resp.status_code == 200
    assert "Prompt suites" in resp.text


def test_prompts_page_shows_imported_suite(client):
    client.post(
        "/api/suites/import",
        files={"file": ("suite.json", SAMPLE_JSON.encode(), "application/json")},
    )
    resp = client.get("/prompts")
    assert "Test Suite" in resp.text
    assert "test-suite" in resp.text
