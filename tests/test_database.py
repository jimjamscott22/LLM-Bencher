from __future__ import annotations

import anyio
from sqlalchemy import create_engine, inspect, select, text

from llm_bencher.app import create_app
from llm_bencher.database import get_engine, session_scope
from llm_bencher.models import Provider


def test_database_tables_exist(client, test_settings) -> None:
    engine = get_engine(test_settings.database_url)
    table_names = set(inspect(engine).get_table_names())

    assert "providers" in table_names
    assert "runs" in table_names
    assert "run_results" in table_names
    assert "prompt_suites" in table_names


def test_default_providers_are_seeded(client, test_settings) -> None:
    with session_scope(test_settings.database_url) as session:
        provider_names = session.scalars(select(Provider.name).order_by(Provider.name)).all()

    assert provider_names == ["LM Studio", "Ollama"]


def test_startup_upgrades_older_sqlite_schema(test_settings) -> None:
    test_settings.ensure_directories()
    engine = create_engine(test_settings.database_url)
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE providers (
                id INTEGER PRIMARY KEY,
                slug VARCHAR(50) NOT NULL UNIQUE,
                name VARCHAR(100) NOT NULL,
                kind VARCHAR(20) NOT NULL,
                base_url VARCHAR(255) NOT NULL,
                is_enabled BOOLEAN NOT NULL,
                is_connected BOOLEAN NOT NULL,
                last_health_check_at DATETIME,
                last_error TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                provider_id INTEGER NOT NULL,
                provider_model_id INTEGER,
                prompt_id INTEGER,
                status VARCHAR(20) NOT NULL,
                model_identifier VARCHAR(255) NOT NULL,
                model_name VARCHAR(255),
                system_prompt TEXT,
                user_prompt TEXT NOT NULL,
                template_inputs JSON NOT NULL,
                temperature FLOAT,
                max_tokens INTEGER,
                started_at DATETIME,
                completed_at DATETIME,
                failure_message TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """))

    app = create_app(test_settings)

    async def run_startup() -> None:
        async with app.router.lifespan_context(app):
            return None

    anyio.run(run_startup)

    inspector = inspect(get_engine(test_settings.database_url))
    provider_columns = {column["name"] for column in inspector.get_columns("providers")}
    run_columns = {column["name"] for column in inspector.get_columns("runs")}

    assert "api_key" in provider_columns
    assert "is_default" in provider_columns
    assert "batch_id" in run_columns
