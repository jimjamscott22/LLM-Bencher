from __future__ import annotations

from sqlalchemy import inspect, select

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
