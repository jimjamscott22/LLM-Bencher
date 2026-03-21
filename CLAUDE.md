# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-Bencher is a local-only FastAPI web application for manual testing of local LLM services (LM Studio and Ollama). Users select available models, run pre-written test prompts, and review saved outputs. Server-rendered with Jinja2 templates, SQLite persistence, and no external cloud dependencies.

## Commands

```bash
# Install dependencies
uv sync

# Run the development server (default: http://127.0.0.1:8000)
uv run python -m llm_bencher

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_app.py::test_name

# Database migrations (Alembic)
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```

Python 3.12+ required. Package manager is `uv` (not pip).

## Architecture

**App startup flow** (`app.py` lifespan → `bootstrap.py`): Settings loaded → directories ensured → SQLite tables created → default providers seeded (LM Studio, Ollama) → Uvicorn serves.

**Key layers:**
- `config.py`: `Settings` dataclass with env var overrides (prefix `LLM_BENCHER_`), cached via `@lru_cache`
- `models.py`: SQLAlchemy ORM — Provider, ProviderModel, PromptSuite, PromptDefinition, Run, RunResult. All use `TimestampMixin` for UTC timestamps. JSON columns for flexible metadata.
- `schemas.py`: Pydantic transfer objects
- `providers/base.py`: Abstract `ProviderAdapter` with `health_check()`, `list_models()`, `run_chat()` — implementations not yet built
- `web/routes.py`: Jinja2-rendered pages (home dashboard, providers, prompts, runs, history)
- `database.py`: Engine/session factory with `session_scope` context manager for transaction safety

**Database:** SQLite with named constraint conventions for Alembic compatibility (FK, UQ, CK, IX, PK prefixes). Schema created via `Base.metadata.create_all` on startup; Alembic configured for future migrations.

## Testing

Tests use temporary SQLite databases created in `.test-tmp/` per test case. Key fixtures in `tests/conftest.py`:
- `test_settings`: isolated `Settings` pointing to temp directory
- `client`: FastAPI `TestClient` wired to test settings

## Configuration

All env vars prefixed with `LLM_BENCHER_`. Key ones:
- `LLM_BENCHER_LM_STUDIO_URL` (default: `http://127.0.0.1:1234/v1`)
- `LLM_BENCHER_OLLAMA_URL` (default: `http://127.0.0.1:11434`)
- `LLM_BENCHER_DATA_DIR` (default: `{project_root}/data`)
- `LLM_BENCHER_SQLITE_ECHO` (default: false) — enables SQL query logging

## Reference Documents

- `LLM_Tester_PLAN.md`: Detailed architectural plan with schemas, implementation slices, and design decisions
