# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-Bencher is a FastAPI web application for testing and benchmarking LLM services. It supports local providers (LM Studio, Ollama), cloud providers (OpenAI), and any OpenAI-compatible endpoint. Users can run individual prompts or batch tests across multiple models, compare outputs side-by-side, rate results, view performance analytics, and export data as CSV. Server-rendered with Jinja2 templates, SQLite persistence, and no external cloud dependencies required.

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

**App startup flow** (`app.py` lifespan → `bootstrap.py`): Settings loaded → directories ensured → SQLite tables created → default providers seeded (LM Studio, Ollama, marked `is_default=True`) → Uvicorn serves.

**Key layers:**
- `config.py`: `Settings` dataclass with env var overrides (prefix `LLM_BENCHER_`), cached via `@lru_cache`
- `models.py`: SQLAlchemy ORM — Provider, ProviderModel, PromptSuite, PromptDefinition, PromptImportRecord, BatchRun, Run, RunResult, RunRating, Comparison, ComparisonItem. All use `TimestampMixin` for UTC timestamps. JSON columns for flexible metadata (tags, variables, template_inputs).
- `schemas.py`: Pydantic transfer objects (ProviderHealth, DiscoveredModel, RunRequest, RunResult)
- `providers/`: Abstract `ProviderAdapter` base with implementations:
  - `openai_compat.py`: OpenAI-compatible API adapter (used by LM Studio)
  - `lm_studio.py`: LM Studio adapter (extends OpenAICompat)
  - `ollama.py`: Ollama native API adapter
  - `openai_cloud.py`: OpenAI cloud adapter with Bearer auth (extends OpenAICompat)
  - `registry.py`: Routes `ProviderKind` to correct adapter class
- `web/api.py`: JSON API endpoints — provider CRUD, health checks, runs, ratings, batches, comparisons, tags, suite import/export, CSV export
- `web/analytics_api.py`: Analytics endpoints — summary, latency, tokens, success-rate, timeline
- `web/routes.py`: Jinja2-rendered pages (home, providers, prompts, runs, batch, history, analytics, compare)
- `batch_runner.py`: Async batch execution with `asyncio.Semaphore(3)` concurrency control
- `database.py`: Engine/session factory with `session_scope` context manager for transaction safety

**Session pattern:** All API endpoints that perform async I/O follow a 3-phase pattern: (1) read from DB → (2) async network call (no session held) → (3) write results to DB. This prevents holding sessions open during network waits.

**Database:** SQLite with named constraint conventions for Alembic compatibility (FK, UQ, CK, IX, PK prefixes). Schema created via `Base.metadata.create_all` on startup; Alembic configured for future migrations.

**Enums:**
- `ProviderKind`: `lm_studio`, `ollama`, `openai`, `openai_compat`
- `RunStatus`: `pending`, `running`, `succeeded`, `failed`
- `BatchStatus`: `pending`, `running`, `completed`, `partial`, `failed`

## Testing

209 tests across 12 test files. Tests use temporary SQLite databases created in `.test-tmp/` per test case. Key fixtures in `tests/conftest.py`:
- `test_settings`: isolated `Settings` pointing to temp directory
- `client`: FastAPI `TestClient` wired to test settings

Provider adapters are mocked with `unittest.mock.AsyncMock` in tests — no real network calls.

Test files:
- `test_app.py`: Basic page rendering and health check
- `test_providers.py`: Provider health check and model discovery
- `test_provider_crud.py`: Provider CRUD, registry wiring, template rendering
- `test_runs.py`: Run creation, execution, persistence
- `test_ratings.py`: Rating CRUD, validation, template integration
- `test_tags.py`: Tag API, prompt/history filtering
- `test_variables.py`: Template variables, template_inputs persistence
- `test_batch.py`: Batch creation, execution, partial failure
- `test_comparison.py`: Comparison CRUD, from-batch creation
- `test_analytics.py`: Analytics aggregation endpoints
- `test_export.py`: CSV export for history, batch, comparison
- `test_integration.py`: End-to-end pipelines
- `test_edge_cases.py`: Unicode, long content, boundary conditions

## Configuration

All env vars prefixed with `LLM_BENCHER_`. Key ones:
- `LLM_BENCHER_LM_STUDIO_URL` (default: `http://127.0.0.1:1234/v1`)
- `LLM_BENCHER_OLLAMA_URL` (default: `http://127.0.0.1:11434`)
- `LLM_BENCHER_OPENAI_API_KEY` (default: empty) — OpenAI API key for cloud provider
- `LLM_BENCHER_OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `LLM_BENCHER_DATA_DIR` (default: `{project_root}/data`)
- `LLM_BENCHER_SQLITE_ECHO` (default: false) — enables SQL query logging
- `LLM_BENCHER_PROVIDER_TIMEOUT` (default: 30.0) — HTTP timeout for provider calls

## Pages

| Route | Template | Description |
|-------|----------|-------------|
| `/` | `home.html` | Dashboard with stats, quick actions, provider status, recent runs |
| `/providers` | `providers.html` | Provider list with add/edit/delete, health check, model discovery |
| `/prompts` | `prompts.html` | Prompt suite library with import, tag filtering, export |
| `/runs/new` | `runs.html` | Single prompt execution form with variable substitution |
| `/runs/batch` | `batch_new.html` | Batch run creation (models × prompts) |
| `/batches/{id}` | `batch_detail.html` | Batch results with progress bar, export, compare |
| `/history` | `history.html` | Run history with filters, pagination, compare, export |
| `/runs/{id}` | `run_detail.html` | Full run detail with rating widget |
| `/compare/{id}` | `compare.html` | Side-by-side model comparison with export |
| `/analytics` | `analytics.html` | Performance dashboard with latency, tokens, success rates |

## Reference Documents

- `LLM_TESTING_PLATFORM_PLAN.md`: Detailed implementation plan with 10 slices, dependency graph, and file-level change summary
