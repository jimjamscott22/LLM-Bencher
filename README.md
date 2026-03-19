# LLM-Bencher

A local-only FastAPI app that lets one user connect to LM Studio and Ollama
running on the same machine, select an available model, run pre-written test
prompts, and review saved outputs.

## Development

This project is set up for the `uv` package manager.

```bash
uv sync
uv run pytest
uv run python -m llm_bencher
```

## Current status

The backend foundation is now scaffolded:

- FastAPI app factory with Jinja2-rendered starter pages
- SQLite persistence with SQLAlchemy models for providers, prompts, runs, and results
- Default provider bootstrapping for LM Studio and Ollama
- Alembic configuration for future migrations

## Planned next slices

- Provider adapters and connection checks
- Prompt-suite import and export
- Prompt execution workflow
- Run history filters and result details
