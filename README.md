# LLM-Bencher

A FastAPI web application for testing, comparing, and benchmarking LLM services. Connect to LM Studio, Ollama, OpenAI, or any OpenAI-compatible endpoint — run prompts, batch test across models, compare outputs side-by-side, and track performance over time.

## Features

- **Multi-provider support** — LM Studio, Ollama, OpenAI cloud, and generic OpenAI-compatible endpoints. Add custom providers with optional API keys.
- **Prompt library** — Import/export JSON prompt suites with tags, categories, and template variables.
- **Single & batch runs** — Execute one prompt at a time or batch across multiple models simultaneously with bounded concurrency.
- **Side-by-side comparison** — Compare outputs from different models in a column layout. Auto-create comparisons from batch runs.
- **Ratings & annotations** — Rate run outputs 1–5 stars with notes.
- **Performance analytics** — Latency, token usage, and success rate dashboards grouped by model.
- **CSV export** — Export history, batch results, and comparisons as CSV files.
- **Tag-based filtering** — Filter prompts and history by tags.
- **Local-first** — SQLite persistence, server-rendered pages, no cloud dependencies required.

## Quick Start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Start the development server
uv run python -m llm_bencher
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Usage

1. **Set up providers** — Go to `/providers` to check connections to LM Studio or Ollama (make sure they're running). Add custom OpenAI or OpenAI-compatible providers with the form.
2. **Import prompts** — Go to `/prompts` and upload a JSON prompt suite file.
3. **Run prompts** — Go to `/runs/new` to execute a single prompt, or `/runs/batch` to run across multiple models at once.
4. **Review results** — Browse `/history` to see all runs. Click into a run to see full output and rate it.
5. **Compare models** — Select runs from history or use "Compare outputs" on a batch to see side-by-side results.
6. **View analytics** — Go to `/analytics` for latency, token, and success rate breakdowns.
7. **Export data** — Use "Export CSV" buttons on history, batch detail, and comparison pages.

## Configuration

All settings configurable via environment variables (prefix `LLM_BENCHER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BENCHER_LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio API URL |
| `LLM_BENCHER_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `LLM_BENCHER_OPENAI_API_KEY` | *(empty)* | OpenAI API key for cloud provider |
| `LLM_BENCHER_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `LLM_BENCHER_DATA_DIR` | `./data` | Data directory for SQLite DB and prompt suites |
| `LLM_BENCHER_PROVIDER_TIMEOUT` | `30.0` | HTTP timeout (seconds) for provider calls |
| `LLM_BENCHER_SQLITE_ECHO` | `false` | Enable SQL query logging |

## Testing

```bash
# Run all 209 tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_batch.py

# Run a specific test
uv run pytest tests/test_analytics.py::test_summary_with_runs
```

## Project Structure

```
llm_bencher/
├── app.py                  # FastAPI app factory and startup
├── config.py               # Settings with env var overrides
├── models.py               # SQLAlchemy ORM models
├── schemas.py              # Pydantic transfer objects
├── database.py             # Engine/session factory
├── bootstrap.py            # Default provider seeding
├── runner.py               # Single run execution
├── batch_runner.py         # Batch execution with concurrency control
├── prompt_io.py            # Prompt suite import/export
├── providers/
│   ├── base.py             # Abstract ProviderAdapter
│   ├── openai_compat.py    # OpenAI-compatible API adapter
│   ├── openai_cloud.py     # OpenAI cloud adapter (with auth)
│   ├── lm_studio.py        # LM Studio adapter
│   ├── ollama.py           # Ollama native API adapter
│   └── registry.py         # Provider kind → adapter routing
├── web/
│   ├── api.py              # JSON API endpoints
│   ├── analytics_api.py    # Analytics endpoints
│   └── routes.py           # HTML page routes
├── templates/              # Jinja2 templates
└── static/css/app.css      # Stylesheet
```

## Prompt Suite Format

```json
{
  "slug": "my-suite",
  "name": "My Test Suite",
  "prompts": [
    {
      "slug": "greeting",
      "title": "Greeting Test",
      "user_prompt_template": "Say hello to {name}",
      "tags": ["basic", "greeting"],
      "variables": [
        {"name": "name", "description": "Name to greet", "required": true}
      ]
    }
  ]
}
```
