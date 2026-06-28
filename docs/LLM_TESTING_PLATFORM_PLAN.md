# LLM Testing Platform — Implementation Plan

## Current State

LLM-Bencher is a working FastAPI + Jinja2 server-rendered app with 86 passing tests. It supports:

- **2 providers**: LM Studio (OpenAI-compat) and Ollama (native API)
- **Provider management**: health checks, model discovery, connection status
- **Prompt library**: JSON import/export, suites with categories/tags/variables
- **Single prompt execution**: select provider + model, run one prompt, view result inline
- **Run history**: filtering (provider, status, date range), pagination (25/page)
- **Run detail**: full output, prompts sent, token usage, latency
- **7 DB tables**: Provider, ProviderModel, PromptSuite, PromptDefinition, PromptImportRecord, Run, RunResult

---

## Implementation Slices

### Slice 1: Run Ratings & Annotations

**Goal:** Let users rate and annotate individual run outputs (1–5 stars + notes).

**DB Changes (`models.py`):**
- New `RunRating` model:
  - `id` (PK), `run_id` (FK → runs.id, unique), `score` (Integer 1–5), `notes` (Text, nullable)
  - Uses `TimestampMixin`
  - Relationship: `Run.rating` (uselist=False)

**New Schemas (`schemas.py`):**
- `RatingRequest`: `score: int = Field(ge=1, le=5)`, `notes: str | None = None`
- `RatingResponse`: `run_id`, `score`, `notes`, `created_at`

**New API Endpoints (`web/api.py`):**
- `POST /api/runs/{run_id}/rating` — create/update rating (upsert)
- `GET /api/runs/{run_id}/rating` — get rating
- `DELETE /api/runs/{run_id}/rating` — remove rating

**Template Changes:**
- `run_detail.html`: star-rating widget in sidebar (5 clickable stars, notes textarea, save button)
- `history.html`: small star indicator next to status badge when run has a rating

**Tests (`tests/test_ratings.py`):**
- Create/update/delete rating on succeeded and failed runs
- Invalid score (0, 6) returns 422
- Rating for nonexistent run returns 404
- Rating appears on run detail page
- Rating indicator on history page

**Dependencies:** None — can start immediately.

---

### Slice 2: Tag-Based Prompt Filtering

**Goal:** Make the existing `tags` JSON column on `PromptDefinition` filterable in the UI.

**DB Changes:** None — tags already stored as JSON arrays.

**New API Endpoint:**
- `GET /api/tags` — return distinct tags across all active prompt definitions

**Modified Routes (`web/routes.py`):**
- `prompts_page`: accept `?tag=X` query param, filter prompts containing that tag
- `history_page`: accept `?tag=X` query param, join Run → PromptDefinition and filter

**Template Changes:**
- `prompts.html`: render tags as clickable pill badges; clicking filters by that tag; active filter banner with "clear" link
- `history.html`: add tag dropdown in filter bar

**Tests (`tests/test_tags.py`):**
- `/api/tags` returns unique tags from imported suites
- Prompts page filters by tag
- History page filters by prompt tag
- Nonexistent tag returns empty results

**Dependencies:** None — can start immediately.

---

### Slice 3: Prompt Template Variables UI

**Goal:** Dynamic form fields for prompt variables so `template_inputs` is actually populated.

**DB Changes:** None — `Run.template_inputs` JSON column and `PromptDefinition.variables` already exist.

**Modified Template (`runs.html`):**
- When a prompt is selected, parse `variables` from `/api/prompts/{id}` response
- Dynamically render labeled text inputs per variable (required attribute if `required: true`, default values pre-filled)
- Before submit: collect variable values into `template_inputs`, substitute `{variable_name}` in `user_prompt_template` to produce `user_prompt`

**Modified API (`web/api.py`):**
- `RunCreateBody`: add `template_inputs: dict[str, Any] = Field(default_factory=dict)`
- `create_run`: pass `template_inputs` to the `Run` row

**Tests (`tests/test_variables.py`):**
- Run with template_inputs stores them in DB
- API accepts and persists variable values
- Prompt with variables populates form (check response text)

**Dependencies:** None — can start immediately.

---

### Slice 4: Batch Run Execution

**Goal:** Run a prompt (or suite) against multiple models simultaneously with progress tracking.

**DB Changes (`models.py`):**
- New enum `BatchStatus(StrEnum)`: `PENDING`, `RUNNING`, `COMPLETED`, `PARTIAL`, `FAILED`
- New `BatchRun` model:
  - `id` (PK), `name` (String, nullable — auto-generated if omitted)
  - `status` (Enum BatchStatus, default PENDING)
  - `total_runs`, `completed_runs`, `failed_runs` (Integer)
  - `started_at`, `completed_at` (DateTime, nullable)
  - Uses `TimestampMixin`
  - Relationship: `runs` → list of Run
- Add to `Run`: `batch_id` (FK → batch_runs.id, nullable)

**New Schemas (`schemas.py`):**
- `ModelTarget`: `provider_id: int`, `model_external_id: str`, `model_name: str | None`
- `BatchCreateRequest`: `name: str | None`, `models: list[ModelTarget]`, `prompt_ids: list[int]`, `temperature: float | None`, `max_tokens: int | None`
- `BatchStatusResponse`: batch metadata + run summaries

**New API Endpoints:**
- `POST /api/batches` — create batch, generate Run rows for each (prompt, model) combo
- `POST /api/batches/{batch_id}/execute` — execute all PENDING runs with `asyncio.gather` + `Semaphore(3)` concurrency limit
- `GET /api/batches/{batch_id}` — batch status + run summaries
- `GET /api/batches` — list all batches

**New Execution Logic (`batch_runner.py`):**
- `execute_batch()`: load PENDING runs, group by provider for adapter reuse, execute concurrently, update Run + BatchRun counters, handle partial failures

**New Routes + Templates:**
- `GET /runs/batch` → `batch_new.html`: multi-select models (checkboxes by provider) + multi-select prompts (grouped by suite), submit creates + executes batch
- `GET /batches/{batch_id}` → `batch_detail.html`: run table with status, progress bar, summary stats

**Nav Update (`base.html`):** Add "Batch Run" link.

**Tests (`tests/test_batch.py`):**
- Create batch with 2 models × 2 prompts = 4 runs
- Execute batch succeeds with mocked adapters
- Partial failure (1 of 4 fails → status=PARTIAL)
- Batch status tracking (PENDING → RUNNING → COMPLETED)
- Batch with no models returns 422
- Batch detail page renders

**Dependencies:** None — can start immediately, but benefits from Slices 1–3 being done first.

---

### Slice 5: Side-by-Side Comparison

**Goal:** Compare outputs from 2+ models for the same prompt in a side-by-side view.

**DB Changes (`models.py`):**
- New `Comparison` model:
  - `id` (PK), `name` (String, nullable), `prompt_id` (FK, nullable), `batch_id` (FK, nullable)
  - Uses `TimestampMixin`
- New `ComparisonItem` model:
  - `id` (PK), `comparison_id` (FK), `run_id` (FK), `position` (Integer)
  - UniqueConstraint on (comparison_id, run_id)

**New API Endpoints:**
- `POST /api/comparisons` — create from selected run IDs
- `POST /api/comparisons/from-batch/{batch_id}` — auto-create comparisons grouping runs by prompt
- `GET /api/comparisons/{id}` — comparison with full run data
- `GET /api/comparisons` — list comparisons

**New Routes + Templates:**
- `GET /compare/{id}` → `compare.html`: CSS grid, 2–4 columns, each showing model name, provider, latency, tokens, output text with scrollable areas
- `GET /compare/select` → `compare_select.html`: searchable run list with checkboxes + "Compare selected" button

**Integration with Existing Views:**
- `batch_detail.html`: "Compare outputs" button
- `history.html`: checkboxes on runs + "Compare selected" button
- `run_detail.html`: "Compare with..." link

**Tests (`tests/test_comparison.py`):**
- Create comparison from 2 run IDs
- Create comparison from batch (auto-groups by prompt)
- Comparison page renders side-by-side
- Edge cases: single run, different prompts

**Dependencies:** Slice 4 (batch runs).

---

### Slice 6: Additional Provider Support

**Goal:** Add OpenAI cloud API, generic OpenAI-compat endpoints, and provider CRUD.

**DB Changes (`models.py`):**
- Extend `ProviderKind`: add `OPENAI = "openai"`, `OPENAI_COMPAT = "openai_compat"`
- Add to `Provider`: `api_key` (String(255), nullable), `is_default` (Boolean, default False)

**Config (`config.py`):**
- Add `openai_api_key: str = ""`, `openai_base_url: str = "https://api.openai.com/v1"`

**New Provider (`providers/openai_cloud.py`):**
- `OpenAICloudAdapter(OpenAICompatAdapter)`: adds `Authorization: Bearer {api_key}` header

**Modified Registry (`providers/registry.py`):**
- Route `OPENAI` → `OpenAICloudAdapter`, `OPENAI_COMPAT` → `OpenAICompatAdapter`

**New API Endpoints:**
- `POST /api/providers` — create custom provider (name, kind, base_url, api_key)
- `PUT /api/providers/{id}` — update provider settings
- `DELETE /api/providers/{id}` — delete (only non-default; soft-delete if runs reference it)

**Template Changes (`providers.html`):**
- "Add Provider" button with form (name, kind, base URL, optional API key)
- Edit/delete buttons on non-default providers

**Bootstrap (`bootstrap.py`):** Mark seeded providers with `is_default=True`.

**Tests (`tests/test_provider_crud.py`):**
- Create/update/delete custom provider
- Cannot delete default provider
- OpenAI adapter includes auth header (mock httpx)
- OpenAI-compat with custom URL works

**Dependencies:** None — can start immediately.

---

### Slice 7: Performance Analytics Dashboard

**Goal:** Charts and aggregations for latency, token usage, and success rates across models.

**DB Changes:** None — aggregation queries on existing Run + RunResult tables.

**Frontend Dependency:** Chart.js bundled at `/static/js/chart.min.js` for offline use.

**New API Endpoints (`web/analytics_api.py`):**
- `GET /api/analytics/latency?group_by=model&period=7d` — avg, p50, p95, min, max latency
- `GET /api/analytics/tokens?group_by=model&period=7d` — token usage aggregation
- `GET /api/analytics/success-rate?group_by=model&period=7d` — success/failure counts
- `GET /api/analytics/timeline?model_id=X&metric=latency&period=30d` — time-series data
- `GET /api/analytics/summary` — overall stats for dashboard header

**Query Patterns:** SQLAlchemy `func.avg`, `func.count`, `func.sum`, `case()` for conditional aggregation, grouped by `Run.model_identifier` or `Run.provider_id`.

**New Route + Template:**
- `GET /analytics` → `analytics.html`: summary cards (total runs, avg latency, total tokens, success rate), latency bar chart per model, token usage chart, success rate donut, latency trend line chart, date range + provider filter controls

**Nav Update (`base.html`):** Add "Analytics" link.

**Tests (`tests/test_analytics.py`):**
- Analytics endpoints return correct aggregations with known test data
- Empty data returns sensible defaults (zeros)
- Date range filtering works
- Analytics page renders

**Dependencies:** None technically, but more useful after Slice 4 generates data.

---

### Slice 8: Export & Reporting

**Goal:** CSV export for history, comparisons, batches, and analytics.

**New API Endpoints:**
- `GET /api/export/history?format=csv&provider_id=X&status=Y` — filtered run history as CSV (reuse history filter logic, `StreamingResponse`)
- `GET /api/export/comparison/{id}?format=csv` — one row per model
- `GET /api/export/batch/{id}?format=csv` — batch results
- `GET /api/export/analytics?format=csv&group_by=model` — aggregated analytics

**Template Changes:** Add "Export CSV" buttons to history, compare, batch_detail, and analytics pages.

**Tests (`tests/test_export.py`):**
- CSV has correct headers and content type `text/csv`
- CSV export respects filters
- Empty data returns headers only
- Comparison and batch CSV exports

**Dependencies:** Slices 4, 5, 7.

---

### Slice 9: Enhanced Home Dashboard

**Goal:** Update home page to reflect all new capabilities.

**Modified Route (`web/routes.py`):**
- Expand summary counts: batch count, comparison count, rated runs count

**Template Changes (`home.html`):**
- Quick-action cards: "Run Single Prompt", "Start Batch Run", "View Analytics"
- Recent activity feed (last 5 runs with status badges)
- Mini latency sparkline chart (last 24h)
- Provider status indicators (connected/disconnected)

**Dependencies:** All feature slices.

---

### Slice 10: Comprehensive Test Expansion

**Goal:** Cross-feature integration tests and edge case coverage.

**New Test Files:**
- `tests/test_integration.py`: end-to-end workflows:
  - Import suite → batch run → comparison → export
  - Create provider → health check → run → rate → analytics
- `tests/test_edge_cases.py`:
  - Very large prompt text, Unicode in prompts/outputs
  - Empty batch, comparison with deleted runs
  - SQLite JSON column edge cases
  - Concurrent batch execution behavior

**Fixture Additions (`conftest.py`):**
- `seeded_client`: pre-populated providers, models, suites, and runs for complex test scenarios
- Batch creation helper

**Dependencies:** All feature slices.

---

## Dependency Graph

```
Independent (Group A — can all start in parallel):
  Slice 1: Ratings
  Slice 2: Tags
  Slice 3: Variables UI
  Slice 4: Batch Runs
  Slice 6: Provider CRUD

After Slice 4 (Group B):
  Slice 5: Comparisons
  Slice 7: Analytics

After Group B (Group C):
  Slice 8: Export

After all features (Group D):
  Slice 9: Enhanced Home
  Slice 10: Test Expansion
```

## Recommended Implementation Order

| Phase | Slices | Rationale |
|-------|--------|-----------|
| **Phase 1** | 1, 2, 3 | Small, independent, low-risk. Build confidence. |
| **Phase 2** | 4, 6 | Core new capabilities: batch execution + provider CRUD. |
| **Phase 3** | 5, 7 | Comparison and analytics depend on batch data. |
| **Phase 4** | 8, 9, 10 | Polish: export, dashboard, comprehensive tests. |

## File-Level Change Summary

| File | Slices | Changes |
|------|--------|---------|
| `models.py` | 1, 4, 5, 6 | +4 models (RunRating, BatchRun, Comparison, ComparisonItem), modify Run (+batch_id), Provider (+api_key, +is_default), extend ProviderKind |
| `schemas.py` | 1, 3, 4, 5 | +5 schemas (RatingRequest/Response, ModelTarget, BatchCreateRequest, BatchStatusResponse) |
| `web/api.py` | 1, 2, 3, 4, 5, 6, 8 | +15 endpoints across ratings, tags, batches, comparisons, provider CRUD, export |
| `web/analytics_api.py` | 7 | New file: 5 analytics endpoints |
| `web/routes.py` | 2, 4, 5, 7, 9 | +4 routes (batch, comparison, analytics pages), modify prompts/history (tags), home |
| `batch_runner.py` | 4 | New file: async batch execution with concurrency control |
| `providers/openai_cloud.py` | 6 | New file: OpenAI cloud adapter with auth |
| `providers/registry.py` | 6 | Add OPENAI + OPENAI_COMPAT routing |
| `config.py` | 6 | Add OpenAI settings |
| `bootstrap.py` | 6 | Mark default providers |
| Templates (new) | 4, 5, 7 | `batch_new.html`, `batch_detail.html`, `compare.html`, `compare_select.html`, `analytics.html` |
| Templates (modified) | 1, 2, 3, 8, 9 | `run_detail.html`, `history.html`, `prompts.html`, `runs.html`, `home.html`, `base.html` |
| `static/css/app.css` | All | Star ratings, tag pills, multi-select, comparison grid, chart containers |
| `static/js/chart.min.js` | 7 | Bundled Chart.js for offline analytics |
| Tests (new) | All | 8 new test files: ratings, tags, variables, batch, comparison, provider_crud, analytics, export, integration, edge_cases |
| `pyproject.toml` | — | No new Python dependencies |

## Migration Strategy

Each slice that adds DB tables needs an Alembic migration:
1. Slice 1: `alembic revision --autogenerate -m "add run_ratings table"`
2. Slice 4: `alembic revision --autogenerate -m "add batch_runs table and run.batch_id"`
3. Slice 5: `alembic revision --autogenerate -m "add comparisons and comparison_items tables"`
4. Slice 6: `alembic revision --autogenerate -m "extend provider_kind enum add api_key and is_default columns"`

Since `Base.metadata.create_all()` runs on startup, new tables are created automatically in dev. Migrations are for existing databases that need to be upgraded.
