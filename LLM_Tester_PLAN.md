# Local LLM Prompt Lab (FastAPI)

## Summary
Build a fresh local-only FastAPI app that lets one user connect to LM Studio and Ollama running on the same machine, select an available model, run pre-written test prompts, and review saved outputs. The first version should optimize for a manual testing workflow, not automated grading or multi-user collaboration.

## Key Changes
### App shape
- Use a monolithic FastAPI app with server-rendered pages via Jinja2 and light client-side behavior via HTMX or vanilla JS.
- Treat the existing static quiz files in this repo as unrelated legacy content; the new app should live in its own Python app structure.
- Run as a localhost desktop-style web app with no authentication in v1.

### Core product behavior
- Add a provider management layer for `LM Studio` and `Ollama` with configurable local base URLs and a connection test action.
- Add model discovery per provider so the UI can show currently available local models and let the user choose one active target.
- Add a prompt-suite library with curated, pre-written prompts grouped into suites/categories and tagged for easy filtering.
- Support running one prompt at a time against one selected model, saving the full request/response record so the user can compare runs later from history.
- Add result history views with filters for provider, model, prompt suite, prompt, and run time.

### Data and interfaces
- Store app state in SQLite for providers, runs, run results, prompt metadata, and import/export bookkeeping.
- Keep prompt suites portable as versionable JSON or YAML files on disk, with import/export support so users can edit suites outside the app and sync them back in.
- Define a normalized internal run request shape:
  - provider id
  - model id/name
  - prompt id
  - system prompt
  - user prompt/template inputs
  - inference settings such as temperature and max tokens
- Define a normalized run result shape:
  - raw output text
  - response metadata
  - latency
  - token usage when available
  - provider-specific raw payload snapshot for troubleshooting
- Implement a provider adapter interface with a shared chat-generation method and model-list method.
- Start with an OpenAI-style adapter for LM Studio and use the same contract for Ollama when possible, with a thin Ollama-native fallback for chat/generate responses if its local API shape differs.

### Suggested implementation slices
- Backend foundation: FastAPI app factory, settings, SQLite models/migrations, Jinja templates, static assets, local config for provider URLs.
- Prompt library: file parser/importer, suite listing, detail view, CRUD-lite management if needed, and export endpoint.
- Provider adapters: connection test, model listing, request normalization, response normalization, provider error handling.
- Run workflow: choose provider/model/prompt, edit runtime settings, execute prompt, persist result, show output detail page and history page.
- UX polish: clear local-only messaging, status badges for provider availability, helpful error states when LM Studio/Ollama is not running.

## Public Interfaces / Types
- `ProviderAdapter` interface with methods equivalent to `health_check()`, `list_models()`, and `run_chat(request)`.
- `RunRequest` and `RunResult` schemas shared between the web layer and provider adapters.
- Prompt-suite file schema with suite metadata, prompt records, optional variables, and default generation settings.
- FastAPI routes/pages for:
  - provider settings and connection checks
  - model refresh/listing
  - prompt suite browse/import/export
  - run creation/execution
  - run history/detail

## Test Plan
- Unit tests for prompt-suite parsing, adapter normalization, and SQLite persistence.
- Adapter tests using mocked LM Studio and Ollama responses for health, model listing, and chat execution.
- Integration tests for the main flow: configure provider, refresh models, import suite, run prompt, view saved result.
- Error-path tests for provider offline, invalid base URL, timeout, malformed response, missing model, and prompt import validation failures.
- Manual acceptance checks on a local machine with LM Studio and Ollama actually running:
  - both providers can connect
  - model lists load
  - a prompt executes successfully
  - the result is saved and visible in history
  - imported prompt suites round-trip through export without data loss

## Assumptions And Defaults
- Default stack: Python, FastAPI, Jinja2, SQLAlchemy/SQLModel-style ORM, Alembic, SQLite.
- v1 is single-user and localhost-only; no auth, user accounts, or remote deployment concerns.
- v1 does not include rubric scoring, automated eval judgments, pairwise model comparison in the same run, or batch tournament execution.
- Default provider strategy is unified OpenAI-compatible chat handling first, with an Ollama-specific fallback only where required.
- Prompt suites are primarily file-authored but mirrored into SQLite for app features like history linkage, filtering, and import tracking.
