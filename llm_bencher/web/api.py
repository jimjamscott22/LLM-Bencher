from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from llm_bencher.models import (
    BatchRun,
    BatchStatus,
    Comparison,
    ComparisonItem,
    Provider,
    ProviderKind,
    PromptDefinition,
    PromptSuite,
    ProviderModel,
    Run,
    RunRating,
    RunResult as RunResultModel,
    RunStatus,
)
from llm_bencher.prompt_io import (
    compute_checksum,
    export_suite,
    export_suite_to_json,
    import_suite,
    load_suite_from_string,
)
from llm_bencher.providers.registry import get_adapter
from llm_bencher.runner import build_run_request, execute_adapter


router = APIRouter(prefix="/api")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _model_dict(m: ProviderModel) -> dict:
    return {
        "id": m.id,
        "external_id": m.external_id,
        "display_name": m.display_name,
        "is_available": m.is_available,
        "last_seen_at": m.last_seen_at.isoformat() if m.last_seen_at else None,
    }


@router.post("/providers/{provider_id}/check")
async def check_provider(provider_id: int, request: Request) -> JSONResponse:
    """
    Run a health check on the provider, discover its models, and persist the
    results. Returns the updated connection state and full model list.

    Async I/O (HTTP calls) is done before the write session opens so that no
    session is held open during network waits.
    """
    session_factory = request.app.state.session_factory
    settings = request.app.state.settings

    # Phase 1: load provider fields needed to build the adapter (read-only).
    with session_factory() as session:
        provider = session.get(Provider, provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")
        adapter = get_adapter(provider, settings)

    # Phase 2: async network I/O — no session is open here.
    health = await adapter.health_check()
    discovered = await adapter.list_models() if health.is_available else []

    # Phase 3: persist results synchronously.
    with session_factory() as session:
        provider = session.get(Provider, provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        provider.is_connected = health.is_available
        provider.last_health_check_at = health.checked_at
        provider.last_error = health.detail if not health.is_available else None

        if discovered:
            discovered_ids = {dm.id for dm in discovered}

            for dm in discovered:
                existing = session.scalar(
                    select(ProviderModel).where(
                        ProviderModel.provider_id == provider_id,
                        ProviderModel.external_id == dm.id,
                    )
                )
                if existing:
                    existing.display_name = dm.name
                    existing.is_available = True
                    existing.last_seen_at = _utc_now()
                else:
                    session.add(
                        ProviderModel(
                            provider_id=provider_id,
                            external_id=dm.id,
                            display_name=dm.name,
                            is_available=True,
                            last_seen_at=_utc_now(),
                        )
                    )

            # Mark models no longer returned as unavailable.
            stale = session.scalars(
                select(ProviderModel).where(
                    ProviderModel.provider_id == provider_id,
                    ProviderModel.external_id.not_in(discovered_ids),
                )
            ).all()
            for m in stale:
                m.is_available = False

        session.flush()
        all_models = session.scalars(
            select(ProviderModel)
            .where(ProviderModel.provider_id == provider_id)
            .order_by(ProviderModel.display_name)
        ).all()
        models = [_model_dict(m) for m in all_models]
        session.commit()

    return JSONResponse(
        {
            "is_connected": health.is_available,
            "detail": health.detail,
            "checked_at": health.checked_at.isoformat() if health.checked_at else None,
            "models": models,
        }
    )


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------

def _provider_dict(p: Provider) -> dict:
    return {
        "id": p.id,
        "slug": p.slug,
        "name": p.name,
        "kind": p.kind.value,
        "base_url": p.base_url,
        "has_api_key": bool(p.api_key),
        "is_enabled": p.is_enabled,
        "is_default": p.is_default,
        "is_connected": p.is_connected,
    }


class ProviderCreateBody(BaseModel):
    slug: str
    name: str
    kind: str
    base_url: str
    api_key: str | None = None
    is_enabled: bool = True


class ProviderUpdateBody(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    is_enabled: bool | None = None


@router.get("/providers")
def list_providers(request: Request) -> JSONResponse:
    """List all providers."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        providers = session.scalars(
            select(Provider).order_by(Provider.name)
        ).all()
    return JSONResponse([_provider_dict(p) for p in providers])


@router.post("/providers")
def create_provider(body: ProviderCreateBody, request: Request) -> JSONResponse:
    """Add a custom provider."""
    try:
        kind = ProviderKind(body.kind)
    except ValueError:
        valid = [k.value for k in ProviderKind]
        raise HTTPException(status_code=422, detail=f"Invalid kind. Must be one of: {valid}")

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        existing = session.scalar(
            select(Provider).where(Provider.slug == body.slug)
        )
        if existing:
            raise HTTPException(status_code=409, detail=f"Provider with slug '{body.slug}' already exists")

        provider = Provider(
            slug=body.slug,
            name=body.name,
            kind=kind,
            base_url=body.base_url,
            api_key=body.api_key,
            is_enabled=body.is_enabled,
            is_default=False,
        )
        session.add(provider)
        session.flush()
        data = _provider_dict(provider)
        session.commit()

    return JSONResponse(data, status_code=201)


@router.put("/providers/{provider_id}")
def update_provider(provider_id: int, body: ProviderUpdateBody, request: Request) -> JSONResponse:
    """Update an existing provider."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        provider = session.get(Provider, provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        if body.name is not None:
            provider.name = body.name
        if body.base_url is not None:
            provider.base_url = body.base_url
        if body.api_key is not None:
            provider.api_key = body.api_key if body.api_key else None
        if body.is_enabled is not None:
            provider.is_enabled = body.is_enabled

        session.flush()
        data = _provider_dict(provider)
        session.commit()

    return JSONResponse(data)


@router.delete("/providers/{provider_id}")
def delete_provider(provider_id: int, request: Request) -> JSONResponse:
    """Delete a custom (non-default) provider."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        provider = session.get(Provider, provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")
        if provider.is_default:
            raise HTTPException(status_code=403, detail="Cannot delete a default provider")

        # Check for existing runs.
        run_count = session.scalar(
            select(func.count()).select_from(Run).where(Run.provider_id == provider_id)
        ) or 0
        if run_count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete provider with {run_count} existing run(s). Disable it instead.",
            )

        session.delete(provider)
        session.commit()

    return JSONResponse({"id": provider_id, "deleted": True})


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

@router.get("/tags")
def list_tags(request: Request) -> JSONResponse:
    """Return distinct tags across all active prompt definitions."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        prompts = session.scalars(
            select(PromptDefinition)
            .join(PromptSuite)
            .where(PromptSuite.is_active.is_(True))
        ).all()

    tags: set[str] = set()
    for p in prompts:
        if p.tags:
            tags.update(p.tags)

    return JSONResponse(sorted(tags))


# ---------------------------------------------------------------------------
# Prompt suites
# ---------------------------------------------------------------------------

def _suite_dict(suite: PromptSuite) -> dict:
    return {
        "id": suite.id,
        "slug": suite.slug,
        "name": suite.name,
        "description": suite.description,
        "version": suite.version,
        "is_active": suite.is_active,
        "prompt_count": len(suite.prompts),
        "imported_at": suite.imported_at.isoformat() if suite.imported_at else None,
    }


@router.get("/suites")
def list_suites(request: Request) -> JSONResponse:
    """Return all active prompt suites."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        suites = session.scalars(
            select(PromptSuite)
            .options(selectinload(PromptSuite.prompts))
            .where(PromptSuite.is_active.is_(True))
            .order_by(PromptSuite.name)
        ).all()
    return JSONResponse([_suite_dict(s) for s in suites])


@router.post("/suites/import")
async def import_suite_endpoint(request: Request, file: UploadFile) -> JSONResponse:
    """
    Import a prompt suite from an uploaded JSON file.
    Upserts the suite and its prompts by slug.
    """
    session_factory = request.app.state.session_factory

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
        suite_file = load_suite_from_string(content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    checksum = compute_checksum(content)
    source_path = file.filename or "<upload>"

    with session_factory() as session:
        suite, action = import_suite(session, suite_file, source_path, checksum)
        session.flush()
        suite_id = suite.id
        suite_name = suite.name
        prompt_count = len(suite_file.prompts)
        session.commit()

    return JSONResponse(
        {
            "id": suite_id,
            "slug": suite_file.slug,
            "name": suite_name,
            "prompt_count": prompt_count,
            "action": action,
        },
        status_code=201 if action == "created" else 200,
    )


@router.get("/suites/{suite_id}")
def get_suite(suite_id: int, request: Request) -> JSONResponse:
    """Return suite details including its prompt list."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        suite = session.scalar(
            select(PromptSuite)
            .options(selectinload(PromptSuite.prompts))
            .where(PromptSuite.id == suite_id)
        )
        if suite is None:
            raise HTTPException(status_code=404, detail="Suite not found")

        prompts = [
            {
                "id": p.id,
                "slug": p.slug,
                "title": p.title,
                "category": p.category,
                "description": p.description,
                "tags": p.tags,
            }
            for p in suite.prompts
        ]

    return JSONResponse({**_suite_dict(suite), "prompts": prompts})


@router.get("/suites/{suite_id}/export")
def export_suite_endpoint(suite_id: int, request: Request) -> Response:
    """Download a suite as a JSON file."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        suite = session.scalar(
            select(PromptSuite)
            .options(selectinload(PromptSuite.prompts))
            .where(PromptSuite.id == suite_id)
        )
        if suite is None:
            raise HTTPException(status_code=404, detail="Suite not found")

        suite_file = export_suite(suite)
        suite.exported_at = _utc_now()
        session.commit()

    json_str = export_suite_to_json(suite_file)
    filename = f"{suite_file.slug}.json"
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/suites/{suite_id}")
def delete_suite(suite_id: int, request: Request) -> JSONResponse:
    """Soft-delete a suite (sets is_active=False)."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        suite = session.get(PromptSuite, suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail="Suite not found")
        suite.is_active = False
        session.commit()
    return JSONResponse({"id": suite_id, "deleted": True})


# ---------------------------------------------------------------------------
# Prompts (individual prompt detail for form pre-population)
# ---------------------------------------------------------------------------

@router.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: int, request: Request) -> JSONResponse:
    """Return a single prompt's details for pre-populating the run form."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        prompt = session.get(PromptDefinition, prompt_id)
        if prompt is None:
            raise HTTPException(status_code=404, detail="Prompt not found")
    return JSONResponse(
        {
            "id": prompt.id,
            "slug": prompt.slug,
            "title": prompt.title,
            "system_prompt": prompt.system_prompt,
            "user_prompt_template": prompt.user_prompt_template,
            "variables": prompt.variables or [],
            "default_temperature": prompt.default_temperature,
            "default_max_tokens": prompt.default_max_tokens,
        }
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunCreateBody(BaseModel):
    provider_id: int
    model_external_id: str
    model_name: str | None = None
    prompt_id: int | None = None
    system_prompt: str | None = None
    user_prompt: str
    template_inputs: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None


@router.post("/runs")
async def create_run(body: RunCreateBody, request: Request) -> JSONResponse:
    """
    Create a Run row, execute it against the provider adapter, and persist the
    result. Returns the run outcome inline so the form can display it immediately.

    Session phases:
      1. Validate provider → build adapter + PENDING run row → commit.
      2. Async adapter call — no session open.
      3. Write outcome (status + RunResult) → commit.
    """
    session_factory = request.app.state.session_factory
    settings = request.app.state.settings

    # Phase 1: validate and create PENDING run.
    with session_factory() as session:
        provider = session.get(Provider, body.provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        # Resolve ProviderModel row if one exists for this external_id.
        pm = session.scalar(
            select(ProviderModel).where(
                ProviderModel.provider_id == body.provider_id,
                ProviderModel.external_id == body.model_external_id,
            )
        )

        adapter = get_adapter(provider, settings)

        run = Run(
            provider_id=body.provider_id,
            provider_model_id=pm.id if pm else None,
            prompt_id=body.prompt_id,
            status=RunStatus.PENDING,
            model_identifier=body.model_external_id,
            model_name=body.model_name or body.model_external_id,
            system_prompt=body.system_prompt,
            user_prompt=body.user_prompt,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            template_inputs=body.template_inputs,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        run_request = build_run_request(run)
        session.commit()

    # Phase 2: async execution — no session held open.
    result_schema, failure_message, started_at, completed_at = await execute_adapter(
        adapter, run_request
    )
    status = RunStatus.SUCCEEDED if result_schema else RunStatus.FAILED

    # Phase 3: persist outcome.
    with session_factory() as session:
        run = session.get(Run, run_id)
        run.status = status
        run.started_at = started_at
        run.completed_at = completed_at
        run.failure_message = failure_message

        if result_schema:
            session.add(
                RunResultModel(
                    run_id=run_id,
                    raw_output_text=result_schema.output_text,
                    response_metadata=result_schema.response_metadata,
                    latency_ms=result_schema.latency_ms,
                    prompt_tokens=result_schema.prompt_tokens,
                    completion_tokens=result_schema.completion_tokens,
                    total_tokens=result_schema.total_tokens,
                    raw_payload=result_schema.raw_payload,
                )
            )
        session.commit()

    return JSONResponse(
        {
            "run_id": run_id,
            "status": status.value,
            "output_text": result_schema.output_text if result_schema else None,
            "failure_message": failure_message,
            "latency_ms": result_schema.latency_ms if result_schema else None,
            "prompt_tokens": result_schema.prompt_tokens if result_schema else None,
            "completion_tokens": result_schema.completion_tokens if result_schema else None,
            "total_tokens": result_schema.total_tokens if result_schema else None,
        },
        status_code=201,
    )


# ---------------------------------------------------------------------------
# Run ratings
# ---------------------------------------------------------------------------

class RatingBody(BaseModel):
    score: int
    notes: str | None = None


@router.post("/runs/{run_id}/rating")
def upsert_rating(run_id: int, body: RatingBody, request: Request) -> JSONResponse:
    """Create or update a rating for a run."""
    if not (1 <= body.score <= 5):
        raise HTTPException(status_code=422, detail="Score must be between 1 and 5")

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        rating = session.scalar(
            select(RunRating).where(RunRating.run_id == run_id)
        )
        if rating:
            rating.score = body.score
            rating.notes = body.notes
            action = "updated"
        else:
            rating = RunRating(run_id=run_id, score=body.score, notes=body.notes)
            session.add(rating)
            action = "created"

        session.flush()
        data = {
            "run_id": run_id,
            "score": rating.score,
            "notes": rating.notes,
            "created_at": rating.created_at.isoformat(),
            "action": action,
        }
        session.commit()

    return JSONResponse(data, status_code=201 if action == "created" else 200)


@router.get("/runs/{run_id}/rating")
def get_rating(run_id: int, request: Request) -> JSONResponse:
    """Get the rating for a run."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        rating = session.scalar(
            select(RunRating).where(RunRating.run_id == run_id)
        )
        if rating is None:
            raise HTTPException(status_code=404, detail="No rating for this run")

    return JSONResponse({
        "run_id": run_id,
        "score": rating.score,
        "notes": rating.notes,
        "created_at": rating.created_at.isoformat(),
    })


@router.delete("/runs/{run_id}/rating")
def delete_rating(run_id: int, request: Request) -> JSONResponse:
    """Delete the rating for a run."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        rating = session.scalar(
            select(RunRating).where(RunRating.run_id == run_id)
        )
        if rating is None:
            raise HTTPException(status_code=404, detail="No rating for this run")

        session.delete(rating)
        session.commit()

    return JSONResponse({"run_id": run_id, "deleted": True})


# ---------------------------------------------------------------------------
# Batch runs
# ---------------------------------------------------------------------------

class ModelTarget(BaseModel):
    provider_id: int
    model_external_id: str
    model_name: str | None = None


class BatchCreateBody(BaseModel):
    name: str | None = None
    models: list[ModelTarget]
    prompt_ids: list[int]
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def _batch_dict(batch: BatchRun) -> dict:
    return {
        "id": batch.id,
        "name": batch.name,
        "status": batch.status.value,
        "total_runs": batch.total_runs,
        "completed_runs": batch.completed_runs,
        "failed_runs": batch.failed_runs,
        "started_at": batch.started_at.isoformat() if batch.started_at else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
        "created_at": batch.created_at.isoformat(),
    }


@router.post("/batches")
async def create_batch(body: BatchCreateBody, request: Request) -> JSONResponse:
    """Create a batch and generate Run rows for each (prompt, model) combination,
    then immediately execute all runs."""
    if not body.models:
        raise HTTPException(status_code=422, detail="At least one model is required")
    if not body.prompt_ids:
        raise HTTPException(status_code=422, detail="At least one prompt is required")

    session_factory = request.app.state.session_factory
    settings = request.app.state.settings

    with session_factory() as session:
        # Validate all prompt IDs exist.
        for pid in body.prompt_ids:
            prompt = session.get(PromptDefinition, pid)
            if prompt is None:
                raise HTTPException(status_code=404, detail=f"Prompt {pid} not found")

        # Validate all providers exist.
        for mt in body.models:
            provider = session.get(Provider, mt.provider_id)
            if provider is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Provider {mt.provider_id} not found",
                )

        batch_name = body.name or f"Batch {_utc_now().strftime('%Y-%m-%d %H:%M')}"
        batch = BatchRun(
            name=batch_name,
            status=BatchStatus.PENDING,
            total_runs=len(body.models) * len(body.prompt_ids),
        )
        session.add(batch)
        session.flush()
        batch_id = batch.id

        # Create a Run for each (prompt, model) pair.
        for pid in body.prompt_ids:
            prompt = session.get(PromptDefinition, pid)
            for mt in body.models:
                pm = session.scalar(
                    select(ProviderModel).where(
                        ProviderModel.provider_id == mt.provider_id,
                        ProviderModel.external_id == mt.model_external_id,
                    )
                )
                session.add(
                    Run(
                        batch_id=batch_id,
                        provider_id=mt.provider_id,
                        provider_model_id=pm.id if pm else None,
                        prompt_id=pid,
                        status=RunStatus.PENDING,
                        model_identifier=mt.model_external_id,
                        model_name=mt.model_name or mt.model_external_id,
                        system_prompt=prompt.system_prompt or body.system_prompt,
                        user_prompt=prompt.user_prompt_template,
                        temperature=body.temperature or prompt.default_temperature,
                        max_tokens=body.max_tokens or prompt.default_max_tokens,
                        template_inputs={},
                    )
                )
        session.commit()

    # Execute the batch asynchronously.
    from llm_bencher.batch_runner import execute_batch
    await execute_batch(batch_id, session_factory, settings)

    # Return final state.
    with session_factory() as session:
        batch = session.get(BatchRun, batch_id)
        return JSONResponse(_batch_dict(batch), status_code=201)


@router.get("/batches")
def list_batches(request: Request) -> JSONResponse:
    """List all batches ordered by most recent first."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        batches = session.scalars(
            select(BatchRun).order_by(BatchRun.created_at.desc())
        ).all()
    return JSONResponse([_batch_dict(b) for b in batches])


@router.get("/batches/{batch_id}")
def get_batch(batch_id: int, request: Request) -> JSONResponse:
    """Return batch details with individual run summaries."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        batch = session.get(BatchRun, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found")

        runs = session.scalars(
            select(Run)
            .options(
                selectinload(Run.provider),
                selectinload(Run.prompt),
                selectinload(Run.result),
            )
            .where(Run.batch_id == batch_id)
            .order_by(Run.id)
        ).all()

        run_summaries = []
        for r in runs:
            run_summaries.append({
                "run_id": r.id,
                "provider_name": r.provider.name if r.provider else None,
                "model_identifier": r.model_identifier,
                "prompt_title": r.prompt.title if r.prompt else "ad-hoc",
                "status": r.status.value,
                "latency_ms": r.result.latency_ms if r.result else None,
                "total_tokens": r.result.total_tokens if r.result else None,
                "output_text": r.result.raw_output_text if r.result else None,
                "failure_message": r.failure_message,
            })

    return JSONResponse({**_batch_dict(batch), "runs": run_summaries})


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

class ComparisonCreateBody(BaseModel):
    run_ids: list[int]
    name: str | None = None


def _run_comparison_dict(run: Run) -> dict:
    return {
        "run_id": run.id,
        "provider_name": run.provider.name if run.provider else None,
        "model_identifier": run.model_identifier,
        "model_name": run.model_name,
        "prompt_title": run.prompt.title if run.prompt else "ad-hoc",
        "status": run.status.value,
        "output_text": run.result.raw_output_text if run.result else None,
        "latency_ms": run.result.latency_ms if run.result else None,
        "total_tokens": run.result.total_tokens if run.result else None,
        "failure_message": run.failure_message,
    }


@router.post("/comparisons")
def create_comparison(body: ComparisonCreateBody, request: Request) -> JSONResponse:
    """Create a comparison from selected run IDs."""
    if len(body.run_ids) < 2:
        raise HTTPException(status_code=422, detail="At least 2 runs required for comparison")

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        # Validate all run IDs.
        for rid in body.run_ids:
            run = session.get(Run, rid)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run {rid} not found")

        # Determine a common prompt_id if all runs share one.
        prompt_ids = set()
        for rid in body.run_ids:
            run = session.get(Run, rid)
            if run.prompt_id:
                prompt_ids.add(run.prompt_id)
        common_prompt_id = prompt_ids.pop() if len(prompt_ids) == 1 else None

        comparison = Comparison(
            name=body.name,
            prompt_id=common_prompt_id,
        )
        session.add(comparison)
        session.flush()

        for pos, rid in enumerate(body.run_ids):
            session.add(ComparisonItem(
                comparison_id=comparison.id,
                run_id=rid,
                position=pos,
            ))

        session.flush()
        comp_id = comparison.id
        session.commit()

    return JSONResponse({"id": comp_id}, status_code=201)


@router.post("/comparisons/from-batch/{batch_id}")
def create_comparisons_from_batch(batch_id: int, request: Request) -> JSONResponse:
    """Auto-create comparisons grouping batch runs by prompt."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        batch = session.get(BatchRun, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found")

        runs = session.scalars(
            select(Run).where(Run.batch_id == batch_id).order_by(Run.id)
        ).all()

        # Group runs by prompt_id.
        groups: dict[int | None, list[Run]] = {}
        for r in runs:
            groups.setdefault(r.prompt_id, []).append(r)

        created_ids: list[int] = []
        for prompt_id, group_runs in groups.items():
            if len(group_runs) < 2:
                continue
            comparison = Comparison(
                name=f"{batch.name} — prompt comparison",
                prompt_id=prompt_id,
                batch_id=batch_id,
            )
            session.add(comparison)
            session.flush()
            for pos, r in enumerate(group_runs):
                session.add(ComparisonItem(
                    comparison_id=comparison.id,
                    run_id=r.id,
                    position=pos,
                ))
            created_ids.append(comparison.id)
        session.commit()

    return JSONResponse({"comparison_ids": created_ids}, status_code=201)


@router.get("/comparisons")
def list_comparisons(request: Request) -> JSONResponse:
    """List all comparisons."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        comparisons = session.scalars(
            select(Comparison)
            .options(selectinload(Comparison.items))
            .order_by(Comparison.created_at.desc())
        ).all()
    return JSONResponse([
        {
            "id": c.id,
            "name": c.name,
            "prompt_id": c.prompt_id,
            "batch_id": c.batch_id,
            "run_count": len(c.items),
            "created_at": c.created_at.isoformat(),
        }
        for c in comparisons
    ])


@router.get("/comparisons/{comparison_id}")
def get_comparison(comparison_id: int, request: Request) -> JSONResponse:
    """Return comparison with full run data."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        comparison = session.scalar(
            select(Comparison)
            .options(
                selectinload(Comparison.items).selectinload(ComparisonItem.run).selectinload(Run.provider),
                selectinload(Comparison.items).selectinload(ComparisonItem.run).selectinload(Run.prompt),
                selectinload(Comparison.items).selectinload(ComparisonItem.run).selectinload(Run.result),
            )
            .where(Comparison.id == comparison_id)
        )
        if comparison is None:
            raise HTTPException(status_code=404, detail="Comparison not found")

        runs_data = [_run_comparison_dict(item.run) for item in comparison.items]

    return JSONResponse({
        "id": comparison.id,
        "name": comparison.name,
        "prompt_id": comparison.prompt_id,
        "batch_id": comparison.batch_id,
        "created_at": comparison.created_at.isoformat(),
        "runs": runs_data,
    })


@router.get("/providers/{provider_id}/models")
def get_provider_models(provider_id: int, request: Request) -> JSONResponse:
    """Return the stored model list for a provider (no live check)."""
    session_factory = request.app.state.session_factory

    with session_factory() as session:
        provider = session.get(Provider, provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        models = session.scalars(
            select(ProviderModel)
            .where(ProviderModel.provider_id == provider_id)
            .order_by(ProviderModel.display_name)
        ).all()

    return JSONResponse([_model_dict(m) for m in models])


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

import csv
import io


def _csv_response(rows: list[dict], filename: str) -> Response:
    """Build a CSV Response from a list of dicts (keys = headers)."""
    if not rows:
        return Response(
            content="",
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/history")
def export_history(request: Request) -> Response:
    """Export run history as CSV."""
    session_factory = request.app.state.session_factory
    params = request.query_params
    filter_provider_id = params.get("provider_id", "")
    filter_status = params.get("status", "")

    with session_factory() as session:
        q = (
            select(Run)
            .options(
                selectinload(Run.provider),
                selectinload(Run.prompt),
                selectinload(Run.result),
                selectinload(Run.rating),
            )
        )
        if filter_provider_id:
            q = q.where(Run.provider_id == int(filter_provider_id))
        if filter_status:
            q = q.where(Run.status == filter_status)
        q = q.order_by(Run.created_at.desc())

        runs = session.scalars(q).all()
        rows = []
        for r in runs:
            rows.append({
                "run_id": r.id,
                "date": r.created_at.isoformat() if r.created_at else "",
                "provider": r.provider.name if r.provider else "",
                "model": r.model_identifier,
                "prompt": r.prompt.title if r.prompt else "ad-hoc",
                "status": r.status.value,
                "latency_ms": r.result.latency_ms if r.result else "",
                "total_tokens": r.result.total_tokens if r.result else "",
                "rating": r.rating.score if r.rating else "",
                "output": r.result.raw_output_text if r.result else "",
            })

    return _csv_response(rows, "history.csv")


@router.get("/export/batch/{batch_id}")
def export_batch(batch_id: int, request: Request) -> Response:
    """Export batch results as CSV."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        batch = session.get(BatchRun, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found")

        runs = session.scalars(
            select(Run)
            .options(
                selectinload(Run.provider),
                selectinload(Run.prompt),
                selectinload(Run.result),
            )
            .where(Run.batch_id == batch_id)
            .order_by(Run.id)
        ).all()

        rows = []
        for r in runs:
            rows.append({
                "run_id": r.id,
                "provider": r.provider.name if r.provider else "",
                "model": r.model_identifier,
                "prompt": r.prompt.title if r.prompt else "ad-hoc",
                "status": r.status.value,
                "latency_ms": r.result.latency_ms if r.result else "",
                "total_tokens": r.result.total_tokens if r.result else "",
                "output": r.result.raw_output_text if r.result else "",
            })

    return _csv_response(rows, f"batch-{batch_id}.csv")


@router.get("/export/comparison/{comparison_id}")
def export_comparison(comparison_id: int, request: Request) -> Response:
    """Export comparison results as CSV."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        comparison = session.scalar(
            select(Comparison)
            .options(
                selectinload(Comparison.items).selectinload(ComparisonItem.run).selectinload(Run.provider),
                selectinload(Comparison.items).selectinload(ComparisonItem.run).selectinload(Run.prompt),
                selectinload(Comparison.items).selectinload(ComparisonItem.run).selectinload(Run.result),
            )
            .where(Comparison.id == comparison_id)
        )
        if comparison is None:
            raise HTTPException(status_code=404, detail="Comparison not found")

        rows = []
        for item in comparison.items:
            r = item.run
            rows.append({
                "position": item.position,
                "run_id": r.id,
                "provider": r.provider.name if r.provider else "",
                "model": r.model_identifier,
                "status": r.status.value,
                "latency_ms": r.result.latency_ms if r.result else "",
                "total_tokens": r.result.total_tokens if r.result else "",
                "output": r.result.raw_output_text if r.result else "",
            })

    return _csv_response(rows, f"comparison-{comparison_id}.csv")
