from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from llm_bencher.models import (
    Provider,
    PromptDefinition,
    PromptSuite,
    ProviderModel,
    Run,
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
            template_inputs={},
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
