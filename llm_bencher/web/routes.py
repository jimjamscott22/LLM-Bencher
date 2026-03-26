from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from llm_bencher.models import BatchRun, Comparison, ComparisonItem, PromptDefinition, Provider, ProviderKind, ProviderModel, PromptSuite, Run, RunRating, RunStatus


router = APIRouter()


def _render(request: Request, template_name: str, **context: object) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=context,
    )


def _summary_counts(session: Session) -> dict[str, int]:
    return {
        "providers": session.scalar(select(func.count()).select_from(Provider)) or 0,
        "prompts": session.scalar(select(func.count()).select_from(PromptDefinition)) or 0,
        "runs": session.scalar(select(func.count()).select_from(Run)) or 0,
        "batches": session.scalar(select(func.count()).select_from(BatchRun)) or 0,
        "comparisons": session.scalar(select(func.count()).select_from(Comparison)) or 0,
        "rated": session.scalar(select(func.count()).select_from(RunRating)) or 0,
    }


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        counts = _summary_counts(session)
        recent_runs = session.scalars(
            select(Run)
            .options(
                selectinload(Run.provider),
                selectinload(Run.result),
            )
            .order_by(Run.created_at.desc())
            .limit(5)
        ).all()
        providers = session.scalars(
            select(Provider).order_by(Provider.name)
        ).all()
    return _render(
        request, "home.html",
        counts=counts,
        recent_runs=recent_runs,
        providers=providers,
    )


@router.get("/health", response_class=JSONResponse)
def health(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    return JSONResponse(
        {
            "status": "ok",
            "app": settings.app_name,
            "environment": settings.environment,
        }
    )


@router.get("/providers", response_class=HTMLResponse)
def providers_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        providers = session.scalars(
            select(Provider)
            .options(selectinload(Provider.models))
            .order_by(Provider.name)
        ).all()
    return _render(
        request, "providers.html",
        providers=providers,
        provider_kinds=[k.value for k in ProviderKind],
    )


@router.get("/prompts", response_class=HTMLResponse)
def prompts_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    filter_tag = request.query_params.get("tag", "")

    with session_factory() as session:
        suites = session.scalars(
            select(PromptSuite)
            .options(selectinload(PromptSuite.prompts))
            .where(PromptSuite.is_active.is_(True))
            .order_by(PromptSuite.name)
        ).all()

        # Collect all distinct tags for the filter UI.
        all_tags: set[str] = set()
        for s in suites:
            for p in s.prompts:
                if p.tags:
                    all_tags.update(p.tags)

        # If a tag filter is active, narrow suites to only those containing
        # at least one prompt with that tag, and filter prompts within each suite.
        if filter_tag:
            filtered_suites = []
            for s in suites:
                matching = [p for p in s.prompts if filter_tag in (p.tags or [])]
                if matching:
                    # Attach filtered prompt list for the template.
                    s._filtered_prompts = matching
                    filtered_suites.append(s)
            suites = filtered_suites

    return _render(
        request, "prompts.html",
        suites=suites,
        all_tags=sorted(all_tags),
        filter_tag=filter_tag,
    )


@router.get("/runs/new", response_class=HTMLResponse)
def new_run_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        providers = session.scalars(select(Provider).order_by(Provider.name)).all()
        suites = session.scalars(
            select(PromptSuite)
            .options(selectinload(PromptSuite.prompts))
            .where(PromptSuite.is_active.is_(True))
            .order_by(PromptSuite.name)
        ).all()
    return _render(request, "runs.html", providers=providers, suites=suites)


@router.get("/runs/batch", response_class=HTMLResponse)
def batch_run_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        providers = session.scalars(
            select(Provider)
            .options(selectinload(Provider.models))
            .order_by(Provider.name)
        ).all()
        suites = session.scalars(
            select(PromptSuite)
            .options(selectinload(PromptSuite.prompts))
            .where(PromptSuite.is_active.is_(True))
            .order_by(PromptSuite.name)
        ).all()
    return _render(request, "batch_new.html", providers=providers, suites=suites)


@router.get("/batches/{batch_id}", response_class=HTMLResponse)
def batch_detail_page(request: Request, batch_id: int) -> HTMLResponse:
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
    return _render(request, "batch_detail.html", batch=batch, runs=runs)


@router.get("/compare/{comparison_id}", response_class=HTMLResponse)
def comparison_page(request: Request, comparison_id: int) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        comparison = session.scalar(
            select(Comparison)
            .options(
                selectinload(Comparison.items)
                .selectinload(ComparisonItem.run)
                .selectinload(Run.provider),
                selectinload(Comparison.items)
                .selectinload(ComparisonItem.run)
                .selectinload(Run.prompt),
                selectinload(Comparison.items)
                .selectinload(ComparisonItem.run)
                .selectinload(Run.result),
            )
            .where(Comparison.id == comparison_id)
        )
        if comparison is None:
            raise HTTPException(status_code=404, detail="Comparison not found")
    return _render(request, "compare.html", comparison=comparison)


_PER_PAGE = 25


@router.get("/history", response_class=HTMLResponse)
def history_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    params = request.query_params

    filter_provider_id = params.get("provider_id", "")
    filter_model_id = params.get("model_id", "")
    filter_status = params.get("status", "")
    filter_tag = params.get("tag", "")
    filter_date_from = params.get("date_from", "")
    filter_date_to = params.get("date_to", "")
    try:
        page = max(1, int(params.get("page", "1")))
    except ValueError:
        page = 1

    with session_factory() as session:
        # Base filter query (no eager loads — used for count).
        base_q = select(Run)
        if filter_provider_id:
            base_q = base_q.where(Run.provider_id == int(filter_provider_id))
        if filter_model_id:
            base_q = base_q.where(Run.provider_model_id == int(filter_model_id))
        if filter_status:
            base_q = base_q.where(Run.status == filter_status)
        if filter_tag:
            base_q = base_q.join(PromptDefinition, Run.prompt_id == PromptDefinition.id)
            base_q = base_q.where(PromptDefinition.tags.contains(filter_tag))
        if filter_date_from:
            base_q = base_q.where(Run.created_at >= filter_date_from)
        if filter_date_to:
            base_q = base_q.where(Run.created_at <= filter_date_to + "T23:59:59")

        total = session.scalar(
            select(func.count()).select_from(base_q.subquery())
        ) or 0

        runs = session.scalars(
            base_q
            .options(
                selectinload(Run.provider),
                selectinload(Run.provider_model),
                selectinload(Run.prompt),
                selectinload(Run.result),
                selectinload(Run.rating),
            )
            .order_by(Run.created_at.desc())
            .offset((page - 1) * _PER_PAGE)
            .limit(_PER_PAGE)
        ).all()

        providers = session.scalars(select(Provider).order_by(Provider.name)).all()

        # Models for the model filter dropdown (only those with runs).
        models = session.scalars(
            select(ProviderModel).order_by(ProviderModel.display_name)
        ).all()

        # Collect distinct tags for the tag filter dropdown.
        all_prompts = session.scalars(
            select(PromptDefinition)
            .join(PromptSuite)
            .where(PromptSuite.is_active.is_(True))
        ).all()
        all_tags: set[str] = set()
        for p in all_prompts:
            if p.tags:
                all_tags.update(p.tags)

    total_pages = max(1, (total + _PER_PAGE - 1) // _PER_PAGE)
    filters = {
        "provider_id": filter_provider_id,
        "model_id": filter_model_id,
        "status": filter_status,
        "tag": filter_tag,
        "date_from": filter_date_from,
        "date_to": filter_date_to,
    }
    return _render(
        request,
        "history.html",
        runs=runs,
        providers=providers,
        models=models,
        statuses=[s.value for s in RunStatus],
        all_tags=sorted(all_tags),
        filters=filters,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(request: Request, run_id: int) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    back_params = request.query_params.get("back", "")

    with session_factory() as session:
        run = session.scalar(
            select(Run)
            .options(
                selectinload(Run.provider),
                selectinload(Run.provider_model),
                selectinload(Run.prompt).selectinload(PromptDefinition.suite),
                selectinload(Run.result),
                selectinload(Run.rating),
            )
            .where(Run.id == run_id)
        )
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

    back_url = f"/history?{back_params}" if back_params else "/history"
    return _render(request, "run_detail.html", run=run, back_url=back_url)


@router.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request) -> HTMLResponse:
    return _render(request, "analytics.html")
