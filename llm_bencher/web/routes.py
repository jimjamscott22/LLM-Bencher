from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from llm_bencher.models import PromptDefinition, Provider, ProviderModel, Run


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
    }


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        counts = _summary_counts(session)
    return _render(request, "home.html", counts=counts)


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
    return _render(request, "providers.html", providers=providers)


@router.get("/prompts", response_class=HTMLResponse)
def prompts_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        prompt_count = session.scalar(select(func.count()).select_from(PromptDefinition)) or 0
    return _render(request, "prompts.html", prompt_count=prompt_count)


@router.get("/runs/new", response_class=HTMLResponse)
def new_run_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        providers = session.scalars(select(Provider).order_by(Provider.name)).all()
        prompt_count = session.scalar(select(func.count()).select_from(PromptDefinition)) or 0
    return _render(
        request,
        "runs.html",
        providers=providers,
        prompt_count=prompt_count,
    )


@router.get("/history", response_class=HTMLResponse)
def history_page(request: Request) -> HTMLResponse:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        recent_runs = session.scalars(
            select(Run).order_by(Run.created_at.desc()).limit(10)
        ).all()
    return _render(request, "history.html", recent_runs=recent_runs)
