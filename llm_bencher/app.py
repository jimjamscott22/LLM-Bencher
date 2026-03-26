from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm_bencher.bootstrap import seed_default_providers
from llm_bencher.config import Settings, get_settings
from llm_bencher.database import get_session_factory, initialize_database, session_scope
from llm_bencher.web.api import router as api_router
from llm_bencher.web.analytics_api import router as analytics_router
from llm_bencher.web.routes import router as web_router


def _lifespan_factory(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_directories()
        initialize_database(settings.database_url, echo=settings.sqlite_echo)

        with session_scope(settings.database_url, echo=settings.sqlite_echo) as session:
            seed_default_providers(session, settings)

        yield

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=_lifespan_factory(settings),
    )

    templates = Jinja2Templates(directory=str(settings.templates_dir))
    templates.env.globals["app_name"] = settings.app_name

    app.state.settings = settings
    app.state.templates = templates
    app.state.session_factory = get_session_factory(
        settings.database_url,
        echo=settings.sqlite_echo,
    )

    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    app.include_router(api_router)
    app.include_router(analytics_router)
    app.include_router(web_router)
    return app


app = create_app()
