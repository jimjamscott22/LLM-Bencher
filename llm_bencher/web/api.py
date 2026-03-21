from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from llm_bencher.models import Provider, ProviderModel
from llm_bencher.providers.registry import get_adapter


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
