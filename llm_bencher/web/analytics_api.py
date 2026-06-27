from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import case, func, select

from llm_bencher.models import (
    BatchRun,
    Comparison,
    Provider,
    Run,
    RunRating,
    RunResult,
    RunStatus,
)


router = APIRouter(prefix="/api/analytics")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _period_start(period: str) -> datetime | None:
    """Convert a period string like '7d' or '30d' to a UTC start datetime."""
    now = _utc_now()
    if not period:
        return None
    if period.endswith("d"):
        try:
            days = int(period[:-1])
            return now - timedelta(days=days)
        except ValueError:
            return None
    return None


@router.get("/summary")
def analytics_summary(request: Request) -> JSONResponse:
    """Overall stats for dashboard header."""
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        # Run counts by status in a single grouped scan.
        status_counts = dict(
            session.execute(
                select(Run.status, func.count()).group_by(Run.status)
            ).all()
        )
        total_runs = sum(status_counts.values())
        succeeded = status_counts.get(RunStatus.SUCCEEDED, 0)
        failed = status_counts.get(RunStatus.FAILED, 0)

        # RunResult aggregates in one pass (avg/sum already skip NULLs).
        avg_latency, total_tokens = session.execute(
            select(func.avg(RunResult.latency_ms), func.sum(RunResult.total_tokens))
        ).one()
        total_tokens = total_tokens or 0

        # Standalone table counts via scalar subqueries in one round-trip.
        rated_runs, batch_count, comparison_count, provider_count = session.execute(
            select(
                select(func.count()).select_from(RunRating).scalar_subquery(),
                select(func.count()).select_from(BatchRun).scalar_subquery(),
                select(func.count()).select_from(Comparison).scalar_subquery(),
                select(func.count()).select_from(Provider).scalar_subquery(),
            )
        ).one()

    success_rate = (succeeded / total_runs * 100) if total_runs > 0 else 0.0

    return JSONResponse({
        "total_runs": total_runs,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": round(success_rate, 1),
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0,
        "total_tokens": total_tokens,
        "rated_runs": rated_runs,
        "batch_count": batch_count,
        "comparison_count": comparison_count,
        "provider_count": provider_count,
    })


@router.get("/latency")
def analytics_latency(request: Request) -> JSONResponse:
    """Average, min, max, p50 latency grouped by model."""
    params = request.query_params
    period = params.get("period", "")
    period_start = _period_start(period)

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        q = (
            select(
                Run.model_identifier.label("model"),
                func.avg(RunResult.latency_ms).label("avg"),
                func.min(RunResult.latency_ms).label("min"),
                func.max(RunResult.latency_ms).label("max"),
                func.count(RunResult.id).label("count"),
            )
            .join(RunResult, RunResult.run_id == Run.id)
            .where(Run.status == RunStatus.SUCCEEDED)
            .where(RunResult.latency_ms.isnot(None))
        )
        if period_start:
            q = q.where(Run.created_at >= period_start)
        q = q.group_by(Run.model_identifier)

        rows = session.execute(q).all()

    return JSONResponse([
        {
            "model": r.model,
            "avg_ms": round(r.avg, 1) if r.avg else 0,
            "min_ms": r.min,
            "max_ms": r.max,
            "count": r.count,
        }
        for r in rows
    ])


@router.get("/tokens")
def analytics_tokens(request: Request) -> JSONResponse:
    """Token usage grouped by model."""
    params = request.query_params
    period = params.get("period", "")
    period_start = _period_start(period)

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        q = (
            select(
                Run.model_identifier.label("model"),
                func.sum(RunResult.prompt_tokens).label("prompt_tokens"),
                func.sum(RunResult.completion_tokens).label("completion_tokens"),
                func.sum(RunResult.total_tokens).label("total_tokens"),
                func.count(RunResult.id).label("count"),
            )
            .join(RunResult, RunResult.run_id == Run.id)
            .where(Run.status == RunStatus.SUCCEEDED)
        )
        if period_start:
            q = q.where(Run.created_at >= period_start)
        q = q.group_by(Run.model_identifier)

        rows = session.execute(q).all()

    return JSONResponse([
        {
            "model": r.model,
            "prompt_tokens": r.prompt_tokens or 0,
            "completion_tokens": r.completion_tokens or 0,
            "total_tokens": r.total_tokens or 0,
            "count": r.count,
        }
        for r in rows
    ])


@router.get("/success-rate")
def analytics_success_rate(request: Request) -> JSONResponse:
    """Success/failure counts grouped by model."""
    params = request.query_params
    period = params.get("period", "")
    period_start = _period_start(period)

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        q = (
            select(
                Run.model_identifier.label("model"),
                func.count().label("total"),
                func.sum(case((Run.status == RunStatus.SUCCEEDED, 1), else_=0)).label("succeeded"),
                func.sum(case((Run.status == RunStatus.FAILED, 1), else_=0)).label("failed"),
            )
            .where(Run.status.in_([RunStatus.SUCCEEDED, RunStatus.FAILED]))
        )
        if period_start:
            q = q.where(Run.created_at >= period_start)
        q = q.group_by(Run.model_identifier)

        rows = session.execute(q).all()

    return JSONResponse([
        {
            "model": r.model,
            "total": r.total,
            "succeeded": r.succeeded,
            "failed": r.failed,
            "rate": round(r.succeeded / r.total * 100, 1) if r.total > 0 else 0,
        }
        for r in rows
    ])


@router.get("/timeline")
def analytics_timeline(request: Request) -> JSONResponse:
    """Per-day latency averages for trend line."""
    params = request.query_params
    period = params.get("period", "30d")
    period_start = _period_start(period)

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        # SQLite date function.
        day_label = func.strftime("%Y-%m-%d", Run.created_at)
        q = (
            select(
                day_label.label("day"),
                func.avg(RunResult.latency_ms).label("avg_latency"),
                func.count(RunResult.id).label("count"),
            )
            .join(RunResult, RunResult.run_id == Run.id)
            .where(Run.status == RunStatus.SUCCEEDED)
            .where(RunResult.latency_ms.isnot(None))
        )
        if period_start:
            q = q.where(Run.created_at >= period_start)
        q = q.group_by(day_label).order_by(day_label)

        rows = session.execute(q).all()

    return JSONResponse([
        {
            "day": r.day,
            "avg_latency_ms": round(r.avg_latency, 1) if r.avg_latency else 0,
            "count": r.count,
        }
        for r in rows
    ])
