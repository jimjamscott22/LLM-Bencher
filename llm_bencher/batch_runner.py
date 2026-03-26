from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from llm_bencher.config import Settings
from llm_bencher.models import (
    BatchRun,
    BatchStatus,
    Run,
    RunResult as RunResultModel,
    RunStatus,
)
from llm_bencher.providers.registry import get_adapter
from llm_bencher.runner import build_run_request, execute_adapter


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_MAX_CONCURRENCY = 3


async def execute_batch(
    batch_id: int,
    session_factory,
    settings: Settings,
) -> None:
    """Execute all PENDING runs in a batch with bounded concurrency.

    Each run follows the same 3-phase pattern as single runs:
    read → async IO → write.
    """
    # Phase 1: load pending runs and build adapters.
    with session_factory() as session:
        batch = session.get(BatchRun, batch_id)
        if batch is None:
            return
        batch.status = BatchStatus.RUNNING
        batch.started_at = _utc_now()
        session.commit()

    with session_factory() as session:
        pending_runs = (
            session.query(Run)
            .filter(Run.batch_id == batch_id, Run.status == RunStatus.PENDING)
            .all()
        )
        # Build (run_id, adapter, run_request) tuples while session is open.
        tasks: list[tuple[int, object, object]] = []
        for run in pending_runs:
            provider = run.provider
            adapter = get_adapter(provider, settings)
            run_request = build_run_request(run)
            tasks.append((run.id, adapter, run_request))

    # Phase 2: execute concurrently.
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def _run_one(run_id: int, adapter, run_request):
        async with sem:
            return run_id, await execute_adapter(adapter, run_request)

    results = await asyncio.gather(
        *[_run_one(rid, a, rr) for rid, a, rr in tasks],
        return_exceptions=True,
    )

    # Phase 3: persist outcomes.
    completed = 0
    failed = 0
    with session_factory() as session:
        for item in results:
            if isinstance(item, Exception):
                failed += 1
                continue
            run_id, (result_schema, failure_message, started_at, completed_at) = item
            run = session.get(Run, run_id)
            if run is None:
                continue
            status = RunStatus.SUCCEEDED if result_schema else RunStatus.FAILED
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
                completed += 1
            else:
                failed += 1

        batch = session.get(BatchRun, batch_id)
        if batch:
            batch.completed_runs = completed
            batch.failed_runs = failed
            batch.completed_at = _utc_now()
            if failed == 0:
                batch.status = BatchStatus.COMPLETED
            elif completed == 0:
                batch.status = BatchStatus.FAILED
            else:
                batch.status = BatchStatus.PARTIAL
        session.commit()
