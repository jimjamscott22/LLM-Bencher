from __future__ import annotations

from datetime import datetime, timezone

import httpx

from llm_bencher.models import Run
from llm_bencher.providers.base import ProviderAdapter
from llm_bencher.schemas import RunRequest
from llm_bencher.schemas import RunResult as RunResultSchema


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_timeout_label(seconds: float) -> str:
    if seconds == int(seconds):
        return str(int(seconds))
    return str(seconds)


def format_adapter_error(
    exc: Exception,
    *,
    timeout_seconds: float | None = None,
) -> str:
    """Turn adapter exceptions into user-facing failure messages."""
    if isinstance(exc, httpx.TimeoutException):
        if timeout_seconds is not None:
            label = _format_timeout_label(timeout_seconds)
            return f"Provider request timed out after {label}s"
        return "Provider request timed out"

    message = str(exc).strip()
    if message:
        return message
    return type(exc).__name__


def build_run_request(run: Run) -> RunRequest:
    """Map a Run ORM row to the RunRequest schema the adapter expects."""
    return RunRequest(
        provider_id=run.provider_id,
        model_id=run.model_identifier,
        model_name=run.model_name,
        prompt_id=run.prompt_id,
        system_prompt=run.system_prompt,
        user_prompt=run.user_prompt,
        template_inputs=run.template_inputs or {},
        temperature=run.temperature,
        max_tokens=run.max_tokens,
    )


async def execute_adapter(
    adapter: ProviderAdapter,
    run_request: RunRequest,
) -> tuple[RunResultSchema | None, str | None, datetime, datetime]:
    """
    Call adapter.run_chat() and return
    (result_schema, failure_message, started_at, completed_at).

    Never raises — adapter failures are captured as (None, message, ...).
    """
    started_at = _utc_now()
    try:
        result = await adapter.run_chat(run_request)
        return result, None, started_at, _utc_now()
    except Exception as exc:
        return (
            None,
            format_adapter_error(exc, timeout_seconds=adapter.timeout_seconds),
            started_at,
            _utc_now(),
        )
