from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_bencher.models import PromptDefinition, PromptImportRecord, PromptSuite
from llm_bencher.schemas import PromptRecord, PromptSuiteFile, PromptVariable


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_checksum(content: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(content.encode()).hexdigest()


def load_suite_file(path: Path) -> PromptSuiteFile:
    """Parse a JSON file into a PromptSuiteFile. Raises ValueError on bad input."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    return load_suite_from_string(content)


def load_suite_from_string(content: str) -> PromptSuiteFile:
    """Parse a JSON string into a PromptSuiteFile. Raises ValueError on bad input."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    try:
        return PromptSuiteFile.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Invalid suite schema: {exc}") from exc


def import_suite(
    session: Session,
    suite_file: PromptSuiteFile,
    source_path: str,
    checksum: str,
) -> tuple[PromptSuite, str]:
    """
    Upsert a PromptSuite and its PromptDefinition children.

    Matches suites by slug. Within a suite, matches prompts by slug.
    Returns (suite, action) where action is "created" or "updated".
    """
    existing_suite = session.scalar(
        select(PromptSuite).where(PromptSuite.slug == suite_file.slug)
    )

    if existing_suite:
        action = "updated"
        suite = existing_suite
        suite.name = suite_file.name
        suite.description = suite_file.description
        suite.version = suite_file.version
        suite.source_path = source_path
        suite.checksum = checksum
        suite.is_active = True
        suite.imported_at = _utc_now()
    else:
        action = "created"
        suite = PromptSuite(
            slug=suite_file.slug,
            name=suite_file.name,
            description=suite_file.description,
            version=suite_file.version,
            source_path=source_path,
            checksum=checksum,
            is_active=True,
            imported_at=_utc_now(),
        )
        session.add(suite)
        session.flush()  # populate suite.id

    # Load existing prompts for this suite keyed by slug.
    existing_prompts: dict[str, PromptDefinition] = {
        p.slug: p
        for p in session.scalars(
            select(PromptDefinition).where(PromptDefinition.suite_id == suite.id)
        ).all()
    }

    for rec in suite_file.prompts:
        if rec.slug in existing_prompts:
            _apply_prompt_fields(existing_prompts[rec.slug], rec)
        else:
            session.add(PromptDefinition(suite_id=suite.id, **_prompt_kwargs(rec)))

    session.add(
        PromptImportRecord(
            suite_id=suite.id,
            source_path=source_path,
            checksum=checksum,
            action=action,
            status="success",
            message=f"Imported {len(suite_file.prompts)} prompt(s)",
        )
    )

    return suite, action


def export_suite(suite: PromptSuite) -> PromptSuiteFile:
    """Convert a PromptSuite ORM row (prompts must be loaded) to PromptSuiteFile."""
    prompts = [
        PromptRecord(
            slug=p.slug,
            title=p.title,
            category=p.category,
            description=p.description,
            system_prompt=p.system_prompt,
            user_prompt_template=p.user_prompt_template,
            tags=p.tags or [],
            variables=[PromptVariable(**v) for v in (p.variables or [])],
            default_temperature=p.default_temperature,
            default_max_tokens=p.default_max_tokens,
        )
        for p in suite.prompts
    ]
    return PromptSuiteFile(
        slug=suite.slug,
        name=suite.name,
        description=suite.description,
        version=suite.version,
        prompts=prompts,
    )


def export_suite_to_json(suite_file: PromptSuiteFile) -> str:
    """Serialize a PromptSuiteFile to an indented JSON string."""
    return suite_file.model_dump_json(indent=2, exclude_none=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prompt_kwargs(rec: PromptRecord) -> dict:
    return {
        "slug": rec.slug,
        "title": rec.title,
        "category": rec.category,
        "description": rec.description,
        "system_prompt": rec.system_prompt,
        "user_prompt_template": rec.user_prompt_template,
        "tags": rec.tags,
        "variables": [v.model_dump() for v in rec.variables],
        "default_temperature": rec.default_temperature,
        "default_max_tokens": rec.default_max_tokens,
    }


def _apply_prompt_fields(p: PromptDefinition, rec: PromptRecord) -> None:
    p.title = rec.title
    p.category = rec.category
    p.description = rec.description
    p.system_prompt = rec.system_prompt
    p.user_prompt_template = rec.user_prompt_template
    p.tags = rec.tags
    p.variables = [v.model_dump() for v in rec.variables]
    p.default_temperature = rec.default_temperature
    p.default_max_tokens = rec.default_max_tokens
