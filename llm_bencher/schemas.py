from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProviderHealth(BaseModel):
    is_available: bool
    detail: str | None = None
    checked_at: datetime | None = None


class DiscoveredModel(BaseModel):
    id: str
    name: str
    provider_slug: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptVariable(BaseModel):
    name: str
    description: str | None = None
    required: bool = False
    default: str | None = None


class PromptRecord(BaseModel):
    slug: str
    title: str
    category: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    user_prompt_template: str
    variables: list[PromptVariable] = Field(default_factory=list)
    default_temperature: float | None = None
    default_max_tokens: int | None = None


class PromptSuiteFile(BaseModel):
    slug: str
    name: str
    description: str | None = None
    version: str | None = None
    prompts: list[PromptRecord] = Field(default_factory=list)


class RunRequest(BaseModel):
    provider_id: int
    model_id: str
    model_name: str | None = None
    prompt_id: int | None = None
    system_prompt: str | None = None
    user_prompt: str
    template_inputs: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None


class RunResult(BaseModel):
    output_text: str
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class RatingRequest(BaseModel):
    score: int = Field(ge=1, le=5)
    notes: str | None = None


class RatingResponse(BaseModel):
    run_id: int
    score: int
    notes: str | None = None
    created_at: datetime
