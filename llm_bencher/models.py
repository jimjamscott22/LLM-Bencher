from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from llm_bencher.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderKind(StrEnum):
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPAT = "openai_compat"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BatchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class Provider(TimestampMixin, Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    kind: Mapped[ProviderKind] = mapped_column(Enum(ProviderKind), nullable=False)
    base_url: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key: Mapped[str | None] = mapped_column(String(255))
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_connected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_health_check_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)

    models: Mapped[list["ProviderModel"]] = relationship(
        back_populates="provider",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list["Run"]] = relationship(back_populates="provider")


class ProviderModel(TimestampMixin, Base):
    __tablename__ = "provider_models"
    __table_args__ = (
        UniqueConstraint("provider_id", "external_id", name="uq_provider_models_provider_external"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    provider: Mapped["Provider"] = relationship(back_populates="models")
    runs: Mapped[list["Run"]] = relationship(back_populates="provider_model")


class PromptSuite(TimestampMixin, Base):
    __tablename__ = "prompt_suites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[str | None] = mapped_column(String(50))
    source_path: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str | None] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    prompts: Mapped[list["PromptDefinition"]] = relationship(
        back_populates="suite",
        cascade="all, delete-orphan",
    )
    import_records: Mapped[list["PromptImportRecord"]] = relationship(
        back_populates="suite",
        cascade="all, delete-orphan",
    )


class PromptDefinition(TimestampMixin, Base):
    __tablename__ = "prompt_definitions"
    __table_args__ = (
        UniqueConstraint("suite_id", "slug", name="uq_prompt_definitions_suite_slug"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("prompt_suites.id"), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    user_prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    variables: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    default_temperature: Mapped[float | None] = mapped_column(Float)
    default_max_tokens: Mapped[int | None] = mapped_column(Integer)

    suite: Mapped["PromptSuite"] = relationship(back_populates="prompts")
    runs: Mapped[list["Run"]] = relationship(back_populates="prompt")


class PromptImportRecord(Base):
    __tablename__ = "prompt_import_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite_id: Mapped[int | None] = mapped_column(ForeignKey("prompt_suites.id"))
    source_path: Mapped[str] = mapped_column(String(500), nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    suite: Mapped["PromptSuite"] = relationship(back_populates="import_records")


class BatchRun(TimestampMixin, Base):
    __tablename__ = "batch_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[BatchStatus] = mapped_column(
        Enum(BatchStatus),
        default=BatchStatus.PENDING,
        nullable=False,
    )
    total_runs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_runs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_runs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    runs: Mapped[list["Run"]] = relationship(back_populates="batch")


class Run(TimestampMixin, Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"), nullable=False)
    provider_model_id: Mapped[int | None] = mapped_column(ForeignKey("provider_models.id"))
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompt_definitions.id"))
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("batch_runs.id"))
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus),
        default=RunStatus.PENDING,
        nullable=False,
    )
    model_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(255))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    template_inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    temperature: Mapped[float | None] = mapped_column(Float)
    max_tokens: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_message: Mapped[str | None] = mapped_column(Text)

    provider: Mapped["Provider"] = relationship(back_populates="runs")
    provider_model: Mapped["ProviderModel"] = relationship(back_populates="runs")
    prompt: Mapped["PromptDefinition"] = relationship(back_populates="runs")
    batch: Mapped["BatchRun | None"] = relationship(back_populates="runs")
    result: Mapped["RunResult"] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
    )
    rating: Mapped["RunRating | None"] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
    )


class RunResult(Base):
    __tablename__ = "run_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), unique=True, nullable=False)
    raw_output_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    run: Mapped["Run"] = relationship(back_populates="result")


class RunRating(TimestampMixin, Base):
    __tablename__ = "run_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), unique=True, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    run: Mapped["Run"] = relationship(back_populates="rating")


class Comparison(TimestampMixin, Base):
    __tablename__ = "comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompt_definitions.id"))
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("batch_runs.id"))

    prompt: Mapped["PromptDefinition | None"] = relationship()
    batch: Mapped["BatchRun | None"] = relationship()
    items: Mapped[list["ComparisonItem"]] = relationship(
        back_populates="comparison",
        cascade="all, delete-orphan",
        order_by="ComparisonItem.position",
    )


class ComparisonItem(Base):
    __tablename__ = "comparison_items"
    __table_args__ = (
        UniqueConstraint("comparison_id", "run_id", name="uq_comparison_items_comparison_run"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comparison_id: Mapped[int] = mapped_column(ForeignKey("comparisons.id"), nullable=False)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    comparison: Mapped["Comparison"] = relationship(back_populates="items")
    run: Mapped["Run"] = relationship()

