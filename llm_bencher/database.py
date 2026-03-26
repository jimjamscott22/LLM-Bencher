from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.sql.schema import Column


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


_engines: dict[tuple[str, bool], Engine] = {}
_sessionmakers: dict[tuple[str, bool], sessionmaker[Session]] = {}


def get_engine(database_url: str, *, echo: bool = False) -> Engine:
    cache_key = (database_url, echo)
    if cache_key not in _engines:
        connect_args: dict[str, Any] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        _engines[cache_key] = create_engine(
            database_url,
            connect_args=connect_args,
            echo=echo,
        )
    return _engines[cache_key]


def get_session_factory(database_url: str, *, echo: bool = False) -> sessionmaker[Session]:
    cache_key = (database_url, echo)
    if cache_key not in _sessionmakers:
        _sessionmakers[cache_key] = sessionmaker(
            bind=get_engine(database_url, echo=echo),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _sessionmakers[cache_key]


def _render_sqlite_default(column: Column[Any]) -> str | None:
    default = column.default
    if default is None or not default.is_scalar:
        return None

    value = default.arg
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return None


def _render_sqlite_add_column(column: Column[Any], engine: Engine) -> str:
    parts = [
        f'ALTER TABLE "{column.table.name}" ADD COLUMN "{column.name}"',
        column.type.compile(dialect=engine.dialect),
    ]
    if not column.nullable:
        parts.append("NOT NULL")

    default_sql = _render_sqlite_default(column)
    if default_sql is not None:
        parts.append(f"DEFAULT {default_sql}")
    elif not column.nullable:
        raise RuntimeError(
            f"Cannot auto-migrate missing non-null column {column.table.name}.{column.name} without a scalar default"
        )

    return " ".join(parts)


def upgrade_sqlite_schema(database_url: str, *, echo: bool = False) -> None:
    engine = get_engine(database_url, echo=echo)
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as connection:
        inspector = inspect(connection)
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue

            existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
            missing_columns = [
                column for column in table.columns if column.name not in existing_columns
            ]
            for column in missing_columns:
                connection.execute(text(_render_sqlite_add_column(column, engine)))
                inspector = inspect(connection)


def initialize_database(database_url: str, *, echo: bool = False) -> None:
    from llm_bencher import models  # noqa: F401

    engine = get_engine(database_url, echo=echo)
    Base.metadata.create_all(bind=engine)
    upgrade_sqlite_schema(database_url, echo=echo)


@contextmanager
def session_scope(database_url: str, *, echo: bool = False) -> Iterator[Session]:
    session_factory = get_session_factory(database_url, echo=echo)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
