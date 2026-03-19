from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


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


def initialize_database(database_url: str, *, echo: bool = False) -> None:
    from llm_bencher import models  # noqa: F401

    engine = get_engine(database_url, echo=echo)
    Base.metadata.create_all(bind=engine)


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
