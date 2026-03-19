from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_bencher.app import create_app
from llm_bencher.config import Settings


TMP_ROOT = Path(__file__).resolve().parent.parent / ".test-tmp"


@pytest.fixture()
def test_settings() -> Iterator[Settings]:
    case_root = TMP_ROOT / uuid4().hex
    data_dir = case_root / "data"
    settings = Settings(
        data_dir=data_dir,
        database_path=data_dir / "test.db",
        prompt_library_dir=data_dir / "prompt_suites",
        sqlite_echo=False,
    )
    yield settings
    shutil.rmtree(case_root, ignore_errors=True)


@pytest.fixture()
def client(test_settings: Settings) -> Iterator[TestClient]:
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client
