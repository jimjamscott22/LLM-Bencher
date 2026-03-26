from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return Path(raw_value).expanduser().resolve()


@dataclass(slots=True)
class Settings:
    app_name: str = "LLM Bencher"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    sqlite_echo: bool = False
    provider_timeout_seconds: float = 30.0
    data_dir: Path = PROJECT_ROOT / "data"
    database_path: Path = PROJECT_ROOT / "data" / "llm_bencher.db"
    prompt_library_dir: Path = PROJECT_ROOT / "data" / "prompt_suites"
    templates_dir: Path = PACKAGE_ROOT / "templates"
    static_dir: Path = PACKAGE_ROOT / "static"
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("LLM_BENCHER_DATA_DIR", PROJECT_ROOT / "data")
        return cls(
            app_name=os.getenv("LLM_BENCHER_APP_NAME", "LLM Bencher"),
            environment=os.getenv("LLM_BENCHER_ENV", "development"),
            host=os.getenv("LLM_BENCHER_HOST", "127.0.0.1"),
            port=int(os.getenv("LLM_BENCHER_PORT", "8099")),
            sqlite_echo=_env_flag("LLM_BENCHER_SQLITE_ECHO", False),
            provider_timeout_seconds=_env_float("LLM_BENCHER_PROVIDER_TIMEOUT", 30.0),
            data_dir=data_dir,
            database_path=_env_path(
                "LLM_BENCHER_DB_PATH",
                data_dir / "llm_bencher.db",
            ),
            prompt_library_dir=_env_path(
                "LLM_BENCHER_PROMPT_LIBRARY_DIR",
                data_dir / "prompt_suites",
            ),
            lm_studio_base_url=os.getenv(
                "LLM_BENCHER_LM_STUDIO_URL",
                "http://127.0.0.1:1234/v1",
            ),
            ollama_base_url=os.getenv(
                "LLM_BENCHER_OLLAMA_URL",
                "http://127.0.0.1:11434",
            ),
            openai_api_key=os.getenv("LLM_BENCHER_OPENAI_API_KEY", ""),
            openai_base_url=os.getenv(
                "LLM_BENCHER_OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ),
        )

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path.as_posix()}"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_library_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
