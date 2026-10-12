"""API / worker settings from environment variables (prefix ``FMI_``). See ``infra/.env.example``."""

from __future__ import annotations

from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - minimal installs
    from pydantic import BaseModel as BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FMI_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///storage/fmi.db"
    redis_url: str = "redis://localhost:6379/0"
    #: run Celery tasks inline (dev/tests without Redis)
    celery_eager: bool = False
    storage_root: str = "storage"
    s3_bucket: str | None = None
    s3_endpoint: str | None = None
    s3_prefix: str = "fmi"
    opendata_root: str = "data/opendata"
    #: optional static bearer token; empty = no auth (local dev)
    api_token: str | None = None
    cors_origins: str = "*"
    max_upload_gb: float = 20.0
    agent_model: str = "claude-opus-5-5"
    agent_effort: str = "high"
    #: "default" enables the server-side refusal fallback on the Claude API; "off" disables it
    agent_fallbacks: str = "default"
    pipeline_config: str | None = None   # YAML path overriding PipelineConfig


@lru_cache
def get_settings() -> Settings:
    return Settings()
