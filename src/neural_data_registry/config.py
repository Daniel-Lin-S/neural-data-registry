from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Deployment settings for the registry.

    Parameters
    ----------
    data_root : pathlib.Path
        Root directory containing managed storage and the registry database.
    database_url : str or None, optional
        SQLAlchemy URL; defaults to SQLite below ``data_root``.
    lock_timeout_seconds : int
        Timeout reserved for ingestion coordination.
    download_proxy : str or None, optional
        Proxy URL used for provider downloads. Existing standard proxy
        environment variables are inherited when this is omitted.
    download_mirror : str or None, optional
        Optional provider download mirror URL or URL template.
    """
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NDR_", extra="ignore")

    data_root: Path = Field(description="Root folder for managed dataset platform")
    database_url: str | None = Field(
        default=None,
        description="SQLAlchemy database URL. Defaults to sqlite under <data_root>/registry/registry.db",
    )
    lock_timeout_seconds: int = 1800
    download_proxy: str | None = Field(
        default=None,
        description="Proxy URL for provider downloads",
    )
    download_mirror: str | None = Field(
        default=None,
        description="Provider download mirror URL or template",
    )
    @property
    def datasets_dir(self) -> Path:
        return self.data_root / "datasets"

    @property
    def incoming_dir(self) -> Path:
        return self.data_root / "incoming"


    @property
    def quarantine_dir(self) -> Path:
        return self.data_root / "quarantine"

    @property
    def registry_dir(self) -> Path:
        return self.data_root / "registry"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def ingestion_lock_dir(self) -> Path:
        return self.registry_dir / "locks"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = (self.registry_dir / "registry.db").resolve()
        return f"sqlite:///{db_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load deployment settings lazily from NDR_* environment variables."""
    return Settings()
