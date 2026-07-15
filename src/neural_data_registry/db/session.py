from __future__ import annotations
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from neural_data_registry.config import Settings, get_settings
from .models import Base

@lru_cache(maxsize=8)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    url = database_url or get_settings().resolved_database_url
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    return sessionmaker(engine, expire_on_commit=False)

def create_database(config: Settings | None = None) -> None:
    config = config or get_settings()
    config.registry_dir.mkdir(parents=True, exist_ok=True)
    factory = get_session_factory(config.resolved_database_url)
    Base.metadata.create_all(factory.kw["bind"])
