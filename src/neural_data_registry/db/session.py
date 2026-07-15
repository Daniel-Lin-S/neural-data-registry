from __future__ import annotations
from functools import lru_cache
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateColumn
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
    engine = factory.kw["bind"]
    _reconcile_sqlite_schema(engine)
    Base.metadata.create_all(engine)


def _reconcile_sqlite_schema(engine: Engine) -> None:
    """Add every model column missing from an existing SQLite registry."""
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    preparer = engine.dialect.identifier_preparer

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue

            existing_columns = {
                column["name"] for column in inspector.get_columns(table.name)
            }
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                definition = str(CreateColumn(column).compile(dialect=engine.dialect))
                connection.exec_driver_sql(
                    f"ALTER TABLE {preparer.quote(table.name)} ADD COLUMN {definition}"
                )
