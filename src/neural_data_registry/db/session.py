from __future__ import annotations
from functools import lru_cache
from sqlalchemy import Column, Engine, MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateColumn, CreateTable
from neural_data_registry.config import Settings, get_settings
from .models import Base

@lru_cache(maxsize=8)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a cached SQLAlchemy session factory for a database URL."""
    url = database_url or get_settings().resolved_database_url
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    return sessionmaker(engine, expire_on_commit=False)

def create_database(config: Settings | None = None) -> None:
    """Create or reconcile the registry schema without dropping stored fields.

    Parameters
    ----------
    config : Settings or None, optional
        Registry configuration.
    """
    config = config or get_settings()
    config.registry_dir.mkdir(parents=True, exist_ok=True)
    factory = get_session_factory(config.resolved_database_url)
    engine = factory.kw["bind"]
    _reconcile_sqlite_schema(engine)
    Base.metadata.create_all(engine)


def _reconcile_sqlite_schema(engine: Engine) -> None:
    """Reconcile SQLite tables while preserving retired columns and their data."""
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as connection:
        foreign_keys_enabled = bool(
            connection.exec_driver_sql("PRAGMA foreign_keys").scalar()
        )
        connection.commit()
        if foreign_keys_enabled:
            connection.exec_driver_sql("PRAGMA foreign_keys = OFF")
            connection.commit()

        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                inspector = inspect(connection)
                existing_tables = set(inspector.get_table_names())
                for table in Base.metadata.sorted_tables:
                    if table.name in existing_tables:
                        _reconcile_sqlite_table(connection, inspector, table)
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()
        finally:
            if foreign_keys_enabled:
                connection.exec_driver_sql("PRAGMA foreign_keys = ON")
                connection.commit()


def _reconcile_sqlite_table(connection: Connection, inspector, table) -> None:
    existing_columns = inspector.get_columns(table.name)
    existing_names = {column["name"] for column in existing_columns}
    current_names = set(table.columns.keys())
    missing_columns = [
        column for column in table.columns if column.name not in existing_names
    ]
    retired_columns = [
        column for column in existing_columns if column["name"] not in current_names
    ]
    retired_columns_to_relax = [
        column
        for column in retired_columns
        if not column["nullable"] or column["default"] is not None
    ]
    missing_columns_requiring_rebuild = [
        column
        for column in missing_columns
        if not column.nullable and column.server_default is None
    ]
    unfillable_columns = [
        column
        for column in missing_columns_requiring_rebuild
        if column.default is None
    ]

    if unfillable_columns and _table_has_rows(connection, table.name):
        names = ", ".join(column.name for column in unfillable_columns)
        raise RuntimeError(
            f"Cannot upgrade non-empty table {table.name!r}: new required columns "
            f"{names} need a default or must be nullable"
        )

    if retired_columns_to_relax or missing_columns_requiring_rebuild:
        _rebuild_sqlite_table(connection, table, existing_columns)
        return

    preparer = connection.dialect.identifier_preparer
    for column in missing_columns:
        definition = str(CreateColumn(column).compile(dialect=connection.dialect))
        connection.exec_driver_sql(
            f"ALTER TABLE {preparer.quote(table.name)} ADD COLUMN {definition}"
        )


def _table_has_rows(connection: Connection, table_name: str) -> bool:
    quoted_name = connection.dialect.identifier_preparer.quote(table_name)
    return (
        connection.exec_driver_sql(f"SELECT 1 FROM {quoted_name} LIMIT 1").first()
        is not None
    )


def _rebuild_sqlite_table(connection: Connection, table, existing_columns) -> None:
    """Rebuild SQLite while preserving every existing column and value.

    SQLite has limited ALTER TABLE support. The compatibility rebuild copies
    all reflected columns, including columns no longer represented by the ORM,
    so reconciliation never silently removes existing metadata.
    """
    metadata = MetaData()
    for foreign_key in table.foreign_keys:
        foreign_key.column.table.to_metadata(metadata)

    temporary_name = f"__ndr_compat_{table.name}"
    replacement = table.to_metadata(metadata, name=temporary_name)
    for reflected in existing_columns:
        if reflected["name"] not in replacement.columns:
            replacement.append_column(
                Column(reflected["name"], reflected["type"], nullable=True)
            )

    preparer = connection.dialect.identifier_preparer
    quoted_table = preparer.quote(table.name)
    quoted_temporary = preparer.quote(temporary_name)
    copied_names = [column["name"] for column in existing_columns]
    source = Table(
        table.name,
        MetaData(),
        *[
            Column(column["name"], column["type"])
            for column in existing_columns
        ],
    )

    connection.exec_driver_sql(f"DROP TABLE IF EXISTS {quoted_temporary}")
    connection.execute(CreateTable(replacement))
    if copied_names:
        connection.execute(
            replacement.insert().from_select(
                copied_names,
                select(*(source.columns[name] for name in copied_names)),
            )
        )
    connection.exec_driver_sql(f"DROP TABLE {quoted_table}")
    connection.exec_driver_sql(
        f"ALTER TABLE {quoted_temporary} RENAME TO {quoted_table}"
    )
