from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Sequence
import json, shutil
from pathlib import Path
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from neural_data_registry.config import Settings, get_settings
from neural_data_registry.db.models import Dataset, DatasetAlias, IngestionJob
from neural_data_registry.db.session import create_database, get_session_factory
from neural_data_registry.enums import DatasetStatus, JobStatus, Modality, Provider, StorageMode, normalize_modalities
from neural_data_registry.provider import download_from_url, provider_for_url
from neural_data_registry.storage import dataset_destination, directory_size, ensure_layout, ingestion_lock, move_into_managed_storage

def session(config: Settings | None = None) -> Session:
    """Open a session for the configured SQL database.

    Parameters
    ----------
    config : Settings or None, optional
        Registry configuration.

    Returns
    -------
    sqlalchemy.orm.Session
        Caller-owned database session.
    """
    config = config or get_settings()
    create_database(config)
    return get_session_factory(config.resolved_database_url)()
def find_datasets(db: Session, query: str | None = None, url: str | None = None, modality: str | None = None, provider: str | None = None) -> list[Dataset]:
    """Return registered datasets matching the supplied filters.

    Parameters
    ----------
    db : sqlalchemy.orm.Session
        Open registry session.
    query, url, modality, provider : str or None
        Optional dataset search filters.

    Returns
    -------
    list[Dataset]
        Matching rows ordered newest first.
    """
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if url: stmt = stmt.outerjoin(DatasetAlias).where(or_(Dataset.source_url == url, DatasetAlias.value == url))
    if query:
        needle = f"%{query}%"; stmt = stmt.outerjoin(DatasetAlias).where(or_(Dataset.name.ilike(needle), DatasetAlias.value.ilike(needle)))
    if modality: stmt = stmt.where(Dataset.modalities.ilike(f"%{modality}%"))
    if provider:
        provider_value = Provider(provider).name if isinstance(provider, str) else provider
        stmt = stmt.where(Dataset.provider == provider_value)
    return list(db.scalars(stmt.distinct()))

def resolve_dataset(db: Session, identifier: str | None = None, *, name: str | None = None, url: str | None = None) -> Dataset | None:
    """Resolve one dataset by its unique ID, name, or source URL."""
    values = [value for value in (identifier, name, url) if value]
    if len(values) != 1:
        raise ValueError("Provide exactly one dataset id, name, or URL")
    value = values[0]
    if identifier:
        conditions = [Dataset.id == value, func.lower(Dataset.name) == value.casefold(), Dataset.source_url == value]
    elif name:
        conditions = [func.lower(Dataset.name) == value.casefold()]
    else:
        conditions = [Dataset.source_url == value]
    matches = list(db.scalars(select(Dataset).where(or_(*conditions))))
    if len(matches) > 1:
        raise ValueError(f"Dataset identifier is ambiguous: {value}")
    return matches[0] if matches else None


def ingest_local(source: Path, name: str, provider: Provider, url: str | None, version: str | None, modalities: Sequence[str | Modality], config: Settings | None = None, storage_mode: StorageMode = StorageMode.REFERENCE) -> Dataset:
    """Register a local dataset as a new append-only registry row.

    Parameters
    ----------
    source : pathlib.Path
        Existing local dataset directory.
    name, provider, url, version, modalities : object
        Dataset identity and metadata.
    config : Settings or None, optional
        Registry configuration.
    storage_mode : StorageMode
        Reference the source or explicitly move it into managed storage.

    Returns
    -------
    Dataset
        The newly committed dataset row.
    """
    config = config or get_settings()
    version = version or "unknown"
    modalities = normalize_modalities(modalities)
    source = source.expanduser().resolve()
    # A local path is retained as the source reference when no remote URL exists;
    # that makes repeated ingestion of the same legacy directory detectable.
    source_reference = url or source.as_uri()
    with ingestion_lock(f"{provider.value}-{source_reference}-{version}", config):
        ensure_layout(config); create_database(config); db = get_session_factory(config.resolved_database_url)()
        try:
            existing = db.scalar(
                select(Dataset).where(
                    (Dataset.name.ilike(name)) | (Dataset.source_url == source_reference)
                )
            )
            if existing:
                reason = "name" if existing.name.casefold() == name.casefold() else "source URL/path"
                raise RuntimeError(
                    f"Ingestion rejected: dataset {reason} is already registered as "
                    f"{existing.id} ({existing.name} {existing.version}). "
                    f"Existing storage path: {existing.storage_path}"
                )
            if not source.is_dir(): raise ValueError(f"Local source is not a directory: {source}")
            item = Dataset(name=name, provider=provider, source_url=source_reference, version=version, modalities=",".join(sorted(set(modalities))), storage_mode=storage_mode, status=DatasetStatus.INGESTING); db.add(item); db.flush()
            job = IngestionJob(dataset_id=item.id, mode="local", status=JobStatus.RUNNING); db.add(job)
            db.add(DatasetAlias(dataset_id=item.id, kind="url" if url else "path", value=source_reference))
            db.add(DatasetAlias(dataset_id=item.id, kind="name", value=name))
            if storage_mode == StorageMode.REFERENCE:
                managed_path = source
            elif storage_mode == StorageMode.MOVE:
                managed_path = dataset_destination(item.id, name, version, config)
                move_into_managed_storage(source, managed_path)
            else:
                raise ValueError(f"Unsupported storage mode: {storage_mode}")
            item.storage_path = str(managed_path)
            item.size_bytes = directory_size(managed_path)
            item.status = DatasetStatus.AVAILABLE
            job.status = JobStatus.SUCCEEDED
            db.commit(); db.refresh(item); (config.registry_dir / f"{item.id}.json").write_text(json.dumps(dataset_dict(item), indent=2), encoding="utf-8"); return item
        except Exception:
            db.rollback(); raise
        finally: db.close()

def download(url: str, version: str, config: Settings | None = None) -> Dataset:
    """Download and register one provider dataset.

    Parameters
    ----------
    url : str
        Supported provider URL.
    version : str
        Provider version or branch.
    config : Settings or None, optional
        Registry configuration.

    Returns
    -------
    Dataset
        Newly registered row.
    """
    config = config or get_settings()
    provider = provider_for_url(url); source = config.staging_dir / f"download-{provider.value}-{version}"
    if source.exists(): raise RuntimeError(f"Staging directory already exists: {source}")
    ensure_layout(config)
    try:
        download_from_url(url, version, source); return ingest_local(source, url.rstrip("/").split("/")[-1], provider, url, version, [], config, StorageMode.MOVE)
    except Exception:
        if source.exists(): shutil.move(str(source), str(config.quarantine_dir / source.name))
        raise

def _dataset_output_columns(*, cli: bool = False):
    for column in Dataset.__table__.columns:
        if not column.info.get("serialize", True):
            continue
        if cli and column.info.get("cli_hidden", False):
            continue
        output_name = column.info.get("output_name", column.name)
        label = column.info.get(
            "label", output_name.replace("_", " ").title()
        )
        yield column, output_name, label


def dataset_output_fields(*, cli: bool = False) -> list[tuple[str, str]]:
    """Return current model fields and labels exposed to registry consumers."""
    return [
        (output_name, label)
        for _, output_name, label in _dataset_output_columns(cli=cli)
    ]


def _serialize_dataset_value(column, value):
    if column.name == "modalities":
        return [item for item in (value or "").split(",") if item]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def dataset_dict(item: Dataset) -> dict[str, object]:
    """Serialize current dataset fields without mutating the database.

    Parameters
    ----------
    item : Dataset
        Dataset row to serialize.

    Returns
    -------
    dict[str, object]
        Public current-model fields. Retired SQL columns remain stored but
        are intentionally not exposed.
    """
    return {
        output_name: _serialize_dataset_value(column, getattr(item, column.name))
        for column, output_name, _ in _dataset_output_columns()
    }
