from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Sequence
import json, re
from pathlib import Path
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session
from urllib.parse import urlparse
from neural_data_registry.config import Settings, get_settings
from neural_data_registry.db.models import Dataset, DatasetAlias, IngestionJob
from neural_data_registry.db.session import create_database, get_session_factory
from neural_data_registry.enums import DatasetStatus, JobStatus, Modality, Provider, StorageMode, normalize_modalities
from neural_data_registry.provider import download_from_url, provider_for_url
from neural_data_registry.storage import copy_into_managed_storage, dataset_destination, directory_size, ensure_layout, ingestion_lock, move_into_managed_storage, safe_component
def _append_ingestion_log(path: Path, message: str) -> None:
    """Append one timestamped ingestion event to a persistent workspace log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp} {message}\n")



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


class DatasetConflictError(RuntimeError):
    """Raised when a dataset name or source identity is already registered."""




def _raise_if_dataset_identity_conflicts(
    db: Session, name: str, source_reference: str | None
) -> None:
    """Reject a registered name or canonical URL/path before dataset processing."""
    conditions = [
        func.lower(Dataset.name) == name.casefold(),
        and_(
            DatasetAlias.kind == "name",
            func.lower(DatasetAlias.value) == name.casefold(),
        ),
    ]
    if source_reference is not None:
        conditions.extend(
            [
                Dataset.source_url == source_reference,
                and_(
                    DatasetAlias.kind.in_(("url", "path")),
                    DatasetAlias.value == source_reference,
                ),
            ]
        )
    existing = db.scalar(
        select(Dataset)
        .outerjoin(DatasetAlias)
        .where(or_(*conditions))
        .order_by(Dataset.created_at.desc())
    )
    if existing is None:
        return
    reason = "name" if existing.name.casefold() == name.casefold() else "source URL/path"
    raise DatasetConflictError(
        f"Ingestion rejected: dataset {reason} is already registered as "
        f"{existing.id} ({existing.name} {existing.version}). "
        f"Existing storage path: {existing.storage_path}"
    )


def _preflight_dataset_identity(
    name: str, source_reference: str | None, config: Settings
) -> None:
    """Check registry identity before provider, workspace, or dataset-file work."""
    create_database(config)
    db = get_session_factory(config.resolved_database_url)()
    try:
        _raise_if_dataset_identity_conflicts(db, name, source_reference)
    finally:
        db.close()


def _ingest_local_locked(
    source: Path,
    name: str,
    provider: Provider,
    source_reference: str,
    version: str,
    modalities: Sequence[str | Modality],
    config: Settings,
    storage_mode: StorageMode,
    *,
    has_remote_url: bool,
) -> Dataset:
    """Ingest a local source while the registry-wide intake lock is held."""
    ensure_layout(config)
    create_database(config)
    db = get_session_factory(config.resolved_database_url)()
    try:
        _raise_if_dataset_identity_conflicts(db, name, source_reference)
        if not source.is_dir():
            raise ValueError(f"Local source is not a directory: {source}")
        item = Dataset(name=name, provider=provider, source_url=source_reference, version=version, modalities=",".join(sorted(set(modalities))), storage_mode=storage_mode, status=DatasetStatus.INGESTING); db.add(item); db.flush()
        job = IngestionJob(dataset_id=item.id, mode="local", status=JobStatus.RUNNING); db.add(job)
        db.add(DatasetAlias(dataset_id=item.id, kind="url" if has_remote_url else "path", value=source_reference))
        db.add(DatasetAlias(dataset_id=item.id, kind="name", value=name))
        if storage_mode == StorageMode.REFERENCE:
            managed_path = source
        elif storage_mode == StorageMode.MOVE:
            managed_path = dataset_destination(item.id, name, version, config)
            move_into_managed_storage(source, managed_path)
        elif storage_mode == StorageMode.COPY:
            managed_path = dataset_destination(item.id, name, version, config)
            copy_into_managed_storage(source, managed_path)
        else:
            raise ValueError(f"Unsupported storage mode: {storage_mode}")
        item.storage_path = str(managed_path)
        item.size_bytes = directory_size(managed_path)
        item.status = DatasetStatus.AVAILABLE
        job.status = JobStatus.SUCCEEDED
        db.commit(); db.refresh(item); (config.registry_dir / f"{item.id}.json").write_text(json.dumps(dataset_dict(item), indent=2), encoding="utf-8"); return item
    except Exception:
        db.rollback(); raise
    finally:
        db.close()


def ingest_local(source: Path, name: str, provider: Provider, url: str | None, version: str | None, modalities: Sequence[str | Modality], config: Settings | None = None, storage_mode: StorageMode = StorageMode.REFERENCE) -> Dataset:
    """Register a local dataset after identity preflight and under intake lock."""
    config = config or get_settings()
    name = name.strip()
    if not name:
        raise ValueError("A non-empty dataset name is required")
    version = version or "unknown"
    _preflight_dataset_identity(name, url, config)
    source = source.expanduser().resolve()
    source_reference = url or source.as_uri()
    if url is None:
        _preflight_dataset_identity(name, source_reference, config)
    normalized_modalities = normalize_modalities(modalities)
    with ingestion_lock("registry-intake", config):
        _preflight_dataset_identity(name, source_reference, config)
        ensure_layout(config)
        log_path = config.logs_dir / f"ingest-{safe_component(name)}-{safe_component(version)}.log"
        _append_ingestion_log(log_path, f"START provider={provider.value} version={version} source={source}")
        try:
            item = _ingest_local_locked(
                source, name, provider, source_reference, version, normalized_modalities,
                config, storage_mode, has_remote_url=url is not None,
            )
            _append_ingestion_log(log_path, f"SUCCEEDED dataset_id={item.id} storage_path={item.storage_path}")
            return item
        except Exception as exc:
            _append_ingestion_log(log_path, f"FAILED {type(exc).__name__}: {exc}")
            raise
def resolve_download_version(url: str, requested: str | None = None) -> str:
    """Resolve an explicit version or an OpenNeuro version embedded in the URL."""
    if requested is not None:
        version = requested.strip()
        if version:
            return version
        raise ValueError("Version cannot be blank")
    provider = provider_for_url(url)
    if provider is Provider.OPENNEURO:
        match = re.search(r"/versions/([0-9]+(?:\.[0-9]+)*)", urlparse(url).path)
        if match:
            return match.group(1)
    raise ValueError(
        "A version is required when it is not present in the dataset URL; "
        "provide --version or the API version field"
    )



def download(
    url: str,
    version: str | None,
    config: Settings | None = None,
    *,
    name: str,
    modalities: Sequence[str | Modality],
    proxy: str | None = None,
    mirror: str | None = None,
) -> Dataset:
    """Download and register one dataset after identity preflight."""
    name = name.strip()
    if not name:
        raise ValueError("A non-empty dataset name is required for downloads")
    config = config or get_settings()
    _preflight_dataset_identity(name, url, config)
    normalized_modalities = normalize_modalities(modalities)
    if not normalized_modalities:
        raise ValueError("At least one modality is required for downloads")
    provider = provider_for_url(url)
    resolved_version = resolve_download_version(url, version)
    with ingestion_lock("registry-intake", config):
        # Repeat the preflight while serialized with all other intake actions.
        _preflight_dataset_identity(name, url, config)
        source = config.incoming_dir / (
            f"download-{provider.value}-{safe_component(url.rstrip(chr(47)).split(chr(47))[-1])}-"
            f"{safe_component(resolved_version)}"
        )
        ensure_layout(config)
        log_path = config.logs_dir / f"{source.name}.log"
        was_present = source.exists()
        resumable = (source / ".git").exists()
        source.mkdir(parents=True, exist_ok=True)
        action = "resume" if resumable else ("retry" if was_present else "new")
        _append_ingestion_log(log_path, f"START provider={provider.value} version={resolved_version} workspace={source} action={action}")
        try:
            download_from_url(
                url,
                resolved_version,
                source,
                proxy=proxy if proxy is not None else config.download_proxy,
                mirror=mirror if mirror is not None else config.download_mirror,
            )
            item = _ingest_local_locked(
                source, name, provider, url, resolved_version, normalized_modalities,
                config, StorageMode.MOVE, has_remote_url=True,
            )
            _append_ingestion_log(log_path, f"SUCCEEDED dataset_id={item.id} storage_path={item.storage_path}")
            return item
        except Exception as exc:
            _append_ingestion_log(log_path, f"FAILED {type(exc).__name__}: {exc}")
            raise RuntimeError(f"Download failed; workspace retained at {source}. Details were written to {log_path}. {exc}") from exc

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
