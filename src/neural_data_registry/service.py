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


CANONICAL_NAME_ALIAS_KIND = "name"
USER_NAME_ALIAS_KIND = "name_alias"
NAME_ALIAS_KINDS = (CANONICAL_NAME_ALIAS_KIND, USER_NAME_ALIAS_KIND)
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
def find_datasets(db: Session, query: str | None = None, url: str | None = None, modality: str | None = None, provider: str | None = None, *, show_all: bool = False) -> list[Dataset]:
    """Return registered datasets matching the supplied filters."""
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if not show_all:
        stmt = stmt.where(
            Dataset.status.notin_((DatasetStatus.MISSING, DatasetStatus.BROKEN))
        )
    if url:
        stmt = stmt.outerjoin(DatasetAlias).where(
            or_(
                Dataset.source_url == url,
                and_(DatasetAlias.kind.in_(("url", "path")), DatasetAlias.value == url),
            )
        )
    if query:
        needle = f"%{query}%"
        stmt = stmt.outerjoin(DatasetAlias).where(
            or_(
                Dataset.name.ilike(needle),
                and_(DatasetAlias.kind.in_(NAME_ALIAS_KINDS), DatasetAlias.value.ilike(needle)),
            )
        )
    if modality:
        stmt = stmt.where(Dataset.modalities.ilike(f"%{modality}%"))
    if provider:
        provider_value = Provider(provider).name if isinstance(provider, str) else provider
        stmt = stmt.where(Dataset.provider == provider_value)
    return list(db.scalars(stmt.distinct()))


def resolve_dataset(db: Session, identifier: str | None = None, *, name: str | None = None, url: str | None = None) -> Dataset | None:
    """Resolve one dataset by its unique ID, canonical name, alias, or source URL."""
    values = [value for value in (identifier, name, url) if value]
    if len(values) != 1:
        raise ValueError("Provide exactly one dataset id, name, or URL")
    value = values[0]
    if identifier:
        conditions = [
            Dataset.id == value,
            func.lower(Dataset.name) == value.casefold(),
            Dataset.source_url == value,
            and_(DatasetAlias.kind.in_(NAME_ALIAS_KINDS), func.lower(DatasetAlias.value) == value.casefold()),
            and_(DatasetAlias.kind.in_(("url", "path")), DatasetAlias.value == value),
        ]
    elif name:
        conditions = [
            func.lower(Dataset.name) == value.casefold(),
            and_(DatasetAlias.kind.in_(NAME_ALIAS_KINDS), func.lower(DatasetAlias.value) == value.casefold()),
        ]
    else:
        conditions = [
            Dataset.source_url == value,
            and_(DatasetAlias.kind.in_(("url", "path")), DatasetAlias.value == value),
        ]
    matches = list(db.scalars(select(Dataset).outerjoin(DatasetAlias).where(or_(*conditions)).distinct()))
    if len(matches) > 1:
        raise ValueError(f"Dataset identifier is ambiguous: {value}")
    return matches[0] if matches else None


class DatasetConflictError(RuntimeError):
    """Raised when a dataset name or source identity is already registered."""




def _raise_dataset_conflict(existing: Dataset, identity: str) -> None:
    raise DatasetConflictError(
        f"Ingestion rejected: dataset {identity} is already registered as "
        f"{existing.id} ({existing.name} {existing.version}). "
        f"Existing storage path: {existing.storage_path}"
    )


def _normalize_name_aliases(
    aliases: Sequence[str], canonical_name: str | None = None
) -> list[str]:
    """Normalize user-provided aliases while retaining their display spelling."""
    normalized: list[str] = []
    seen: set[str] = set()
    canonical = canonical_name.casefold() if canonical_name else None
    for raw_alias in aliases:
        alias = raw_alias.strip()
        if not alias:
            raise ValueError("Dataset aliases must be non-empty")
        key = alias.casefold()
        if key == canonical or key in seen:
            continue
        seen.add(key)
        normalized.append(alias)
    return normalized


def _name_alias_owner(db: Session, alias: str) -> Dataset | None:
    """Return a dataset owning an identity which a user name alias cannot reuse."""
    return db.scalar(
        select(Dataset)
        .outerjoin(DatasetAlias)
        .where(
            or_(
                Dataset.id == alias,
                func.lower(Dataset.name) == alias.casefold(),
                Dataset.source_url == alias,
                and_(
                    DatasetAlias.kind.in_(NAME_ALIAS_KINDS),
                    func.lower(DatasetAlias.value) == alias.casefold(),
                ),
                and_(
                    DatasetAlias.kind.in_(("url", "path")),
                    DatasetAlias.value == alias,
                ),
            )
        )
        .order_by(Dataset.created_at.desc())
    )


def _raise_if_name_aliases_conflict(
    db: Session, aliases: Sequence[str], *, dataset_id: str | None = None
) -> None:
    for alias in aliases:
        owner = _name_alias_owner(db, alias)
        if owner is not None and owner.id != dataset_id:
            _raise_dataset_conflict(owner, f"name alias {alias!r}")


def _raise_if_dataset_identity_conflicts(
    db: Session,
    name: str,
    source_reference: str | None,
    name_aliases: Sequence[str] = (),
) -> None:
    """Reject registered identities before dataset processing."""
    conditions = [
        func.lower(Dataset.name) == name.casefold(),
        and_(
            DatasetAlias.kind.in_(NAME_ALIAS_KINDS),
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
        _raise_if_name_aliases_conflict(db, name_aliases)
        return
    name_match = existing.name.casefold() == name.casefold() or db.scalar(
        select(DatasetAlias.id).where(
            DatasetAlias.dataset_id == existing.id,
            DatasetAlias.kind.in_(NAME_ALIAS_KINDS),
            func.lower(DatasetAlias.value) == name.casefold(),
        )
    ) is not None
    _raise_dataset_conflict(existing, "name" if name_match else "source URL/path")


def add_name_aliases(
    identifier: str, aliases: Sequence[str], config: Settings | None = None
) -> Dataset:
    """Append searchable aliases without replacing registered metadata."""
    config = config or get_settings()
    requested_aliases = _normalize_name_aliases(aliases)
    if not requested_aliases:
        raise ValueError("Provide at least one non-empty dataset alias")
    with ingestion_lock("registry-intake", config):
        create_database(config)
        db = get_session_factory(config.resolved_database_url)()
        try:
            item = resolve_dataset(db, identifier)
            if item is None:
                raise ValueError("No dataset matched the supplied ID, name, alias, or URL")
            aliases_to_add = _normalize_name_aliases(requested_aliases, item.name)
            _raise_if_name_aliases_conflict(db, aliases_to_add, dataset_id=item.id)
            existing_aliases = {
                row.value.casefold()
                for row in item.aliases
                if row.kind == USER_NAME_ALIAS_KIND
            }
            for alias in aliases_to_add:
                if alias.casefold() not in existing_aliases:
                    item.aliases.append(
                        DatasetAlias(kind=USER_NAME_ALIAS_KIND, value=alias)
                    )
            db.commit()
            db.refresh(item)
            list(item.aliases)
            return item
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def _preflight_dataset_identity(
    name: str,
    source_reference: str | None,
    config: Settings,
    name_aliases: Sequence[str] = (),
) -> None:
    """Check registry identity before provider, workspace, or dataset-file work."""
    create_database(config)
    db = get_session_factory(config.resolved_database_url)()
    try:
        _raise_if_dataset_identity_conflicts(db, name, source_reference, name_aliases)
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
    name_aliases: Sequence[str],
) -> Dataset:
    """Ingest a local source while the registry-wide intake lock is held."""
    ensure_layout(config)
    create_database(config)
    db = get_session_factory(config.resolved_database_url)()
    try:
        _raise_if_dataset_identity_conflicts(db, name, source_reference, name_aliases)
        if not source.is_dir():
            raise ValueError(f"Local source is not a directory: {source}")
        item = Dataset(name=name, provider=provider, source_url=source_reference, version=version, modalities=",".join(sorted(set(modalities))), storage_mode=storage_mode, status=DatasetStatus.INGESTING); db.add(item); db.flush()
        job = IngestionJob(dataset_id=item.id, mode="local", status=JobStatus.RUNNING); db.add(job)
        item.aliases.extend(
            [
                DatasetAlias(kind="url" if has_remote_url else "path", value=source_reference),
                DatasetAlias(kind=CANONICAL_NAME_ALIAS_KIND, value=name),
                *[
                    DatasetAlias(kind=USER_NAME_ALIAS_KIND, value=alias)
                    for alias in name_aliases
                ],
            ]
        )
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


def ingest_local(source: Path, name: str, provider: Provider, url: str | None, version: str | None, modalities: Sequence[str | Modality], config: Settings | None = None, storage_mode: StorageMode = StorageMode.REFERENCE, *, name_aliases: Sequence[str] = ()) -> Dataset:
    """Register a local dataset after identity preflight and under intake lock."""
    config = config or get_settings()
    name = name.strip()
    if not name:
        raise ValueError("A non-empty dataset name is required")
    name_aliases = _normalize_name_aliases(name_aliases, name)
    version = version or "unknown"
    _preflight_dataset_identity(name, url, config, name_aliases)
    source = source.expanduser().resolve()
    source_reference = url or source.as_uri()
    if url is None:
        _preflight_dataset_identity(name, source_reference, config, name_aliases)
    normalized_modalities = normalize_modalities(modalities)
    with ingestion_lock("registry-intake", config):
        _preflight_dataset_identity(name, source_reference, config, name_aliases)
        ensure_layout(config)
        log_path = config.logs_dir / f"ingest-{safe_component(name)}-{safe_component(version)}.log"
        _append_ingestion_log(log_path, f"START provider={provider.value} version={version} source={source}")
        try:
            item = _ingest_local_locked(
                source, name, provider, source_reference, version, normalized_modalities,
                config, storage_mode, has_remote_url=url is not None, name_aliases=name_aliases,
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
    name_aliases: Sequence[str] = (),
) -> Dataset:
    """Download and register one dataset after identity preflight."""
    name = name.strip()
    if not name:
        raise ValueError("A non-empty dataset name is required for downloads")
    name_aliases = _normalize_name_aliases(name_aliases, name)
    config = config or get_settings()
    _preflight_dataset_identity(name, url, config, name_aliases)
    normalized_modalities = normalize_modalities(modalities)
    if not normalized_modalities:
        raise ValueError("At least one modality is required for downloads")
    provider = provider_for_url(url)
    resolved_version = resolve_download_version(url, version)
    with ingestion_lock("registry-intake", config):
        # Repeat the preflight while serialized with all other intake actions.
        _preflight_dataset_identity(name, url, config, name_aliases)
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
                config, StorageMode.MOVE, has_remote_url=True, name_aliases=name_aliases,
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
    data = {
        output_name: _serialize_dataset_value(column, getattr(item, column.name))
        for column, output_name, _ in _dataset_output_columns()
    }
    data["aliases"] = sorted(
        alias.value for alias in item.aliases if alias.kind == USER_NAME_ALIAS_KIND
    )
    return data
