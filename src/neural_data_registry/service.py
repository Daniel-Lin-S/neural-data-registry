from __future__ import annotations
import json, shutil
from pathlib import Path
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from neural_data_registry.config import Settings, get_settings
from neural_data_registry.db.models import Dataset, DatasetAlias, IngestionJob
from neural_data_registry.db.session import create_database, get_session_factory
from neural_data_registry.enums import DatasetStatus, JobStatus, Provider, StorageMode, normalize_modalities
from neural_data_registry.provider import download_from_url, provider_for_url
from neural_data_registry.storage import dataset_destination, directory_size, ensure_layout, ingestion_lock, move_into_managed_storage

def session(config: Settings | None = None) -> Session:
    config = config or get_settings()
    create_database(config)
    return get_session_factory(config.resolved_database_url)()
def find_datasets(db: Session, query: str | None = None, url: str | None = None, modality: str | None = None) -> list[Dataset]:
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if url: stmt = stmt.outerjoin(DatasetAlias).where(or_(Dataset.source_url == url, DatasetAlias.value == url))
    if query:
        needle = f"%{query}%"; stmt = stmt.outerjoin(DatasetAlias).where(or_(Dataset.name.ilike(needle), DatasetAlias.value.ilike(needle)))
    if modality: stmt = stmt.where(Dataset.modalities.ilike(f"%{modality}%"))
    return list(db.scalars(stmt.distinct()))

def ingest_local(source: Path, name: str, provider: Provider, url: str | None, version: str | None, modalities: list[str], config: Settings | None = None, storage_mode: StorageMode = StorageMode.REFERENCE) -> Dataset:
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
        except Exception as exc:
            db.rollback(); raise
        finally: db.close()

def download(url: str, version: str, config: Settings | None = None) -> Dataset:
    config = config or get_settings()
    provider = provider_for_url(url); source = config.staging_dir / f"download-{provider.value}-{version}"
    if source.exists(): raise RuntimeError(f"Staging directory already exists: {source}")
    ensure_layout(config)
    try:
        download_from_url(url, version, source); return ingest_local(source, url.rstrip("/").split("/")[-1], provider, url, version, [], config, StorageMode.MOVE)
    except Exception:
        if source.exists(): shutil.move(str(source), str(config.quarantine_dir / source.name))
        raise

def dataset_dict(item: Dataset) -> dict:
    return {"dataset_id": item.id, "name": item.name, "provider": item.provider.value, "version": item.version, "source_url": item.source_url, "modalities": [x for x in item.modalities.split(",") if x], "size_bytes": item.size_bytes, "status": item.status.value, "storage_mode": item.storage_mode.value, "storage_path": item.storage_path}
