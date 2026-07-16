from __future__ import annotations
import os, re, shutil
from contextlib import contextmanager
from pathlib import Path
from neural_data_registry.config import Settings, get_settings

class IngestionConflictError(RuntimeError): pass

def ensure_layout(config: Settings | None = None) -> None:
    """Create the registry storage layout if it does not exist.

    Parameters
    ----------
    config : Settings or None, optional
        Registry configuration.
    """
    config = config or get_settings()
    for path in (config.datasets_dir, config.incoming_dir, config.quarantine_dir, config.registry_dir, config.logs_dir, config.ingestion_lock_dir): path.mkdir(parents=True, exist_ok=True)

def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
def safe_component(value: str) -> str: return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-").lower() or "dataset"

@contextmanager
def ingestion_lock(key: str, config: Settings | None = None):
    config = config or get_settings()
    ensure_layout(config); path = config.ingestion_lock_dir / f"{safe_component(key)}.lock"
    try: os.mkdir(path)
    except FileExistsError as exc: raise IngestionConflictError(f"An ingestion for {key!r} is already running") from exc
    try: yield
    finally: shutil.rmtree(path, ignore_errors=True)

def dataset_destination(dataset_id: str, name: str, version: str, config: Settings | None = None) -> Path:
    """Build the managed storage path for a dataset without creating it."""
    config = config or get_settings()
    return config.datasets_dir / f"{safe_component(name)}-{safe_component(version)}-{dataset_id}"
def move_into_managed_storage(source: Path, destination: Path) -> None:
    """Explicitly move a source directory into managed storage."""
    if destination.exists(): raise IngestionConflictError(f"Managed destination already exists: {destination}")
    if not source.is_dir(): raise ValueError(f"Source must be an existing directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(source), str(destination))
