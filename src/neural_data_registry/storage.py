from __future__ import annotations
import fcntl, os, re, shutil, stat
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
    for path in (config.datasets_dir, config.incoming_dir, config.quarantine_dir, config.registry_dir, config.logs_dir, config.ingestion_lock_dir, config.health_cooldown_dir): path.mkdir(parents=True, exist_ok=True)

def directory_size(path: Path) -> int:
    """Return logical file bytes below *path*, counting each inode once.

    ``Path.stat`` follows file symlinks, which lets a DataLad checkout count
    annexed content as payload. Tracking the resolved device/inode prevents
    that content from being counted again through both its working-tree link
    and its ``.git/annex/objects`` path. Dangling links are not payload.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    for item in path.rglob("*"):
        try:
            item_stat = item.stat()
        except FileNotFoundError:
            # A dangling link (or a file removed during traversal) contributes
            # no readable payload.
            continue
        if not stat.S_ISREG(item_stat.st_mode):
            continue
        identity = (item_stat.st_dev, item_stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += item_stat.st_size
    return total
def safe_component(value: str) -> str: return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-").lower() or "dataset"

@contextmanager
def ingestion_lock(key: str, config: Settings | None = None):
    config = config or get_settings()
    ensure_layout(config); path = config.ingestion_lock_dir / f"{safe_component(key)}.lock"
    try: os.mkdir(path)
    except FileExistsError as exc: raise IngestionConflictError(f"An ingestion for {key!r} is already running") from exc
    try: yield
    finally: shutil.rmtree(path, ignore_errors=True)


@contextmanager
def process_lock(key: str, config: Settings | None = None):
    """Try to hold a process-safe registry lock without blocking."""
    config = config or get_settings()
    ensure_layout(config)
    path = config.ingestion_lock_dir / f"{safe_component(key)}.process.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def dataset_destination(dataset_id: str, name: str, version: str, config: Settings | None = None) -> Path:
    """Build the managed storage path for a dataset without creating it."""
    config = config or get_settings()
    return config.datasets_dir / f"{safe_component(name)}-{safe_component(version)}-{dataset_id}"
def copy_into_managed_storage(source: Path, destination: Path) -> None:
    """Copy a source directory into managed storage, preserving the source."""
    if destination.exists(): raise IngestionConflictError(f"Managed destination already exists: {destination}")
    if not source.is_dir(): raise ValueError(f"Source must be an existing directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(source, destination)

def move_into_managed_storage(source: Path, destination: Path) -> None:
    """Explicitly move a source directory into managed storage."""
    if destination.exists(): raise IngestionConflictError(f"Managed destination already exists: {destination}")
    if not source.is_dir(): raise ValueError(f"Source must be an existing directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(source), str(destination))
