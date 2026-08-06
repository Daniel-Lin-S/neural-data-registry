from __future__ import annotations

import errno
import fcntl
import os
import re
import shutil
import stat
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from neural_data_registry.config import Settings, get_settings

EXECUTE_PERMISSION_BITS = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
IGNORED_ACL_ERROR_NUMBERS = {
    errno.ENODATA,
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
}
MANAGED_DIRECTORY_MODE = 0o755
MANAGED_EXECUTABLE_FILE_MODE = 0o755
MANAGED_FILE_MODE = 0o644
POSIX_ACL_NAMES = (
    "system.posix_acl_access",
    "system.posix_acl_default",
)


class IngestionConflictError(RuntimeError):
    pass

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

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current_root, directory_names, file_names in os.walk(
        path,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current = Path(current_root)
        for name in (*directory_names, *file_names):
            item = current / name
            try:
                item_stat = item.stat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(item_stat.st_mode):
                continue
            identity = (item_stat.st_dev, item_stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += item_stat.st_size
    return total


def _remove_posix_acls(path: Path) -> None:
    """Remove access rules that could override public-read-only modes."""
    for acl_name in POSIX_ACL_NAMES:
        try:
            os.removexattr(
                path,
                acl_name,
                follow_symlinks=False,
            )
        except OSError as exc:
            if exc.errno not in IGNORED_ACL_ERROR_NUMBERS:
                raise


def _normalize_managed_entry(
    path: Path,
    entry_mode: int,
    owner_uid: int | None,
    owner_gid: int | None,
) -> None:
    """Apply managed ownership and access to one filesystem entry."""
    if owner_uid is not None and owner_gid is not None:
        os.chown(
            path,
            owner_uid,
            owner_gid,
            follow_symlinks=False,
        )
    if stat.S_ISLNK(entry_mode):
        return
    _remove_posix_acls(path)
    if stat.S_ISDIR(entry_mode):
        mode = MANAGED_DIRECTORY_MODE
    elif entry_mode & EXECUTE_PERMISSION_BITS:
        mode = MANAGED_EXECUTABLE_FILE_MODE
    else:
        mode = MANAGED_FILE_MODE
    os.chmod(path, mode, follow_symlinks=False)


def normalize_managed_dataset_access(
    path: Path,
    *,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    progress_callback: Callable[[int, Path], None] | None = None,
) -> None:
    """Make one managed tree publicly readable and service-writable.

    The traversal is limited to path and never follows symbolic links.
    Directories become mode 0755. Regular and special files become mode 0644,
    except files that already have an executable bit become mode 0755.
    Extended and default POSIX ACLs are removed. Optional ownership is applied
    without following links.

    Parameters
    ----------
    path : pathlib.Path
        Managed dataset directory to normalize.
    owner_uid : int or None, optional
        Service account UID, by default None.
    owner_gid : int or None, optional
        Service account GID, by default None.
    progress_callback : callable or None, optional
        Receives the normalized entry count and absolute entry path, by
        default None.

    Raises
    ------
    ValueError
        If ownership is incomplete or path is not a real directory.
    OSError
        If an entry cannot be inspected or updated.
    """
    if (owner_uid is None) != (owner_gid is None):
        raise ValueError(
            "Managed ownership requires both owner_uid and owner_gid"
        )
    requested_path = path.expanduser()
    if requested_path.is_symlink():
        raise ValueError(
            "Managed dataset root must not be a symbolic link: "
            f"{requested_path.absolute()}"
        )
    try:
        root = requested_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "Managed dataset root could not be resolved: "
            f"{requested_path.absolute()}"
        ) from exc
    if not root.is_dir():
        raise ValueError(
            f"Managed dataset root must be a directory: {root}"
        )

    pending = [root]
    normalized_count = 0

    def report_progress(entry_path: Path) -> None:
        nonlocal normalized_count
        normalized_count += 1
        if progress_callback is not None:
            progress_callback(normalized_count, entry_path)

    while pending:
        current = pending.pop()
        current_stat = current.lstat()
        _normalize_managed_entry(
            current,
            current_stat.st_mode,
            owner_uid,
            owner_gid,
        )
        report_progress(current)
        with os.scandir(current) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(entry_stat.st_mode):
                    pending.append(entry_path)
                else:
                    _normalize_managed_entry(
                        entry_path,
                        entry_stat.st_mode,
                        owner_uid,
                        owner_gid,
                    )
                    report_progress(entry_path)


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
