"""Coordinate protected local ingestion across caller and service identities.

Input is one parsed ``brainctl ingest-local`` request plus trusted ``SUDO_*``
identity variables. Reference mode publishes and records the source path. Copy and
move modes stage only the submitted directory below the registry ``incoming``
folder. Output is a committed dataset row and an optional size warning.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import pwd
import shutil
import tempfile
from typing import Iterator, Sequence

from neural_data_registry.config import Settings
from neural_data_registry.db.models import Dataset
from neural_data_registry.enums import (
    DatasetStatus,
    Modality,
    Provider,
    StorageMode,
)
from neural_data_registry.service import (
    LocalIngestionRequest,
    commit_reregistration,
    commit_local_ingestion_request,
    preflight_local_ingestion_request,
    prepare_local_ingestion_request,
    resolve_dataset,
    session,
    unregister_moved_dataset,
    validate_privileged_source,
)
from neural_data_registry.storage import (
    directory_size,
    ingestion_lock,
    normalize_managed_dataset_access,
    normalize_reference_dataset_access,
)

SUDO_GID_VARIABLE = "SUDO_GID"
SUDO_UID_VARIABLE = "SUDO_UID"
SUDO_USER_VARIABLE = "SUDO_USER"
STAGING_PREFIX = "protected-local-"
STAGING_PAYLOAD_NAME = "payload"


@dataclass(frozen=True)
class UnixIdentity:
    """Filesystem identity used for one protected operation."""

    name: str
    uid: int
    gid: int
    groups: tuple[int, ...]


@dataclass(frozen=True)
class PreparedCallerSource:
    """Caller-prepared source handed to the registry service."""

    source: Path
    staging_root: Path | None
    size_bytes: int | None
    warning: str | None


def _identity_for_account(name: str) -> UnixIdentity:
    """Resolve one account and all of its supplementary groups."""
    try:
        account = pwd.getpwnam(name)
    except KeyError as exc:
        message = f"Configured account does not exist: {name}"
        raise RuntimeError(message) from exc
    groups = tuple(sorted(set(os.getgrouplist(name, account.pw_gid))))
    return UnixIdentity(name, account.pw_uid, account.pw_gid, groups)


def caller_identity_from_sudo() -> UnixIdentity:
    """Return the invoking identity from sudo-controlled environment values."""
    sudo_user = os.environ.get(SUDO_USER_VARIABLE)
    sudo_uid = os.environ.get(SUDO_UID_VARIABLE)
    sudo_gid = os.environ.get(SUDO_GID_VARIABLE)
    if not sudo_user or sudo_uid is None or sudo_gid is None:
        raise RuntimeError(
            "Protected ingestion must be invoked through the installed "
            "brainctl sudo wrapper"
        )
    try:
        expected_uid = int(sudo_uid)
        expected_gid = int(sudo_gid)
    except ValueError as exc:
        raise RuntimeError("Sudo caller UID and GID must be integers") from exc
    identity = _identity_for_account(sudo_user)
    if identity.uid != expected_uid:
        raise RuntimeError(
            "Sudo caller identity does not match the local account database"
        )
    groups = tuple(sorted(set((*identity.groups, expected_gid))))
    return UnixIdentity(sudo_user, expected_uid, expected_gid, groups)


@contextmanager
def effective_identity(identity: UnixIdentity) -> Iterator[None]:
    """Temporarily use one identity while retaining a root saved UID."""
    original_uid = os.geteuid()
    original_gid = os.getegid()
    original_groups = tuple(os.getgroups())
    try:
        os.seteuid(0)
    except PermissionError as exc:
        raise RuntimeError(
            "Protected ingestion coordinator must retain a root saved UID"
        ) from exc
    os.setgroups(identity.groups)
    os.setegid(identity.gid)
    os.seteuid(identity.uid)
    try:
        yield
    finally:
        os.seteuid(0)
        os.setegid(original_gid)
        os.setgroups(original_groups)
        os.seteuid(original_uid)



def _stage_source_as_caller(
    request: LocalIngestionRequest,
    config: Settings,
    caller: UnixIdentity,
    service: UnixIdentity,
) -> PreparedCallerSource:
    """Stage only the submitted copy or move using caller permissions."""
    staging_root = Path(
        tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=config.incoming_dir)
    )
    os.chown(staging_root, caller.uid, caller.gid)
    payload = staging_root / STAGING_PAYLOAD_NAME
    warning = None
    try:
        with effective_identity(caller):
            if request.storage_mode is StorageMode.COPY:
                shutil.copytree(request.source, payload)
            elif request.storage_mode is StorageMode.MOVE:
                shutil.move(str(request.source), str(payload))
            else:
                raise ValueError(
                    "Caller staging requires copy or move storage mode"
                )
            try:
                size_bytes = directory_size(payload)
            except OSError as exc:
                size_bytes = None
                warning = (
                    "Dataset size could not be measured after staging; "
                    f"the managed dataset has unknown size: {exc}"
                )
    except Exception as exc:
        raise RuntimeError(
            "Source staging failed; partial data was retained at "
            f"{staging_root}: {exc}"
        ) from exc
    try:
        normalize_managed_dataset_access(
            staging_root,
            owner_uid=service.uid,
            owner_gid=service.gid,
        )
    except OSError as exc:
        raise RuntimeError(
            "Staging access preparation failed; data was retained at "
            f"{staging_root}: {exc}"
        ) from exc
    return PreparedCallerSource(
        payload,
        staging_root,
        size_bytes,
        warning,
    )


def _prepare_reference_as_caller(
    request: LocalIngestionRequest,
    caller: UnixIdentity,
) -> PreparedCallerSource:
    """Publish and measure one reference using caller permissions."""
    try:
        with effective_identity(caller):
            normalize_reference_dataset_access(request.source)
    except OSError as exc:
        failed_path = exc.filename or str(request.source)
        raise RuntimeError(
            "Could not make reference dataset publicly readable at "
            f"{failed_path}: {exc}"
        ) from exc
    try:
        with effective_identity(caller):
            size_bytes = directory_size(request.source)
    except OSError as exc:
        warning = (
            "Dataset size could not be measured with caller permissions; "
            f"the reference was registered with unknown size: {exc}"
        )
        return PreparedCallerSource(request.source, None, None, warning)
    return PreparedCallerSource(request.source, None, size_bytes, None)


def coordinate_protected_ingestion(
    source: Path,
    name: str,
    provider: Provider,
    url: str | None,
    version: str | None,
    modalities: Sequence[str | Modality],
    storage_mode: StorageMode,
    name_aliases: Sequence[str],
    config: Settings,
) -> tuple[Dataset, str | None]:
    """Run source work as the caller and registry work as the service.

    Parameters
    ----------
    source : pathlib.Path
        Submitted dataset directory.
    name : str
        Canonical dataset name.
    provider : neural_data_registry.enums.Provider
        Requested provider, overridden by a recognized URL.
    url : str or None
        Canonical remote URL, optional.
    version : str or None
        Dataset version, optional.
    modalities : sequence of str or Modality
        Dataset modalities.
    storage_mode : neural_data_registry.enums.StorageMode
        Reference, copy, or move behavior.
    name_aliases : sequence of str
        Alternate dataset names.
    config : neural_data_registry.config.Settings
        Protected deployment settings.

    Returns
    -------
    tuple[neural_data_registry.db.models.Dataset, str or None]
        Committed dataset and an optional non-fatal size warning.
    """
    if os.geteuid() != 0:
        raise RuntimeError("Protected ingestion coordinator must run as root")
    if not config.service_user:
        raise RuntimeError(
            "NDR_SERVICE_USER is required for protected ingestion"
        )

    caller = caller_identity_from_sudo()
    service = _identity_for_account(config.service_user)
    validated_source = validate_privileged_source(source, config)
    request = prepare_local_ingestion_request(
        validated_source,
        name,
        provider,
        url,
        version,
        modalities,
        storage_mode,
        name_aliases,
    )

    with effective_identity(service):
        preflight_local_ingestion_request(request, config)
        with ingestion_lock("registry-intake", config):
            preflight_local_ingestion_request(request, config)
            if request.storage_mode is StorageMode.REFERENCE:
                prepared = _prepare_reference_as_caller(request, caller)
            else:
                with effective_identity(
                    UnixIdentity("root", 0, 0, (0,))
                ):
                    prepared = _stage_source_as_caller(
                        request,
                        config,
                        caller,
                        service,
                    )
            try:
                item = commit_local_ingestion_request(
                    request,
                    config,
                    prepared_source=(
                        prepared.source
                        if prepared.staging_root is not None
                        else None
                    ),
                    source_prevalidated=True,
                    prepared_access_normalized=(
                        prepared.staging_root is not None
                    ),
                    size_bytes=prepared.size_bytes,
                )
            except Exception as exc:
                if (
                    prepared.staging_root is not None
                    and prepared.source.exists()
                ):
                    raise RuntimeError(
                        "Registry update failed; staged data was retained at "
                        f"{prepared.staging_root}: {exc}"
                    ) from exc
                raise

    if prepared.staging_root is not None:
        prepared.staging_root.rmdir()
    return item, prepared.warning


def coordinate_protected_unregistration(
    identifier: str,
    restore_to: Path,
    config: Settings,
) -> tuple[Dataset, str]:
    """Restore one moved dataset to the invoking user's source tree.

    Parameters
    ----------
    identifier : str
        Dataset ID, name, alias, or source URL.
    restore_to : pathlib.Path
        Destination below an approved source root.
    config : neural_data_registry.config.Settings
        Protected deployment settings.

    Returns
    -------
    tuple[neural_data_registry.db.models.Dataset, str]
        Preserved inactive record and a permissions warning.
    """
    if os.geteuid() != 0:
        raise RuntimeError(
            "Protected unregistration coordinator must run as root"
        )
    if not config.service_user:
        raise RuntimeError(
            "NDR_SERVICE_USER is required for protected unregistration"
        )

    caller = caller_identity_from_sudo()
    service = _identity_for_account(config.service_user)
    destination = restore_to.expanduser()
    if not destination.is_absolute():
        raise ValueError(
            f"Restore destination must be an absolute path: {destination}"
        )
    if destination.exists():
        raise ValueError(
            f"Restore destination already exists: {destination}"
        )
    parent = destination.parent.resolve(strict=True)
    resolved_destination = parent / destination.name
    if not any(
        resolved_destination.is_relative_to(root)
        for root in config.privileged_source_root_paths
    ):
        allowed_roots = ", ".join(
            str(root) for root in config.privileged_source_root_paths
        )
        raise ValueError(
            "Restore destination must be below an allowed source root: "
            f"{allowed_roots}"
        )
    with effective_identity(caller):
        if not os.access(parent, os.W_OK | os.X_OK):
            raise PermissionError(
                errno.EACCES,
                "Caller cannot write to the restore destination parent",
                str(parent),
            )

    item = unregister_moved_dataset(
        identifier,
        resolved_destination,
        caller.uid,
        caller.gid,
        config,
        managed_owner_uid=service.uid,
        managed_owner_gid=service.gid,
    )
    warning = (
        "The original per-entry modes and ACLs were not recorded by this "
        "registration. Ownership was restored to "
        f"{caller.name}; managed file and directory modes were retained."
    )
    return item, warning


def coordinate_protected_reregistration(
    identifier: str,
    config: Settings,
) -> tuple[Dataset, str | None]:
    """Re-register an inactive moved dataset through caller staging.

    Parameters
    ----------
    identifier : str
        Dataset ID, name, alias, or source URL.
    config : neural_data_registry.config.Settings
        Protected deployment settings.

    Returns
    -------
    tuple[neural_data_registry.db.models.Dataset, str or None]
        Reactivated dataset and an optional size warning.
    """
    if os.geteuid() != 0:
        raise RuntimeError(
            "Protected re-registration coordinator must run as root"
        )
    if not config.service_user:
        raise RuntimeError(
            "NDR_SERVICE_USER is required for protected re-registration"
        )

    caller = caller_identity_from_sudo()
    service = _identity_for_account(config.service_user)
    with effective_identity(service):
        with session(config) as db:
            item = resolve_dataset(db, identifier)
        if item is None:
            raise ValueError(
                "No dataset matched the supplied ID, name, alias, or URL"
            )
        if item.status is not DatasetStatus.UNREGISTERED:
            raise RuntimeError(
                "Only an unregistered dataset can be re-registered"
            )
        if item.storage_mode is not StorageMode.REFERENCE:
            raise RuntimeError(
                "Unregistered dataset must reference its restored source"
            )
        if item.storage_path is None:
            raise ValueError("Unregistered dataset has no restored source path")
        source = validate_privileged_source(Path(item.storage_path), config)
        request = LocalIngestionRequest(
            source=source,
            name=item.name,
            provider=item.provider,
            source_reference=item.source_url or source.as_uri(),
            version=item.version,
            modalities=tuple(
                value
                for value in item.modalities.split(",")
                if value
            ),
            storage_mode=StorageMode.MOVE,
            has_remote_url=item.source_url is not None,
            name_aliases=(),
        )
        with ingestion_lock("registry-intake", config):
            with effective_identity(UnixIdentity("root", 0, 0, (0,))):
                prepared = _stage_source_as_caller(
                    request,
                    config,
                    caller,
                    service,
                )
            try:
                item = commit_reregistration(
                    identifier,
                    config,
                    prepared_source=prepared.source,
                    prepared_access_normalized=True,
                    size_bytes=prepared.size_bytes,
                    managed_owner_uid=service.uid,
                    managed_owner_gid=service.gid,
                )
            except Exception as exc:
                if prepared.source.exists():
                    raise RuntimeError(
                        "Re-registration failed; staged data was retained at "
                        f"{prepared.staging_root}: {exc}"
                    ) from exc
                raise

    assert prepared.staging_root is not None
    prepared.staging_root.rmdir()
    return item, prepared.warning
