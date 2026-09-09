"""Download and verify one OpenNeuro dataset snapshot.

Input
-----
Configuration is read from ``DOWNLOAD_*`` variables exported by
``download_openneuro.sh``. ``DOWNLOAD_REPO_ID`` accepts ``dsNNNNNN``, a
dataset URL, or a version URL ending in ``/versions/TAG``.

Output
------
A DataLad dataset is installed beneath the absolute destination. Snapshot tags
are pinned when supplied, annexed content is retrieved with bounded jobs, and
existing git-annex state is reused by later attempts and invocations. Optional
repository-relative git-annex globs can omit content from the current run while
leaving its tracked paths available for later retrieval.
"""

from __future__ import annotations

import ctypes
import errno
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from download_diagnostics import (
    DownloadInterrupted,
    configure_diagnostics,
    current_log_path,
    log_event,
    log_exception,
    sanitize_text,
)
from download_retry import (
    TerminalDownloadError,
    classify_download_error,
    run_with_retries,
)
from mihomo_ranker import MihomoConfig, MihomoNodeManager

DEFAULT_MIHOMO_PROBE_TIMEOUT = 8.0
DATASET_ID_PATTERN = re.compile(r"^ds[0-9]+$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_URL_PORTS = {"http": 80, "https": 443}
TERMINAL_COMMAND_EXIT_CODES = frozenset({2, 126, 127, 129, 130})
AT_FDCWD = -100
RENAME_NOREPLACE = 1
STAGED_DATASET_NAME = "dataset"
STAGING_DIRECTORY_SUFFIX = ".openneuro-clone-"
REMOTE_BRANCH_NOT_FOUND_PATTERN = re.compile(
    r"\bremote branch [A-Za-z0-9._-]+ not found in upstream origin\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
TERMINAL_COMMAND_MARKERS = (
    "all offered socks5 authentication methods were rejected",
    "ambiguous argument",
    "already exists and is not an empty directory",
    "authentication failed",
    "authorization failed",
    "command not found",
    "connect tunnel failed, response 407",
    "could not read username",
    "could not find remote ref",
    "couldn't find remote ref",
    "curl: (3)",
    "curl: (60)",
    "did not match any file",
    "disk quota exceeded",
    "file name too long",
    "filename too long",
    "http code 407 from proxy after connect",
    "invalid argument",
    "invalid path",
    "invalid username or password",
    "is a directory",
    "malformed url",
    "is not associated with a dataset",
    "is not associated with any dataset",
    "no space left on device",
    "no such file or directory",
    "not enough free space",
    "no dataset found at",
    "not a git repository",
    "not a valid object name",
    "operation not permitted",
    "path not associated with any dataset",
    "pathspec",
    "permission denied",
    "read-only file system",
    "remote head refers to nonexistent ref",
    "reference is not a tree",
    "repository not found",
    "server certificate verification failed",
    "ssl peer certificate or ssh remote key was not ok",
    "target path already exists",
    "the following arguments are required",
    "assertionerror:",
    "attributeerror:",
    "importerror:",
    "indexerror:",
    "keyerror:",
    "memoryerror:",
    "nameerror:",
    "notimplementederror:",
    "syntaxerror:",
    "typeerror:",
    "unrecognized argument",
    "unrecognized option",
    "unknown option",
    "unknown revision or path",
    "unsupported option",
    "url using bad/illegal format",
    "usage:",
    "zerodivisionerror:",
)

CommandRunner = Callable[[Sequence[str], dict[str, str]], str]


class IncompleteCloneError(FileExistsError):
    """Represent an invocation-owned clone that is not yet reusable."""


class NetworkCommandError(ConnectionError):
    """Represent a retryable DataLad or git-annex transfer failure."""


@dataclass(frozen=True)
class RepositoryReference:
    """Describe a canonical OpenNeuro dataset and optional snapshot tag."""

    dataset_id: str
    version: str | None


@dataclass(frozen=True)
class DownloadConfig:
    """Store validated OpenNeuro download configuration.

    Parameters
    ----------
    dataset_id : str
        Canonical OpenNeuro accession number.
    version : str or None
        Snapshot tag parsed from the repository URL, optional.
    destination : str
        Absolute DataLad dataset destination.
    endpoint : str
        OpenNeuro service endpoint.
    max_workers : int
        Number of concurrent DataLad jobs.
    timeout : float
        HTTP low-speed timeout in seconds.
    dry_run : bool
        Whether to inspect pending state without network mutation.
    retry_attempts : int
        Maximum attempts per provider operation; zero is unlimited.
    retry_base_delay : float
        Initial retry delay in seconds.
    retry_max_delay : float
        Maximum retry delay in seconds.
    proxy_url : str or None
        Explicit HTTP or SOCKS proxy URL, optional.
    mihomo : MihomoConfig or None
        Mihomo ranking configuration, optional.
    exclude_patterns : tuple of str
        Repository-relative git-annex globs omitted from retrieval.
    """

    dataset_id: str
    version: str | None
    destination: str
    endpoint: str
    max_workers: int
    timeout: float
    dry_run: bool
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float
    proxy_url: str | None
    mihomo: MihomoConfig | None
    exclude_patterns: tuple[str, ...]


@dataclass(frozen=True)
class CloneWorkspace:
    """Track a final destination and invocation-owned clone staging."""

    destination: Path
    staging_root: Path | None
    staged_destination: Path | None


def optional_environment(name: str) -> str | None:
    """Return a non-empty environment variable, or ``None``."""

    value = os.environ.get(name, "")
    return value or None


def load_exclude_patterns() -> tuple[str, ...]:
    """Load newline-delimited OpenNeuro annex exclusion patterns."""

    value = os.environ.get("DOWNLOAD_EXCLUDE_PATTERNS", "")
    return tuple(line for line in value.splitlines() if line)


def parse_repository(value: str) -> RepositoryReference:
    """Parse an OpenNeuro accession number or dataset/version URL."""

    candidate = value.strip()
    version = None
    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "expected an HTTP or HTTPS OpenNeuro URL, but got "
                f"{value!r}."
            )
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) not in {2, 4} or parts[0] != "datasets":
            raise ValueError(
                "expected an OpenNeuro /datasets/ID URL, optionally ending "
                f"in /versions/TAG, but got {value!r}."
            )
        candidate = parts[1]
        if len(parts) == 4:
            if parts[2] != "versions":
                raise ValueError(
                    f"expected /versions/TAG in OpenNeuro URL {value!r}."
                )
            version = parts[3]
    if DATASET_ID_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            "expected an OpenNeuro dataset ID such as ds005261, but got "
            f"{candidate!r}."
        )
    if version is not None and VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(
            f"OpenNeuro version contains unsupported characters: {version!r}."
        )
    return RepositoryReference(candidate, version)


def load_mihomo_config() -> MihomoConfig | None:
    """Load optional Mihomo ranking settings exported by the shell."""

    speed_test_url = optional_environment(
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL"
    )
    if speed_test_url is None:
        return None
    controller = optional_environment("DOWNLOAD_MIHOMO_CONTROLLER")
    if controller is None:
        raise ValueError(
            "Mihomo ranking requires --mihomo-controller or "
            "MIHOMO_CONTROLLER."
        )
    return MihomoConfig(
        controller_url=controller,
        group_name=optional_environment("DOWNLOAD_MIHOMO_GROUP"),
        node_marker=(
            optional_environment("DOWNLOAD_MIHOMO_NODE_MARKER") or ""
        ),
        speed_test_url=speed_test_url,
        probe_timeout=float(
            optional_environment("DOWNLOAD_MIHOMO_PROBE_TIMEOUT")
            or DEFAULT_MIHOMO_PROBE_TIMEOUT
        ),
        secret=optional_environment("MIHOMO_SECRET"),
    )


def load_config_from_environment() -> DownloadConfig:
    """Load and validate downloader configuration from the environment."""

    reference = parse_repository(os.environ["DOWNLOAD_REPO_ID"])
    config = DownloadConfig(
        dataset_id=reference.dataset_id,
        version=reference.version,
        destination=os.environ["DOWNLOAD_DEST"],
        endpoint=os.environ["DOWNLOAD_ENDPOINT"].rstrip("/"),
        max_workers=int(os.environ["DOWNLOAD_MAX_WORKERS"]),
        timeout=float(os.environ["DOWNLOAD_TIMEOUT"]),
        dry_run=os.environ["DOWNLOAD_DRY_RUN"] == "1",
        retry_attempts=int(os.environ["DOWNLOAD_RETRY_ATTEMPTS"]),
        retry_base_delay=float(
            os.environ["DOWNLOAD_RETRY_BASE_DELAY"]
        ),
        retry_max_delay=float(os.environ["DOWNLOAD_RETRY_MAX_DELAY"]),
        proxy_url=optional_environment("DOWNLOAD_PROXY_URL"),
        mihomo=load_mihomo_config(),
        exclude_patterns=load_exclude_patterns(),
    )
    validate_config(config)
    return config


def validate_config(config: DownloadConfig) -> None:
    """Reject invalid OpenNeuro downloader configuration."""

    if not os.path.isabs(config.destination):
        raise ValueError(
            "expected an absolute destination path, but got "
            f"{config.destination!r}."
        )
    parsed_endpoint = urlparse(config.endpoint)
    if (
        parsed_endpoint.scheme not in {"http", "https"}
        or not parsed_endpoint.netloc
    ):
        raise ValueError(
            "expected an HTTP or HTTPS OpenNeuro endpoint, but got "
            f"{config.endpoint!r}."
        )
    positive_values = {
        "max_workers": config.max_workers,
        "timeout": config.timeout,
        "retry_base_delay": config.retry_base_delay,
        "retry_max_delay": config.retry_max_delay,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(
                f"expected {name} to be positive, but got {value}."
            )
    if config.retry_attempts < 0:
        raise ValueError(
            "expected retry_attempts to be non-negative, but got "
            f"{config.retry_attempts}."
        )
    if config.mihomo is not None and config.proxy_url is None:
        raise ValueError(
            "Mihomo ranking requires an explicit local proxy URL."
        )
    if config.mihomo is not None and config.max_workers != 1:
        raise ValueError(
            "Mihomo ranking requires max_workers=1 so node failover does "
            "not interrupt concurrent DataLad jobs."
        )
    if len(set(config.exclude_patterns)) != len(config.exclude_patterns):
        raise ValueError("OpenNeuro exclusion patterns must be unique.")
    for pattern in config.exclude_patterns:
        if not pattern:
            raise ValueError("OpenNeuro exclusion patterns must not be empty.")
        if pattern.startswith("/"):
            raise ValueError(
                "OpenNeuro exclusion patterns must be repository-relative, "
                f"but got {pattern!r}."
            )
        if ".." in Path(pattern).parts:
            raise ValueError(
                "OpenNeuro exclusion patterns must not contain a parent "
                f"directory component, but got {pattern!r}."
            )


def check_dependencies() -> None:
    """Require DataLad, git, and git-annex executables."""

    missing = [
        command
        for command in ("datalad", "git", "git-annex")
        if shutil.which(command) is None
    ]
    if missing:
        raise RuntimeError(
            "install required OpenNeuro downloader commands: "
            f"{', '.join(missing)}."
        )


def repository_url(config: DownloadConfig) -> str:
    """Build the configured git-mirror URL for one dataset."""

    dataset_id = quote(config.dataset_id, safe="")
    if config.endpoint.endswith("/git/0"):
        return f"{config.endpoint}/{dataset_id}"
    return f"{config.endpoint}/{dataset_id}.git"


def subprocess_environment(config: DownloadConfig) -> dict[str, str]:
    """Build a proxy-controlled environment for DataLad and git-annex."""

    environment = os.environ.copy()
    proxy_names = (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    )
    for name in proxy_names:
        environment.pop(name, None)
    if config.proxy_url is not None:
        for name in proxy_names:
            environment[name] = config.proxy_url
    environment["GIT_HTTP_LOW_SPEED_LIMIT"] = "1"
    environment["GIT_HTTP_LOW_SPEED_TIME"] = str(int(config.timeout))
    return environment


def command_failure_records(message: str) -> list[str]:
    """Return non-warning records that can explain command failure."""

    records = [
        line.strip()
        for line in message.splitlines()
        if line.strip()
    ]
    if not records:
        return [message]
    non_warning_records = [
        record
        for record in records
        if not record.casefold().startswith("[warning]")
    ]
    return non_warning_records or records


def is_terminal_command_record(record: str) -> bool:
    """Return whether one provider record proves deterministic failure."""

    lower_record = record.lower()
    return (
        any(
            marker in lower_record
            for marker in TERMINAL_COMMAND_MARKERS
        )
        or REMOTE_BRANCH_NOT_FOUND_PATTERN.search(record) is not None
    )


def record_has_retryable_http_status(record: str) -> bool:
    """Return whether one record reports a transient HTTP response."""

    decision = classify_download_error(
        RuntimeError(record),
        "datalad",
    )
    return (
        decision.category in {"rate_limit", "server"}
        or decision.reason == "transient_http"
    )


def classify_command_failure(
    command: Sequence[str],
    return_code: int,
    output: str,
) -> TerminalDownloadError | NetworkCommandError:
    """Translate provider output into deterministic or retryable failure."""

    sanitized_output = sanitize_text(output).strip()
    message = sanitized_output or "command produced no diagnostic output"
    command_name = Path(command[0]).name
    if return_code < 0:
        status = f"command terminated by signal {-return_code}"
    elif return_code in TERMINAL_COMMAND_EXIT_CODES:
        status = f"invalid command exit status {return_code}"
    else:
        status = f"exited with status {return_code}"
    records = command_failure_records(message)
    detail = f"{command_name} {status}: {message}"
    is_terminal_status = (
        return_code < 0
        or return_code in TERMINAL_COMMAND_EXIT_CODES
    )
    if is_terminal_status:
        return TerminalDownloadError(detail)
    for record in records:
        if not is_terminal_command_record(record):
            continue
        if record_has_retryable_http_status(record):
            continue
        return TerminalDownloadError(detail)
    retry_message = "\n".join(records)
    retry_detail = f"{command_name} {status}: {retry_message}"
    return NetworkCommandError(retry_detail)


def run_command(
    command: Sequence[str],
    environment: dict[str, str],
) -> str:
    """Run one provider command and return its combined textual output."""

    log_event("provider_command_started", command=list(command))
    result = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode != 0:
        raise classify_command_failure(command, result.returncode, output)
    log_event("provider_command_completed", command=list(command))
    return output


def normalized_repository_url(value: str) -> str:
    """Return a credential-free Git URL for identity comparison."""

    candidate = value.strip()
    scp_match = None
    if "://" not in candidate:
        scp_match = re.fullmatch(
            r"(?:[^@/]+@)?([^:/]+):(.+)",
            candidate,
        )
    if scp_match is not None:
        host, path = scp_match.groups()
        return (
            f"https://{host.lower()}/"
            f"{path.rstrip('/').removesuffix('.git')}"
        )
    try:
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return candidate.rstrip("/").removesuffix(".git")
    scheme = parsed.scheme.lower()
    if scheme not in DEFAULT_URL_PORTS or not hostname:
        return candidate.rstrip("/").removesuffix(".git")
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = DEFAULT_URL_PORTS[scheme]
    authority = (
        f"{hostname}:{port}"
        if port is not None and port != default_port
        else hostname
    )
    path = parsed.path.rstrip("/").removesuffix(".git")
    return f"{scheme}://{authority}{path}"


def read_required_git_value(
    config: DownloadConfig,
    arguments: Sequence[str],
    field_name: str,
) -> str:
    """Read required checkout metadata or reject an incomplete clone."""

    command = ["git", "-C", config.destination, *arguments]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_environment(config),
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        diagnostic = sanitize_text(result.stderr).strip() or "<none>"
        raise IncompleteCloneError(
            "OpenNeuro destination is incomplete or conflicting: "
            f"could not read {field_name}; git reported {diagnostic}."
        )
    return value


def validate_existing_dataset(config: DownloadConfig) -> None:
    """Reject a conflicting destination or mismatched snapshot checkout."""

    destination = Path(config.destination)
    if not os.path.lexists(destination):
        return
    if not destination.is_dir() or not (destination / ".git").is_dir():
        raise IncompleteCloneError(
            "OpenNeuro destination exists but is not a DataLad dataset: "
            f"{destination}."
        )
    read_required_git_value(
        config,
        ["rev-parse", "--verify", "HEAD"],
        "Git HEAD",
    )
    current_origin = read_required_git_value(
        config,
        ["config", "--get", "remote.origin.url"],
        "origin URL",
    )
    expected_origin = repository_url(config)
    if normalized_repository_url(current_origin) != (
        normalized_repository_url(expected_origin)
    ):
        safe_current = sanitize_text(current_origin)
        safe_expected = sanitize_text(expected_origin)
        raise FileExistsError(
            "OpenNeuro destination belongs to a different repository: "
            f"expected {safe_expected}, but found {safe_current}."
        )
    read_required_git_value(
        config,
        ["config", "--get", "annex.uuid"],
        "git-annex UUID",
    )
    if config.version is None:
        return
    current_version = read_required_git_value(
        config,
        ["describe", "--tags", "--exact-match"],
        "exact version tag",
    )
    if current_version != config.version:
        raise FileExistsError(
            f"OpenNeuro destination is not at requested version "
            f"{config.version!r}; current exact tag is "
            f"{current_version!r}."
        )


def clone_command(config: DownloadConfig) -> list[str]:
    """Build the version-aware DataLad clone command."""

    command = ["datalad", "install"]
    if config.version is not None:
        command.extend(["--branch", config.version])
    command.extend(
        ["--source", repository_url(config), config.destination]
    )
    return command




def create_clone_workspace(config: DownloadConfig) -> CloneWorkspace:
    """Create invocation-owned sibling staging when cloning is required."""

    destination = Path(config.destination)
    if os.path.lexists(destination):
        return CloneWorkspace(destination, None, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{destination.name}{STAGING_DIRECTORY_SUFFIX}"
    staging_root = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=prefix,
        )
    )
    staged_destination = staging_root / STAGED_DATASET_NAME
    log_event(
        "clone_staging_created",
        path=str(staging_root),
    )
    return CloneWorkspace(
        destination,
        staging_root,
        staged_destination,
    )


def remove_owned_staging_path(
    workspace: CloneWorkspace,
    path: Path,
) -> None:
    """Remove one path only when it belongs to this clone workspace."""

    staging_root = workspace.staging_root
    if staging_root is None or (
        path != staging_root and staging_root not in path.parents
    ):
        raise ValueError(
            "refusing to remove a path outside invocation-owned staging: "
            f"{path}."
        )
    if not os.path.lexists(path):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def cleanup_clone_workspace(workspace: CloneWorkspace) -> None:
    """Remove only the temporary root owned by this invocation."""

    if workspace.staging_root is None:
        return
    remove_owned_staging_path(workspace, workspace.staging_root)
    log_event(
        "clone_staging_removed",
        path=str(workspace.staging_root),
    )


def staged_clone_is_valid(
    config: DownloadConfig,
    workspace: CloneWorkspace,
) -> bool:
    """Return whether staged clone metadata is complete and compatible."""

    staged_destination = workspace.staged_destination
    if staged_destination is None or not os.path.lexists(staged_destination):
        return False
    staged_config = replace(
        config,
        destination=str(staged_destination),
    )
    try:
        validate_existing_dataset(staged_config)
    except IncompleteCloneError:
        return False
    return os.path.lexists(staged_destination)


def destination_race_error(workspace: CloneWorkspace) -> FileExistsError:
    """Build the terminal error used when the final path has appeared."""

    return FileExistsError(
        "OpenNeuro destination appeared while cloning; preserving it "
        f"without promotion: {workspace.destination}."
    )


def rename_without_replacement(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing another path."""

    if sys.platform == "win32":
        source.rename(destination)
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "atomic no-clobber clone promotion is unsupported on "
            f"platform {sys.platform!r}."
        )
    libc = ctypes.CDLL(None, use_errno=True)
    rename_at_two = getattr(libc, "renameat2", None)
    if rename_at_two is None:
        raise RuntimeError(
            "the system C library does not provide atomic no-clobber "
            "clone promotion."
        )
    rename_at_two.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_at_two.restype = ctypes.c_int
    result = rename_at_two(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def promote_staged_dataset(workspace: CloneWorkspace) -> None:
    """Promote a validated staged clone without replacing a destination."""

    staged_destination = workspace.staged_destination
    if staged_destination is None:
        raise ValueError("clone workspace does not contain staging.")
    if os.path.lexists(workspace.destination):
        raise destination_race_error(workspace)
    try:
        rename_without_replacement(
            staged_destination,
            workspace.destination,
        )
    except OSError as error:
        if os.path.lexists(workspace.destination):
            raise destination_race_error(workspace) from error
        raise
    log_event(
        "clone_staging_promoted",
        destination=str(workspace.destination),
    )


def install_dataset_once(
    config: DownloadConfig,
    runner: CommandRunner,
    workspace: CloneWorkspace,
) -> None:
    """Install through owned staging or reuse the initial destination."""

    staged_destination = workspace.staged_destination
    if staged_destination is None:
        if not os.path.lexists(workspace.destination):
            raise FileNotFoundError(
                "OpenNeuro destination present at startup was removed: "
                f"{workspace.destination}."
            )
        validate_existing_dataset(config)
        return
    if os.path.lexists(workspace.destination):
        raise destination_race_error(workspace)
    if os.path.lexists(staged_destination):
        if staged_clone_is_valid(config, workspace):
            promote_staged_dataset(workspace)
            return
        log_event(
            "incomplete_clone_staging_removed",
            path=str(staged_destination),
        )
        remove_owned_staging_path(workspace, staged_destination)
    staged_config = replace(
        config,
        destination=str(staged_destination),
    )
    try:
        runner(
            clone_command(staged_config),
            subprocess_environment(staged_config),
        )
    except NetworkCommandError:
        if staged_clone_is_valid(config, workspace):
            promote_staged_dataset(workspace)
            return
        raise
    if not staged_clone_is_valid(config, workspace):
        raise NetworkCommandError(
            "OpenNeuro clone command completed without a reusable staged "
            f"dataset at {staged_destination}."
        )
    promote_staged_dataset(workspace)


def retrieve_content_once(
    config: DownloadConfig,
    runner: CommandRunner,
) -> None:
    """Retrieve selected annexed dataset content exactly once."""

    command = [
        "git",
        "-C",
        config.destination,
        "annex",
        "get",
        "--jobs",
        str(config.max_workers),
        *annex_exclude_arguments(config.exclude_patterns),
        ".",
    ]
    runner(command, subprocess_environment(config))


def annex_exclude_arguments(patterns: tuple[str, ...]) -> list[str]:
    """Build git-annex matching options for excluded relative paths."""

    return [f"--exclude={pattern}" for pattern in patterns]


def missing_annex_content(
    config: DownloadConfig,
    runner: CommandRunner = run_command,
) -> list[str]:
    """Return annexed file paths whose content is not present locally."""

    command = [
        "git",
        "-C",
        config.destination,
        "annex",
        "find",
        "--not",
        "--in=here",
        *annex_exclude_arguments(config.exclude_patterns),
    ]
    output = runner(command, subprocess_environment(config))
    return [line for line in output.splitlines() if line.strip()]


def retrieve_and_verify_content_once(
    config: DownloadConfig,
    runner: CommandRunner,
) -> None:
    """Retrieve annex content and report incomplete transfer as retryable."""

    retrieve_content_once(config, runner)
    missing = missing_annex_content(config, runner)
    if missing:
        raise NetworkCommandError(
            "download verification found "
            f"{len(missing)} unavailable annex files; first missing "
            f"file: {missing[0]}."
        )


def create_node_manager(
    config: DownloadConfig,
) -> MihomoNodeManager | None:
    """Create the optional shared Mihomo ranking manager."""

    if config.mihomo is None:
        return None
    if config.proxy_url is None:
        raise ValueError("Mihomo ranking requires a proxy URL.")
    return MihomoNodeManager(
        config.mihomo,
        config.proxy_url,
        repository_url(config),
        {},
    )


def print_dry_run(config: DownloadConfig) -> None:
    """Describe locally discoverable pending OpenNeuro work."""

    destination = Path(config.destination)
    print()
    print("Dry-run summary")
    print("----------------------------------------")
    print(f"Dataset          : {config.dataset_id}")
    print(f"Version          : {config.version or 'default snapshot'}")
    if not destination.exists():
        print("Repository clone : pending")
        print("Annex content    : unknown until cloned")
        return
    validate_existing_dataset(config)
    missing = missing_annex_content(config)
    print("Repository clone : complete")
    print(f"Annex files pending: {len(missing)}")
    for path in missing:
        print(path)


def run(
    config: DownloadConfig,
    runner: CommandRunner = run_command,
) -> None:
    """Execute the configured OpenNeuro inspection or download."""

    check_dependencies()
    if config.dry_run:
        print_dry_run(config)
        return
    workspace = create_clone_workspace(config)
    node_manager = None
    try:
        node_manager = create_node_manager(config)
        run_with_retries(
            config,
            "OpenNeuro repository clone",
            lambda: install_dataset_once(config, runner, workspace),
            transport="datalad",
            node_manager=node_manager,
        )
        run_with_retries(
            config,
            "OpenNeuro annex retrieval",
            lambda: retrieve_and_verify_content_once(config, runner),
            transport="datalad",
            node_manager=node_manager,
        )
        print()
        print(f"Download location: {config.destination}")
        print("Verification     : complete")
    finally:
        try:
            if node_manager is not None:
                node_manager.close()
        finally:
            cleanup_clone_workspace(workspace)


def diagnostic_configuration(config: DownloadConfig) -> dict[str, Any]:
    """Return safe OpenNeuro configuration fields for diagnostics."""

    return {
        "destination": config.destination,
        "dry_run": config.dry_run,
        "endpoint": config.endpoint,
        "exclude_patterns": config.exclude_patterns,
        "max_workers": config.max_workers,
        "provider": "openneuro",
        "proxy_url": config.proxy_url,
        "repo": config.dataset_id,
        "retry_attempts": config.retry_attempts,
        "retry_base_delay": config.retry_base_delay,
        "retry_max_delay": config.retry_max_delay,
        "timeout": config.timeout,
        "transport": "datalad",
        "version": config.version,
    }


def main() -> int:
    """Run the OpenNeuro downloader and return a shell-compatible status."""

    try:
        config = load_config_from_environment()
        log_path = configure_diagnostics(
            "openneuro",
            diagnostic_configuration(config),
        )
        print(f"Debug log     : {log_path}")
        run(config)
        log_event("download_completed", status="success")
    except (DownloadInterrupted, KeyboardInterrupt) as error:
        log_exception("download_interrupted", error)
        path = current_log_path()
        print("ERROR: download interrupted.", file=sys.stderr)
        if path is not None:
            print(f"Debug log: {path}", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        log_exception("download_failed", error)
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        path = current_log_path()
        if path is not None:
            print(f"Debug log: {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
