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
existing git-annex state is reused by later attempts and invocations.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from download_retry import run_with_retries
from mihomo_ranker import MihomoConfig, MihomoNodeManager

DEFAULT_MIHOMO_PROBE_TIMEOUT = 8.0
DATASET_ID_PATTERN = re.compile(r"^ds[0-9]+$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
NETWORK_COMMAND_MARKERS = (
    "broken pipe",
    "connection reset",
    "curl 5",
    "could not resolve",
    "connection timed out",
    "could not resolve host",
    "failed to connect",
    "gnutls",
    "http/2 stream",
    "network is unreachable",
    "operation timed out",
    "remote end hung up",
    "rpc failed",
    "temporary failure in name resolution",
    "tls",
    "unexpected disconnect",
)
TERMINAL_COMMAND_MARKERS = (
    "authentication failed",
    "could not read username",
    "disk quota exceeded",
    "no space left on device",
    "permission denied",
    "repository not found",
    "unknown option",
)

CommandRunner = Callable[[Sequence[str], dict[str, str]], str]


class NetworkCommandError(ConnectionError):
    """Represent a network-related DataLad or git-annex failure."""


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


def optional_environment(name: str) -> str | None:
    """Return a non-empty environment variable, or ``None``."""

    value = os.environ.get(name, "")
    return value or None


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


def classify_command_failure(
    command: Sequence[str],
    return_code: int,
    output: str,
) -> RuntimeError:
    """Translate provider command output into network or terminal failure."""

    sanitized_output = sanitize_text(output).strip()
    message = sanitized_output or "command produced no diagnostic output"
    lower_message = message.lower()
    command_name = Path(command[0]).name
    detail = (
        f"{command_name} exited with status {return_code}: {message}"
    )
    if any(marker in lower_message for marker in TERMINAL_COMMAND_MARKERS):
        return RuntimeError(detail)
    if any(marker in lower_message for marker in NETWORK_COMMAND_MARKERS):
        return NetworkCommandError(detail)
    return RuntimeError(detail)


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


def validate_existing_dataset(config: DownloadConfig) -> None:
    """Reject a conflicting destination or mismatched snapshot checkout."""

    destination = Path(config.destination)
    if not destination.exists():
        return
    if not destination.is_dir() or not (destination / ".git").is_dir():
        raise FileExistsError(
            "OpenNeuro destination exists but is not a DataLad dataset: "
            f"{destination}."
        )
    if config.version is None:
        return
    result = subprocess.run(
        [
            "git",
            "-C",
            config.destination,
            "describe",
            "--tags",
            "--exact-match",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_environment(config),
    )
    current_version = result.stdout.strip()
    if result.returncode != 0 or current_version != config.version:
        raise FileExistsError(
            f"OpenNeuro destination is not at requested version "
            f"{config.version!r}; current exact tag is "
            f"{current_version or '<none>'!r}."
        )


def clone_command(config: DownloadConfig) -> list[str]:
    """Build the version-aware DataLad clone command."""

    command = ["datalad", "install"]
    if config.version is not None:
        command.extend(["--branch", config.version])
    command.extend([repository_url(config), config.destination])
    return command


def install_dataset_once(
    config: DownloadConfig,
    runner: CommandRunner,
) -> None:
    """Install the DataLad repository once or reuse a valid checkout."""

    destination = Path(config.destination)
    if destination.exists():
        validate_existing_dataset(config)
        return
    parent = destination.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    runner(clone_command(config), subprocess_environment(config))
    validate_existing_dataset(config)


def retrieve_content_once(
    config: DownloadConfig,
    runner: CommandRunner,
) -> None:
    """Retrieve all annexed dataset content exactly once."""

    command = [
        "datalad",
        "-C",
        config.destination,
        "get",
        "--recursive",
        "--jobs",
        str(config.max_workers),
        ".",
    ]
    runner(command, subprocess_environment(config))


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
    ]
    output = runner(command, subprocess_environment(config))
    return [line for line in output.splitlines() if line.strip()]


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
    node_manager = create_node_manager(config)
    try:
        run_with_retries(
            config,
            "OpenNeuro repository clone",
            lambda: install_dataset_once(config, runner),
            node_manager=node_manager,
        )
        run_with_retries(
            config,
            "OpenNeuro annex retrieval",
            lambda: retrieve_content_once(config, runner),
            node_manager=node_manager,
        )
        missing = missing_annex_content(config, runner)
        if missing:
            raise RuntimeError(
                "download verification found "
                f"{len(missing)} unavailable annex files; first missing "
                f"file: {missing[0]}."
            )
        print()
        print(f"Download location: {config.destination}")
        print("Verification     : complete")
    finally:
        if node_manager is not None:
            node_manager.close()


def diagnostic_configuration(config: DownloadConfig) -> dict[str, Any]:
    """Return safe OpenNeuro configuration fields for diagnostics."""

    return {
        "destination": config.destination,
        "dry_run": config.dry_run,
        "endpoint": config.endpoint,
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
