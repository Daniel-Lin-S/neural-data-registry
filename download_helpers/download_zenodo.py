"""Download and verify one Zenodo record.

Input
-----
Configuration is read from ``DOWNLOAD_*`` variables exported by
``download_zenodo.sh``. ``DOWNLOAD_REPO_ID`` accepts a numeric record ID or a
Zenodo record URL. Private-record authentication inherits ``ZENODO_TOKEN``.

Output
------
Every advertised record file is written beneath the absolute destination.
Transfers resume from sibling ``.part`` files and are verified against the
advertised size and checksum before atomic publication.
"""

from __future__ import annotations

import os
import ssl
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from download_diagnostics import (
    DownloadInterrupted,
    configure_diagnostics,
    current_log_path,
    log_event,
    log_exception,
)
from download_http_manifest import (
    HTTPFile,
    download_manifest,
    pending_files,
    print_dry_run_summary,
    validate_checksum,
    validate_download_url,
    validate_relative_path,
)
from download_retry import run_with_retries
from mihomo_ranker import MihomoConfig, MihomoNodeManager

DEFAULT_MIHOMO_PROBE_TIMEOUT = 8.0
ZENODO_RECORD_PATH_MARKERS = frozenset({"record", "records"})

ClientFactory = Callable[[], httpx.Client]


@dataclass(frozen=True)
class DownloadConfig:
    """Store validated Zenodo download configuration.

    Parameters
    ----------
    record_id : str
        Canonical numeric Zenodo record ID.
    destination : str
        Absolute destination directory.
    api_base : str
        Zenodo-compatible API base URL.
    max_workers : int
        Number of concurrent file transfers.
    timeout : float
        HTTP request timeout in seconds.
    dry_run : bool
        Whether to report pending files without downloading.
    retry_attempts : int
        Maximum attempts per operation; zero is unlimited.
    retry_base_delay : float
        Initial retry delay in seconds.
    retry_max_delay : float
        Maximum retry delay in seconds.
    proxy_url : str or None
        Explicit proxy URL, optional.
    token : str or None
        Zenodo access token, optional.
    mihomo : MihomoConfig or None
        Mihomo ranking configuration, optional.
    """

    record_id: str
    destination: str
    api_base: str
    max_workers: int
    timeout: float
    dry_run: bool
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float
    proxy_url: str | None
    token: str | None
    mihomo: MihomoConfig | None


def optional_environment(name: str) -> str | None:
    """Return a non-empty environment variable, or ``None``."""

    value = os.environ.get(name, "")
    return value or None


def normalize_api_base(value: str) -> str:
    """Return a Zenodo API base for an endpoint or explicit API URL."""

    endpoint = value.rstrip("/")
    if endpoint.endswith("/api"):
        return endpoint
    return f"{endpoint}/api"


def parse_record_id(value: str) -> str:
    """Extract a numeric Zenodo record ID from an ID or record URL."""

    candidate = value.strip()
    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "expected an HTTP or HTTPS Zenodo record URL, but got "
                f"{value!r}."
            )
        parts = [part for part in parsed.path.split("/") if part]
        matches = [
            parts[index + 1]
            for index, part in enumerate(parts[:-1])
            if part in ZENODO_RECORD_PATH_MARKERS
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected a Zenodo record ID in URL {value!r}."
            )
        candidate = matches[0]
    if not candidate.isdigit():
        raise ValueError(
            f"expected a numeric Zenodo record ID, but got {candidate!r}."
        )
    return candidate


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

    config = DownloadConfig(
        record_id=parse_record_id(os.environ["DOWNLOAD_REPO_ID"]),
        destination=os.environ["DOWNLOAD_DEST"],
        api_base=normalize_api_base(os.environ["DOWNLOAD_ENDPOINT"]),
        max_workers=int(os.environ["DOWNLOAD_MAX_WORKERS"]),
        timeout=float(os.environ["DOWNLOAD_TIMEOUT"]),
        dry_run=os.environ["DOWNLOAD_DRY_RUN"] == "1",
        retry_attempts=int(os.environ["DOWNLOAD_RETRY_ATTEMPTS"]),
        retry_base_delay=float(
            os.environ["DOWNLOAD_RETRY_BASE_DELAY"]
        ),
        retry_max_delay=float(os.environ["DOWNLOAD_RETRY_MAX_DELAY"]),
        proxy_url=optional_environment("DOWNLOAD_PROXY_URL"),
        token=optional_environment("ZENODO_TOKEN"),
        mihomo=load_mihomo_config(),
    )
    validate_config(config)
    return config


def validate_config(config: DownloadConfig) -> None:
    """Reject invalid Zenodo downloader configuration."""

    if not os.path.isabs(config.destination):
        raise ValueError(
            "expected an absolute destination path, but got "
            f"{config.destination!r}."
        )
    parsed_api = urlparse(config.api_base)
    if parsed_api.scheme not in {"http", "https"} or not parsed_api.netloc:
        raise ValueError(
            "expected an HTTP or HTTPS Zenodo API base URL, but got "
            f"{config.api_base!r}."
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
            "not interrupt concurrent file transfers."
        )


def request_headers(token: str | None) -> dict[str, str]:
    """Build Zenodo headers without exposing authentication material."""

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def create_tls_context() -> ssl.SSLContext:
    """Create the TLS 1.2 context used by Zenodo requests."""

    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def create_client(config: DownloadConfig) -> httpx.Client:
    """Create one explicit Zenodo HTTP client."""

    return httpx.Client(
        headers=request_headers(config.token),
        proxy=config.proxy_url,
        verify=create_tls_context(),
        follow_redirects=True,
        timeout=httpx.Timeout(config.timeout),
        trust_env=False,
    )


def record_url(config: DownloadConfig) -> str:
    """Build the Zenodo record API URL."""

    record_id = quote(config.record_id, safe="")
    return f"{config.api_base}/records/{record_id}"


def require_payload(response: httpx.Response) -> dict[str, Any]:
    """Return a validated JSON object from a Zenodo response."""

    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(
            "expected the Zenodo API response to be an object, but got "
            f"{type(payload).__name__}."
        )
    return payload


def file_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return files from current or legacy Zenodo response layouts."""

    files = payload.get("files")
    if isinstance(files, list):
        items = files
    elif isinstance(files, dict):
        entries = files.get("entries")
        if not isinstance(entries, dict):
            raise TypeError(
                "expected Zenodo files.entries to be an object."
            )
        items = list(entries.values())
    else:
        raise TypeError(
            "expected Zenodo files to be a list or entries object."
        )
    if not all(isinstance(item, dict) for item in items):
        raise TypeError("expected every Zenodo file to be an object.")
    return items


def file_from_item(item: dict[str, Any]) -> HTTPFile:
    """Build and validate one Zenodo manifest record."""

    name = item.get("key") or item.get("filename")
    relative_path = validate_relative_path(name, "Zenodo")
    size = item.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(
            f"expected a non-negative size for {relative_path}, but got "
            f"{size!r}."
        )
    links = item.get("links")
    if not isinstance(links, dict):
        raise TypeError(
            f"expected links for Zenodo file {relative_path}."
        )
    download_url = validate_download_url(
        links.get("content") or links.get("self"),
        relative_path,
    )
    checksum = validate_checksum(item.get("checksum"), relative_path)
    return HTTPFile(relative_path, size, download_url, checksum)


def collect_manifest_once(
    config: DownloadConfig,
    client_factory: ClientFactory,
) -> list[HTTPFile]:
    """Fetch and validate the Zenodo record manifest exactly once."""

    with client_factory() as client:
        payload = require_payload(client.get(record_url(config)))
    manifest = {}
    for item in file_items(payload):
        record = file_from_item(item)
        if record.relative_path in manifest:
            raise ValueError(
                "Zenodo advertised duplicate path "
                f"{record.relative_path!r}."
            )
        manifest[record.relative_path] = record
    if not manifest:
        raise ValueError(
            f"Zenodo record {config.record_id!r} has no downloadable files."
        )
    return [manifest[path] for path in sorted(manifest)]


def collect_manifest(
    config: DownloadConfig,
    node_manager: MihomoNodeManager | None,
    client_factory: ClientFactory | None = None,
) -> list[HTTPFile]:
    """Collect a Zenodo manifest with retry and failover behavior."""

    factory = client_factory or (lambda: create_client(config))
    return run_with_retries(
        config,
        "Zenodo manifest",
        lambda: collect_manifest_once(config, factory),
        node_manager=node_manager,
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
        record_url(config),
        request_headers(config.token),
    )


def authentication_status(token: str | None) -> str:
    """Describe Zenodo authentication without exposing token material."""

    return "configured" if token else "not configured"


def run(config: DownloadConfig) -> None:
    """Execute the configured Zenodo inspection or download."""

    print(
        f"Authentication: {authentication_status(config.token)}",
        flush=True,
    )
    node_manager = create_node_manager(config)
    factory = lambda: create_client(config)
    try:
        manifest = collect_manifest(config, node_manager, factory)
        if config.dry_run:
            print_dry_run_summary(config, manifest)
            return
        Path(config.destination).mkdir(parents=True, exist_ok=True)
        download_manifest(
            config,
            manifest,
            node_manager,
            factory,
            "Zenodo",
        )
        incomplete = pending_files(config, manifest)
        if incomplete:
            raise RuntimeError(
                "download verification found "
                f"{len(incomplete)} pending files; first pending file: "
                f"{incomplete[0].relative_path}."
            )
        print()
        print(f"Download location: {config.destination}")
        print("Verification     : complete")
    finally:
        if node_manager is not None:
            node_manager.close()


def diagnostic_configuration(config: DownloadConfig) -> dict[str, Any]:
    """Return safe Zenodo configuration fields for diagnostics."""

    return {
        "destination": config.destination,
        "dry_run": config.dry_run,
        "endpoint": config.api_base,
        "max_workers": config.max_workers,
        "provider": "zenodo",
        "proxy_url": config.proxy_url,
        "repo": config.record_id,
        "retry_attempts": config.retry_attempts,
        "retry_base_delay": config.retry_base_delay,
        "retry_max_delay": config.retry_max_delay,
        "timeout": config.timeout,
        "transport": "http",
    }


def main() -> int:
    """Run the Zenodo downloader and return a shell-compatible status."""

    try:
        config = load_config_from_environment()
        log_path = configure_diagnostics(
            "zenodo",
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
