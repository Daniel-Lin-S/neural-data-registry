"""Download and verify one OSF storage tree.

Input
-----
Configuration is read from ``DOWNLOAD_*`` environment variables exported by
``download_osf.sh``. ``DOWNLOAD_PROJECT`` accepts an OSF node ID or project
URL. Public projects need no credentials; private access inherits
``OSF_TOKEN``.

Output
------
Files advertised by the selected OSF storage provider are written beneath the
absolute destination. Each transfer uses a sibling ``.part`` file, resumes
with an HTTP range request, verifies the advertised byte size, and is renamed
atomically after completion. A final manifest check rejects missing or
size-mismatched files.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import sys
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlparse

import httpx

from download_retry import run_with_retries
from mihomo_ranker import MihomoConfig, MihomoNodeManager


DEFAULT_PAGE_SIZE = 100
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
OSF_PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
OSF_URL_HOSTS = frozenset({"osf.io", "www.osf.io"})

ClientFactory = Callable[[], httpx.Client]


@dataclass(frozen=True)
class OSFFile:
    """Describe one file advertised by the OSF API.

    Parameters
    ----------
    relative_path : str
        POSIX path relative to the destination.
    size : int
        Expected file size in bytes.
    download_url : str
        OSF download action URL.
    """

    relative_path: str
    size: int
    download_url: str


@dataclass(frozen=True)
class DownloadConfig:
    """Store validated OSF download configuration.

    Parameters
    ----------
    project_id : str
        Canonical OSF node ID.
    destination : str
        Absolute destination directory.
    api_base : str
        OSF-compatible API v2 base URL.
    storage : str
        OSF storage provider name.
    max_workers : int
        Number of files downloaded concurrently.
    timeout : float
        HTTP timeout in seconds.
    dry_run : bool
        Whether to inspect pending files without downloading.
    retry_attempts : int
        Maximum attempts per manifest or file operation; zero is unlimited.
    retry_base_delay : float
        Initial retry delay in seconds.
    retry_max_delay : float
        Maximum retry delay in seconds.
    proxy_url : str or None
        Explicit HTTPX proxy URL, optional.
    token : str or None
        OSF personal access token, optional.
    mihomo : MihomoConfig or None
        Mihomo ranking configuration, optional.
    """

    project_id: str
    destination: str
    api_base: str
    storage: str
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
    if value:
        return value
    return None


def parse_project_id(value: str) -> str:
    """Extract and validate an OSF node ID from an ID or project URL.

    Parameters
    ----------
    value : str
        OSF node ID or ``https://osf.io/ID/...`` URL.

    Returns
    -------
    str
        Lower-case OSF node ID.
    """

    candidate = value.strip()
    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "expected an HTTP or HTTPS OSF project URL, but got "
                f"{value!r}."
            )
        if parsed.hostname not in OSF_URL_HOSTS:
            raise ValueError(
                "expected an osf.io project URL, but got "
                f"host {parsed.hostname!r}."
            )
        segments = [part for part in parsed.path.split("/") if part]
        if not segments:
            raise ValueError(
                f"expected an OSF project ID in URL {value!r}."
            )
        candidate = segments[0]
    if not OSF_PROJECT_ID_PATTERN.fullmatch(candidate):
        raise ValueError(
            "expected an alphanumeric OSF project ID, but got "
            f"{candidate!r}."
        )
    return candidate.lower()


def load_mihomo_config() -> MihomoConfig | None:
    """Load optional Mihomo ranking settings exported by the shell."""

    group = optional_environment("DOWNLOAD_MIHOMO_GROUP")
    speed_test_url = optional_environment(
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL"
    )
    if group is None or speed_test_url is None:
        return None
    controller = optional_environment("DOWNLOAD_MIHOMO_CONTROLLER")
    if controller is None:
        raise ValueError(
            "Mihomo ranking requires DOWNLOAD_MIHOMO_CONTROLLER."
        )
    return MihomoConfig(
        controller_url=controller,
        group_name=group,
        node_marker=(
            optional_environment("DOWNLOAD_MIHOMO_NODE_MARKER") or ""
        ),
        speed_test_url=speed_test_url,
        probe_timeout=float(
            os.environ["DOWNLOAD_MIHOMO_PROBE_TIMEOUT"]
        ),
        secret=optional_environment("MIHOMO_SECRET"),
    )


def load_config_from_environment() -> DownloadConfig:
    """Load and validate downloader configuration from the environment."""

    config = DownloadConfig(
        project_id=parse_project_id(os.environ["DOWNLOAD_PROJECT"]),
        destination=os.environ["DOWNLOAD_DEST"],
        api_base=os.environ["DOWNLOAD_ENDPOINT"].rstrip("/"),
        storage=os.environ["DOWNLOAD_STORAGE"],
        max_workers=int(os.environ["DOWNLOAD_MAX_WORKERS"]),
        timeout=float(os.environ["DOWNLOAD_TIMEOUT"]),
        dry_run=os.environ["DOWNLOAD_DRY_RUN"] == "1",
        retry_attempts=int(os.environ["DOWNLOAD_RETRY_ATTEMPTS"]),
        retry_base_delay=float(
            os.environ["DOWNLOAD_RETRY_BASE_DELAY"]
        ),
        retry_max_delay=float(os.environ["DOWNLOAD_RETRY_MAX_DELAY"]),
        proxy_url=optional_environment("DOWNLOAD_PROXY_URL"),
        token=optional_environment("OSF_TOKEN"),
        mihomo=load_mihomo_config(),
    )
    validate_config(config)
    return config


def validate_config(config: DownloadConfig) -> None:
    """Reject invalid OSF downloader configuration."""

    if not os.path.isabs(config.destination):
        raise ValueError(
            "expected an absolute destination path, but got "
            f"{config.destination!r}."
        )
    parsed_api = urlparse(config.api_base)
    if parsed_api.scheme not in {"http", "https"} or not parsed_api.netloc:
        raise ValueError(
            "expected an HTTP or HTTPS OSF API base URL, but got "
            f"{config.api_base!r}."
        )
    if not OSF_PROJECT_ID_PATTERN.fullmatch(config.storage):
        raise ValueError(
            "expected an alphanumeric OSF storage name, but got "
            f"{config.storage!r}."
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
    """Build OSF API headers without logging authentication material."""

    headers = {"Accept": "application/vnd.api+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def create_tls_context() -> ssl.SSLContext:
    """Create the TLS 1.2 context used by OSF HTTP requests."""

    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def create_client(config: DownloadConfig) -> httpx.Client:
    """Create one explicit OSF HTTP client."""

    return httpx.Client(
        headers=request_headers(config.token),
        proxy=config.proxy_url,
        verify=create_tls_context(),
        follow_redirects=True,
        timeout=httpx.Timeout(config.timeout),
        trust_env=False,
    )


def storage_root_url(config: DownloadConfig) -> str:
    """Build the OSF storage root listing URL."""

    project_id = quote(config.project_id, safe="")
    storage = quote(config.storage, safe="")
    return (
        f"{config.api_base}/nodes/{project_id}/files/{storage}/"
    )


def project_probe_url(config: DownloadConfig) -> str:
    """Build the OSF project metadata URL used for Mihomo probes."""

    project_id = quote(config.project_id, safe="")
    return f"{config.api_base}/nodes/{project_id}/"


def paginated_url(url: str) -> str:
    """Add the maximum supported page size to an OSF listing URL."""

    return str(
        httpx.URL(url).copy_merge_params(
            {"page[size]": str(DEFAULT_PAGE_SIZE)}
        )
    )


def require_payload(response: httpx.Response) -> dict[str, Any]:
    """Return a validated JSON object from an OSF API response."""

    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(
            "expected the OSF API response to be an object, but got "
            f"{type(payload).__name__}."
        )
    return payload


def related_files_url(entry: dict[str, Any]) -> str | None:
    """Return a folder's related-files URL from a JSON:API entry."""

    relationships = entry.get("relationships")
    if not isinstance(relationships, dict):
        return None
    files = relationships.get("files")
    if not isinstance(files, dict):
        return None
    links = files.get("links")
    if not isinstance(links, dict):
        return None
    related = links.get("related")
    if isinstance(related, str):
        return related
    if isinstance(related, dict):
        href = related.get("href")
        if isinstance(href, str):
            return href
    return None


def validate_relative_path(value: Any) -> str:
    """Return a safe, normalized POSIX path advertised by OSF."""

    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError(
            f"expected a non-empty OSF materialized path, but got {value!r}."
        )
    relative = value.lstrip("/")
    path = PurePosixPath(relative)
    unsafe_part = any(part in {"", ".", ".."} for part in path.parts)
    if path.is_absolute() or unsafe_part:
        raise ValueError(
            f"refusing unsafe OSF materialized path {value!r}."
        )
    return path.as_posix()


def file_from_entry(entry: dict[str, Any]) -> OSFFile:
    """Build and validate one downloadable file record."""

    attributes = entry.get("attributes")
    links = entry.get("links")
    if not isinstance(attributes, dict) or not isinstance(links, dict):
        raise ValueError(
            "expected every OSF file to contain attributes and links objects."
        )
    relative_path = validate_relative_path(
        attributes.get("materialized_path")
    )
    size = attributes.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(
            f"expected a non-negative size for {relative_path}, but got "
            f"{size!r}."
        )
    download_url = links.get("download")
    parsed_url = urlparse(download_url) if isinstance(download_url, str) \
        else None
    if (
        parsed_url is None
        or parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
    ):
        raise ValueError(
            f"expected a download URL for {relative_path}, but got "
            f"{download_url!r}."
        )
    return OSFFile(relative_path, size, download_url)


def next_page_url(payload: dict[str, Any]) -> str | None:
    """Return and validate the next JSON:API collection page."""

    links = payload.get("links")
    if links is None:
        return None
    if not isinstance(links, dict):
        raise ValueError("expected OSF pagination links to be an object.")
    next_url = links.get("next")
    if next_url is None:
        return None
    if not isinstance(next_url, str) or not next_url:
        raise ValueError(
            f"expected a non-empty OSF next-page URL, but got {next_url!r}."
        )
    return next_url


def collect_manifest_once(
    config: DownloadConfig,
    client_factory: ClientFactory,
) -> list[OSFFile]:
    """Walk the selected OSF storage tree exactly once."""

    pending_urls = [paginated_url(storage_root_url(config))]
    visited_urls: set[str] = set()
    manifest: dict[str, OSFFile] = {}
    with client_factory() as client:
        while pending_urls:
            current_url = pending_urls.pop()
            if current_url in visited_urls:
                continue
            visited_urls.add(current_url)
            payload = require_payload(client.get(current_url))
            entries = payload.get("data")
            if not isinstance(entries, list):
                raise ValueError(
                    "expected OSF collection data to be a list, but got "
                    f"{type(entries).__name__}."
                )
            for raw_entry in entries:
                if not isinstance(raw_entry, dict):
                    raise ValueError(
                        "expected every OSF collection entry to be an object."
                    )
                attributes = raw_entry.get("attributes")
                if not isinstance(attributes, dict):
                    raise ValueError(
                        "expected every OSF entry to contain attributes."
                    )
                kind = attributes.get("kind")
                if kind == "file":
                    record = file_from_entry(raw_entry)
                    if record.relative_path in manifest:
                        raise ValueError(
                            "OSF advertised duplicate path "
                            f"{record.relative_path!r}."
                        )
                    manifest[record.relative_path] = record
                elif kind == "folder":
                    folder_url = related_files_url(raw_entry)
                    if folder_url is None:
                        raise ValueError(
                            "expected every OSF folder to advertise its "
                            "related files URL."
                        )
                    pending_urls.append(paginated_url(folder_url))
                else:
                    raise ValueError(
                        "expected OSF entry kind 'file' or 'folder', but got "
                        f"{kind!r}."
                    )
            next_url = next_page_url(payload)
            if next_url is not None:
                pending_urls.append(next_url)
    if not manifest:
        raise ValueError(
            f"OSF project {config.project_id!r} storage "
            f"{config.storage!r} contains no downloadable files."
        )
    return [manifest[path] for path in sorted(manifest)]


def collect_manifest(
    config: DownloadConfig,
    node_manager: MihomoNodeManager | None,
    client_factory: ClientFactory | None = None,
) -> list[OSFFile]:
    """Collect an OSF manifest with shared retry and failover behavior."""

    factory = client_factory or (lambda: create_client(config))
    return run_with_retries(
        config,
        "OSF manifest",
        lambda: collect_manifest_once(config, factory),
        node_manager=node_manager,
    )


def destination_path(destination: str, relative_path: str) -> Path:
    """Resolve a manifest path while preventing destination traversal."""

    root = Path(destination).resolve()
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    resolved = candidate.resolve()
    if os.path.commonpath([root, resolved]) != str(root):
        raise ValueError(
            f"refusing OSF path outside destination: {relative_path!r}."
        )
    return candidate


def partial_path(destination: Path) -> Path:
    """Return the sibling partial-transfer path for a destination file."""

    return destination.with_name(f"{destination.name}.part")


def validate_existing_file(path: Path, expected_size: int) -> bool:
    """Return whether a completed destination file can be skipped."""

    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise FileExistsError(
            f"expected a regular destination file, but got {path}."
        )
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise FileExistsError(
            f"existing file {path} has {actual_size} bytes; expected "
            f"{expected_size}. Move it aside before downloading."
        )
    return True


def validate_content_range(response: httpx.Response, offset: int) -> None:
    """Require a resumed response to begin at the requested byte offset."""

    value = response.headers.get("Content-Range", "")
    if not value.startswith(f"bytes {offset}-"):
        raise httpx.RemoteProtocolError(
            "expected resumed response Content-Range to start with "
            f"'bytes {offset}-', but got {value!r}."
        )


def write_response_body(
    response: httpx.Response,
    output_path: Path,
    mode: str,
) -> None:
    """Stream one HTTP response body to a partial file."""

    with output_path.open(mode) as output:
        for chunk in response.iter_bytes(DOWNLOAD_CHUNK_SIZE):
            if chunk:
                output.write(chunk)


def download_file_once(
    config: DownloadConfig,
    record: OSFFile,
    client_factory: ClientFactory,
) -> Path:
    """Resume, verify, and atomically publish one OSF file once."""

    destination = destination_path(
        config.destination,
        record.relative_path,
    )
    if validate_existing_file(destination, record.size):
        return destination
    temporary = partial_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists() and (
        temporary.is_symlink() or not temporary.is_file()
    ):
        raise FileExistsError(
            f"expected a regular partial file, but got {temporary}."
        )
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > record.size:
        raise OSError(
            f"partial file {temporary} has {offset} bytes; expected at most "
            f"{record.size}. Move it aside before downloading."
        )
    if offset == record.size:
        os.replace(temporary, destination)
        return destination

    headers = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with client_factory() as client:
        with client.stream(
            "GET",
            record.download_url,
            headers=headers,
        ) as response:
            response.raise_for_status()
            mode = "wb"
            if offset and response.status_code == 206:
                validate_content_range(response, offset)
                mode = "ab"
            write_response_body(response, temporary, mode)

    actual_size = temporary.stat().st_size
    if actual_size != record.size:
        raise httpx.RemoteProtocolError(
            f"incomplete OSF file {record.relative_path!r}: received "
            f"{actual_size} bytes, expected {record.size}."
        )
    os.replace(temporary, destination)
    return destination


def download_file(
    config: DownloadConfig,
    record: OSFFile,
    node_manager: MihomoNodeManager | None,
    client_factory: ClientFactory | None = None,
) -> Path:
    """Download one OSF file with shared retry and failover behavior."""

    factory = client_factory or (lambda: create_client(config))
    return run_with_retries(
        config,
        f"OSF file {record.relative_path}",
        lambda: download_file_once(config, record, factory),
        node_manager=node_manager,
    )


def pending_files(
    config: DownloadConfig,
    manifest: Iterable[OSFFile],
) -> list[OSFFile]:
    """Return files absent from the destination or having a wrong size."""

    pending = []
    for record in manifest:
        path = destination_path(config.destination, record.relative_path)
        if not path.is_file() or path.stat().st_size != record.size:
            pending.append(record)
    return pending


def print_dry_run_summary(
    config: DownloadConfig,
    manifest: list[OSFFile],
) -> None:
    """Print pending OSF paths and aggregate byte size."""

    pending = pending_files(config, manifest)
    total_bytes = sum(record.size for record in pending)
    print()
    print("Dry-run summary")
    print("----------------------------------------")
    print(f"Files in manifest : {len(manifest)}")
    print(f"Files to download : {len(pending)}")
    print(f"GiB to download   : {total_bytes / 1024**3:.2f}")
    print()
    for record in pending:
        print(
            f"{record.size / 1024**2:10.2f} MiB  "
            f"{record.relative_path}"
        )


def download_manifest(
    config: DownloadConfig,
    manifest: list[OSFFile],
    node_manager: MihomoNodeManager | None,
) -> None:
    """Download every pending manifest file with bounded concurrency."""

    pending = pending_files(config, manifest)
    if not pending:
        return
    print(f"Downloading {len(pending)} OSF file(s)...")
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(download_file, config, record, node_manager)
            for record in pending
        ]
        for index, (record, future) in enumerate(
            zip(pending, futures),
            1,
        ):
            path = future.result()
            print(
                f"[{index}/{len(pending)}] {record.relative_path} -> "
                f"{path}"
            )


def authentication_status(token: str | None) -> str:
    """Describe OSF authentication without exposing token material."""

    if token:
        return "configured"
    return "not configured"


def create_node_manager(config: DownloadConfig) -> MihomoNodeManager | None:
    """Create the optional shared Mihomo ranking manager."""

    if config.mihomo is None:
        return None
    if config.proxy_url is None:
        raise ValueError("Mihomo ranking requires a proxy URL.")
    return MihomoNodeManager(
        config.mihomo,
        config.proxy_url,
        project_probe_url(config),
        request_headers(config.token),
    )


def run(config: DownloadConfig) -> None:
    """Execute the configured OSF manifest inspection or download."""

    print(
        f"Authentication: {authentication_status(config.token)}",
        flush=True,
    )
    node_manager = create_node_manager(config)
    try:
        manifest = collect_manifest(config, node_manager)
        if config.dry_run:
            print_dry_run_summary(config, manifest)
            return

        Path(config.destination).mkdir(parents=True, exist_ok=True)
        download_manifest(config, manifest, node_manager)
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


def main() -> int:
    """Run the OSF downloader and return a shell-compatible status."""

    try:
        run(load_config_from_environment())
    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
