"""Download and verify provider-advertised HTTP file manifests.

Input
-----
A validated downloader configuration, a manifest of safe relative paths,
advertised sizes and optional checksums, and an HTTPX client factory.

Output
------
Files are streamed into sibling ``.part`` files, resumed with HTTP ranges,
verified, and atomically renamed beneath the configured destination.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import httpx
from download_retry import run_with_retries

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
CHECKSUM_PATTERN = re.compile(
    r"^(md5|sha1|sha256|sha512):([0-9a-fA-F]+)$"
)

ClientFactory = Callable[[], httpx.Client]


class ManifestDownloadConfig(Protocol):
    """Describe configuration used by the shared manifest downloader."""

    destination: str
    max_workers: int
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float


class NodeManager(Protocol):
    """Describe Mihomo operations used during manifest downloads."""

    def prepare_attempt(self) -> str:
        """Prepare a node for an attempted provider operation."""

    def failover(self) -> str | None:
        """Choose another node after a transient network failure."""


@dataclass(frozen=True)
class HTTPFile:
    """Describe one downloadable provider file.

    Parameters
    ----------
    relative_path : str
        POSIX path relative to the destination.
    size : int
        Expected file size in bytes.
    download_url : str
        HTTP or HTTPS content URL.
    checksum : str or None, optional
        ``algorithm:hex-digest`` integrity value, default ``None``.
    """

    relative_path: str
    size: int
    download_url: str
    checksum: str | None = None


def validate_relative_path(value: object, provider: str) -> str:
    """Return a safe normalized provider path below the destination."""

    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError(
            f"expected a non-empty {provider} path, but got {value!r}."
        )
    path = PurePosixPath(value.lstrip("/"))
    unsafe_part = any(part in {"", ".", ".."} for part in path.parts)
    if path.is_absolute() or unsafe_part:
        raise ValueError(f"refusing unsafe {provider} path {value!r}.")
    return path.as_posix()


def validate_download_url(value: object, relative_path: str) -> str:
    """Return a validated HTTP content URL for one manifest record."""

    if not isinstance(value, str):
        raise TypeError(
            f"expected a download URL for {relative_path}, but got {value!r}."
        )
    parsed = httpx.URL(value)
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise ValueError(
            f"expected an HTTP download URL for {relative_path}, but got "
            f"{value!r}."
        )
    return value


def validate_checksum(value: object, relative_path: str) -> str | None:
    """Return a normalized checksum or reject malformed metadata."""

    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"expected a checksum for {relative_path}, but got {value!r}."
        )
    match = CHECKSUM_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"expected algorithm:hex checksum for {relative_path}, but got "
            f"{value!r}."
        )
    algorithm, digest = match.groups()
    expected_length = hashlib.new(algorithm).digest_size * 2
    if len(digest) != expected_length:
        raise ValueError(
            f"expected {expected_length} hexadecimal characters for "
            f"{relative_path}, but got {len(digest)}."
        )
    return f"{algorithm}:{digest.lower()}"


def destination_path(destination: str, relative_path: str) -> Path:
    """Resolve a manifest path while preventing destination traversal."""

    root = Path(destination).resolve()
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    resolved = candidate.resolve()
    if os.path.commonpath([root, resolved]) != str(root):
        raise ValueError(
            f"refusing path outside destination: {relative_path!r}."
        )
    return candidate


def partial_path(destination: Path) -> Path:
    """Return the sibling partial-transfer path for a destination file."""

    return destination.with_name(f"{destination.name}.part")


def calculate_checksum(path: Path, checksum: str) -> str:
    """Calculate the requested checksum for one regular file."""

    algorithm = checksum.partition(":")[0]
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


def validate_file(path: Path, record: HTTPFile) -> bool:
    """Return whether an existing regular file matches its manifest."""

    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise FileExistsError(
            f"expected a regular destination file, but got {path}."
        )
    actual_size = path.stat().st_size
    if actual_size != record.size:
        raise FileExistsError(
            f"existing file {path} has {actual_size} bytes; expected "
            f"{record.size}. Move it aside before downloading."
        )
    if record.checksum is not None:
        actual_checksum = calculate_checksum(path, record.checksum)
        if actual_checksum != record.checksum:
            raise FileExistsError(
                f"existing file {path} has checksum {actual_checksum}; "
                f"expected {record.checksum}. Move it aside before "
                "downloading."
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
    config: ManifestDownloadConfig,
    record: HTTPFile,
    client_factory: ClientFactory,
    provider: str,
) -> Path:
    """Resume, verify, and atomically publish one manifest file once."""

    destination = destination_path(
        config.destination,
        record.relative_path,
    )
    if validate_file(destination, record):
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
        if record.checksum is not None:
            actual_checksum = calculate_checksum(temporary, record.checksum)
            if actual_checksum != record.checksum:
                raise OSError(
                    f"partial file {temporary} has checksum "
                    f"{actual_checksum}; expected {record.checksum}."
                )
        os.replace(temporary, destination)
        return destination

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client_factory() as client, client.stream(
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
            f"incomplete {provider} file {record.relative_path!r}: received "
            f"{actual_size} bytes, expected {record.size}."
        )
    if record.checksum is not None:
        actual_checksum = calculate_checksum(temporary, record.checksum)
        if actual_checksum != record.checksum:
            raise ValueError(
                f"checksum mismatch for {provider} file "
                f"{record.relative_path!r}: expected {record.checksum}, "
                f"but got {actual_checksum}."
            )
    os.replace(temporary, destination)
    return destination


def download_file(
    config: ManifestDownloadConfig,
    record: HTTPFile,
    node_manager: NodeManager | None,
    client_factory: ClientFactory,
    provider: str,
) -> Path:
    """Download one file with shared retry and failover behavior."""

    return run_with_retries(
        config,
        f"{provider} file {record.relative_path}",
        lambda: download_file_once(
            config,
            record,
            client_factory,
            provider,
        ),
        node_manager=node_manager,
    )


def pending_files(
    config: ManifestDownloadConfig,
    manifest: Iterable[HTTPFile],
) -> list[HTTPFile]:
    """Return manifest files absent or inconsistent with local content."""

    pending = []
    for record in manifest:
        path = destination_path(config.destination, record.relative_path)
        if not path.exists():
            pending.append(record)
            continue
        try:
            complete = validate_file(path, record)
        except FileExistsError:
            complete = False
        if not complete:
            pending.append(record)
    return pending


def print_dry_run_summary(
    config: ManifestDownloadConfig,
    manifest: list[HTTPFile],
) -> None:
    """Print pending paths and aggregate byte size."""

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
    config: ManifestDownloadConfig,
    manifest: list[HTTPFile],
    node_manager: NodeManager | None,
    client_factory: ClientFactory,
    provider: str,
) -> None:
    """Download every pending manifest file with bounded concurrency."""

    pending = pending_files(config, manifest)
    if not pending:
        return
    print(f"Downloading {len(pending)} {provider} file(s)...")
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                download_file,
                config,
                record,
                node_manager,
                client_factory,
                provider,
            )
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
