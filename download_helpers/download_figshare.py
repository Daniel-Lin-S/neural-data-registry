"""Download public Figshare article files, with resumable unlimited retries.

Input
-----
``download_figshare.sh`` exports DOWNLOAD_* options. The article may be a
numeric ID, a Figshare landing-page URL, or a versioned article URL. Registry
settings use NDR_* variables or the repository .env file. The destination's
basename is the canonical dataset name used by the shared intake preflight.

Output
------
The absolute destination contains original files and sibling .part files for
unfinished transfers. Each file must match the API size and computed MD5
before publication. ``.figshare-manifest.json`` pins article ID, version, and
files (name, size, download_url, computed_md5) across restarts. Diagnostic logs
are stored in the repository's logs/downloads directory. No registration is
performed. The registry intake lock is held unless --standalone explicitly
selects read-only brainctl checks and separate process locks instead.

Transient HTTP/network failures retry indefinitely by default, with bounded
exponential delays. Invalid metadata, conflicting files, checksum failures,
and local storage errors fail explicitly and preserve existing data.
Optional Mihomo ranking and failover reuse the shared node manager for both
metadata requests and file transfers; one worker is required when enabled.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from download_diagnostics import (
    DownloadInterrupted,
    configure_diagnostics,
    log_event,
    log_exception,
)
from download_http_manifest import (
    HTTPFile,
    download_manifest,
    print_dry_run_summary,
    validate_checksum,
    validate_download_url,
    validate_relative_path,
)
from download_retry import run_with_retries
from download_standalone import standalone_download_intake
from mihomo_ranker import (
    MihomoConfig,
    MihomoNodeManager,
    load_mihomo_config,
)
from neural_data_registry.config import get_settings
from neural_data_registry.service import external_download_intake

MANIFEST_NAME = ".figshare-manifest.json"
ARTICLE_PATH = re.compile(
    r"^/(?:v2/)?articles/(?:[A-Za-z_]+/[^/]+/)?"
    r"([0-9]+)(?:/(?:versions/)?([0-9]+))?/?$"
)
GIB = 1024 ** 3


@dataclass(frozen=True)
class DownloadConfig:
    """Validated HTTP settings; retry_attempts=0 means unlimited attempts."""

    article_id: str
    version: str | None
    source_url: str
    destination: str
    api_base: str
    max_workers: int
    timeout: float
    dry_run: bool
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float
    proxy_url: str | None
    standalone: bool = False
    mihomo: MihomoConfig | None = None


def parse_article(value: str) -> tuple[str, str | None]:
    """Resolve a public Figshare ID or URL without performing network I/O.

    Parameters
    ----------
    value : str
        Numeric article ID or Figshare article URL, optionally versioned.

    Returns
    -------
    tuple[str, str or None]
        Canonical article ID and optional explicit version.
    """
    value = value.strip()
    if value.isascii() and value.isdigit() and int(value) > 0:
        return str(int(value)), None
    parsed = urlparse(value)
    host = parsed.hostname or ""
    match = ARTICLE_PATH.fullmatch(parsed.path)
    if (
        parsed.scheme not in {"http", "https"}
        or not (host == "figshare.com" or host.endswith(".figshare.com"))
        or match is None
        or int(match[1]) <= 0
        or (match[2] is not None and int(match[2]) <= 0)
    ):
        raise ValueError(f"Expected a public Figshare article, got {value!r}.")
    return str(int(match[1])), match[2]


def load_config() -> DownloadConfig:
    """Validate shell options, rejecting non-finite numbers and unsafe paths."""
    source = os.environ["DOWNLOAD_REPO_ID"]
    article_id, version = parse_article(source)
    destination = Path(os.environ["DOWNLOAD_DEST"])
    if not destination.is_absolute() or not destination.name:
        raise ValueError("Expected an absolute, named destination directory.")
    config = DownloadConfig(
        article_id=article_id,
        version=version,
        source_url=source,
        destination=str(destination.resolve()),
        api_base=validate_download_url(
            os.environ["DOWNLOAD_ENDPOINT"].rstrip("/"), "Figshare API"
        ),
        max_workers=int(os.environ["DOWNLOAD_MAX_WORKERS"]),
        timeout=float(os.environ["DOWNLOAD_TIMEOUT"]),
        dry_run=os.environ["DOWNLOAD_DRY_RUN"] == "1",
        retry_attempts=int(os.environ["DOWNLOAD_RETRY_ATTEMPTS"]),
        retry_base_delay=float(os.environ["DOWNLOAD_RETRY_BASE_DELAY"]),
        retry_max_delay=float(os.environ["DOWNLOAD_RETRY_MAX_DELAY"]),
        proxy_url=os.environ.get("DOWNLOAD_PROXY_URL") or None,
        standalone=os.environ.get("DOWNLOAD_STANDALONE") == "1",
        mihomo=load_mihomo_config(),
    )
    for name in (
        "max_workers", "timeout", "retry_base_delay", "retry_max_delay"
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Expected finite positive {name}, got {value}.")
    if config.retry_attempts < 0:
        raise ValueError("Expected non-negative retry_attempts.")
    if config.mihomo is not None:
        if config.proxy_url is None:
            raise ValueError("Mihomo ranking requires an explicit proxy URL.")
        if config.max_workers != 1:
            raise ValueError("Mihomo ranking requires max_workers=1.")
    return config


def create_client(config: DownloadConfig) -> httpx.Client:
    """Create explicit transport; ambient proxies are never inherited."""
    return httpx.Client(
        trust_env=False,
        proxy=config.proxy_url,
        follow_redirects=True,
        timeout=config.timeout,
        headers={"Accept-Encoding": "identity"},
    )


def parse_manifest(payload: dict[str, Any]) -> list[HTTPFile]:
    """Validate all file metadata before any transfer starts.

    Parameters
    ----------
    payload : dict
        Figshare article JSON containing a non-empty files array.

    Returns
    -------
    list[HTTPFile]
        Safe, unique paths with positive sizes and required MD5 digests.
    """
    items = payload.get("files")
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a non-empty Figshare files array.")
    records = []
    paths = {MANIFEST_NAME, MANIFEST_NAME + ".tmp"}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Expected each Figshare file to be an object.")
        name = item.get("name")
        if not isinstance(name, str) or name.startswith("/"):
            raise ValueError(f"Expected a relative file name, got {name!r}.")
        name = validate_relative_path(name, "Figshare")
        if item.get("is_link_only"):
            raise ValueError(f"Cannot verify link-only Figshare file {name!r}.")
        size = item.get("size")
        if type(size) is not int or size <= 0:
            raise ValueError(
                f"Expected positive size for {name!r}, got {size}."
            )
        checksum = validate_checksum(
            f"md5:{item.get('computed_md5', '')}", name
        )
        if name in paths or name + ".part" in paths:
            raise ValueError(f"Conflicting Figshare file name: {name!r}.")
        paths.update((name, name + ".part"))
        records.append(HTTPFile(
            name, size,
            validate_download_url(item.get("download_url"), name),
            checksum,
        ))
    for name in paths:
        if any(str(parent) in paths for parent in Path(name).parents):
            raise ValueError(f"Conflicting Figshare directory: {name!r}.")
    return sorted(records, key=lambda record: record.relative_path)


def fetch_article(config: DownloadConfig) -> dict[str, Any]:
    """Fetch a requested article version in one bounded HTTP attempt."""
    url = f"{config.api_base}/articles/{config.article_id}"
    if config.version is not None:
        url += f"/versions/{config.version}"
    with create_client(config) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    validate_article(config, payload)
    return payload


def validate_article(config: DownloadConfig, payload: Any) -> None:
    """Reject a cached or remote snapshot belonging to another article."""
    if not isinstance(payload, dict):
        raise ValueError("Expected Figshare article metadata to be an object.")
    if str(payload.get("id")) != config.article_id:
        raise ValueError("Figshare manifest article ID does not match request.")
    version = payload.get("version")
    if type(version) is not int or version <= 0:
        raise ValueError(f"Expected positive article version, got {version!r}.")
    if config.version is not None and str(version) != config.version:
        raise ValueError("Figshare manifest version does not match request.")


def run(config: DownloadConfig) -> None:
    """Pin metadata and transfer files while the shared intake guard is held."""
    root = Path(config.destination)
    if config.standalone:
        urls = [
            f"https://api.figshare.com/v2/articles/{config.article_id}",
            f"https://figshare.com/articles/{config.article_id}",
        ]
        if "://" in config.source_url:
            urls.append(config.source_url)
        guard = standalone_download_intake(
            root.name, f"figshare:{config.article_id}", urls, root
        )
    else:
        guard = external_download_intake(
            root.name, config.source_url, root, get_settings()
        )
    with guard, ExitStack() as resources:
        diagnostics = {
            key: value for key, value in vars(config).items()
            if key != "mihomo"
        }
        diagnostics["mihomo_enabled"] = config.mihomo is not None
        log = configure_diagnostics("figshare", diagnostics)
        print(f"Debug log: {log}", flush=True)
        node_manager = None
        if config.mihomo is not None:
            node_manager = run_with_retries(
                config, "Mihomo setup",
                lambda: MihomoNodeManager(
                    config.mihomo, config.proxy_url,
                    f"{config.api_base}/articles/{config.article_id}",
                ),
            )
            resources.callback(node_manager.close)
        path = root / MANIFEST_NAME
        if path.is_symlink():
            raise ValueError(f"Refusing manifest symlink: {path}")
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_article(config, payload)
        else:
            payload = run_with_retries(
                config, "Figshare manifest", lambda: fetch_article(config),
                node_manager=node_manager,
            )
        manifest = parse_manifest(payload)
        print(
            f"Article {config.article_id}, version {payload['version']}: "
            f"{len(manifest)} files, "
            f"{sum(item.size for item in manifest) / GIB:.2f} GiB",
            flush=True,
        )
        if config.dry_run:
            print_dry_run_summary(config, manifest)
            return
        root.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        download_manifest(
            config, manifest, node_manager,
            lambda: create_client(config), "Figshare"
        )
        log_event("download_completed", status="success")
        print(f"Verified download complete: {root}", flush=True)


def main() -> int:
    """Return a shell exit status, retaining partial files on failure."""
    try:
        run(load_config())
    except (DownloadInterrupted, KeyboardInterrupt):
        print("Download interrupted; partial files retained.", file=sys.stderr)
        return 130
    except Exception as error:
        log_exception("download_failed", error)
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
