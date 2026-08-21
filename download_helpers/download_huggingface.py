"""Download and verify a Hugging Face dataset snapshot.

Input
-----
Configuration is read from ``DOWNLOAD_*`` environment variables exported by
``download_huggingface.sh``. Hugging Face authentication is inherited from
``HF_TOKEN`` or the user's Hugging Face token store.

Output
------
Repository files are written beneath ``DOWNLOAD_DEST``. A metadata dry run
verifies that no files remain pending after a successful download. Existing
partial files are retained for retry and resume operations.
"""

from __future__ import annotations

import importlib.util
import os
import ssl
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from download_diagnostics import (
    DownloadInterrupted,
    configure_diagnostics,
    current_log_path,
    log_event,
    log_exception,
)
from download_retry import (
    is_retryable_download_error,  # noqa: F401
    retry_delay,  # noqa: F401
    run_with_retries,
)
from huggingface_hub import (
    close_session,
    get_token,
    set_client_factory,
    snapshot_download,
)
from mihomo_ranker import (
    MihomoConfig,
    MihomoNodeManager,
)

DEFAULT_MIHOMO_PROBE_TIMEOUT = 8.0

SnapshotFunction = Callable[..., str | list[Any]]
SleepFunction = Callable[[float], None]
CloseSessionFunction = Callable[[], None]
AbortXetSessionFunction = Callable[[], None]


@dataclass(frozen=True)
class DownloadConfig:
    """Store validated download configuration.

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset repository identifier.
    destination : str
        Absolute destination directory.
    endpoint : str
        Hugging Face-compatible HTTP endpoint.
    max_workers : int
        Number of files downloaded concurrently.
    timeout : float
        HTTP timeout in seconds.
    dry_run : bool
        Whether to report pending files without downloading.
    transport : str
        File transport, either ``http`` or ``xet``.
    xet_range_concurrency : int
        Concurrent Xet range requests per file.
    retry_attempts : int
        Maximum complete-snapshot attempts.
    retry_base_delay : float
        Initial retry delay in seconds.
    retry_max_delay : float
        Maximum retry delay in seconds.
    proxy_url : str or None
        Explicit proxy URL used by Mihomo throughput tests, optional.
    mihomo : MihomoConfig or None
        Mihomo node-ranking configuration, optional.
    """

    repo_id: str
    destination: str
    endpoint: str
    max_workers: int
    timeout: float
    dry_run: bool
    transport: str
    xet_range_concurrency: int
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float
    proxy_url: str | None
    mihomo: MihomoConfig | None


def optional_environment(name: str) -> str | None:
    """Return a non-empty environment variable, or ``None``."""

    value = os.environ.get(name, "")
    if value:
        return value
    return None


def load_mihomo_config() -> MihomoConfig | None:
    """Load optional Mihomo configuration exported by the shell."""

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
    group = optional_environment("DOWNLOAD_MIHOMO_GROUP")
    return MihomoConfig(
        controller_url=controller,
        group_name=group,
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
    """Load downloader configuration from environment variables.

    Returns
    -------
    DownloadConfig
        Validated configuration exported by the shell entrypoint.
    """

    config = DownloadConfig(
        repo_id=os.environ["DOWNLOAD_REPO_ID"],
        destination=os.environ["DOWNLOAD_DEST"],
        endpoint=os.environ["DOWNLOAD_ENDPOINT"],
        max_workers=int(os.environ["DOWNLOAD_MAX_WORKERS"]),
        timeout=float(os.environ["DOWNLOAD_TIMEOUT"]),
        dry_run=os.environ["DOWNLOAD_DRY_RUN"] == "1",
        transport=os.environ["DOWNLOAD_TRANSPORT"],
        xet_range_concurrency=int(
            os.environ["DOWNLOAD_XET_RANGE_CONCURRENCY"]
        ),
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
    """Reject invalid downloader configuration.

    Parameters
    ----------
    config : DownloadConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        If a required value is empty, unsupported, or non-positive.
    """

    if not config.repo_id:
        raise ValueError("expected a non-empty repository ID.")
    if not os.path.isabs(config.destination):
        raise ValueError(
            "expected an absolute destination path, but got "
            f"{config.destination!r}."
        )
    if config.transport not in {"http", "xet"}:
        raise ValueError(
            "expected transport 'http' or 'xet', but got "
            f"{config.transport!r}."
        )

    positive_values = {
        "max_workers": config.max_workers,
        "timeout": config.timeout,
        "retry_base_delay": config.retry_base_delay,
        "retry_max_delay": config.retry_max_delay,
        "xet_range_concurrency": config.xet_range_concurrency,
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


def ensure_transport_available(transport: str) -> None:
    """Verify that the selected file transport is available.

    Parameters
    ----------
    transport : str
        Selected transport, either ``http`` or ``xet``.

    Raises
    ------
    RuntimeError
        If Xet is selected but ``hf_xet`` is not installed.
    """

    if transport != "xet":
        return
    if importlib.util.find_spec("hf_xet") is None:
        raise RuntimeError(
            "Xet transport requires hf_xet. Install it with "
            "'pip install -U hf_xet'."
        )


def configure_http_client(timeout: float) -> None:
    """Configure Hugging Face metadata requests for the Mihomo route.

    Parameters
    ----------
    timeout : float
        HTTP timeout in seconds.

    Notes
    -----
    File bodies also use this client in HTTP mode. Xet file bodies use Xet's
    separate network stack, while metadata continues to use this client.
    """

    ssl_context = ssl.create_default_context()
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2

    def hf_client_factory() -> httpx.Client:
        """Create the shared Hugging Face HTTPX client."""

        return httpx.Client(
            verify=ssl_context,
            trust_env=True,
            follow_redirects=True,
            timeout=httpx.Timeout(timeout),
        )

    set_client_factory(hf_client_factory)


def authentication_status(token: str | None) -> str:
    """Describe authentication without exposing credential material.

    Parameters
    ----------
    token : str or None
        Resolved Hugging Face token, if one is configured.

    Returns
    -------
    str
        ``configured`` when a token exists, otherwise ``not configured``.
    """

    if token:
        return "configured"
    return "not configured"


def repository_probe_url(endpoint: str, repo_id: str) -> str:
    """Build the dataset metadata URL used for Mihomo node probes."""

    encoded_repo = quote(repo_id, safe="/")
    return f"{endpoint.rstrip('/')}/api/datasets/{encoded_repo}"


def call_snapshot(
    config: DownloadConfig,
    dry_run: bool,
    snapshot_fn: SnapshotFunction,
) -> str | list[Any]:
    """Invoke one Hugging Face snapshot operation.

    Parameters
    ----------
    config : DownloadConfig
        Active downloader configuration.
    dry_run : bool
        Whether to report pending files without downloading.
    snapshot_fn : callable
        Snapshot function, injectable for focused tests.

    Returns
    -------
    str or list
        Download path or Hugging Face dry-run records.
    """

    return snapshot_fn(
        repo_id=config.repo_id,
        repo_type="dataset",
        local_dir=config.destination,
        endpoint=config.endpoint,
        max_workers=config.max_workers,
        dry_run=dry_run,
    )


def reset_download_sessions(
    transport: str,
    close_session_fn: CloseSessionFunction,
    abort_xet_session_fn: AbortXetSessionFunction | None,
) -> None:
    """Discard transport state after a retryable network failure.

    Parameters
    ----------
    transport : str
        Active transport, either ``http`` or ``xet``.
    close_session_fn : callable
        Hugging Face HTTP session cleanup.
    abort_xet_session_fn : callable or None
        Native Xet session cleanup, optional. When omitted, load the cleanup
        supplied by the installed Hugging Face Hub version.

    Notes
    -----
    Cleanup failures are logged without replacing the retryable download
    error. A stale native session otherwise returns its previous task error
    immediately on every later snapshot attempt.
    """

    cleanup_functions: list[tuple[str, CloseSessionFunction]] = [
        ("http", close_session_fn),
    ]
    if transport == "xet":
        if abort_xet_session_fn is None:
            try:
                from huggingface_hub.utils._xet import (
                    abort_xet_session,
                )
            except (AttributeError, ImportError) as error:
                log_exception("xet_session_reset_unavailable", error)
            else:
                cleanup_functions.append(("xet", abort_xet_session))
        else:
            cleanup_functions.append(("xet", abort_xet_session_fn))

    for session_name, cleanup_fn in cleanup_functions:
        try:
            cleanup_fn()
        except Exception as error:  # noqa: BLE001
            log_exception(
                "download_session_reset_failed",
                error,
                session=session_name,
            )
        else:
            log_event(
                "download_session_reset",
                session=session_name,
            )


def download_with_retries(
    config: DownloadConfig,
    dry_run: bool,
    snapshot_fn: SnapshotFunction = snapshot_download,
    sleep_fn: SleepFunction = time.sleep,
    close_session_fn: CloseSessionFunction = close_session,
    abort_xet_session_fn: AbortXetSessionFunction | None = None,
    node_manager: MihomoNodeManager | None = None,
) -> str | list[Any]:
    """Run a resumable snapshot operation with bounded or unlimited retries.

    Parameters
    ----------
    config : DownloadConfig
        Active downloader configuration.
    dry_run : bool
        Whether to report pending files without downloading.
    snapshot_fn : callable, optional
        Snapshot function, injectable for focused tests.
    sleep_fn : callable, optional
        Sleep function, injectable for focused tests.
    close_session_fn : callable, optional
        Hugging Face HTTP session cleanup, injectable for focused tests.
    abort_xet_session_fn : callable or None, optional
        Native Xet session cleanup, injectable for focused tests.

    Returns
    -------
    str or list
        Download path or Hugging Face dry-run records.

    Raises
    ------
    Exception
        The original terminal error, or a retry-exhaustion error chained from
        the final transient failure.
    """

    operation = "verification" if dry_run else "download"
    return run_with_retries(
        config,
        operation,
        lambda: call_snapshot(config, dry_run, snapshot_fn),
        transport=config.transport,
        node_manager=node_manager,
        before_retry=lambda: reset_download_sessions(
            config.transport,
            close_session_fn,
            abort_xet_session_fn,
        ),
        sleep_fn=sleep_fn,
    )


def require_dry_run_records(result: str | list[Any]) -> list[Any]:
    """Require the list returned by ``snapshot_download(dry_run=True)``.

    Parameters
    ----------
    result : str or list
        Result returned by the snapshot function.

    Returns
    -------
    list
        Hugging Face dry-run records.

    Raises
    ------
    TypeError
        If the dry-run result has an unexpected type.
    """

    if not isinstance(result, list):
        raise TypeError(
            "expected snapshot_download(dry_run=True) to return a list, "
            f"but got {type(result).__name__}."
        )
    return result


def pending_records(records: list[Any]) -> list[Any]:
    """Return dry-run records that still require download.

    Parameters
    ----------
    records : list
        Hugging Face dry-run records.

    Returns
    -------
    list
        Records whose ``will_download`` flag is true.
    """

    return [record for record in records if record.will_download]


def file_size(record: Any) -> int:
    """Return a validated dry-run file size.

    Parameters
    ----------
    record : object
        Hugging Face dry-run record.

    Returns
    -------
    int
        Non-negative size in bytes.

    Raises
    ------
    ValueError
        If the record does not contain a valid file size.
    """

    size = getattr(record, "file_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        filename = getattr(record, "filename", "<unknown>")
        raise ValueError(
            f"expected a non-negative file size for {filename}, "
            f"but got {size!r}."
        )
    return size


def print_dry_run_summary(records: list[Any]) -> None:
    """Print pending file names and aggregate byte size.

    Parameters
    ----------
    records : list
        Hugging Face dry-run records.
    """

    pending = pending_records(records)
    total_bytes = sum(file_size(record) for record in pending)

    print()
    print("Dry-run summary")
    print("----------------------------------------")
    print(f"Files to download : {len(pending)}")
    print(f"GiB to download   : {total_bytes / 1024**3:.2f}")
    print()

    for record in pending:
        print(
            f"{file_size(record) / 1024**2:10.2f} MiB  "
            f"{record.filename}"
        )


def verify_complete(records: list[Any]) -> None:
    """Fail when verification reports files still pending.

    Parameters
    ----------
    records : list
        Hugging Face dry-run records collected after download.

    Raises
    ------
    RuntimeError
        If one or more files still require download.
    """

    pending = pending_records(records)
    if not pending:
        return

    first_filename = getattr(pending[0], "filename", "<unknown>")
    raise RuntimeError(
        "download verification found "
        f"{len(pending)} pending files; first pending file: "
        f"{first_filename}."
    )


def run(config: DownloadConfig) -> None:
    """Execute the configured download or dry run.

    Parameters
    ----------
    config : DownloadConfig
        Active downloader configuration.
    """

    ensure_transport_available(config.transport)
    configure_http_client(config.timeout)
    token = get_token()
    print(f"Authentication: {authentication_status(token)}")

    node_manager = None
    if config.mihomo is not None:
        if config.proxy_url is None:
            raise ValueError("Mihomo ranking requires a proxy URL.")
        request_headers = {}
        if token:
            request_headers["Authorization"] = f"Bearer {token}"
        node_manager = MihomoNodeManager(
            config.mihomo,
            config.proxy_url,
            repository_probe_url(config.endpoint, config.repo_id),
            request_headers,
        )

    try:
        if config.dry_run:
            result = download_with_retries(
                config,
                dry_run=True,
                node_manager=node_manager,
            )
            print_dry_run_summary(require_dry_run_records(result))
            return

        result = download_with_retries(
            config,
            dry_run=False,
            node_manager=node_manager,
        )
        if not isinstance(result, str) or not result:
            raise TypeError(
                "expected snapshot_download() to return a non-empty path, "
                f"but got {result!r}."
            )

        verification_result = download_with_retries(
            config,
            dry_run=True,
            node_manager=node_manager,
        )
        verification_records = require_dry_run_records(
            verification_result
        )
        verify_complete(verification_records)

        print()
        print(f"Download location: {result}")
        print("Verification     : complete")
    finally:
        if node_manager is not None:
            node_manager.close()


def diagnostic_configuration(config: DownloadConfig) -> dict[str, Any]:
    """Return safe Hugging Face configuration fields for diagnostics."""

    return {
        "destination": config.destination,
        "dry_run": config.dry_run,
        "endpoint": config.endpoint,
        "max_workers": config.max_workers,
        "provider": "huggingface",
        "proxy_url": config.proxy_url,
        "repo": config.repo_id,
        "retry_attempts": config.retry_attempts,
        "retry_base_delay": config.retry_base_delay,
        "retry_max_delay": config.retry_max_delay,
        "timeout": config.timeout,
        "transport": config.transport,
        "xet_adaptive_concurrency": True,
        "xet_range_concurrency": config.xet_range_concurrency,
    }


def main() -> int:
    """Run the downloader and translate failures into a clear exit status."""

    try:
        config = load_config_from_environment()
        log_path = configure_diagnostics(
            "huggingface",
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
