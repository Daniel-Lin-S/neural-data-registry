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

from dataclasses import dataclass
import importlib.util
import os
import ssl
import sys
import time
from typing import Any, Callable

import httpx
from huggingface_hub import get_token, set_client_factory, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429})
RETRYABLE_XET_ERROR_MARKERS = (
    "cas service error",
    "connection",
    "network",
    "reqwest",
    "timed out",
    "timeout",
)

SnapshotFunction = Callable[..., str | list[Any]]
SleepFunction = Callable[[float], None]


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
    retry_attempts : int
        Maximum complete-snapshot attempts.
    retry_base_delay : float
        Initial retry delay in seconds.
    retry_max_delay : float
        Maximum retry delay in seconds.
    """

    repo_id: str
    destination: str
    endpoint: str
    max_workers: int
    timeout: float
    dry_run: bool
    transport: str
    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float


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
        retry_attempts=int(os.environ["DOWNLOAD_RETRY_ATTEMPTS"]),
        retry_base_delay=float(
            os.environ["DOWNLOAD_RETRY_BASE_DELAY"]
        ),
        retry_max_delay=float(os.environ["DOWNLOAD_RETRY_MAX_DELAY"]),
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
        "retry_attempts": config.retry_attempts,
        "retry_base_delay": config.retry_base_delay,
        "retry_max_delay": config.retry_max_delay,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(
                f"expected {name} to be positive, but got {value}."
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


def http_status_code(error: BaseException) -> int | None:
    """Return an HTTP response status associated with an exception.

    Parameters
    ----------
    error : BaseException
        Exception raised by Hugging Face or HTTPX.

    Returns
    -------
    int or None
        HTTP status code when a response is attached.
    """

    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def is_retryable_download_error(
    error: BaseException,
    transport: str,
) -> bool:
    """Classify whether a failed snapshot attempt should be retried.

    Parameters
    ----------
    error : BaseException
        Failure raised by the download operation.
    transport : str
        Active file transport.

    Returns
    -------
    bool
        Whether retrying the complete snapshot is appropriate.
    """

    status_code = http_status_code(error)
    if status_code is not None:
        return (
            status_code in RETRYABLE_HTTP_STATUS_CODES
            or 500 <= status_code <= 599
        )

    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    if transport == "xet" and isinstance(error, RuntimeError):
        message = str(error).lower()
        return any(
            marker in message
            for marker in RETRYABLE_XET_ERROR_MARKERS
        )

    return False


def retry_delay(config: DownloadConfig, failed_attempt: int) -> float:
    """Calculate a capped exponential retry delay.

    Parameters
    ----------
    config : DownloadConfig
        Active downloader configuration.
    failed_attempt : int
        One-based attempt number that just failed.

    Returns
    -------
    float
        Delay in seconds before the next attempt.
    """

    delay = config.retry_base_delay * (2 ** (failed_attempt - 1))
    return min(delay, config.retry_max_delay)


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


def download_with_retries(
    config: DownloadConfig,
    dry_run: bool,
    snapshot_fn: SnapshotFunction = snapshot_download,
    sleep_fn: SleepFunction = time.sleep,
) -> str | list[Any]:
    """Run a resumable snapshot operation with bounded retries.

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

    for attempt in range(1, config.retry_attempts + 1):
        try:
            return call_snapshot(config, dry_run, snapshot_fn)
        except Exception as error:
            retryable = is_retryable_download_error(
                error,
                config.transport,
            )
            if not retryable:
                raise
            if attempt == config.retry_attempts:
                raise RuntimeError(
                    f"{operation} failed after {attempt} attempts; "
                    f"last error was {type(error).__name__}: {error}"
                ) from error

            delay = retry_delay(config, attempt)
            print(
                f"WARNING: {operation} attempt {attempt} of "
                f"{config.retry_attempts} failed with "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            print(
                f"Retrying in {delay:g} seconds using existing "
                "partial files.",
                file=sys.stderr,
            )
            sleep_fn(delay)

    raise RuntimeError("retry loop ended without a result.")


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
    print(f"Authentication: {authentication_status(get_token())}")

    if config.dry_run:
        result = download_with_retries(config, dry_run=True)
        print_dry_run_summary(require_dry_run_records(result))
        return

    result = download_with_retries(config, dry_run=False)
    if not isinstance(result, str) or not result:
        raise TypeError(
            "expected snapshot_download() to return a non-empty path, "
            f"but got {result!r}."
        )

    verification_result = download_with_retries(config, dry_run=True)
    verification_records = require_dry_run_records(verification_result)
    verify_complete(verification_records)

    print()
    print(f"Download location: {result}")
    print("Verification     : complete")


def main() -> int:
    """Run the downloader and translate failures into a clear exit status.

    Returns
    -------
    int
        Zero for success and one for failure.
    """

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
