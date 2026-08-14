"""Tests for the resumable Hugging Face dataset downloader."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "download_huggingface.sh"
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "download_huggingface.py"


def load_downloader_module() -> Any:
    """Load the downloader companion module from its repository path."""

    spec = importlib.util.spec_from_file_location(
        "download_huggingface",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load downloader from {MODULE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


downloader = load_downloader_module()


def make_config(
    tmp_path: Path,
    *,
    retry_attempts: int = 3,
    transport: str = "http",
) -> Any:
    """Create a valid downloader configuration for focused tests."""

    return downloader.DownloadConfig(
        repo_id="owner/dataset",
        destination=str(tmp_path.resolve()),
        endpoint="https://huggingface.co",
        max_workers=2,
        timeout=300.0,
        dry_run=False,
        transport=transport,
        retry_attempts=retry_attempts,
        retry_base_delay=5.0,
        retry_max_delay=60.0,
    )


def hub_error(status_code: int) -> HfHubHTTPError:
    """Create a Hugging Face HTTP error with a response status."""

    request = httpx.Request("GET", "https://huggingface.co/test")
    response = httpx.Response(status_code, request=request)
    return HfHubHTTPError("test failure", response=response)


def run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the shell entrypoint and capture its user-facing output."""

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_transient_failure_resumes_snapshot(tmp_path: Path) -> None:
    """Retry a truncated response against the same destination."""

    config = make_config(tmp_path)
    calls: list[dict[str, Any]] = []
    delays: list[float] = []

    def snapshot_fn(**kwargs: Any) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("incomplete message body")
        return str(tmp_path)

    result = downloader.download_with_retries(
        config,
        dry_run=False,
        snapshot_fn=snapshot_fn,
        sleep_fn=delays.append,
    )

    assert result == str(tmp_path)
    assert len(calls) == 2
    assert calls[0]["local_dir"] == calls[1]["local_dir"]
    assert delays == [5.0]


def test_retry_exhaustion_is_nonzero_failure(tmp_path: Path) -> None:
    """Raise a clear terminal error after all transient retries fail."""

    config = make_config(tmp_path, retry_attempts=3)
    delays: list[float] = []

    def snapshot_fn(**kwargs: Any) -> str:
        del kwargs
        raise httpx.RemoteProtocolError("incomplete message body")

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        downloader.download_with_retries(
            config,
            dry_run=False,
            snapshot_fn=snapshot_fn,
            sleep_fn=delays.append,
        )

    assert delays == [5.0, 10.0]


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_terminal_http_status_is_not_retried(status_code: int) -> None:
    """Fail immediately for authentication and missing-resource errors."""

    error = hub_error(status_code)
    assert not downloader.is_retryable_download_error(error, "http")


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 599])
def test_transient_http_status_is_retried(status_code: int) -> None:
    """Retry throttling, timeout, and server-side HTTP responses."""

    error = hub_error(status_code)
    assert downloader.is_retryable_download_error(error, "http")


def test_filesystem_error_is_not_retried() -> None:
    """Do not hide destination or disk errors behind network retries."""

    error = OSError("no space left on device")
    assert not downloader.is_retryable_download_error(error, "http")


def test_xet_requires_installed_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject Xet mode when the optional transport is unavailable."""

    monkeypatch.setattr(
        downloader.importlib.util,
        "find_spec",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="requires hf_xet"):
        downloader.ensure_transport_available("xet")


def test_authentication_status_never_returns_token() -> None:
    """Report token presence without returning credential material."""

    token = "hf_secret_value"
    status = downloader.authentication_status(token)

    assert status == "configured"
    assert token not in status


def test_verification_rejects_pending_files() -> None:
    """Reject an apparently successful snapshot with pending files."""

    record = SimpleNamespace(
        filename="large.fif",
        file_size=100,
        will_download=True,
    )

    with pytest.raises(RuntimeError, match="1 pending files"):
        downloader.verify_complete([record])


def test_retry_delay_is_capped(tmp_path: Path) -> None:
    """Cap exponential retry delays at the configured maximum."""

    config = replace(make_config(tmp_path), retry_max_delay=60.0)

    assert downloader.retry_delay(config, 1) == 5.0
    assert downloader.retry_delay(config, 2) == 10.0
    assert downloader.retry_delay(config, 3) == 20.0
    assert downloader.retry_delay(config, 4) == 40.0
    assert downloader.retry_delay(config, 5) == 60.0
    assert downloader.retry_delay(config, 8) == 60.0


def test_help_reports_balanced_defaults() -> None:
    """Document required inputs and portable operational defaults."""

    result = run_script("--help")

    assert result.returncode == 0
    assert "Default: 2" in result.stdout
    assert "Default: 300" in result.stdout
    assert "File transport: http or xet" in result.stdout
    assert "--repo REPO_ID --dest PATH" in result.stdout
    assert "pnpl/LibriBrain" not in result.stdout
    assert "7893" not in result.stdout


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (("--dest",), "--dest requires a value"),
        (("--dest", "/tmp/test"), "--repo is required"),
        (("--repo", "owner/dataset"), "--dest is required"),
        (("--repo", "owner/dataset", "--dest", "relative/path"),
         "--dest must be an absolute path"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--transport", "bad"),
         "--transport must be http or xet"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--max-workers", "0"),
         "--max-workers must be a positive integer"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--timeout", "none"),
         "--timeout must be a positive integer"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--retry-attempts", "0"),
         "--retry-attempts must be a positive integer"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-scheme", "ftp"),
         "--proxy-scheme must be http, https, socks5, or socks5h"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-port", "65536", "--proxy-host", "localhost"),
         "--proxy-port must not exceed 65535"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-port", "7893"),
         "--proxy-host is required when proxying is enabled"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-host", "localhost"),
         "--proxy-port is required when proxying is enabled"),
        (("--unknown",), "Unknown option: --unknown"),
    ],
)
def test_invalid_shell_arguments_fail_before_download(
    arguments: tuple[str, ...],
    expected_error: str,
) -> None:
    """Reject invalid CLI arguments before filesystem or network work."""

    result = run_script(*arguments)

    assert result.returncode == 1
    assert expected_error in result.stderr
