"""Tests for the resumable Hugging Face dataset downloader."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import (
    DryRunError,
    HfHubHTTPError,
    IncompleteSnapshotError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "download_huggingface.sh"
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "download_huggingface.py"


def load_downloader_module() -> Any:
    """Load the downloader companion module from its repository path."""

    sys.path.insert(0, str(MODULE_PATH.parent))
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
        max_workers=1,
        timeout=300.0,
        dry_run=False,
        transport=transport,
        retry_attempts=retry_attempts,
        retry_base_delay=5.0,
        retry_max_delay=300.0,
        proxy_url=None,
        mihomo=None,
    )


@pytest.mark.parametrize(
    "missing_name",
    [
        "DOWNLOAD_MIHOMO_GROUP",
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL",
    ],
)
def test_load_mihomo_config_skips_incomplete_ranking(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    """Skip node selection when either operational input is absent."""

    values = {
        "DOWNLOAD_MIHOMO_CONTROLLER": "http://127.0.0.1:9091",
        "DOWNLOAD_MIHOMO_GROUP": "download",
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL": (
            "https://example.com/large.bin"
        ),
        "DOWNLOAD_MIHOMO_PROBE_TIMEOUT": "15",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing_name, raising=False)

    assert downloader.load_mihomo_config() is None


def test_load_mihomo_config_allows_empty_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow every direct selector node when no marker is configured."""

    monkeypatch.setenv(
        "DOWNLOAD_MIHOMO_CONTROLLER",
        "http://127.0.0.1:9091",
    )
    monkeypatch.setenv("DOWNLOAD_MIHOMO_GROUP", "download")
    monkeypatch.setenv(
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL",
        "https://example.com/large.bin",
    )
    monkeypatch.setenv("DOWNLOAD_MIHOMO_PROBE_TIMEOUT", "15")
    monkeypatch.delenv("DOWNLOAD_MIHOMO_NODE_MARKER", raising=False)

    config = downloader.load_mihomo_config()

    assert config is not None
    assert config.node_marker == ""


def hub_error(status_code: int) -> HfHubHTTPError:
    """Create a Hugging Face HTTP error with a response status."""

    request = httpx.Request("GET", "https://huggingface.co/test")
    response = httpx.Response(status_code, request=request)
    return HfHubHTTPError("test failure", response=response)


def run_script(
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the shell entrypoint and capture its user-facing output."""

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def python_stub_environment(tmp_path: Path) -> dict[str, str]:
    """Return an environment whose Python command performs no work."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    return environment


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


def test_incomplete_snapshot_with_connection_cause_is_retried(
    tmp_path: Path,
) -> None:
    """Retry the Hugging Face wrapper observed on the unstable proxy."""

    request = httpx.Request("GET", "https://huggingface.co/api/test")
    cause = httpx.ConnectError("SSL EOF", request=request)
    wrapped = IncompleteSnapshotError("incomplete", str(tmp_path))
    wrapped.__cause__ = cause

    assert downloader.is_retryable_download_error(wrapped, "http")


def test_dry_run_wrapper_with_connection_cause_is_retried() -> None:
    """Retry verification metadata failures through their cause chain."""

    request = httpx.Request("GET", "https://huggingface.co/api/test")
    cause = httpx.ConnectError("SSL EOF", request=request)
    wrapped = DryRunError("metadata unavailable")
    wrapped.__cause__ = cause

    assert downloader.is_retryable_download_error(wrapped, "http")


def test_unlimited_retries_continue_until_success(tmp_path: Path) -> None:
    """Treat zero attempts as retry-until-interrupted mode."""

    config = make_config(tmp_path, retry_attempts=0)
    calls = 0

    def snapshot_fn(**kwargs: Any) -> str:
        nonlocal calls
        del kwargs
        calls += 1
        if calls < 5:
            raise httpx.RemoteProtocolError("truncated")
        return str(tmp_path)

    result = downloader.download_with_retries(
        config,
        dry_run=False,
        snapshot_fn=snapshot_fn,
        sleep_fn=lambda delay: None,
        close_session_fn=lambda: None,
    )

    assert result == str(tmp_path)
    assert calls == 5


def test_retry_closes_stale_hugging_face_session(tmp_path: Path) -> None:
    """Close pooled connections before the next snapshot attempt."""

    config = make_config(tmp_path)
    calls = 0
    close_calls = 0

    def snapshot_fn(**kwargs: Any) -> str:
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("failed")
        return str(tmp_path)

    def close_fn() -> None:
        nonlocal close_calls
        close_calls += 1

    downloader.download_with_retries(
        config,
        dry_run=False,
        snapshot_fn=snapshot_fn,
        sleep_fn=lambda delay: None,
        close_session_fn=close_fn,
    )

    assert close_calls == 1


def test_repeated_server_errors_trigger_node_failover(
    tmp_path: Path,
) -> None:
    """Keep one 5xx on-node, then rotate after a repeated server failure."""

    config = make_config(tmp_path)
    calls = 0
    manager = SimpleNamespace(
        prepare_attempt=lambda: "allowed-0.1倍",
        failover_calls=0,
    )

    def failover() -> str:
        manager.failover_calls += 1
        return "next-0.1倍"

    manager.failover = failover

    def snapshot_fn(**kwargs: Any) -> str:
        nonlocal calls
        del kwargs
        calls += 1
        if calls <= 2:
            raise hub_error(503)
        return str(tmp_path)

    downloader.download_with_retries(
        config,
        dry_run=False,
        snapshot_fn=snapshot_fn,
        sleep_fn=lambda delay: None,
        close_session_fn=lambda: None,
        node_manager=manager,
    )

    assert manager.failover_calls == 1


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

    config = replace(make_config(tmp_path), retry_max_delay=300.0)

    assert downloader.retry_delay(config, 1) == 5.0
    assert downloader.retry_delay(config, 2) == 10.0
    assert downloader.retry_delay(config, 3) == 20.0
    assert downloader.retry_delay(config, 4) == 40.0
    assert downloader.retry_delay(config, 5) == 80.0
    assert downloader.retry_delay(config, 7) == 300.0
    assert downloader.retry_delay(config, 8) == 300.0


def test_help_reports_balanced_defaults() -> None:
    """Document required inputs and portable operational defaults."""

    result = run_script("--help")

    assert result.returncode == 0
    assert "Default: 1" in result.stdout
    assert "Default: 300" in result.stdout
    assert "File transport: http or xet" in result.stdout
    assert "Default: xet" in result.stdout
    assert "Default: 127.0.0.1" in result.stdout
    assert "--repo REPO_ID --dest PATH" in result.stdout
    assert "pnpl/LibriBrain" not in result.stdout
    assert "7893" not in result.stdout


@pytest.mark.parametrize(
    ("ranking_arguments", "ranking_enabled"),
    [
        (
            (
                "--mihomo-controller",
                "http://127.0.0.1:9091",
                "--mihomo-speed-test-url",
                "https://example.com/large.bin",
                "--mihomo-probe-timeout",
                "unused",
            ),
            False,
        ),
        (
            (
                "--mihomo-controller",
                "http://127.0.0.1:9091",
                "--mihomo-group",
                "download",
                "--mihomo-probe-timeout",
                "unused",
            ),
            False,
        ),
        (
            (
                "--mihomo-controller",
                "http://127.0.0.1:9091",
                "--mihomo-group",
                "download",
                "--mihomo-speed-test-url",
                "https://example.com/large.bin",
            ),
            True,
        ),
    ],
)
def test_mihomo_ranking_is_optional_and_marker_is_not_required(
    tmp_path: Path,
    ranking_arguments: tuple[str, ...],
    ranking_enabled: bool,
) -> None:
    """Keep proxying active while enabling ranking only with both inputs."""

    result = run_script(
        "--repo",
        "owner/dataset",
        "--dest",
        str((tmp_path / "dataset").resolve()),
        "--proxy-port",
        "7893",
        *ranking_arguments,
        env=python_stub_environment(tmp_path),
    )

    assert result.returncode == 0
    assert "Proxy         : http://127.0.0.1:7893" in result.stdout
    assert ("Mihomo API" in result.stdout) is ranking_enabled
    if ranking_enabled:
        assert "Node filter  : all direct nodes" in result.stdout


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
          "--retry-attempts", "-1"),
         "--retry-attempts must be a non-negative integer"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-scheme", "ftp"),
         "--proxy-scheme must be http, https, socks5, or socks5h"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-port", "65536", "--proxy-host", "localhost"),
         "--proxy-port must not exceed 65535"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-host", "localhost"),
         "--proxy-port is required when proxying is enabled"),
        (("--repo", "owner/dataset", "--dest", "/tmp/test",
          "--proxy-port", "7893", "--mihomo-group", "download",
          "--mihomo-speed-test-url", "https://example.com/large.bin"),
         "Mihomo ranking requires --mihomo-controller"),
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
