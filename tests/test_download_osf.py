"""Tests for the resumable OSF dataset downloader."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "download_osf.sh"
MODULE_PATH = REPOSITORY_ROOT / "download_helpers" / "download_osf.py"


def load_downloader_module() -> Any:
    """Load the OSF downloader from its repository path."""

    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "download_osf",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OSF downloader from {MODULE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


downloader = load_downloader_module()


def make_config(tmp_path: Path, *, retry_attempts: int = 3) -> Any:
    """Create a valid direct-download configuration."""

    return downloader.DownloadConfig(
        project_id="ag3kj",
        destination=str((tmp_path / "dataset").resolve()),
        api_base="https://api.osf.io/v2",
        storage="osfstorage",
        max_workers=1,
        timeout=300.0,
        dry_run=False,
        retry_attempts=retry_attempts,
        retry_base_delay=5.0,
        retry_max_delay=300.0,
        proxy_url=None,
        token=None,
        mihomo=None,
    )


def test_load_mihomo_config_skips_without_speed_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip node selection when no speed-test URL is configured."""

    monkeypatch.setenv("DOWNLOAD_MIHOMO_GROUP", "download")
    monkeypatch.delenv(
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL",
        raising=False,
    )

    assert downloader.load_mihomo_config() is None


def test_load_mihomo_config_discovers_group_and_uses_empty_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the selector unset for speed-triggered discovery."""

    monkeypatch.setenv(
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL",
        "https://example.com/large.bin",
    )
    monkeypatch.setenv(
        "DOWNLOAD_MIHOMO_CONTROLLER",
        "http://127.0.0.1:9091",
    )
    monkeypatch.delenv("DOWNLOAD_MIHOMO_GROUP", raising=False)
    monkeypatch.delenv("DOWNLOAD_MIHOMO_NODE_MARKER", raising=False)
    monkeypatch.delenv("DOWNLOAD_MIHOMO_PROBE_TIMEOUT", raising=False)

    config = downloader.load_mihomo_config()

    assert config is not None
    assert config.controller_url == "http://127.0.0.1:9091"
    assert config.group_name is None
    assert config.node_marker == ""
    assert config.probe_timeout == downloader.DEFAULT_MIHOMO_PROBE_TIMEOUT


def test_load_mihomo_config_requires_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject speed testing without an explicit controller."""

    monkeypatch.setenv(
        "DOWNLOAD_MIHOMO_SPEED_TEST_URL",
        "https://example.com/large.bin",
    )
    monkeypatch.delenv("DOWNLOAD_MIHOMO_CONTROLLER", raising=False)

    with pytest.raises(ValueError, match="requires --mihomo-controller"):
        downloader.load_mihomo_config()


def client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[[], httpx.Client]:
    """Build HTTPX clients backed by one deterministic mock transport."""

    transport = httpx.MockTransport(handler)
    return lambda: httpx.Client(
        transport=transport,
        follow_redirects=True,
    )


def file_entry(path: str, size: int, url: str) -> dict[str, Any]:
    """Build one OSF JSON:API file fixture."""

    return {
        "id": path,
        "type": "files",
        "attributes": {
            "kind": "file",
            "materialized_path": f"/{path}",
            "name": Path(path).name,
            "size": size,
        },
        "links": {"download": url},
    }


def folder_entry(path: str, url: str) -> dict[str, Any]:
    """Build one OSF JSON:API folder fixture."""

    return {
        "id": path,
        "type": "files",
        "attributes": {
            "kind": "folder",
            "materialized_path": f"/{path}/",
            "name": Path(path).name,
            "size": 0,
        },
        "relationships": {
            "files": {
                "links": {
                    "related": {"href": url}
                }
            }
        },
        "links": {},
    }


def run_script(
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the OSF shell entrypoint and capture user-facing output."""

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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ag3kj", "ag3kj"),
        ("AG3KJ", "ag3kj"),
        ("https://osf.io/ag3kj/overview", "ag3kj"),
        ("https://www.osf.io/ag3kj/files/osfstorage", "ag3kj"),
    ],
)
def test_parse_project_id(value: str, expected: str) -> None:
    """Accept canonical OSF IDs and project-page URLs."""

    assert downloader.parse_project_id(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "bad/id", "ftp://osf.io/ag3kj", "https://example.com/ag3kj"],
)
def test_parse_project_id_rejects_invalid_values(value: str) -> None:
    """Reject malformed IDs and non-OSF URLs before network access."""

    with pytest.raises(ValueError):
        downloader.parse_project_id(value)


def test_manifest_walks_folders_and_pagination(tmp_path: Path) -> None:
    """Preserve file paths across recursive and paginated OSF listings."""

    config = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "folder" in url:
            payload = {
                "data": [
                    file_entry(
                        "sub-01/data.bin",
                        3,
                        "https://files.osf.io/data",
                    )
                ],
                "links": {"next": None},
            }
        elif "page=2" in url:
            payload = {
                "data": [
                    file_entry(
                        "participants.tsv",
                        4,
                        "https://files.osf.io/participants",
                    )
                ],
                "links": {"next": None},
            }
        else:
            payload = {
                "data": [
                    file_entry(
                        "README",
                        2,
                        "https://files.osf.io/readme",
                    ),
                    folder_entry("sub-01", "https://api.osf.io/folder"),
                ],
                "links": {
                    "next": "https://api.osf.io/root?page=2"
                },
            }
        return httpx.Response(200, json=payload)

    manifest = downloader.collect_manifest_once(
        config,
        client_factory(handler),
    )

    assert [record.relative_path for record in manifest] == [
        "README",
        "participants.tsv",
        "sub-01/data.bin",
    ]


def test_transient_incomplete_file_resumes(tmp_path: Path) -> None:
    """Retry a truncated body using the retained partial-file offset."""

    config = make_config(tmp_path)
    record = downloader.OSFFile(
        "sub-01/data.bin",
        6,
        "https://files.osf.io/data",
    )
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("Range")
        calls.append(range_header)
        if len(calls) == 1:
            return httpx.Response(200, content=b"abc")
        return httpx.Response(
            206,
            headers={"Content-Range": "bytes 3-5/6"},
            content=b"def",
        )

    path = downloader.download_file(
        config,
        record,
        None,
        client_factory(handler),
    )

    assert path.read_bytes() == b"abcdef"
    assert calls == [None, "bytes=3-"]
    assert not downloader.partial_path(path).exists()


def test_server_ignoring_range_restarts_partial_file(tmp_path: Path) -> None:
    """Replace a partial body when a server answers a range with HTTP 200."""

    config = make_config(tmp_path)
    record = downloader.OSFFile(
        "data.bin",
        6,
        "https://files.osf.io/data",
    )
    destination = downloader.destination_path(
        config.destination,
        record.relative_path,
    )
    destination.parent.mkdir(parents=True)
    downloader.partial_path(destination).write_bytes(b"abc")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=3-"
        return httpx.Response(200, content=b"abcdef")

    result = downloader.download_file_once(
        config,
        record,
        client_factory(handler),
    )

    assert result.read_bytes() == b"abcdef"


def test_complete_file_is_skipped_without_network(tmp_path: Path) -> None:
    """Reuse a verified destination file on subsequent invocations."""

    config = make_config(tmp_path)
    record = downloader.OSFFile(
        "data.bin",
        3,
        "https://files.osf.io/data",
    )
    destination = downloader.destination_path(
        config.destination,
        record.relative_path,
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"abc")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    result = downloader.download_file_once(
        config,
        record,
        client_factory(handler),
    )

    assert result == destination


def test_wrong_sized_destination_fails_without_overwrite(
    tmp_path: Path,
) -> None:
    """Preserve an existing destination whose size conflicts with OSF."""

    config = make_config(tmp_path)
    record = downloader.OSFFile(
        "data.bin",
        4,
        "https://files.osf.io/data",
    )
    destination = downloader.destination_path(
        config.destination,
        record.relative_path,
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="Move it aside"):
        downloader.download_file_once(
            config,
            record,
            client_factory(
                lambda request: httpx.Response(200, content=b"new!")
            ),
        )

    assert destination.read_bytes() == b"old"


def test_partial_symlink_is_rejected(tmp_path: Path) -> None:
    """Never follow a pre-existing partial-file symlink."""

    config = make_config(tmp_path)
    record = downloader.OSFFile(
        "data.bin",
        3,
        "https://files.osf.io/data",
    )
    destination = downloader.destination_path(
        config.destination,
        record.relative_path,
    )
    destination.parent.mkdir(parents=True)
    external = tmp_path / "external.bin"
    external.write_bytes(b"old")
    downloader.partial_path(destination).symlink_to(external)

    with pytest.raises(FileExistsError, match="regular partial file"):
        downloader.download_file_once(
            config,
            record,
            client_factory(
                lambda request: httpx.Response(200, content=b"new")
            ),
        )

    assert external.read_bytes() == b"old"


def test_authentication_status_never_returns_token() -> None:
    """Report token presence without returning credential material."""

    token = "osf_secret_value"
    status = downloader.authentication_status(token)

    assert status == "configured"
    assert token not in status


def test_mihomo_ranking_requires_serial_downloads(tmp_path: Path) -> None:
    """Reject node switching while concurrent transfers could be active."""

    config = make_config(tmp_path)
    mihomo = downloader.MihomoConfig(
        controller_url="http://127.0.0.1:9091",
        group_name="download",
        node_marker="0.1倍",
        speed_test_url="https://files.osf.io/large",
        probe_timeout=15.0,
        secret=None,
    )
    config = replace(
        config,
        max_workers=2,
        proxy_url="http://127.0.0.1:7893",
        mihomo=mihomo,
    )

    with pytest.raises(ValueError, match="max_workers=1"):
        downloader.validate_config(config)


def test_help_documents_direct_and_mihomo_modes() -> None:
    """Document portable direct access and explicit Mihomo controls."""

    result = run_script("--help")

    assert result.returncode == 0
    assert "--repo ID_OR_URL --dest PATH" in result.stdout
    assert "Omit for direct access" in result.stdout
    assert "--mihomo-controller" in result.stdout
    assert "MIHOMO_CONTROLLER is set" in result.stdout
    assert "discover the selector" in result.stdout
    assert "ag3kj" in result.stdout
    assert "7893" not in result.stdout


@pytest.mark.parametrize(
    ("ranking_arguments", "ranking_enabled"),
    [
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
                "--mihomo-speed-test-url",
                "https://example.com/large.bin",
            ),
            True,
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
    """Use the speed URL as the sole ranking trigger."""

    result = run_script(
        "--repo",
        "ag3kj",
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
        assert "Node filter   : all direct nodes" in result.stdout
        if "--mihomo-group" in ranking_arguments:
            assert "Mihomo group  : download" in result.stdout
        else:
            assert "Mihomo group" not in result.stdout


def test_mihomo_environment_and_cli_precedence(tmp_path: Path) -> None:
    """Use environment overrides and prefer explicit CLI values."""

    environment = python_stub_environment(tmp_path)
    environment["MIHOMO_CONTROLLER"] = "http://127.0.0.1:9090"
    environment["MIHOMO_GROUP"] = "environment-group"
    result = run_script(
        "--repo",
        "ag3kj",
        "--dest",
        str((tmp_path / "dataset").resolve()),
        "--proxy-port",
        "7893",
        "--mihomo-controller",
        "http://127.0.0.1:9091",
        "--mihomo-speed-test-url",
        "https://example.com/large.bin",
        env=environment,
    )

    assert result.returncode == 0
    assert "Mihomo API    : http://127.0.0.1:9091" in result.stdout
    assert "Mihomo group  : environment-group" in result.stdout


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (("--dest",), "--dest requires a value"),
        (("--dest", "/tmp/test"), "--repo is required"),
        (("--project", "ag3kj"), "--dest is required"),
        (
            ("--project", "ag3kj", "--dest", "relative/path"),
            "--dest must be an absolute path",
        ),
        (
            (
                "--project",
                "ag3kj",
                "--dest",
                "/tmp/test",
                "--max-workers",
                "0",
            ),
            "--max-workers must be a positive integer",
        ),
        (
            (
                "--project",
                "ag3kj",
                "--dest",
                "/tmp/test",
                "--proxy-port",
                "65536",
            ),
            "--proxy-port must not exceed 65535",
        ),
        (
            (
                "--project",
                "ag3kj",
                "--dest",
                "/tmp/test",
                "--proxy-port",
                "7893",
                "--mihomo-speed-test-url",
                "https://example.com/large.bin",
            ),
            "Set --mihomo-controller or MIHOMO_CONTROLLER",
        ),
        (
            (
                "--project",
                "ag3kj",
                "--dest",
                "/tmp/test",
                "--proxy-port",
                "7893",
                "--mihomo-controller",
                "ftp://bad",
                "--mihomo-speed-test-url",
                "https://example.com/large.bin",
            ),
            "--mihomo-controller must be an HTTP or HTTPS URL",
        ),
        (("--unknown",), "Unknown option: --unknown"),
    ],
)
def test_invalid_shell_arguments_fail_before_download(
    arguments: tuple[str, ...],
    expected_error: str,
) -> None:
    """Reject invalid shell options before filesystem or network work."""

    result = run_script(*arguments)

    assert result.returncode == 1
    assert expected_error in result.stderr
