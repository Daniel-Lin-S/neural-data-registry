"""Tests for the resumable Zenodo downloader."""

from __future__ import annotations

import hashlib
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
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "download_zenodo.sh"
MODULE_PATH = REPOSITORY_ROOT / "download_helpers" / "download_zenodo.py"


def load_module() -> Any:
    """Load the Zenodo downloader from its repository path."""

    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "download_zenodo",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load Zenodo downloader from {MODULE_PATH}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


downloader = load_module()
import download_http_manifest


def make_config(tmp_path: Path, *, retry_attempts: int = 3) -> Any:
    """Create a valid direct-download configuration."""

    return downloader.DownloadConfig(
        record_id="583331",
        destination=str((tmp_path / "dataset").resolve()),
        api_base="https://zenodo.org/api",
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


def client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[[], httpx.Client]:
    """Build HTTPX clients backed by deterministic mock transport."""

    transport = httpx.MockTransport(handler)
    return lambda: httpx.Client(
        transport=transport,
        follow_redirects=True,
    )


def run_script(
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the Zenodo shell entrypoint and capture output."""

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def python_stub_environment(tmp_path: Path) -> dict[str, str]:
    """Return an environment whose Python prints exported proxy state."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'stub:%s|%s\\n' \"$DOWNLOAD_PROXY_URL\" "
        "\"${HTTPS_PROXY-unset}\"\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    return environment


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("583331", "583331"),
        ("https://zenodo.org/records/583331", "583331"),
        ("https://zenodo.org/record/583331/", "583331"),
    ],
)
def test_parse_record_id(value: str, expected: str) -> None:
    """Accept numeric record IDs and current or legacy record URLs."""

    assert downloader.parse_record_id(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "abc", "https://zenodo.org/records/abc", "ftp://x/record/1"],
)
def test_parse_record_id_rejects_invalid_values(value: str) -> None:
    """Reject malformed IDs and unsupported URLs before network access."""

    with pytest.raises(ValueError):
        downloader.parse_record_id(value)


def test_manifest_supports_current_file_list(tmp_path: Path) -> None:
    """Parse and sort the current Zenodo list representation."""

    config = make_config(tmp_path)
    digest = hashlib.md5(b"abc").hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://zenodo.org/api/records/583331"
        return httpx.Response(
            200,
            json={
                "files": [
                    {
                        "key": "sub-01/data.bin",
                        "size": 3,
                        "checksum": f"md5:{digest}",
                        "links": {"content": "https://files.zenodo.org/a"},
                    },
                    {
                        "key": "README",
                        "size": 2,
                        "checksum": None,
                        "links": {"content": "https://files.zenodo.org/b"},
                    },
                ]
            },
        )

    manifest = downloader.collect_manifest_once(
        config,
        client_factory(handler),
    )

    assert [record.relative_path for record in manifest] == [
        "README",
        "sub-01/data.bin",
    ]
    assert manifest[1].checksum == f"md5:{digest}"


def test_manifest_supports_legacy_entries_layout(tmp_path: Path) -> None:
    """Parse the legacy files.entries mapping."""

    config = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "files": {
                    "entries": {
                        "README": {
                            "filename": "README",
                            "size": 2,
                            "links": {
                                "self": "https://files.zenodo.org/readme"
                            },
                        }
                    }
                }
            },
        )

    manifest = downloader.collect_manifest_once(
        config,
        client_factory(handler),
    )
    assert [record.relative_path for record in manifest] == ["README"]


def test_manifest_rejects_unsafe_paths(tmp_path: Path) -> None:
    """Reject traversal paths before creating destination content."""

    config = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "files": [
                    {
                        "key": "../outside",
                        "size": 1,
                        "links": {"content": "https://files.zenodo.org/x"},
                    }
                ]
            },
        )

    with pytest.raises(ValueError, match="unsafe Zenodo path"):
        downloader.collect_manifest_once(config, client_factory(handler))


def test_resumed_file_is_size_and_checksum_verified(tmp_path: Path) -> None:
    """Resume a partial body and publish only after checksum verification."""

    config = make_config(tmp_path)
    digest = hashlib.md5(b"abcdef").hexdigest()
    record = downloader.HTTPFile(
        "data.bin",
        6,
        "https://files.zenodo.org/data",
        f"md5:{digest}",
    )
    destination = Path(config.destination) / "data.bin"
    destination.parent.mkdir(parents=True)
    destination.with_name("data.bin.part").write_bytes(b"abc")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=3-"
        return httpx.Response(
            206,
            headers={"Content-Range": "bytes 3-5/6"},
            content=b"def",
        )

    result = download_http_manifest.download_file_once(
        config,
        record,
        client_factory(handler),
        "Zenodo",
    )

    assert result.read_bytes() == b"abcdef"
    assert not destination.with_name("data.bin.part").exists()


def test_checksum_mismatch_never_publishes_file(tmp_path: Path) -> None:
    """Keep a bad partial file and reject its advertised checksum."""

    config = make_config(tmp_path)
    record = downloader.HTTPFile(
        "data.bin",
        3,
        "https://files.zenodo.org/data",
        f"md5:{hashlib.md5(b'xyz').hexdigest()}",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_http_manifest.download_file_once(
            config,
            record,
            client_factory(
                lambda request: httpx.Response(200, content=b"abc")
            ),
            "Zenodo",
        )

    assert not (Path(config.destination) / "data.bin").exists()


def test_mihomo_ranking_requires_serial_downloads(tmp_path: Path) -> None:
    """Reject node switching while concurrent transfers are active."""

    config = make_config(tmp_path)
    mihomo = downloader.MihomoConfig(
        controller_url="http://127.0.0.1:9091",
        group_name="download",
        node_marker="",
        speed_test_url="https://files.zenodo.org/large",
        probe_timeout=8.0,
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


def test_help_uses_shared_hugging_face_interface() -> None:
    """Expose common options and omit Hugging Face-only Xet controls."""

    result = run_script("--help")

    assert result.returncode == 0
    assert "--repo ID_OR_URL --dest PATH" in result.stdout
    assert "--mirror URL" in result.stdout
    assert "--proxy-port PORT" in result.stdout
    assert "--retry-base-delay SEC" in result.stdout
    assert "--mihomo-controller URL" in result.stdout
    assert "--transport" not in result.stdout
    assert "--xet-range-concurrency" not in result.stdout


def test_explicit_proxy_is_exported_to_provider(tmp_path: Path) -> None:
    """Pass the common proxy controls through to provider processes."""

    result = run_script(
        "--repo",
        "583331",
        "--dest",
        str((tmp_path / "dataset").resolve()),
        "--proxy-port",
        "7893",
        env=python_stub_environment(tmp_path),
    )

    assert result.returncode == 0
    expected = "http://127.0.0.1:7893"
    assert f"stub:{expected}|{expected}" in result.stdout


def test_no_proxy_clears_ambient_proxy(tmp_path: Path) -> None:
    """Ignore inherited proxy variables in explicit direct mode."""

    environment = python_stub_environment(tmp_path)
    environment["HTTPS_PROXY"] = "http://ambient.example:8080"
    result = run_script(
        "--repo",
        "583331",
        "--dest",
        str((tmp_path / "dataset").resolve()),
        "--no-proxy",
        env=environment,
    )

    assert result.returncode == 0
    assert "stub:|unset" in result.stdout


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (("--dest", "/tmp/test"), "--repo is required"),
        (("--repo", "583331"), "--dest is required"),
        (
            ("--repo", "583331", "--dest", "relative"),
            "--dest must be an absolute path",
        ),
        (
            (
                "--repo",
                "583331",
                "--dest",
                "/tmp/test",
                "--proxy-port",
                "65536",
            ),
            "--proxy-port must not exceed 65535",
        ),
        (("--unknown",), "Unknown option"),
    ],
)
def test_invalid_shell_arguments_fail_before_provider(
    arguments: tuple[str, ...],
    error: str,
) -> None:
    """Reject invalid common options before provider or filesystem work."""

    result = run_script(*arguments)
    assert result.returncode == 1
    assert error in result.stderr
