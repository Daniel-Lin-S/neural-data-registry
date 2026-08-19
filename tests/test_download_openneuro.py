"""Tests for the version-aware OpenNeuro downloader."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "download_openneuro.sh"
MODULE_PATH = REPOSITORY_ROOT / "download_helpers" / "download_openneuro.py"


def load_module() -> Any:
    """Load the OpenNeuro downloader from its repository path."""

    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "download_openneuro",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load OpenNeuro downloader from {MODULE_PATH}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


downloader = load_module()


def make_config(
    tmp_path: Path,
    *,
    version: str | None = "2.0.0",
) -> Any:
    """Create a valid direct-download configuration."""

    return downloader.DownloadConfig(
        dataset_id="ds005261",
        version=version,
        destination=str((tmp_path / "dataset").resolve()),
        endpoint="https://github.com/OpenNeuroDatasets",
        max_workers=2,
        timeout=300.0,
        dry_run=False,
        retry_attempts=3,
        retry_base_delay=5.0,
        retry_max_delay=300.0,
        proxy_url=None,
        mihomo=None,
    )


def run_script(
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the OpenNeuro shell entrypoint and capture output."""

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
        "\"${ALL_PROXY-unset}\"\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    return environment


@pytest.mark.parametrize(
    ("value", "dataset_id", "version"),
    [
        ("ds005261", "ds005261", None),
        (
            "https://openneuro.org/datasets/ds005261",
            "ds005261",
            None,
        ),
        (
            "https://openneuro.org/datasets/ds005261/versions/2.0.0",
            "ds005261",
            "2.0.0",
        ),
    ],
)
def test_parse_repository(
    value: str,
    dataset_id: str,
    version: str | None,
) -> None:
    """Parse IDs and canonical dataset or version URLs."""

    reference = downloader.parse_repository(value)
    assert reference.dataset_id == dataset_id
    assert reference.version == version


@pytest.mark.parametrize(
    "value",
    [
        "",
        "005261",
        "dsABC",
        "ftp://openneuro.org/datasets/ds005261",
        "https://openneuro.org/datasets/ds005261/files",
        "https://openneuro.org/datasets/ds005261/versions/bad/tag",
    ],
)
def test_parse_repository_rejects_invalid_values(value: str) -> None:
    """Reject malformed dataset identifiers and URLs."""

    with pytest.raises(ValueError):
        downloader.parse_repository(value)


def test_repository_url_uses_configured_mirror(tmp_path: Path) -> None:
    """Build the direct git URL from the common mirror option."""

    config = replace(
        make_config(tmp_path),
        endpoint="https://mirror.example/OpenNeuroDatasets",
    )
    assert downloader.repository_url(config) == (
        "https://mirror.example/OpenNeuroDatasets/ds005261.git"
    )


def test_clone_command_pins_version_tag(tmp_path: Path) -> None:
    """Use DataLad install with the version embedded in the dataset URL."""

    config = make_config(tmp_path)
    assert downloader.clone_command(config) == [
        "datalad",
        "install",
        "--branch",
        "2.0.0",
        "--source",
        "https://github.com/OpenNeuroDatasets/ds005261.git",
        config.destination,
    ]


def test_clone_command_uses_default_snapshot_without_version(
    tmp_path: Path,
) -> None:
    """Leave branch selection to OpenNeuro for an unversioned repository."""

    config = make_config(tmp_path, version=None)
    assert "--branch" not in downloader.clone_command(config)


def test_install_retry_reuses_successfully_cloned_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reuse a valid destination created before an install command failed."""

    config = make_config(tmp_path)
    calls: list[list[str]] = []
    validations: list[str] = []

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        del environment
        calls.append(list(command))
        Path(config.destination).mkdir()
        raise downloader.NetworkCommandError("annex setup timed out")

    def validate(candidate: Any) -> None:
        validations.append(candidate.destination)

    monkeypatch.setattr(downloader, "validate_existing_dataset", validate)

    with pytest.raises(downloader.NetworkCommandError):
        downloader.install_dataset_once(config, runner)
    downloader.install_dataset_once(config, runner)

    assert len(calls) == 1
    assert validations == [config.destination]


def test_retrieve_content_passes_worker_count(tmp_path: Path) -> None:
    """Forward common worker configuration to DataLad get."""

    config = make_config(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        calls.append((list(command), environment))
        return ""

    downloader.retrieve_content_once(config, runner)

    assert calls[0][0] == [
        "datalad",
        "-C",
        config.destination,
        "get",
        "--recursive",
        "--jobs",
        "2",
        ".",
    ]


def test_proxy_environment_is_explicit_and_complete(tmp_path: Path) -> None:
    """Pass HTTP and SOCKS proxy settings to git, DataLad, and git-annex."""

    config = replace(
        make_config(tmp_path),
        proxy_url="socks5h://127.0.0.1:7893",
    )
    environment = downloader.subprocess_environment(config)

    assert environment["HTTP_PROXY"] == config.proxy_url
    assert environment["HTTPS_PROXY"] == config.proxy_url
    assert environment["ALL_PROXY"] == config.proxy_url
    assert environment["http_proxy"] == config.proxy_url
    assert environment["GIT_HTTP_LOW_SPEED_TIME"] == "300"


def test_direct_environment_removes_ambient_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep direct mode independent of ambient proxy variables."""

    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.example:8080")
    environment = downloader.subprocess_environment(make_config(tmp_path))
    assert "HTTPS_PROXY" not in environment
    assert "https_proxy" not in environment


@pytest.mark.parametrize(
    "message",
    [
        "Could not resolve host openneuro.org",
        "connection reset by peer",
        "TLS connection timed out",
        "remote end hung up unexpectedly",
    ],
)
def test_network_command_failures_are_retryable(message: str) -> None:
    """Translate DataLad network output into shared retryable errors."""

    error = downloader.classify_command_failure(["datalad"], 1, message)
    assert isinstance(error, downloader.NetworkCommandError)


@pytest.mark.parametrize(
    "message",
    [
        "authentication failed",
        "repository not found",
        "no space left on device",
        "permission denied",
    ],
)
def test_terminal_command_failures_stay_terminal(message: str) -> None:
    """Do not hide credentials, dataset, or local storage errors in retries."""

    error = downloader.classify_command_failure(["datalad"], 1, message)
    assert type(error) is RuntimeError


def test_mihomo_ranking_requires_serial_datalad_jobs(tmp_path: Path) -> None:
    """Reject node switching while concurrent annex jobs are active."""

    config = make_config(tmp_path)
    mihomo = downloader.MihomoConfig(
        controller_url="http://127.0.0.1:9091",
        group_name="download",
        node_marker="",
        speed_test_url="https://openneuro.org/large",
        probe_timeout=8.0,
        secret=None,
    )
    config = replace(
        config,
        proxy_url="http://127.0.0.1:7893",
        mihomo=mihomo,
    )

    with pytest.raises(ValueError, match="max_workers=1"):
        downloader.validate_config(config)


def test_dry_run_does_not_create_missing_destination(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Report pending clone state without changing the destination."""

    config = replace(make_config(tmp_path), dry_run=True)
    downloader.print_dry_run(config)

    output = capsys.readouterr().out
    assert "Repository clone : pending" in output
    assert not Path(config.destination).exists()


def test_help_uses_shared_hugging_face_interface() -> None:
    """Expose common options and omit Hugging Face-only Xet controls."""

    result = run_script("--help")

    assert result.returncode == 0
    assert "--repo ID_OR_URL --dest PATH" in result.stdout
    assert "--mirror URL" in result.stdout
    assert "--proxy-port PORT" in result.stdout
    assert "--retry-max-delay SEC" in result.stdout
    assert "--mihomo-controller URL" in result.stdout
    assert "--transport" not in result.stdout
    assert "--xet-range-concurrency" not in result.stdout


def test_explicit_proxy_is_exported_to_provider(tmp_path: Path) -> None:
    """Pass common proxy controls through to OpenNeuro processes."""

    result = run_script(
        "--repo",
        "https://openneuro.org/datasets/ds005261/versions/2.0.0",
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
    environment["ALL_PROXY"] = "socks5h://ambient.example:1080"
    result = run_script(
        "--repo",
        "ds005261",
        "--dest",
        str((tmp_path / "dataset").resolve()),
        "--no-proxy",
        env=environment,
    )

    assert result.returncode == 0
    assert "stub:|unset" in result.stdout
