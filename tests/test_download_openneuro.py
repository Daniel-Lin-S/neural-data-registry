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
        exclude_patterns=(),
    )


def install_git_metadata_stub(
    monkeypatch: pytest.MonkeyPatch,
    config: Any,
    *,
    head: str = "0123456789abcdef",
    origin: str | None = None,
    annex_uuid: str = "annex-uuid",
    version: str | None = None,
) -> list[list[str]]:
    """Install a deterministic Git metadata runner for one checkout."""

    commands: list[list[str]] = []
    values = {
        ("rev-parse", "--verify", "HEAD"): head,
        ("config", "--get", "remote.origin.url"): (
            origin
            if origin is not None
            else downloader.repository_url(config)
        ),
        ("config", "--get", "annex.uuid"): annex_uuid,
        ("describe", "--tags", "--exact-match"): (
            version
            if version is not None
            else config.version or ""
        ),
    }

    def fake_run(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(list(command))
        arguments = tuple(command[3:])
        value = values.get(arguments, "")
        return subprocess.CompletedProcess(
            command,
            0 if value else 1,
            stdout=f"{value}\n" if value else "",
            stderr="" if value else "metadata unavailable\n",
        )

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    return commands


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


def python_stub_environment(
    tmp_path: Path,
    command_name: str = "python",
) -> dict[str, str]:
    """Return an environment whose Python prints exported proxy state."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / command_name
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


def test_shell_wrapper_falls_back_to_python3(tmp_path: Path) -> None:
    """Use python3 when the unversioned Python command is unavailable."""

    environment = python_stub_environment(
        tmp_path,
        command_name="python3",
    )
    environment["PATH"] = (
        f"{tmp_path / 'bin'}:/usr/bin:/bin"
    )

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


def test_shell_wrapper_honors_explicit_python(
    tmp_path: Path,
) -> None:
    """Use the configured environment instead of an unrelated python3."""

    environment = python_stub_environment(
        tmp_path,
        command_name="provider-python",
    )
    environment["DOWNLOAD_PYTHON"] = str(
        tmp_path / "bin" / "provider-python"
    )
    environment["PATH"] = "/usr/bin:/bin"

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


def test_load_exclude_patterns_preserves_each_glob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode the shell's newline-delimited exclusion transport."""

    monkeypatch.setenv(
        "DOWNLOAD_EXCLUDE_PATTERNS",
        "derivatives/**\ncode/**\nsub-*/func/**\n",
    )

    assert downloader.load_exclude_patterns() == (
        "derivatives/**",
        "code/**",
        "sub-*/func/**",
    )


@pytest.mark.parametrize(
    "pattern",
    ["/derivatives/**", "sub-01/../code/**"],
)
def test_validate_config_rejects_non_relative_exclusion(
    pattern: str,
    tmp_path: Path,
) -> None:
    """Reject exclusion patterns that escape repository-relative matching."""

    config = replace(
        make_config(tmp_path),
        exclude_patterns=(pattern,),
    )

    with pytest.raises(ValueError, match="exclusion patterns"):
        downloader.validate_config(config)


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


def test_clone_workspaces_are_unique_destination_siblings(
    tmp_path: Path,
) -> None:
    """Create a distinct sibling staging root for every invocation."""

    config = make_config(tmp_path)
    first = downloader.create_clone_workspace(config)
    second = downloader.create_clone_workspace(config)
    try:
        assert first.staging_root is not None
        assert second.staging_root is not None
        assert first.staging_root != second.staging_root
        assert first.staging_root.parent == Path(config.destination).parent
        assert second.staging_root.parent == Path(config.destination).parent
        assert first.staged_destination == (
            first.staging_root / downloader.STAGED_DATASET_NAME
        )
    finally:
        downloader.cleanup_clone_workspace(first)
        downloader.cleanup_clone_workspace(second)


def test_partial_staged_clone_is_replaced_on_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delete only an owned partial clone before retrying the install."""

    config = replace(
        make_config(tmp_path),
        retry_attempts=0,
        retry_base_delay=0.001,
        retry_max_delay=0.001,
    )
    install_git_metadata_stub(monkeypatch, config)
    workspace = downloader.create_clone_workspace(config)
    staging_root = workspace.staging_root
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        del environment
        calls.append(list(command))
        staged_destination = Path(command[-1])
        assert staged_destination == workspace.staged_destination
        if len(calls) == 1:
            staged_destination.mkdir()
            (staged_destination / "partial").write_text(
                "incomplete",
                encoding="utf-8",
            )
            raise downloader.NetworkCommandError("clone disconnected")
        assert not staged_destination.exists()
        (staged_destination / ".git").mkdir(parents=True)
        return ""

    try:
        downloader.run_with_retries(
            config,
            "OpenNeuro repository clone",
            lambda: downloader.install_dataset_once(
                config,
                runner,
                workspace,
            ),
            transport="datalad",
            sleep_fn=lambda _delay: None,
        )

        destination = Path(config.destination)
        assert len(calls) == 2
        assert (destination / ".git").is_dir()
        assert not (destination / "partial").exists()
    finally:
        downloader.cleanup_clone_workspace(workspace)

    assert staging_root is not None
    assert not staging_root.exists()


def test_valid_staged_clone_is_promoted_after_runner_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep a complete clone when DataLad reports a trailing error."""

    config = make_config(tmp_path)
    install_git_metadata_stub(monkeypatch, config)
    workspace = downloader.create_clone_workspace(config)
    calls = 0

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        nonlocal calls
        del environment
        calls += 1
        (Path(command[-1]) / ".git").mkdir(parents=True)
        raise downloader.NetworkCommandError("annex setup timed out")

    try:
        downloader.install_dataset_once(config, runner, workspace)
    finally:
        downloader.cleanup_clone_workspace(workspace)

    assert calls == 1
    assert (Path(config.destination) / ".git").is_dir()


def test_preexisting_invalid_destination_is_preserved(
    tmp_path: Path,
) -> None:
    """Reject an initial conflicting destination without modifying it."""

    config = make_config(tmp_path)
    destination = Path(config.destination)
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    workspace = downloader.create_clone_workspace(config)

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        del command, environment
        raise AssertionError("provider command must not run")

    with pytest.raises(FileExistsError, match="not a DataLad dataset"):
        downloader.install_dataset_once(config, runner, workspace)
    downloader.cleanup_clone_workspace(workspace)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert workspace.staging_root is None


def test_destination_race_is_preserved_and_terminal(tmp_path: Path) -> None:
    """Refuse promotion when a destination appears during this invocation."""

    config = make_config(tmp_path)
    workspace = downloader.create_clone_workspace(config)
    destination = Path(config.destination)
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        del command, environment
        raise AssertionError("provider command must not run")

    try:
        with pytest.raises(FileExistsError, match="appeared while cloning"):
            downloader.install_dataset_once(config, runner, workspace)
    finally:
        downloader.cleanup_clone_workspace(workspace)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert workspace.staging_root is not None
    assert not workspace.staging_root.exists()


def test_atomic_promotion_does_not_replace_empty_destination(
    tmp_path: Path,
) -> None:
    """Use a no-clobber rename even when the competing path is empty."""

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError):
        downloader.rename_without_replacement(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_retrieve_content_passes_worker_count(tmp_path: Path) -> None:
    """Forward common worker configuration to git-annex get."""

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
        "git",
        "-C",
        config.destination,
        "-c",
        "annex.stalldetection-download=1KB/300s",
        "annex",
        "get",
        "--jobs",
        "2",
        ".",
    ]


def test_retrieval_and_verification_share_exclusions(
    tmp_path: Path,
) -> None:
    """Apply the same annex filters to transfer and completeness checks."""

    config = replace(
        make_config(tmp_path),
        exclude_patterns=(
            "derivatives/**",
            "code/**",
            "sub-*/func/**",
        ),
    )
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        del environment
        calls.append(list(command))
        return ""

    downloader.retrieve_and_verify_content_once(config, runner)

    expected_options = [
        "--exclude=derivatives/**",
        "--exclude=code/**",
        "--exclude=sub-*/func/**",
    ]
    assert calls[0][-4:] == [*expected_options, "."]
    assert calls[1][-3:] == expected_options


def test_annex_stall_detection_uses_download_timeout(
    tmp_path: Path,
) -> None:
    """Cancel annex downloads that make no useful progress in time."""

    config = replace(make_config(tmp_path), timeout=45.9)

    assert downloader.annex_stall_detection(config) == "1KB/45s"


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
    ["", "transfer helper exited without a recognized diagnostic"],
)
def test_unknown_provider_command_failure_is_retryable(message: str) -> None:
    """Retry provider subprocess failures without known terminal evidence."""

    error = downloader.classify_command_failure(["datalad"], 1, message)

    assert isinstance(error, downloader.NetworkCommandError)


@pytest.mark.parametrize(
    "message",
    [
        "authentication failed",
        "repository not found",
        "no space left on device",
        "permission denied",
        (
            "destination path 'dataset' already exists and is not an "
            "empty directory"
        ),
        "path not associated with any dataset",
        "fatal: not a git repository",
        "Target path already exists and not empty, refuse to clone into",
        "fatal: couldn't find remote ref 9.9.9",
    ],
)
def test_terminal_command_failures_stay_terminal(message: str) -> None:
    """Do not hide credentials, dataset, or local storage errors in retries."""

    error = downloader.classify_command_failure(["datalad"], 1, message)
    assert type(error) is downloader.TerminalDownloadError


def test_missing_annex_content_is_retryable(tmp_path: Path) -> None:
    """Treat an incomplete annex result as an interrupted transfer."""

    config = make_config(tmp_path)
    outputs = iter(["", "sub-01/eeg/file.edf\n"])

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        del command, environment
        return next(outputs)

    with pytest.raises(
        downloader.NetworkCommandError,
        match="1 unavailable annex files",
    ):
        downloader.retrieve_and_verify_content_once(config, runner)


def test_run_retries_retrieval_verification_with_datalad_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep retrieval and annex verification in one DataLad retry unit."""

    config = replace(
        make_config(tmp_path),
        retry_attempts=2,
        retry_base_delay=0.001,
        retry_max_delay=0.001,
    )
    original_run_with_retries = downloader.run_with_retries
    transports: list[str] = []
    retrieval_attempts = 0
    verification_attempts = 0

    def run_immediately(
        retry_config: Any,
        operation: str,
        callback: Any,
        *,
        transport: str = "http",
        node_manager: Any = None,
    ) -> Any:
        transports.append(transport)
        return original_run_with_retries(
            retry_config,
            operation,
            callback,
            transport=transport,
            node_manager=node_manager,
            sleep_fn=lambda _delay: None,
        )

    def runner(
        command: Sequence[str],
        environment: dict[str, str],
    ) -> str:
        nonlocal retrieval_attempts, verification_attempts
        del environment
        if "get" in command:
            retrieval_attempts += 1
            return ""
        if "find" in command:
            verification_attempts += 1
            if verification_attempts == 1:
                return "sub-01/eeg/file.edf\n"
            return ""
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(downloader, "check_dependencies", lambda: None)
    monkeypatch.setattr(downloader, "install_dataset_once", lambda *_: None)
    monkeypatch.setattr(
        downloader,
        "run_with_retries",
        run_immediately,
    )

    downloader.run(config, runner)

    assert transports == ["datalad", "datalad"]
    assert retrieval_attempts == 2
    assert verification_attempts == 2


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
    assert "--exclude GLOB" in result.stdout
    assert "--transport" not in result.stdout
    assert "--xet-range-concurrency" not in result.stdout


def test_shell_exports_repeatable_exclusions(tmp_path: Path) -> None:
    """Preserve each OpenNeuro exclusion as one provider pattern."""

    environment = python_stub_environment(tmp_path)
    stub = tmp_path / "bin" / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s' \"$DOWNLOAD_EXCLUDE_PATTERNS\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    result = run_script(
        "--repo",
        "ds004078",
        "--dest",
        str((tmp_path / "dataset").resolve()),
        "--exclude",
        "derivatives/**",
        "--exclude",
        "sub-*/func/**",
        "--no-proxy",
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout.endswith(
        "derivatives/**\nsub-*/func/**\n"
    )


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


@pytest.mark.parametrize(
    ("return_code", "message"),
    (
        (2, "connection reset while parsing arguments"),
        (126, "permission denied"),
        (127, "command not found"),
        (129, "usage: git clone [options]"),
        (130, "connection reset by peer"),
        (-9, "command produced no diagnostic output"),
    ),
)
def test_deterministic_command_exit_is_terminal(
    return_code: int,
    message: str,
) -> None:
    """Do not retry usage, missing-command, or signal exits."""

    error = downloader.classify_command_failure(
        ["datalad"],
        return_code,
        message,
    )

    assert type(error) is downloader.TerminalDownloadError


@pytest.mark.parametrize(
    "message",
    (
        "remote HEAD refers to nonexistent ref",
        "not enough free space to retrieve content",
        "curl: (3) URL rejected: malformed input",
        "pathspec 'missing' did not match any file known to git",
        "Traceback (most recent call last): TypeError: invalid state",
        (
            "server certificate verification failed. CAfile: none "
            "CRLfile: none"
        ),
        "curl: (60) SSL peer certificate or SSH remote key was not OK",
        "fatal: Operation not permitted",
        "fatal: File name too long",
        "fatal: Filename too long",
        "fatal: Is a directory",
    ),
)
def test_deterministic_command_message_is_terminal(message: str) -> None:
    """Recognize deterministic provider diagnostics before retrying."""

    error = downloader.classify_command_failure(["datalad"], 1, message)

    assert type(error) is downloader.TerminalDownloadError


@pytest.mark.parametrize(
    "message",
    (
        "CONNECT tunnel failed, response 407",
        "curl: (56) Received HTTP code 407 from proxy after CONNECT",
        "All offered SOCKS5 authentication methods were rejected",
    ),
)
def test_proxy_authentication_command_failure_is_terminal(
    message: str,
) -> None:
    """Stop retries for explicit proxy authentication failures."""

    error = downloader.classify_command_failure(["datalad"], 1, message)
    decision = downloader.classify_download_error(error, "datalad")

    assert type(error) is downloader.TerminalDownloadError
    assert not decision.retryable


def test_decorated_missing_remote_branch_is_terminal() -> None:
    """Recognize the missing-branch message inside DataLad decoration."""

    output = (
        "[ERROR] Failed to clone dataset [status=error]\n"
        "stderr='fatal: Remote branch 9.9.9 not found in upstream "
        "origin']]"
    )
    assert "\n" in output

    error = downloader.classify_command_failure(["datalad"], 1, output)

    assert type(error) is downloader.TerminalDownloadError


def test_other_remote_branch_failure_remains_retryable() -> None:
    """Do not make unrelated remote-branch transfer failures terminal."""

    output = (
        "Remote branch 9.9.9 lookup failed after upstream origin timed out"
    )

    error = downloader.classify_command_failure(["datalad"], 1, output)

    assert isinstance(error, downloader.NetworkCommandError)


def test_terminal_command_type_overrides_network_words() -> None:
    """Keep a local provider failure terminal despite proxy text."""

    error = downloader.classify_command_failure(
        ["datalad"],
        1,
        "not enough free space; proxy connection failed",
    )
    decision = downloader.classify_download_error(error, "datalad")

    assert type(error) is downloader.TerminalDownloadError
    assert not decision.retryable
    assert decision.reason == "terminal_exception"


def test_unknown_provider_traceback_is_retryable() -> None:
    """Retry future transfer exceptions not known to be deterministic."""

    error = downloader.classify_command_failure(
        ["datalad"],
        1,
        "Traceback (most recent call last): "
        "NewTransferError: provider stream failed",
    )

    assert isinstance(error, downloader.NetworkCommandError)


def test_warning_filesystem_text_does_not_mask_network_failure() -> None:
    """Ignore DataLad warning records when a later transfer record fails."""

    output = (
        "[WARNING] Failed to (re)set permissions: OSError: [Errno 30] "
        "Read-only file system\n"
        "error: RPC failed; curl 56 Recv failure: Connection reset by peer"
    )

    error = downloader.classify_command_failure(["datalad"], 1, output)

    assert isinstance(error, downloader.NetworkCommandError)


def test_http_status_on_another_record_does_not_override_auth() -> None:
    """Keep fatal authentication terminal despite an earlier HTTP 503."""

    output = "mirror: server returned status 503\nfatal: authentication failed"

    error = downloader.classify_command_failure(["datalad"], 1, output)
    assert type(error) is downloader.TerminalDownloadError


def test_server_status_overrides_terminal_looking_command_text(
    tmp_path: Path,
) -> None:
    """Retry a real server error even when its body mentions a repository."""

    config = replace(
        make_config(tmp_path),
        retry_attempts=2,
        retry_base_delay=0.001,
        retry_max_delay=0.001,
    )
    error = downloader.classify_command_failure(
        ["datalad"],
        1,
        "server returned status 503: repository not found",
    )
    calls = 0

    def callback() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "complete"

    result = downloader.run_with_retries(
        config,
        "OpenNeuro test transfer",
        callback,
        transport="datalad",
        sleep_fn=lambda _delay: None,
    )

    assert result == "complete"
    assert calls == 2


def test_unversioned_existing_checkout_requires_valid_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a clone directory left before Git created HEAD."""

    config = make_config(tmp_path, version=None)
    destination = Path(config.destination)
    (destination / ".git").mkdir(parents=True)
    commands = install_git_metadata_stub(
        monkeypatch,
        config,
        head="",
    )

    with pytest.raises(FileExistsError, match="could not read Git HEAD"):
        downloader.validate_existing_dataset(config)

    assert commands == [
        [
            "git",
            "-C",
            config.destination,
            "rev-parse",
            "--verify",
            "HEAD",
        ]
    ]


def test_unversioned_existing_checkout_reuses_valid_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reuse a successfully cloned unversioned repository."""

    config = make_config(tmp_path, version=None)
    destination = Path(config.destination)
    (destination / ".git").mkdir(parents=True)
    commands = install_git_metadata_stub(monkeypatch, config)

    downloader.validate_existing_dataset(config)

    assert [command[3:] for command in commands] == [
        ["rev-parse", "--verify", "HEAD"],
        ["config", "--get", "remote.origin.url"],
        ["config", "--get", "annex.uuid"],
    ]


def test_existing_checkout_rejects_different_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not reuse a valid Git repository for another dataset."""

    config = make_config(tmp_path, version=None)
    destination = Path(config.destination)
    (destination / ".git").mkdir(parents=True)
    install_git_metadata_stub(
        monkeypatch,
        config,
        origin=(
            "https://github.com/OpenNeuroDatasets/"
            "ds000001.git"
        ),
    )

    with pytest.raises(FileExistsError, match="different repository"):
        downloader.validate_existing_dataset(config)


def test_existing_checkout_requires_git_annex_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject an ordinary Git clone without initialized annex state."""

    config = make_config(tmp_path, version=None)
    destination = Path(config.destination)
    (destination / ".git").mkdir(parents=True)
    install_git_metadata_stub(
        monkeypatch,
        config,
        annex_uuid="",
    )

    with pytest.raises(FileExistsError, match="git-annex UUID"):
        downloader.validate_existing_dataset(config)


def test_repository_url_identity_accepts_git_syntax() -> None:
    """Treat common Git URL spellings as the same repository."""

    https_url = (
        "https://github.com/OpenNeuroDatasets/ds005261.git"
    )
    ssh_url = "git@github.com:OpenNeuroDatasets/ds005261.git"

    assert downloader.normalized_repository_url(https_url) == (
        downloader.normalized_repository_url(ssh_url)
    )
