"""Tests for persistent sanitized downloader diagnostics."""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY_ROOT / "download_helpers" / "download_diagnostics.py"
)


def load_module() -> Any:
    """Load the diagnostic module from its repository path."""

    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "download_diagnostics_test_module",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load downloader diagnostics from {MODULE_PATH}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnostics = load_module()


def configure_at(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Configure diagnostics beneath a temporary repository root."""

    monkeypatch.setattr(diagnostics, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(diagnostics, "install_signal_handlers", lambda: None)
    return diagnostics.configure_diagnostics(
        "huggingface",
        {
            "repo": "owner/dataset",
            "destination": str(tmp_path / "dataset"),
            "proxy_url": "http://user:password@127.0.0.1:7893",
            "token": "configuration-secret",
        },
    )


def test_log_path_is_unique_absolute_and_under_ignored_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create unique absolute logs beneath logs/downloads."""

    first = configure_at(monkeypatch, tmp_path)
    second = configure_at(monkeypatch, tmp_path)

    assert first.is_absolute()
    assert first.parent == tmp_path / "logs" / "downloads"
    assert second.parent == first.parent
    assert second != first
    assert first.is_file()
    assert second.is_file()


def test_log_redacts_credentials_and_signed_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Never persist tokens, proxy credentials, or signed query strings."""

    path = configure_at(monkeypatch, tmp_path)
    diagnostics.log_event(
        "request_failed",
        authorization="Bearer super-secret-token",
        cookie="session=hidden-cookie",
        url=(
            "https://cdn.example/data.bin?X-Amz-Signature=hidden-signature"
        ),
    )
    content = path.read_text(encoding="utf-8")

    assert "super-secret-token" not in content
    assert "hidden-cookie" not in content
    assert "hidden-signature" not in content
    assert "user:password" not in content
    assert "https://cdn.example/data.bin" in content


def test_exception_log_contains_sanitized_complete_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist chained traceback evidence without signed URL parameters."""

    path = configure_at(monkeypatch, tmp_path)
    try:
        try:
            raise ConnectionResetError(
                "reset at https://cdn.example/file?token=hidden"
            )
        except ConnectionResetError as cause:
            raise RuntimeError("snapshot failed") from cause
    except RuntimeError as error:
        diagnostics.log_exception("download_failed", error)

    content = path.read_text(encoding="utf-8")
    assert "ConnectionResetError" in content
    assert "RuntimeError" in content
    assert "snapshot failed" in content
    assert "hidden" not in content


def test_signal_handler_flushes_interruption_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record an interrupt signal before unwinding the process."""

    path = configure_at(monkeypatch, tmp_path)
    with pytest.raises(diagnostics.DownloadInterrupted):
        diagnostics.signal_handler(signal.SIGTERM, None)

    content = path.read_text(encoding="utf-8")
    assert "download_interrupted" in content
    assert "SIGTERM" in content
