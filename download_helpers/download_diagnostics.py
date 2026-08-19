"""Write sanitized diagnostics for standalone dataset downloaders.

Input
-----
A provider name, configuration mapping, retry events, exceptions, and process
signals emitted by scripts beneath ``scripts/``.

Output
------
One append-only UTF-8 log per invocation beneath ``logs/downloads`` at the
repository root. Sensitive headers, credentials, tokens, cookies, and URL
query strings are removed before any value is written.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
import signal
import threading
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

LOGGER_NAME = "neural_data_registry.download"
LOG_DIRECTORY_PARTS = ("logs", "downloads")
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(authorization|cookie|password|secret|token|api[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
URL_PATTERN = re.compile(r"https?://[^\s\]\[\)\(\}\{<>\"']+")
SENSITIVE_FIELD_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
    }
)

SAFE_CONFIGURATION_KEYS = frozenset(
    {
        "destination",
        "dry_run",
        "endpoint",
        "max_workers",
        "provider",
        "proxy_url",
        "repo",
        "retry_attempts",
        "retry_base_delay",
        "retry_max_delay",
        "storage",
        "timeout",
        "transport",
        "version",
    }
)
DEPENDENCY_PACKAGES = (
    "datalad",
    "git-annex",
    "hf-xet",
    "httpx",
    "huggingface-hub",
)

_logger: logging.Logger | None = None
_log_path: Path | None = None
_state_lock = threading.Lock()


class DownloadInterrupted(KeyboardInterrupt):
    """Represent a process signal recorded by the diagnostic logger."""




def sanitize_url(value: str) -> str:
    """Remove credentials, query parameters, and fragments from one URL."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if parsed.scheme not in {"http", "https", "socks5", "socks5h"}:
        return value
    if port is not None:
        hostname = f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def sanitize_text(value: Any) -> str:
    """Return one diagnostic value with likely credentials removed."""

    text = str(value)
    text = SECRET_ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", text)
    text = BEARER_PATTERN.sub("Bearer <redacted>", text)
    return URL_PATTERN.sub(
        lambda match: sanitize_url(match.group(0)),
        text,
    )


def sanitized_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, str]:
    """Return allow-listed, redacted configuration values for logging."""

    return {
        key: sanitize_text(value)
        for key, value in sorted(configuration.items())
        if key in SAFE_CONFIGURATION_KEYS
    }


def dependency_versions() -> dict[str, str]:
    """Return installed downloader dependency versions without failing."""

    versions = {}
    for package in DEPENDENCY_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def repository_root() -> Path:
    """Return the absolute repository root containing this module."""

    return Path(__file__).resolve().parent.parent


def create_log_path(provider: str) -> Path:
    """Create the ignored download-log directory and unique log path."""

    log_directory = repository_root().joinpath(*LOG_DIRECTORY_PARTS)
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return log_directory / f"{provider}-{timestamp}-{os.getpid()}.log"


def configure_diagnostics(
    provider: str,
    configuration: Mapping[str, Any],
) -> Path:
    """Configure one flushed file logger and process signal handlers."""

    global _logger, _log_path
    with _state_lock:
        path = create_log_path(provider).resolve()
        logger = logging.getLogger(LOGGER_NAME)
        for existing_handler in logger.handlers:
            existing_handler.close()
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = logging.FileHandler(path, encoding="utf-8", delay=False)
        formatter = logging.Formatter(
            "%(asctime)sZ %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _logger = logger
        _log_path = path

    log_event(
        "download_started",
        provider=provider,
        pid=os.getpid(),
        configuration=sanitized_configuration(configuration),
        dependencies=dependency_versions(),
    )
    install_signal_handlers()
    return path


def current_log_path() -> Path | None:
    """Return the configured absolute log path, if logging has started."""

    return _log_path


def log_event(event: str, **fields: Any) -> None:
    """Append one sanitized structured event and flush it immediately."""

    if _logger is None:
        return
    rendered = " ".join(
        f"{key}=" + (
            "<redacted>"
            if key.lower() in SENSITIVE_FIELD_KEYS
            else sanitize_text(value)
        )
        for key, value in sorted(fields.items())
    )
    message = event if not rendered else f"{event} {rendered}"
    _logger.info(message)
    for handler in _logger.handlers:
        handler.flush()


def format_exception(error: BaseException) -> str:
    """Return a sanitized traceback including chained exceptions."""

    lines = traceback.format_exception(
        type(error),
        error,
        error.__traceback__,
        chain=True,
    )
    return sanitize_text("".join(lines))


def log_exception(event: str, error: BaseException, **fields: Any) -> None:
    """Append one structured event and its complete sanitized traceback."""

    if _logger is None:
        return
    log_event(
        event,
        exception_type=type(error).__name__,
        exception=sanitize_text(error),
        **fields,
    )
    _logger.error("traceback\n%s", format_exception(error).rstrip())
    for handler in _logger.handlers:
        handler.flush()


def signal_handler(
    signum: int,
    frame: FrameType | None,
) -> None:
    """Record an interrupting signal before unwinding the downloader."""

    del frame
    signal_name = signal.Signals(signum).name
    log_event("download_interrupted", signal=signal_name)
    raise DownloadInterrupted(f"received {signal_name}")


def install_signal_handlers() -> None:
    """Install diagnostic handlers on the main thread where supported."""

    if threading.current_thread() is not threading.main_thread():
        return
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, signal_handler)
