"""Classify and schedule retries for resumable dataset downloads.

Input
-----
An exception raised by HTTPX, a provider client, or Mihomo and the active
transport name. Retry-delay calculation accepts any configuration object with
``retry_base_delay`` and ``retry_max_delay`` numeric attributes.

Output
------
Failures are classified as terminal, rate-limited, server-side, network, or
Mihomo availability errors. Retry decisions state whether a node should be
rotated and preserve a valid HTTP ``Retry-After`` delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import ssl
import sys
import time
from typing import Callable, Protocol, TypeVar

import httpx

from mihomo_ranker import MihomoUnavailableError


RETRYABLE_XET_ERROR_MARKERS = (
    "cas service error",
    "connection",
    "network",
    "reqwest",
    "timed out",
    "timeout",
)
TERMINAL_HTTP_STATUS_CODES = frozenset({400, 401, 403, 404})


Result = TypeVar("Result")


class RetryConfig(Protocol):
    """Describe the configuration fields needed for retry execution."""

    retry_attempts: int
    retry_base_delay: float
    retry_max_delay: float


class NodeManager(Protocol):
    """Describe Mihomo operations used by the shared retry loop."""

    def prepare_attempt(self) -> str:
        """Prepare a permitted node for one provider operation."""

    def failover(self) -> str | None:
        """Select another permitted node after a network failure."""


@dataclass(frozen=True)
class RetryDecision:
    """Describe how one failed download operation should be retried."""

    retryable: bool
    rotate_node: bool
    category: str
    retry_after: float | None = None


def http_status_code(error: BaseException) -> int | None:
    """Return an HTTP response status associated with an exception."""

    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def exception_chain(error: BaseException) -> list[BaseException]:
    """Return an exception and its explicit or implicit causes."""

    chain = []
    pending: BaseException | None = error
    seen: set[int] = set()
    while pending is not None and id(pending) not in seen:
        chain.append(pending)
        seen.add(id(pending))
        pending = pending.__cause__ or pending.__context__
    return chain


def retry_after_seconds(error: BaseException) -> float | None:
    """Parse an HTTP Retry-After header as a non-negative delay."""

    response = getattr(error, "response", None)
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (retry_at - now).total_seconds())


def classify_download_error(
    error: BaseException,
    transport: str,
) -> RetryDecision:
    """Classify chained failures and whether node rotation is appropriate."""

    chain = exception_chain(error)
    statuses = [
        status
        for item in chain
        if (status := http_status_code(item)) is not None
    ]
    if any(status in TERMINAL_HTTP_STATUS_CODES for status in statuses):
        return RetryDecision(False, False, "terminal")
    if 429 in statuses:
        delay = next(
            (
                retry_after_seconds(item)
                for item in chain
                if http_status_code(item) == 429
            ),
            None,
        )
        return RetryDecision(True, False, "rate_limit", delay)
    if any(500 <= status <= 599 for status in statuses):
        return RetryDecision(True, False, "server")
    if 408 in statuses:
        return RetryDecision(True, True, "network")
    if any(isinstance(item, MihomoUnavailableError) for item in chain):
        return RetryDecision(True, False, "mihomo")

    network_types = (
        httpx.TransportError,
        ConnectionError,
        TimeoutError,
        ssl.SSLError,
    )
    if any(isinstance(item, network_types) for item in chain):
        return RetryDecision(True, True, "network")

    if transport == "xet":
        for item in chain:
            if not isinstance(item, RuntimeError):
                continue
            message = str(item).lower()
            if any(
                marker in message
                for marker in RETRYABLE_XET_ERROR_MARKERS
            ):
                return RetryDecision(True, True, "network")
    return RetryDecision(False, False, "terminal")


def is_retryable_download_error(
    error: BaseException,
    transport: str,
) -> bool:
    """Return whether the failed download operation should be retried."""

    return classify_download_error(error, transport).retryable


def retry_delay(
    config: RetryConfig,
    failed_attempt: int,
) -> float:
    """Calculate a capped exponential retry delay in seconds."""

    delay = config.retry_base_delay * (2 ** (failed_attempt - 1))
    return min(delay, config.retry_max_delay)


def run_with_retries(
    config: RetryConfig,
    operation: str,
    callback: Callable[[], Result],
    *,
    transport: str = "http",
    node_manager: NodeManager | None = None,
    before_retry: Callable[[], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Result:
    """Run one provider operation with shared retry and failover policy.

    Parameters
    ----------
    config : RetryConfig
        Retry attempt and delay settings.
    operation : str
        User-facing operation name.
    callback : callable
        Provider operation executed on every attempt.
    transport : str, optional
        Active transport name, default ``"http"``.
    node_manager : NodeManager or None, optional
        Mihomo node manager, default ``None``.
    before_retry : callable or None, optional
        Connection cleanup called after transient failures, default ``None``.
    sleep_fn : callable, optional
        Delay implementation, default :func:`time.sleep`.

    Returns
    -------
    Result
        Value returned by ``callback``.
    """

    attempt = 1
    consecutive_server_failures = 0
    while config.retry_attempts == 0 or attempt <= config.retry_attempts:
        try:
            if node_manager is not None:
                node_manager.prepare_attempt()
            return callback()
        except Exception as error:
            decision = classify_download_error(error, transport)
            if not decision.retryable:
                raise
            if (
                config.retry_attempts > 0
                and attempt == config.retry_attempts
            ):
                raise RuntimeError(
                    f"{operation} failed after {attempt} attempts; "
                    f"last error was {type(error).__name__}: {error}"
                ) from error

            if before_retry is not None:
                before_retry()
            rotate_node = decision.rotate_node
            if decision.category == "server":
                consecutive_server_failures += 1
                rotate_node = consecutive_server_failures >= 2
            else:
                consecutive_server_failures = 0
            if node_manager is not None and rotate_node:
                try:
                    node_manager.failover()
                except Exception as failover_error:
                    print(
                        "WARNING: Mihomo failover preparation failed with "
                        f"{type(failover_error).__name__}: "
                        f"{failover_error}",
                        file=sys.stderr,
                    )

            delay = retry_delay(config, attempt)
            if decision.retry_after is not None:
                delay = max(delay, decision.retry_after)
            attempt_limit = (
                "unlimited"
                if config.retry_attempts == 0
                else str(config.retry_attempts)
            )
            print(
                f"WARNING: {operation} attempt {attempt} of "
                f"{attempt_limit} failed with "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            print(
                f"Retrying in {delay:g} seconds using existing "
                "partial files.",
                file=sys.stderr,
            )
            sleep_fn(delay)
            attempt += 1

    raise RuntimeError("retry loop ended without a result.")
