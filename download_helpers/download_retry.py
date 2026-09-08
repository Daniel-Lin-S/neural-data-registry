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

import builtins
import errno
import http.client
import math
import re
import socket
import ssl
import sys
import time
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol, TypeVar
from urllib.parse import urlsplit

import httpx
from download_diagnostics import log_event, log_exception
from mihomo_ranker import MihomoUnavailableError

HTTP_STATUS_PATTERNS = (
    re.compile(
        r"\b(?:http(?:\s+status)?(?:\s+client\s+error)?|"
        r"status(?:[\s_-]+code)?)\s*[:=(]?\s*"
        r"([1-5][0-9]{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brequested url returned error:\s*([1-5][0-9]{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhttp(?:/[0-9.]+)?\s+([1-5][0-9]{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bserver returned (?:http response code |status )?"
        r"([1-5][0-9]{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([1-5][0-9]{2})\s+(?:bad request|unauthorized|"
        r"forbidden|not found|request timeout|too early|"
        r"too many requests|internal server error|bad gateway|"
        r"service unavailable|gateway timeout)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconnect tunnel failed,?\s*response\s+(407)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breceived http code\s+(407)\s+from proxy\b",
        re.IGNORECASE,
    ),
)
HTTP_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
MAX_FAILURE_SIGNATURES = 128
PROXY_HTTP_STATUS_PATTERN = re.compile(
    r"^\s*(407)(?:\s.*)?$",
    re.IGNORECASE,
)
HOSTNAME_MISMATCH_PATTERN = re.compile(
    r"\bhostname .{0,128} doesn't match\b",
    re.IGNORECASE,
)
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429})
TERMINAL_ERROR_MARKERS = (
    "access token is invalid",
    "all offered socks5 authentication methods were rejected",
    "authentication failed",
    "authorization failed",
    "certificate verify failed",
    "certificate verification failed",
    "checksum mismatch",
    "client private key is encrypted",
    "dataset not found",
    "digest mismatch",
    "disk quota exceeded",
    "entry not found",
    "file name too long",
    "fingerprint of invalid length",
    "fingerprints did not match",
    "filename too long",
    "hash function implementation unavailable for fingerprint length",
    "integrity check failed",
    "is a directory",
    "invalid credentials",
    "invalid proxy url",
    "invalid token",
    "invalid username/password",
    "invalid url",
    "no appropriate subjectaltname fields were found",
    "no host given",
    "no space left on device",
    "offline mode is enabled",
    "operation not permitted",
    "outgoing traffic has been disabled",
    "outgoing traffic is disabled",
    "not a git repository",
    "no such file or directory",
    "not a directory",
    "out of memory",
    "panicked at",
    "path already exists and not empty",
    "path not associated with dataset",
    "permission denied",
    "programming error",
    "proxy authentication required",
    "requested username/password from proxy server",
    "read-only file system",
    "record not found",
    "repository not found",
    "revision not found",
    "self-signed certificate",
    "server requested username/password",
    "ssl peer certificate or ssh remote key was not ok",
    "target path already exists",
    "too many open files",
    "unable to get local issuer certificate",
    "unknown option",
    "unknown url type",
    "unrecognized argument",
    "unsupported protocol",
    "url using bad/illegal format",
)
ROOT_TERMINAL_ERROR_MARKERS = (
    "command terminated by signal",
    "invalid command exit status",
    "assertion failed",
    "index out of bounds",
    "index out of range",
    "install required",
    "invalid argument",
    "malformed metadata",
    "not implemented",
    "unsupported operation",
)
TERMINAL_LOCAL_ERRNOS = frozenset(
    code
    for name in (
        "EACCES",
        "EDQUOT",
        "EEXIST",
        "EFBIG",
        "EMFILE",
        "ENAMETOOLONG",
        "ENFILE",
        "ENOENT",
        "ENOSPC",
        "ENOTDIR",
        "EPERM",
        "EROFS",
        "EISDIR",
    )
    if (code := getattr(errno, name, None)) is not None
)
EXCEPTION_GROUP_TYPES = (
    (builtins.BaseExceptionGroup,)
    if hasattr(builtins, "BaseExceptionGroup")
    else ()
)
NETWORK_ERRNOS = frozenset(
    code
    for name in (
        "EADDRNOTAVAIL",
        "ECOMM",
        "ECONNABORTED",
        "ECONNREFUSED",
        "ECONNRESET",
        "EHOSTDOWN",
        "EHOSTUNREACH",
        "ENETDOWN",
        "ENETRESET",
        "ENETUNREACH",
        "ENOBUFS",
        "ENOTCONN",
        "EPIPE",
        "EPROTO",
        "ETIMEDOUT",
    )
    if (code := getattr(errno, name, None)) is not None
)
ROOT_PROGRAMMING_ERROR_TYPES = (
    TypeError,
    ValueError,
)
TERMINAL_EXCEPTION_TYPES = (
    AssertionError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    MemoryError,
    NameError,
    NotImplementedError,
    OverflowError,
    RecursionError,
    SyntaxError,
    SystemError,
    ZeroDivisionError,
    FileExistsError,
    FileNotFoundError,
    PermissionError,
    httpx.InvalidURL,
    httpx.LocalProtocolError,
    httpx.StreamError,
    httpx.TooManyRedirects,
    httpx.UnsupportedProtocol,
)
TERMINAL_LIBRARY_TYPE_NAMES = frozenset(
    {
        "aiohttp.client_exceptions.InvalidURL",
        "aiohttp.client_exceptions.TooManyRedirects",
        "httpcore.LocalProtocolError",
        "huggingface_hub.errors.OfflineModeIsEnabled",
        "httpcore.UnsupportedProtocol",
        "requests.exceptions.InvalidJSONError",
        "requests.exceptions.InvalidSchema",
        "requests.exceptions.InvalidURL",
        "requests.exceptions.MissingSchema",
        "requests.exceptions.StreamConsumedError",
        "requests.exceptions.TooManyRedirects",
        "requests.exceptions.UnrewindableBodyError",
        "requests.exceptions.URLRequired",
        "urllib3.exceptions.BodyNotHttplibCompatible",
        "urllib3.exceptions.LocationParseError",
        "urllib3.exceptions.ProxySchemeUnknown",
        "urllib3.exceptions.ProxySchemeUnsupported",
        "urllib3.exceptions.TimeoutStateError",
        "urllib3.exceptions.UnrewindableBodyError",
        "urllib3.exceptions.URLSchemeUnknown",
    }
)
NETWORK_LIBRARY_BASE_TYPE_NAMES = frozenset(
    {
        "aiohttp.client_exceptions.ClientError",
        "httpcore.NetworkError",
        "httpcore.TimeoutException",
        "requests.exceptions.RequestException",
        "urllib3.exceptions.HTTPError",
    }
)
NETWORK_LIBRARY_MODULE_PREFIXES = (
    "aiohttp.",
    "httpcore",
    "requests.",
    "urllib3.",
)
NETWORK_LIBRARY_TYPE_NAMES = frozenset(
    {
        "ClientConnectionError",
        "ClientConnectorError",
        "ConnectionNotAvailable",
        "ClientOSError",
        "ClientPayloadError",
        "ClientProxyConnectionError",
        "ChunkedEncodingError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ContentDecodingError",
        "DecodeError",
        "IncompleteRead",
        "MaxRetryError",
        "NewConnectionError",
        "PoolTimeout",
        "ProtocolError",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "ReadTimeoutError",
        "RemoteProtocolError",
        "ResponseError",
        "RetryError",
        "ServerConnectionError",
        "ServerDisconnectedError",
        "ServerTimeoutError",
        "SSLError",
        "Timeout",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
)
RETRYABLE_PROVIDER_WRAPPER_NAMES = frozenset(
    {
        "huggingface_hub.errors.DryRunError",
        "huggingface_hub.errors.IncompleteSnapshotError",
        "huggingface_hub.errors.LocalEntryNotFoundError",
        "huggingface_hub.errors.XetDownloadError",
    }
)
XET_TERMINAL_ERROR_MARKERS = (
    "invalid argument (os error",
    "too many levels of symbolic links (os error",
    "unsupported operation (os error",
)
XET_TRANSFER_ENVELOPE_MARKERS = (
    "cas client error",
    "cas service error",
    "data processing error",
    "file reconstruction error",
    "previous task error",
    "reqwest",
    "hyper error",
    "http2 error",
    "request middleware error",
)
SIGNED_URL_QUERY_MARKERS = (
    "expires=",
    "key-pair-id=",
    "policy=",
    "signature=",
    "x-amz-signature=",
    "x-goog-signature=",
    "x-xet-signed-range=",
)
XET_TRANSFER_PATH_MARKERS = (
    "/xet-bridge",
    "/xorb/",
    "/xorbs/",
)
NETWORK_MESSAGE_PATTERNS = (
    re.compile(
        r"\b(?:connection|peer|socket)\b.{0,48}\b(?:aborted|"
        r"closed|error|failed|lost|refused|reset|timed out|"
        r"unreachable)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:aborted|closed|failed|lost|refused|reset)\b"
        r".{0,32}\b(?:connection|peer|socket)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:could not|failed|unable)\s+to\s+(?:connect|read|"
        r"receive|resolve|send|write)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dns|name resolution|resolve host|"
        r"name or service not known|nodename nor servname)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:low speed time|operation timed out|timed out|timeout)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gnutls|handshake|openssl|ssl_error|tls[a-z0-9_]*)\b"
        r".{0,64}\b(?:closed|eof|error|failed|failure|syscall|"
        r"timed out)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:early|unexpected)?\s*eof\b|"
        r"\bunexpected end of file\b|"
        r"\bincomplete (?:message|read|response|body)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:body|channel|stream)\b.{0,48}\b"
        r"(?:closed|error|failed|failure|incomplete|reset|"
        r"truncated)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\berror\s+(?:decoding|reading|receiving|sending|writing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bad gateway|empty reply from server|gateway timeout|"
        r"remote end hung up|server disconnected|service unavailable)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bproxy\b.{0,48}\b(?:closed|connect|connection|failed|"
        r"failure|reset|timed out|timeout|tunnel|unreachable)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:broken pipe|cannot assign requested address|"
        r"getaddrinfo failed|network is unreachable|no route to host|"
        r"unexpected disconnect)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btransfer\b.{0,48}\b(?:closed|interrupted|reset|truncated)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcurl\s*:?\s*\(?(?:5|6|7|18|28|35|52|55|56|92)\)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\brpc failed\b", re.IGNORECASE),
)


Result = TypeVar("Result")


class TerminalDownloadError(RuntimeError):
    """Represent a deterministic failure that retries cannot resolve."""


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
    reason: str = "unspecified"


@dataclass(frozen=True)
class HTTPFailure:
    """Describe one normalized HTTP failure from an exception."""

    error: BaseException
    status_code: int
    urls: tuple[str, ...]


def safe_attribute(value: object, name: str) -> object | None:
    """Return an attribute without trusting third-party property access."""

    try:
        return getattr(value, name, None)
    except Exception:  # noqa: BLE001
        return None


def safe_message(error: BaseException) -> str:
    """Return a lowercase exception message without raising another error."""

    try:
        return str(error).lower()
    except Exception:  # noqa: BLE001
        return ""


def qualified_mro_names(error: BaseException) -> frozenset[str]:
    """Return module-qualified type names from an exception MRO."""

    return frozenset(
        f"{candidate.__module__}.{candidate.__name__}"
        for candidate in type(error).__mro__
    )


def is_requests_invalid_header(error: BaseException) -> bool:
    """Return whether an exception is Requests' ambiguous header type."""

    return (
        "requests.exceptions.InvalidHeader"
        in qualified_mro_names(error)
    )


def is_local_requests_invalid_header(
    error: BaseException,
) -> bool:
    """Return whether Requests rejected a locally supplied header."""

    if not is_requests_invalid_header(error):
        return False
    nested_values = (
        safe_attribute(error, "__cause__"),
        safe_attribute(error, "__context__"),
    )
    if any(isinstance(value, BaseException) for value in nested_values):
        return False
    arguments = safe_attribute(error, "args")
    if not isinstance(arguments, tuple):
        return True
    return not any(
        isinstance(value, BaseException)
        for value in arguments
    )


def is_wrapped_requests_invalid_header(
    error: BaseException,
) -> bool:
    """Return whether Requests wrapped invalid remote response headers."""

    return (
        is_requests_invalid_header(error)
        and not is_local_requests_invalid_header(error)
    )


def is_provider_wrapper_type(error: BaseException) -> bool:
    """Return whether a provider wrapper represents incomplete I/O."""

    return bool(
        qualified_mro_names(error)
        & RETRYABLE_PROVIDER_WRAPPER_NAMES
    )


def is_xet_terminal_message(
    error: BaseException,
    transport: str,
) -> bool:
    """Return whether native Xet reports deterministic local misuse."""

    is_xet_error = (
        "huggingface_hub.errors.XetDownloadError"
        in qualified_mro_names(error)
    )
    return (
        transport == "xet"
        and is_xet_error
        and any(
            marker in safe_message(error)
            for marker in XET_TERMINAL_ERROR_MARKERS
        )
    )


def exception_chain(error: BaseException) -> list[BaseException]:
    """Return the active, cycle-safe exception graph in root-first order."""

    chain: list[BaseException] = []
    pending = [error]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        chain.append(item)
        children: list[BaseException] = []
        if isinstance(item, EXCEPTION_GROUP_TYPES):
            children.extend(item.exceptions)
        for attribute_name in ("original_error", "reason"):
            nested = safe_attribute(item, attribute_name)
            if isinstance(nested, BaseException):
                children.append(nested)
        cause = safe_attribute(item, "__cause__")
        context = safe_attribute(item, "__context__")
        suppress_context = bool(
            safe_attribute(item, "__suppress_context__")
        )
        if isinstance(cause, BaseException):
            children.append(cause)
        elif (
            not suppress_context
            and isinstance(context, BaseException)
        ):
            children.append(context)
        pending.extend(reversed(children))
    return chain


def failure_signature(
    error: BaseException,
    decision: RetryDecision,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Return a stable signature for repeated failure log suppression."""

    chain_signature = tuple(
        (
            f"{type(item).__module__}.{type(item).__name__}",
            safe_message(item),
        )
        for item in exception_chain(error)
    )
    return decision.category, decision.reason, chain_signature


def log_attempt_failure(
    error: BaseException,
    decision: RetryDecision,
    attempt: int,
    operation: str,
    transport: str,
    failure_counts: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        int,
    ],
) -> None:
    """Log every failed attempt and one traceback per repeated failure."""

    signature = failure_signature(error, decision)
    if (
        signature not in failure_counts
        and len(failure_counts) >= MAX_FAILURE_SIGNATURES
    ):
        oldest_signature = next(iter(failure_counts))
        del failure_counts[oldest_signature]
    occurrence = failure_counts.get(signature, 0) + 1
    failure_counts[signature] = occurrence
    fields = {
        "attempt": attempt,
        "category": decision.category,
        "operation": operation,
        "reason": decision.reason,
        "retryable": decision.retryable,
        "transport": transport,
    }
    if occurrence == 1:
        log_exception(
            "download_attempt_failed",
            error,
            **fields,
        )
        return
    log_event(
        "download_attempt_failed",
        exception=safe_message(error),
        exception_type=type(error).__name__,
        repeated_occurrence=occurrence,
        traceback="suppressed_repeated_failure",
        **fields,
    )


def normalize_http_status(value: object) -> int | None:
    """Return a validated integer HTTP status."""

    if isinstance(value, bool):
        return None
    try:
        status = int(value)
    except Exception:  # noqa: BLE001
        return None
    if 100 <= status <= 599:
        return status
    return None


def structured_http_status_codes(
    error: BaseException,
) -> tuple[int, ...]:
    """Return validated statuses exposed by exception attributes."""

    response = safe_attribute(error, "response")
    candidates = [
        safe_attribute(response, "status_code"),
        safe_attribute(response, "status"),
        safe_attribute(error, "status_code"),
        safe_attribute(error, "status"),
    ]
    if isinstance(error, urllib.error.HTTPError):
        candidates.append(safe_attribute(error, "code"))
    is_proxy_error = any(
        name.endswith(".ProxyError")
        for name in qualified_mro_names(error)
    )
    if is_proxy_error:
        match = PROXY_HTTP_STATUS_PATTERN.fullmatch(
            safe_message(error)
        )
        if match is not None:
            candidates.append(match.group(1))
    statuses = []
    for candidate in candidates:
        status = normalize_http_status(candidate)
        if status is not None and status not in statuses:
            statuses.append(status)
    return tuple(statuses)


def allows_text_inference(error: BaseException) -> bool:
    """Return whether opaque text is valid transport evidence."""

    if is_provider_wrapper_type(error):
        return True
    if isinstance(error, (RuntimeError, OSError)):
        return True
    module = type(error).__module__
    return module.startswith(NETWORK_LIBRARY_MODULE_PREFIXES)


def embedded_http_status_codes(
    error: BaseException,
) -> tuple[int, ...]:
    """Return HTTP statuses from a qualified opaque wrapper message."""

    if not allows_text_inference(error):
        return ()
    message = safe_message(error)
    statuses = []
    for pattern in HTTP_STATUS_PATTERNS:
        for match in pattern.finditer(message):
            status = normalize_http_status(match.group(1))
            if status is not None and status not in statuses:
                statuses.append(status)
    return tuple(statuses)


def exception_urls(error: BaseException) -> tuple[str, ...]:
    """Return HTTP URLs associated with one exception."""

    urls: list[str] = []

    def append_url(value: object) -> None:
        try:
            candidate = str(value)
        except Exception:  # noqa: BLE001
            return
        if (
            candidate.lower().startswith(("http://", "https://"))
            and candidate not in urls
        ):
            urls.append(candidate)

    response = safe_attribute(error, "response")
    request = safe_attribute(error, "request")
    response_request = safe_attribute(response, "request")
    for value in (
        safe_attribute(response, "url"),
        safe_attribute(response_request, "url"),
        safe_attribute(request, "url"),
        safe_attribute(error, "url"),
    ):
        if value is not None:
            append_url(value)
    if allows_text_inference(error):
        for match in HTTP_URL_PATTERN.finditer(safe_message(error)):
            append_url(match.group(0).rstrip(".,;:)]}"))
    return tuple(urls)


def http_failures(chain: list[BaseException]) -> list[HTTPFailure]:
    """Return normalized HTTP failure facts from an exception graph."""

    failures = []
    for item in chain:
        statuses = structured_http_status_codes(item)
        if not statuses:
            statuses = embedded_http_status_codes(item)
        urls = exception_urls(item)
        failures.extend(
            HTTPFailure(item, status, urls)
            for status in statuses
        )
    return failures


def retry_after_seconds(error: BaseException) -> float | None:
    """Parse an HTTP Retry-After header as a non-negative delay."""

    response = safe_attribute(error, "response")
    headers = safe_attribute(response, "headers")
    getter = safe_attribute(headers, "get")
    if not callable(getter):
        return None
    try:
        value = getter("Retry-After")
    except Exception:  # noqa: BLE001
        return None
    if value is None:
        return None
    try:
        delay = float(value)
    except Exception:  # noqa: BLE001
        delay = math.nan
    if math.isfinite(delay):
        return max(0.0, delay)
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (retry_at - now).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def is_provider_api_url(value: str) -> bool:
    """Return whether a URL identifies a provider control-plane request."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    is_hugging_face_api = (
        (
            host == "huggingface.co"
            or host.endswith(".huggingface.co")
        )
        and "/api/" in path
    )
    return (
        is_hugging_face_api
        or "xet-read-token" in path
        or host.startswith("cas-server.")
        or "/reconstruction/" in path
    )


def is_transfer_url(
    value: str,
    transport: str,
) -> bool:
    """Return whether a URL identifies signed or object-storage content."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    query = parsed.query.lower()
    path = parsed.path.lower()
    is_signed = any(
        marker in query
        for marker in SIGNED_URL_QUERY_MARKERS
    )
    is_xet_object = transport == "xet" and any(
        marker in path
        for marker in XET_TRANSFER_PATH_MARKERS
    )
    return is_signed or is_xet_object


def is_xet_transfer_envelope(
    error: BaseException,
    transport: str,
) -> bool:
    """Return whether an opaque native Xet task encloses transfer work."""

    if transport != "xet" or not isinstance(error, RuntimeError):
        return False
    message = safe_message(error)
    return any(
        marker in message
        for marker in XET_TRANSFER_ENVELOPE_MARKERS
    )


def is_retryable_transfer_403(
    failure: HTTPFailure,
    transport: str,
) -> bool:
    """Return whether a 403 belongs to a refreshable content request."""

    if failure.status_code != 403:
        return False
    if any(is_provider_api_url(url) for url in failure.urls):
        return False
    if any(
        is_transfer_url(url, transport)
        for url in failure.urls
    ):
        return True
    return is_xet_transfer_envelope(failure.error, transport)


def is_terminal_message(error: BaseException) -> bool:
    """Return whether an exception reports explicit terminal evidence."""

    message = safe_message(error)
    has_marker = any(
        marker in message
        for marker in TERMINAL_ERROR_MARKERS
    )
    return (
        has_marker
        or HOSTNAME_MISMATCH_PATTERN.search(message) is not None
    )


def root_failures(error: BaseException) -> list[BaseException]:
    """Return outer failures, flattening only exception groups."""

    failures: list[BaseException] = []
    pending = [error]
    while pending:
        item = pending.pop()
        if isinstance(item, EXCEPTION_GROUP_TYPES):
            pending.extend(reversed(item.exceptions))
        else:
            failures.append(item)
    return failures


def is_root_terminal_message(error: BaseException) -> bool:
    """Return whether an outer failure states invalid local input."""

    if is_provider_wrapper_type(error):
        return False
    message = safe_message(error)
    return any(
        marker in message
        for marker in ROOT_TERMINAL_ERROR_MARKERS
    )


def is_explicit_terminal_error(error: BaseException) -> bool:
    """Return whether an exception is deterministic local failure."""

    if is_provider_wrapper_type(error):
        return False
    if isinstance(error, TerminalDownloadError):
        return True
    if isinstance(error, ssl.SSLCertVerificationError):
        return True
    if is_local_requests_invalid_header(error):
        return True
    if isinstance(error, TERMINAL_EXCEPTION_TYPES):
        return True
    if qualified_mro_names(error) & TERMINAL_LIBRARY_TYPE_NAMES:
        return True
    error_number = safe_attribute(error, "errno")
    network_code_types = (
        socket.gaierror,
        socket.herror,
        ssl.SSLError,
    )
    return (
        isinstance(error, OSError)
        and not isinstance(error, network_code_types)
        and error_number in TERMINAL_LOCAL_ERRNOS
    )


def is_typed_network_error(error: BaseException) -> bool:
    """Return whether exception type or errno proves transport failure."""

    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, urllib.error.HTTPError):
        return False
    if isinstance(error, httpx.RequestError):
        return True
    network_types = (
        BrokenPipeError,
        ConnectionAbortedError,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        EOFError,
        TimeoutError,
        http.client.BadStatusLine,
        http.client.IncompleteRead,
        http.client.LineTooLong,
        http.client.RemoteDisconnected,
        socket.gaierror,
        socket.herror,
        ssl.SSLError,
        urllib.error.URLError,
    )
    if isinstance(error, network_types):
        return True
    if qualified_mro_names(error) & NETWORK_LIBRARY_BASE_TYPE_NAMES:
        return True
    error_number = safe_attribute(error, "errno")
    if isinstance(error, OSError) and error_number in NETWORK_ERRNOS:
        return True
    for candidate in type(error).__mro__:
        module = candidate.__module__
        if (
            module.startswith(NETWORK_LIBRARY_MODULE_PREFIXES)
            and candidate.__name__ in NETWORK_LIBRARY_TYPE_NAMES
        ):
            return True
    return False


def is_network_error_message(error: BaseException) -> bool:
    """Return whether qualified opaque text describes transport failure."""

    if not allows_text_inference(error):
        return False
    message = safe_message(error)
    return any(
        pattern.search(message) is not None
        for pattern in NETWORK_MESSAGE_PATTERNS
    )


def is_transient_http_failure(
    failure: HTTPFailure,
    transport: str,
) -> bool:
    """Return whether HTTP semantics describe a retryable transfer."""

    status = failure.status_code
    return (
        status in RETRYABLE_HTTP_STATUS_CODES
        or 500 <= status <= 599
        or is_retryable_transfer_403(failure, transport)
    )


def classify_http_failures(
    failures: list[HTTPFailure],
    transport: str,
) -> RetryDecision | None:
    """Classify normalized HTTP facts with terminal 4xx precedence."""

    for failure in failures:
        status = failure.status_code
        if not 400 <= status <= 499:
            continue
        if status in RETRYABLE_HTTP_STATUS_CODES:
            continue
        if is_retryable_transfer_403(failure, transport):
            continue
        return RetryDecision(
            False,
            False,
            "terminal",
            reason=f"http_{status}",
        )
    rate_limit_failures = [
        failure
        for failure in failures
        if failure.status_code == 429
    ]
    if rate_limit_failures:
        delays = [
            delay
            for failure in rate_limit_failures
            if (
                delay := retry_after_seconds(failure.error)
            ) is not None
        ]
        return RetryDecision(
            True,
            False,
            "rate_limit",
            retry_after=max(delays, default=None),
            reason="http_429",
        )
    if any(
        failure.status_code in {408, 425}
        or is_retryable_transfer_403(failure, transport)
        for failure in failures
    ):
        return RetryDecision(
            True,
            True,
            "network",
            reason="transient_http",
        )
    server_statuses = [
        failure.status_code
        for failure in failures
        if 500 <= failure.status_code <= 599
    ]
    if server_statuses:
        return RetryDecision(
            True,
            False,
            "server",
            reason=f"http_{server_statuses[0]}",
        )
    return None


def classify_download_error(
    error: BaseException,
    transport: str,
) -> RetryDecision:
    """Classify chained failures and whether node rotation is appropriate."""

    chain = exception_chain(error)
    if any(is_explicit_terminal_error(item) for item in chain):
        return RetryDecision(
            False,
            False,
            "terminal",
            reason="terminal_exception",
        )
    roots = root_failures(error)
    if any(
        isinstance(item, ROOT_PROGRAMMING_ERROR_TYPES)
        and not is_wrapped_requests_invalid_header(item)
        for item in roots
    ):
        return RetryDecision(
            False,
            False,
            "terminal",
            reason="root_programming_exception",
        )
    failures = http_failures(chain)
    http_decision = classify_http_failures(failures, transport)
    transient_error_ids = {
        id(failure.error)
        for failure in failures
        if is_transient_http_failure(failure, transport)
    }
    if any(
        id(item) not in transient_error_ids
        and (
            is_terminal_message(item)
            or is_xet_terminal_message(item, transport)
        )
        for item in chain
    ):
        return RetryDecision(
            False,
            False,
            "terminal",
            reason="terminal_message",
        )
    if any(
        id(item) not in transient_error_ids
        and is_root_terminal_message(item)
        for item in roots
    ):
        return RetryDecision(
            False,
            False,
            "terminal",
            reason="root_terminal_message",
        )
    if http_decision is not None:
        return http_decision
    if any(isinstance(item, MihomoUnavailableError) for item in chain):
        return RetryDecision(
            True,
            False,
            "mihomo",
            reason="mihomo_unavailable",
        )
    if any(is_typed_network_error(item) for item in chain):
        return RetryDecision(
            True,
            True,
            "network",
            reason="network_exception",
        )
    if any(is_provider_wrapper_type(item) for item in chain):
        return RetryDecision(
            True,
            True,
            "network",
            reason="provider_wrapper",
        )
    if any(
        is_xet_transfer_envelope(item, transport)
        for item in chain
    ):
        return RetryDecision(
            True,
            True,
            "network",
            reason="xet_transfer_wrapper",
        )
    if transport in {"http", "xet", "datalad"} and any(
        is_network_error_message(item) for item in chain
    ):
        return RetryDecision(
            True,
            True,
            "network",
            reason="transport_message",
        )
    return RetryDecision(
        False,
        False,
        "terminal",
        reason="unknown_error",
    )


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
    """Calculate a capped exponential retry delay without overflow."""

    if failed_attempt < 1:
        raise ValueError(
            "expected failed_attempt to be positive, but got "
            f"{failed_attempt}."
        )
    delay = min(config.retry_base_delay, config.retry_max_delay)
    for _ in range(1, failed_attempt):
        if delay >= config.retry_max_delay / 2:
            return config.retry_max_delay
        delay *= 2
    return delay


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
    failure_counts: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        int,
    ] = {}
    while config.retry_attempts == 0 or attempt <= config.retry_attempts:
        try:
            if node_manager is not None:
                selected_node = node_manager.prepare_attempt()
                log_event("mihomo_attempt_prepared", node=selected_node)
            log_event(
                "download_attempt_started",
                attempt=attempt,
                operation=operation,
                transport=transport,
            )
            return callback()
        except Exception as error:
            decision = classify_download_error(error, transport)
            log_attempt_failure(
                error,
                decision,
                attempt,
                operation,
                transport,
                failure_counts,
            )
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
                    selected_node = node_manager.failover()
                    log_event("mihomo_failover", node=selected_node)
                except Exception as failover_error:  # noqa: BLE001
                    log_exception("mihomo_failover_failed", failover_error)
                    print(
                        "WARNING: Mihomo failover preparation failed with "
                        f"{type(failover_error).__name__}: "
                        f"{failover_error}",
                        file=sys.stderr,
                    )

            delay = retry_delay(config, attempt)
            if decision.retry_after is not None:
                delay = max(
                    delay,
                    min(
                        decision.retry_after,
                        config.retry_max_delay,
                    ),
                )
            attempt_limit = (
                "unlimited"
                if config.retry_attempts == 0
                else str(config.retry_attempts)
            )
            log_event(
                "download_retry_scheduled",
                attempt=attempt,
                attempt_limit=attempt_limit,
                category=decision.category,
                delay_seconds=delay,
                operation=operation,
                reason=decision.reason,
                rotate_node=rotate_node,
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
