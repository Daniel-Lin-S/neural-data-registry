"""Regression tests for shared downloader failure classification."""

from __future__ import annotations

import builtins
import errno
import http.client
import importlib.util
import socket
import ssl
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY_ROOT / "download_helpers" / "download_retry.py"
)
NETWORK_MESSAGE = "connection reset while reading response body"
BROKEN_BROTLI_MESSAGE = (
    "brotli: decoder process called with data when "
    "'can_accept_more_data()' is False"
)
XET_ENVELOPE = (
    "Task error: File reconstruction error: CAS Client Error: "
)
UNLIMITED_RETRY_FAILURE_COUNT = 257


def load_retry_module() -> Any:
    """Load the shared retry helper from its repository path."""

    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "download_retry_regression",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load retry helper from {MODULE_PATH}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retry = load_retry_module()


def request() -> httpx.Request:
    """Return a request suitable for constructing HTTPX failures."""

    return httpx.Request("GET", "https://example.test/object")


def status_error(
    status_code: int,
    url: str = "https://example.test/object",
    message: str = "provider request failed",
) -> httpx.HTTPStatusError:
    """Create an HTTPX failure with structured response metadata."""

    failed_request = httpx.Request("GET", url)
    response = httpx.Response(
        status_code,
        request=failed_request,
    )
    return httpx.HTTPStatusError(
        message,
        request=failed_request,
        response=response,
    )


def assert_network(
    error: BaseException,
    transport: str = "http",
) -> None:
    """Assert that an exception is a node-rotating network failure."""

    decision = retry.classify_download_error(error, transport)

    assert decision.retryable
    assert decision.rotate_node
    assert decision.category == "network"
    assert decision.reason != "unspecified"


def assert_terminal(
    error: BaseException,
    transport: str = "http",
) -> None:
    """Assert that an exception remains a terminal failure."""

    decision = retry.classify_download_error(error, transport)

    assert not decision.retryable
    assert not decision.rotate_node
    assert decision.category == "terminal"
    assert decision.reason != "unspecified"


def test_observed_brotli_decoding_failure_is_network() -> None:
    """Retry the exact response decoder failure from LibriBrain."""

    cause = ValueError(BROKEN_BROTLI_MESSAGE)
    error = httpx.DecodingError(
        BROKEN_BROTLI_MESSAGE,
        request=request(),
    )
    error.__cause__ = cause

    decision = retry.classify_download_error(error, "xet")

    assert decision.retryable
    assert decision.rotate_node
    assert decision.category == "network"
    assert decision.reason == "network_exception"


@pytest.mark.parametrize(
    "error_type",
    (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.DecodingError,
        httpx.PoolTimeout,
        httpx.ProxyError,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        httpx.WriteError,
        httpx.WriteTimeout,
    ),
)
def test_httpx_request_failures_are_network(
    error_type: type[httpx.RequestError],
) -> None:
    """Retry every HTTPX request failure used by transfer clients."""

    error = error_type(NETWORK_MESSAGE, request=request())

    assert_network(error)


@pytest.mark.parametrize(
    "error",
    (
        httpx.InvalidURL("malformed endpoint"),
        httpx.LocalProtocolError(
            "invalid request framing",
            request=request(),
        ),
        httpx.StreamClosed(),
        httpx.StreamConsumed(),
        httpx.TooManyRedirects(
            "redirect limit reached",
            request=request(),
        ),
        httpx.UnsupportedProtocol(
            "cannot issue request",
            request=request(),
        ),
    ),
)
def test_httpx_local_failures_are_terminal(
    error: BaseException,
) -> None:
    """Do not retry deterministic HTTPX usage failures."""

    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_exception"


@pytest.mark.parametrize(
    "error_name",
    (
        "ChunkedEncodingError",
        "ContentDecodingError",
        "ConnectionError",
        "ProxyError",
    ),
)
def test_requests_transfer_failures_are_network(
    error_name: str,
) -> None:
    """Retry decoding and truncated transfers raised by Requests."""

    requests = pytest.importorskip("requests")
    error_type = getattr(requests.exceptions, error_name)

    assert_network(error_type(NETWORK_MESSAGE))


@pytest.mark.parametrize(
    ("error_name", "arguments"),
    (
        ("DecodeError", ("response decoder failed",)),
        ("IncompleteRead", (5, 10)),
        ("ProtocolError", ("response transfer truncated",)),
    ),
)
def test_urllib3_transfer_failures_are_network(
    error_name: str,
    arguments: tuple[object, ...],
) -> None:
    """Retry decoding and truncation failures raised by urllib3."""

    urllib3 = pytest.importorskip("urllib3")
    error_type = getattr(urllib3.exceptions, error_name)

    assert_network(error_type(*arguments))


@pytest.mark.parametrize(
    ("status_code", "category", "rotate_node"),
    (
        (408, "network", True),
        (425, "network", True),
        (429, "rate_limit", False),
        (500, "server", False),
        (502, "server", False),
        (503, "server", False),
        (599, "server", False),
    ),
)
@pytest.mark.parametrize("structured", (False, True))
def test_transient_http_statuses_are_retried(
    status_code: int,
    category: str,
    rotate_node: bool,
    structured: bool,
) -> None:
    """Normalize retryable statuses from metadata and wrappers."""

    if structured:
        error = status_error(status_code)
    else:
        error = RuntimeError(
            f"request failed with HTTP status {status_code}"
        )

    decision = retry.classify_download_error(error, "http")

    assert decision.retryable
    assert decision.rotate_node is rotate_node
    assert decision.category == category
    assert decision.reason != "unspecified"


@pytest.mark.parametrize(
    "status_code",
    (400, 401, 402, 403, 404, 405, 407, 409, 410, 422),
)
def test_terminal_http_statuses_do_not_retry(
    status_code: int,
) -> None:
    """Stop on client responses without transient semantics."""

    error = status_error(status_code)
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == f"http_{status_code}"


@pytest.mark.parametrize(
    "status_code",
    (400, 401, 403, 404, 407, 409, 422),
)
def test_opaque_terminal_http_statuses_do_not_retry(
    status_code: int,
) -> None:
    """Normalize terminal statuses from qualified opaque wrappers."""

    error = RuntimeError(
        f"request failed with HTTP status {status_code}"
    )
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == f"http_{status_code}"


def test_urllib_http_error_404_is_terminal() -> None:
    """Do not treat urllib HTTP errors as generic URL failures."""

    error = urllib.error.HTTPError(
        "https://example.test/missing",
        404,
        "Not Found",
        None,
        None,
    )

    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "http_404"


def test_signed_object_403_is_retried() -> None:
    """Refresh and retry an expired signed object-storage request."""

    error = urllib.error.HTTPError(
        "https://storage.example.test/object"
        "?X-Amz-Signature=redacted",
        403,
        "Forbidden",
        None,
        None,
    )

    decision = retry.classify_download_error(error, "http")

    assert decision.retryable
    assert decision.rotate_node
    assert decision.reason == "transient_http"


def test_provider_api_403_is_terminal() -> None:
    """Keep provider authorization failures terminal."""

    error = status_error(
        403,
        "https://huggingface.co/api/datasets/owner/dataset",
    )
    decision = retry.classify_download_error(error, "xet")

    assert_terminal(error, "xet")
    assert decision.reason == "http_403"


def test_proxy_authentication_407_is_terminal() -> None:
    """Do not rotate forever when the proxy rejects credentials."""

    error = status_error(407)
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "http_407"


def test_future_xet_object_host_403_is_transport_scoped() -> None:
    """Recognize Xet object paths without hard-coding a CDN host."""

    url = "https://future-storage.example.test/xorbs/default/hash"
    error = status_error(403, url)

    assert_network(error, "xet")
    assert_terminal(error, "http")


@pytest.mark.parametrize(
    "error_number",
    (
        errno.EADDRNOTAVAIL,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.ENETUNREACH,
        errno.ENOBUFS,
        errno.ENOTCONN,
        errno.EPIPE,
        errno.EPROTO,
        errno.ETIMEDOUT,
    ),
)
def test_network_errno_is_retried(error_number: int) -> None:
    """Use operating-system network errno as transport evidence."""

    error = OSError(error_number, "operation failed")
    decision = retry.classify_download_error(error, "http")

    assert_network(error)
    assert decision.reason == "network_exception"


@pytest.mark.parametrize(
    "error_number",
    (
        errno.EACCES,
        errno.EEXIST,
        errno.ENOENT,
        errno.ENOSPC,
        errno.ENOTDIR,
        errno.EROFS,
    ),
)
def test_filesystem_errno_is_terminal(error_number: int) -> None:
    """Do not retry deterministic local filesystem failures."""

    error = OSError(error_number, "local operation failed")
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_exception"


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    (
        (AssertionError(NETWORK_MESSAGE), "terminal_exception"),
        (TypeError("proxy connection failed"),
         "root_programming_exception"),
        (ValueError("response stream was truncated"),
         "root_programming_exception"),
    ),
)
def test_root_programming_errors_override_network_words(
    error: BaseException,
    expected_reason: str,
) -> None:
    """Avoid converting root programming defects into retry loops."""

    decision = retry.classify_download_error(error, "xet")

    assert_terminal(error, "xet")
    assert decision.reason == expected_reason


def test_nested_programming_error_does_not_mask_decoding() -> None:
    """Judge a transfer wrapper by its outer operation boundary."""

    error = httpx.DecodingError(
        "response decoder failed",
        request=request(),
    )
    error.__cause__ = ValueError("decoder state rejected bytes")

    assert_network(error, "xet")


def test_future_xet_transfer_envelope_is_network() -> None:
    """Retry new native-Xet details inside a known transfer task."""

    error = RuntimeError(
        XET_ENVELOPE + "future transport implementation detail"
    )
    decision = retry.classify_download_error(error, "xet")

    assert_network(error, "xet")
    assert decision.reason == "xet_transfer_wrapper"


@pytest.mark.parametrize(
    "detail",
    (
        "checksum mismatch",
        "no space left on device",
        "unsupported protocol",
    ),
)
def test_xet_local_failures_override_transfer_envelope(
    detail: str,
) -> None:
    """Keep deterministic Xet failures terminal despite the wrapper."""

    error = RuntimeError(XET_ENVELOPE + detail)
    decision = retry.classify_download_error(error, "xet")

    assert_terminal(error, "xet")
    assert decision.reason == "terminal_message"


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="ExceptionGroup was added in Python 3.11",
)
def test_exception_group_network_children_are_retried() -> None:
    """Traverse grouped transport failures without version breakage."""

    group_type = builtins.ExceptionGroup
    error = group_type(
        "parallel transfers failed",
        [
            httpx.ReadError(NETWORK_MESSAGE, request=request()),
            ConnectionResetError("peer reset connection"),
        ],
    )

    assert_network(error, "http")


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="ExceptionGroup was added in Python 3.11",
)
def test_exception_group_terminal_child_takes_precedence() -> None:
    """Stop a grouped retry when one child is a local failure."""

    group_type = builtins.ExceptionGroup
    error = group_type(
        "parallel transfers failed",
        [
            httpx.ReadError(NETWORK_MESSAGE, request=request()),
            OSError(errno.ENOSPC, "local operation failed"),
        ],
    )

    assert_terminal(error, "http")


class HostileResponse:
    """Expose response metadata that raises during inspection."""

    @property
    def status_code(self) -> int:
        """Raise instead of exposing a status code."""

        raise RuntimeError("status unavailable")

    @property
    def status(self) -> int:
        """Raise instead of exposing a secondary status."""

        raise RuntimeError("status unavailable")


class HostileError(RuntimeError):
    """Raise while the classifier inspects common exception fields."""

    @property
    def response(self) -> HostileResponse:
        """Return a response whose properties cannot be inspected."""

        return HostileResponse()

    @property
    def request(self) -> object:
        """Raise instead of exposing request metadata."""

        raise RuntimeError("request unavailable")

    def __str__(self) -> str:
        """Raise instead of producing exception text."""

        raise RuntimeError("message unavailable")


def test_malformed_exception_attributes_are_total() -> None:
    """Never fail while deciding how to handle a third-party error."""

    error = HostileError()
    decision = retry.classify_download_error(error, "http")

    assert not decision.retryable
    assert decision.category == "terminal"
    assert decision.reason == "unknown_error"


def test_exception_cycle_is_total() -> None:
    """Traverse malformed cause graphs without looping forever."""

    error = RuntimeError("opaque provider failure")
    error.__cause__ = error

    decision = retry.classify_download_error(error, "http")

    assert not decision.retryable
    assert decision.reason == "unknown_error"


@pytest.mark.parametrize(
    "error",
    (
        OSError(errno.ENOSPC, "no space left on device"),
        status_error(404),
    ),
)
def test_unlimited_setting_stops_on_terminal_failure(
    error: Exception,
) -> None:
    """Do not loop on terminal failures when retries are unlimited."""

    config = SimpleNamespace(
        retry_attempts=0,
        retry_base_delay=1.0,
        retry_max_delay=2.0,
    )
    calls = 0
    delays: list[float] = []

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        retry.run_with_retries(
            config,
            "test transfer",
            fail,
            sleep_fn=delays.append,
        )

    assert calls == 1
    assert delays == []


def test_unlimited_retries_have_no_hidden_attempt_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue beyond ordinary retry limits until transport recovers."""

    config = SimpleNamespace(
        retry_attempts=0,
        retry_base_delay=1.0,
        retry_max_delay=2.0,
    )
    calls = 0

    def callback() -> str:
        nonlocal calls
        calls += 1
        if calls <= UNLIMITED_RETRY_FAILURE_COUNT:
            raise httpx.ProxyError(
                "proxy connection reset",
                request=request(),
            )
        return "complete"

    monkeypatch.setattr(
        retry,
        "log_exception",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        retry,
        "log_event",
        lambda *_args, **_kwargs: None,
    )

    result = retry.run_with_retries(
        config,
        "test transfer",
        callback,
        sleep_fn=lambda _delay: None,
    )

    assert result == "complete"
    assert calls == UNLIMITED_RETRY_FAILURE_COUNT + 1


def test_retry_events_include_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record why a failure was retried in both diagnostic events."""

    config = SimpleNamespace(
        retry_attempts=2,
        retry_base_delay=1.0,
        retry_max_delay=2.0,
    )
    failure_fields: list[dict[str, object]] = []
    event_fields: list[tuple[str, dict[str, object]]] = []
    calls = 0

    def callback() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("peer reset connection")
        return "complete"

    monkeypatch.setattr(
        retry,
        "log_exception",
        lambda _event, _error, **fields: failure_fields.append(fields),
    )
    monkeypatch.setattr(
        retry,
        "log_event",
        lambda event, **fields: event_fields.append((event, fields)),
    )

    result = retry.run_with_retries(
        config,
        "test transfer",
        callback,
        sleep_fn=lambda _delay: None,
    )

    retry_event = next(
        fields
        for event, fields in event_fields
        if event == "download_retry_scheduled"
    )
    assert result == "complete"
    assert failure_fields[0]["reason"] == "network_exception"
    assert retry_event["reason"] == "network_exception"


def test_repeated_network_failures_log_one_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every failed attempt without duplicating its traceback."""

    config = SimpleNamespace(
        retry_attempts=0,
        retry_base_delay=1.0,
        retry_max_delay=2.0,
    )
    detailed: list[tuple[str, dict[str, object]]] = []
    events: list[tuple[str, dict[str, object]]] = []
    calls = 0

    def callback() -> str:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise httpx.DecodingError(
                BROKEN_BROTLI_MESSAGE,
                request=request(),
            )
        return "complete"

    monkeypatch.setattr(
        retry,
        "log_exception",
        lambda event, _error, **fields: detailed.append(
            (event, fields)
        ),
    )
    monkeypatch.setattr(
        retry,
        "log_event",
        lambda event, **fields: events.append((event, fields)),
    )

    result = retry.run_with_retries(
        config,
        "test transfer",
        callback,
        sleep_fn=lambda _delay: None,
    )

    compact = [
        fields
        for event, fields in events
        if event == "download_attempt_failed"
    ]
    assert result == "complete"
    assert len(detailed) == 1
    assert [fields["attempt"] for fields in compact] == [2, 3]
    assert [
        fields["repeated_occurrence"] for fields in compact
    ] == [2, 3]
    assert all(
        fields["traceback"] == "suppressed_repeated_failure"
        for fields in compact
    )


def test_failure_signature_cache_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid retaining unbounded error variants during unlimited retries."""

    decision = retry.RetryDecision(
        True,
        True,
        "network",
        reason="network_exception",
    )
    failure_counts: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        int,
    ] = {}
    first_error = ConnectionResetError("connection reset variant 0")
    first_signature = retry.failure_signature(first_error, decision)
    monkeypatch.setattr(
        retry,
        "log_exception",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        retry,
        "log_event",
        lambda *_args, **_kwargs: None,
    )

    for index in range(retry.MAX_FAILURE_SIGNATURES + 1):
        error = ConnectionResetError(
            f"connection reset variant {index}"
        )
        retry.log_attempt_failure(
            error,
            decision,
            index + 1,
            "test transfer",
            "http",
            failure_counts,
        )

    assert len(failure_counts) == retry.MAX_FAILURE_SIGNATURES
    assert first_signature not in failure_counts


@pytest.mark.parametrize(
    ("status_code", "message"),
    (
        (500, "repository not found in upstream response"),
        (502, "no such file or directory in gateway response"),
        (503, "authentication failed for an upstream service"),
    ),
)
def test_http_server_status_overrides_response_text(
    status_code: int,
    message: str,
) -> None:
    """Use server status semantics instead of response-body substrings."""

    error = status_error(status_code, message=message)
    decision = retry.classify_download_error(error, "http")

    assert decision.retryable
    assert decision.category == "server"
    assert decision.reason == f"http_{status_code}"


def test_opaque_server_status_overrides_command_output_text() -> None:
    """Retry a server response embedded in provider command output."""

    error = RuntimeError(
        "server returned status 503: repository not found"
    )
    decision = retry.classify_download_error(error, "datalad")

    assert decision.retryable
    assert decision.category == "server"
    assert decision.reason == "http_503"


@pytest.mark.parametrize("value", ("inf", "-inf", "nan"))
def test_non_finite_retry_after_is_ignored(value: str) -> None:
    """Prevent malformed rate-limit delays from stopping retries."""

    failed_request = request()
    response = httpx.Response(
        429,
        headers={"Retry-After": value},
        request=failed_request,
    )
    error = httpx.HTTPStatusError(
        "rate limited",
        request=failed_request,
        response=response,
    )

    decision = retry.classify_download_error(error, "http")

    assert decision.retryable
    assert decision.category == "rate_limit"
    assert decision.retry_after is None


def test_request_validation_message_is_terminal() -> None:
    """Do not infer transport failure from generic request wording."""

    error = RuntimeError("request validation failed")
    decision = retry.classify_download_error(error, "http")

    assert not decision.retryable
    assert decision.reason == "unknown_error"


def test_nested_memory_error_overrides_transfer_wrapper() -> None:
    """Do not hide process exhaustion behind a request exception."""

    error = httpx.DecodingError(
        "response decoder failed",
        request=request(),
    )
    error.__cause__ = MemoryError("cannot allocate decoder buffer")

    decision = retry.classify_download_error(error, "xet")

    assert not decision.retryable
    assert decision.reason == "terminal_exception"


def test_outer_terminal_message_overrides_nested_server_status() -> None:
    """Do not let an unrelated nested 503 hide an outer auth failure."""

    inner = status_error(503)
    error = RuntimeError("authentication failed")
    error.__cause__ = inner

    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


def test_signed_object_403_ignores_response_permission_text() -> None:
    """Retry an expired signed URL even when its response says denied."""

    error = urllib.error.HTTPError(
        "https://storage.example.test/object"
        "?X-Amz-Signature=redacted",
        403,
        "Permission denied",
        None,
        None,
    )

    decision = retry.classify_download_error(error, "http")

    assert_network(error)
    assert decision.reason == "transient_http"


def test_xet_object_403_ignores_response_auth_text() -> None:
    """Retry Xet object authorization failures that refresh can repair."""

    error = status_error(
        403,
        "https://storage.example.test/xorbs/default/hash",
        message="authorization failed",
    )

    decision = retry.classify_download_error(error, "xet")

    assert_network(error, "xet")
    assert decision.reason == "transient_http"


def test_explicit_terminal_type_overrides_http_words() -> None:
    """Keep provider-certified local failures out of retry loops."""

    error = retry.TerminalDownloadError(
        "invalid command exit status 130 after HTTP status 503"
    )
    decision = retry.classify_download_error(error, "datalad")

    assert_terminal(error, "datalad")
    assert decision.reason == "terminal_exception"


def test_huge_retry_after_is_capped() -> None:
    """Cap a finite but impractical server-provided retry delay."""

    failed_request = request()
    response = httpx.Response(
        429,
        headers={"Retry-After": "1e308"},
        request=failed_request,
    )
    error = httpx.HTTPStatusError(
        "rate limited",
        request=failed_request,
        response=response,
    )
    config = SimpleNamespace(
        retry_attempts=2,
        retry_base_delay=1.0,
        retry_max_delay=2.0,
    )
    calls = 0
    delays: list[float] = []

    def callback() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "complete"

    result = retry.run_with_retries(
        config,
        "test transfer",
        callback,
        sleep_fn=delays.append,
    )

    assert result == "complete"
    assert delays == [2.0]


def test_hugging_face_offline_mode_is_terminal() -> None:
    """Do not retry a provider explicitly disabled by local settings."""

    hub_errors = pytest.importorskip("huggingface_hub.errors")
    cause = hub_errors.OfflineModeIsEnabled(
        "offline mode is enabled"
    )
    error = hub_errors.LocalEntryNotFoundError(
        "outgoing traffic has been disabled"
    )
    error.__cause__ = cause

    decision = retry.classify_download_error(error, "xet")

    assert_terminal(error, "xet")
    assert decision.reason == "terminal_exception"


def test_opaque_xet_download_error_is_retryable() -> None:
    """Retry new native Xet transfer details without message matching."""

    hub_errors = pytest.importorskip("huggingface_hub.errors")
    error = hub_errors.XetDownloadError(
        "future native implementation detail"
    )
    decision = retry.classify_download_error(error, "xet")

    assert_network(error, "xet")
    assert decision.reason == "provider_wrapper"


@pytest.mark.parametrize(
    "message",
    (
        "Invalid argument (os error 22)",
        "Unsupported operation (os error 95)",
        "Too many levels of symbolic links (os error 40)",
    ),
)
def test_xet_local_operation_failures_are_terminal(
    message: str,
) -> None:
    """Do not loop on deterministic native Xet filesystem failures."""

    hub_errors = pytest.importorskip("huggingface_hub.errors")
    error = hub_errors.XetDownloadError(message)
    decision = retry.classify_download_error(error, "xet")

    assert_terminal(error, "xet")
    assert decision.reason == "terminal_message"


@pytest.mark.parametrize(
    "message",
    (
        "server certificate verification failed. CAfile: none",
        "curl: (60) SSL peer certificate or SSH remote key was not OK",
    ),
)
def test_certificate_verification_text_is_terminal(
    message: str,
) -> None:
    """Do not retry a certificate trust configuration failure."""

    error = ConnectionError(message)
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


@pytest.mark.parametrize(
    "message",
    (
        "cannot create directory: Operation not permitted",
        "unable to create file: File name too long",
        "unable to create file: Filename too long",
        "cannot write object: Is a directory",
    ),
)
def test_erased_filesystem_failure_text_is_terminal(
    message: str,
) -> None:
    """Recognize local failures after a subprocess erases errno."""

    error = ConnectionError(message)
    decision = retry.classify_download_error(error, "datalad")

    assert_terminal(error, "datalad")
    assert decision.reason == "terminal_message"


@pytest.mark.parametrize(
    "error",
    (
        socket.herror(1, "Unknown host"),
        socket.herror(2, "Host name lookup failure"),
        ssl.SSLWantReadError(
            ssl.SSL_ERROR_WANT_READ,
            "The operation did not complete (read)",
        ),
        ssl.SSLError(
            ssl.SSL_ERROR_SYSCALL,
            "TLS transport closed",
        ),
    ),
)
def test_network_family_errno_collision_is_retryable(
    error: OSError,
) -> None:
    """Prefer DNS and TLS exception families over colliding errno values."""

    assert_network(error)


@pytest.mark.skipif(
    not hasattr(errno, "ECOMM"),
    reason="ECOMM is not defined on this platform",
)
def test_communication_errno_is_retryable() -> None:
    """Retry an opaque kernel communication failure."""

    error = OSError(errno.ECOMM, "provider operation failed")

    assert_network(error)


def test_httpcore_connection_not_available_is_retryable() -> None:
    """Retry exhausted or stale HTTP connection-pool state."""

    httpcore = pytest.importorskip("httpcore")
    error = httpcore.ConnectionNotAvailable(
        "connection unavailable"
    )

    assert_network(error)


def test_stdlib_remote_header_line_too_long_is_retryable() -> None:
    """Retry malformed or truncated remote HTTP framing."""

    error = http.client.LineTooLong("header line")

    assert_network(error)


def test_requests_invalid_json_body_is_terminal() -> None:
    """Do not retry a request body that cannot be serialized."""

    requests = pytest.importorskip("requests")
    with pytest.raises(requests.exceptions.InvalidJSONError) as raised:
        requests.Request(
            "POST",
            "https://example.test/object",
            json={"invalid": float("nan")},
        ).prepare()

    decision = retry.classify_download_error(raised.value, "http")

    assert_terminal(raised.value)
    assert decision.reason == "terminal_exception"


@pytest.mark.parametrize(
    "error_name",
    (
        "BodyNotHttplibCompatible",
        "TimeoutStateError",
        "UnrewindableBodyError",
    ),
)
def test_urllib3_local_state_error_is_terminal(
    error_name: str,
) -> None:
    """Do not retry deterministic body or timeout-state misuse."""

    urllib3 = pytest.importorskip("urllib3")
    error_type = getattr(urllib3.exceptions, error_name)
    error = error_type("invalid local request state")

    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_exception"


def test_urllib3_proxy_wrapper_exposes_terminal_reason() -> None:
    """Inspect a proxy wrapper's original configuration error."""

    urllib3 = pytest.importorskip("urllib3")
    reason = urllib3.exceptions.ProxySchemeUnknown("ftp")
    error = urllib3.exceptions.ProxyError(
        "unable to connect to proxy",
        reason,
    )

    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_exception"


def test_urllib3_retry_wrapper_exposes_terminal_reason() -> None:
    """Inspect the final reason inside an exhausted-request wrapper."""

    urllib3 = pytest.importorskip("urllib3")
    reason = urllib3.exceptions.UnrewindableBodyError(
        "request body cannot be replayed"
    )
    error = urllib3.exceptions.MaxRetryError(
        None,
        "/object",
        reason,
    )

    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_exception"


def test_urllib_no_host_error_is_terminal() -> None:
    """Do not retry a URL that omits its host."""

    error = urllib.error.URLError("no host given")
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


@pytest.mark.parametrize(
    "message",
    (
        "Invalid username/password",
        "407 ",
        (
            "No username/password supplied. "
            "Server requested username/password"
        ),
    ),
)
def test_httpx_proxy_authentication_is_terminal(
    message: str,
) -> None:
    """Do not retry explicit HTTP or SOCKS proxy credential failures."""

    error = httpx.ProxyError(message, request=request())
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason in {"http_407", "terminal_message"}


def test_httpcore_terse_proxy_407_is_terminal() -> None:
    """Recognize proxy authentication before HTTPX wraps the error."""

    httpcore = pytest.importorskip("httpcore")
    error = httpcore.ProxyError("407 ")
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "http_407"


@pytest.mark.parametrize(
    "message",
    (
        "Fingerprints did not match",
        "Fingerprint of invalid length",
        "Hash function implementation unavailable for fingerprint length",
        "Client private key is encrypted, password is required",
        "hostname 'proxy.test' doesn't match 'other.test'",
    ),
)
@pytest.mark.parametrize(
    "module_name",
    ("requests", "urllib3"),
)
def test_tls_configuration_error_is_terminal(
    module_name: str,
    message: str,
) -> None:
    """Keep certificate pin, key, and hostname errors terminal."""

    library = pytest.importorskip(module_name)
    error = library.exceptions.SSLError(message)
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


def test_exact_incomplete_snapshot_offline_message_is_terminal() -> None:
    """Recognize the installed Hub local-files-only diagnostic."""

    hub_errors = pytest.importorskip("huggingface_hub.errors")
    error = hub_errors.IncompleteSnapshotError(
        "Outgoing traffic is disabled ('local_files_only=True').",
        "/absolute/incomplete-snapshot",
    )
    decision = retry.classify_download_error(error, "xet")

    assert_terminal(error, "xet")
    assert decision.reason == "terminal_message"


@pytest.mark.parametrize(
    "message",
    (
        "Content-Length contained multiple unmatching values",
        "Invalid Retry-After header: invalid server value",
    ),
)
def test_urllib3_invalid_response_header_is_retryable(
    message: str,
) -> None:
    """Retry malformed response headers received from a proxy or server."""

    urllib3 = pytest.importorskip("urllib3")
    error = urllib3.exceptions.InvalidHeader(message)

    assert_network(error)


@pytest.mark.parametrize(
    "message",
    (
        "CONNECT tunnel failed, response 407",
        "curl: (56) Received HTTP code 407 from proxy after CONNECT",
    ),
)
def test_erased_connect_proxy_407_is_terminal(message: str) -> None:
    """Recognize proxy authentication after subprocess type erasure."""

    error = RuntimeError(message)
    decision = retry.classify_download_error(error, "datalad")

    assert_terminal(error, "datalad")
    assert decision.reason == "http_407"


def test_socks_auth_method_rejection_is_terminal() -> None:
    """Do not retry a proxy that rejected all configured auth methods."""

    urllib3 = pytest.importorskip("urllib3")
    error = urllib3.exceptions.NewConnectionError(
        None,
        "All offered SOCKS5 authentication methods were rejected",
    )
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


def test_httpcore_proxy_auth_method_mismatch_is_terminal() -> None:
    """Do not retry an incompatible configured proxy auth method."""

    httpcore = pytest.importorskip("httpcore")
    error = httpcore.ProxyError(
        "Requested USERNAME/PASSWORD from proxy server, "
        "but got NO AUTHENTICATION REQUIRED."
    )
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


def test_urllib3_missing_subject_alt_name_is_terminal() -> None:
    """Keep deterministic certificate identity validation terminal."""

    urllib3 = pytest.importorskip("urllib3")
    reason = urllib3.exceptions.SSLError(
        "no appropriate subjectAltName fields were found"
    )
    error = urllib3.exceptions.MaxRetryError(
        None,
        "/object",
        reason,
    )
    decision = retry.classify_download_error(error, "http")

    assert_terminal(error)
    assert decision.reason == "terminal_message"


def test_local_requests_invalid_header_is_terminal() -> None:
    """Do not retry a malformed caller-supplied request header."""

    requests = pytest.importorskip("requests")
    with pytest.raises(requests.exceptions.InvalidHeader) as raised:
        requests.Request(
            "GET",
            "https://example.test/object",
            headers={"bad\nname": "value"},
        ).prepare()

    decision = retry.classify_download_error(raised.value, "http")

    assert_terminal(raised.value)
    assert decision.reason == "terminal_exception"


def test_wrapped_requests_response_invalid_header_is_retryable() -> None:
    """Retry response framing that Requests wraps as InvalidHeader."""

    requests = pytest.importorskip("requests")
    urllib3 = pytest.importorskip("urllib3")
    reason = urllib3.exceptions.InvalidHeader(
        "Content-Length contained multiple unmatching values"
    )
    error = requests.exceptions.InvalidHeader(
        reason,
        request=requests.Request(
            "GET",
            "https://example.test/object",
        ).prepare(),
    )

    assert_network(error)
