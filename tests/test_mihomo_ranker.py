"""Tests for deterministic Mihomo node ranking and filtering."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "download_helpers" / "mihomo_ranker.py"


def load_ranker_module() -> Any:
    """Load the Mihomo ranking module from its repository path."""

    spec = importlib.util.spec_from_file_location(
        "mihomo_ranker",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Mihomo ranker from {MODULE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ranker = load_ranker_module()


class FakeController:
    """Provide deterministic controller responses for ranking tests."""

    def __init__(self, members: list[str]) -> None:
        self.members = members
        self.selected = members[0]
        self.selections: list[str] = []

    def close(self) -> None:
        """Match the production controller lifecycle."""

    def proxy(self, name: str) -> dict[str, object]:
        """Return a selector or direct-node response."""

        if name == "download":
            return {
                "type": "Selector",
                "all": self.members,
                "now": self.selected,
            }
        return {"type": "Vless", "name": name}

    def delay(self, name: str, url: str, timeout: float) -> float:
        """Return stable synthetic latency."""

        del url, timeout
        return float(self.members.index(name) + 1)

    def select(self, group_name: str, node_name: str) -> None:
        """Record and apply a selector update."""

        assert group_name == "download"
        self.selected = node_name
        self.selections.append(node_name)


class DiscoveryController:
    """Return before-and-after connection snapshots for discovery."""

    def __init__(
        self,
        current: list[dict[str, object]],
        proxy_types: dict[str, str],
    ) -> None:
        self.snapshots = [[], current]
        self.proxy_types = proxy_types

    def close(self) -> None:
        """Match the production controller lifecycle."""

    def connections(self) -> list[dict[str, object]]:
        """Return the next active-connection snapshot."""

        return self.snapshots.pop(0)

    def proxy(self, name: str) -> dict[str, object]:
        """Return the configured type for one chain entry."""

        return {"name": name, "type": self.proxy_types[name]}


class DiscoveryResponse:
    """Keep a synthetic streamed response open during discovery."""

    url = ranker.httpx.URL("https://example.com/large.bin")

    def __enter__(self) -> "DiscoveryResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def raise_for_status(self) -> None:
        """Represent a successful response."""


class ControllerClient:
    """Return one synthetic response from a controller request."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def get(self, path: str) -> Any:
        """Return the configured response for the expected endpoint."""

        assert path == "/connections"
        return self.response


def install_discovery_client(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, dict[str, str]]]:
    """Install a proxy client that records the bounded discovery request."""

    requests = []

    class DiscoveryClient:
        def __init__(self, **kwargs: object) -> None:
            self.options = kwargs

        def __enter__(self) -> "DiscoveryClient":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def stream(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
        ) -> DiscoveryResponse:
            requests.append((method, url, headers))
            return DiscoveryResponse()

    monkeypatch.setattr(ranker.httpx, "Client", DiscoveryClient)
    return requests


def make_connection(
    identifier: str,
    hostname: str,
    chain: list[str],
) -> dict[str, object]:
    """Build one active Mihomo connection fixture."""

    return {
        "id": identifier,
        "metadata": {"host": hostname},
        "chains": chain,
    }


def make_discovery_manager(
    monkeypatch: pytest.MonkeyPatch,
    connections: list[dict[str, object]],
    proxy_types: dict[str, str],
) -> tuple[Any, list[tuple[str, str, dict[str, str]]]]:
    """Create a manager whose selector must be discovered."""

    requests = install_discovery_client(monkeypatch)
    config = ranker.MihomoConfig(
        controller_url="http://127.0.0.1:9091",
        group_name=None,
        node_marker="",
        speed_test_url="https://example.com/large.bin",
        probe_timeout=8.0,
        secret=None,
    )
    controller = DiscoveryController(connections, proxy_types)
    manager = ranker.MihomoNodeManager(
        config,
        "http://127.0.0.1:7893",
        "https://example.com/probe",
        controller=controller,
    )
    return manager, requests


def make_manager(
    controller: FakeController,
    node_marker: str = "0.1倍",
) -> Any:
    """Create a manager with explicit synthetic local configuration."""

    config = ranker.MihomoConfig(
        controller_url="http://127.0.0.1:9091",
        group_name="download",
        node_marker=node_marker,
        speed_test_url="https://huggingface.co/large.bin",
        probe_timeout=15.0,
        secret=None,
    )
    return ranker.MihomoNodeManager(
        config,
        "http://127.0.0.1:7893",
        "https://huggingface.co/api/datasets/owner/dataset",
        controller=controller,
        throughput_fn=lambda *args: 1.0,
    )


def test_eligible_nodes_require_exact_marker_and_direct_type() -> None:
    """Exclude unmarked and 0.01x nodes from every ranking operation."""

    members = [
        "stable-0.1倍",
        "cheap-0.01倍",
        "unmarked",
    ]
    manager = make_manager(FakeController(members))

    assert manager.eligible_nodes() == ["stable-0.1倍"]


def test_empty_marker_allows_every_direct_node() -> None:
    """Rank generic selector members when names have no rating marker."""

    members = [
        "stable-us",
        "fast-jp",
    ]
    manager = make_manager(
        FakeController(members),
        node_marker="",
    )

    assert manager.eligible_nodes() == members


def test_discovers_outermost_selector_with_one_byte_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the outermost live Selector without a throughput sample."""

    connections = [
        make_connection("other", "unrelated.example", ["DIRECT"]),
        make_connection(
            "new",
            "example.com",
            ["node", "automatic", "download"],
        ),
    ]
    proxy_types = {
        "node": "VLESS",
        "automatic": "URLTest",
        "download": "Selector",
    }

    manager, requests = make_discovery_manager(
        monkeypatch,
        connections,
        proxy_types,
    )

    assert manager.group_name == "download"
    assert requests == [
        (
            "GET",
            "https://example.com/large.bin",
            {"Range": "bytes=0-0"},
        )
    ]
    assert manager.ranked_nodes == []


@pytest.mark.parametrize(
    ("connections", "proxy_types", "message"),
    [
        (
            [make_connection("other", "unrelated.example", ["DIRECT"])],
            {"DIRECT": "Direct"},
            "no new connection matched",
        ),
        (
            [
                make_connection("one", "example.com", ["download"]),
                make_connection("two", "example.com", ["download"]),
            ],
            {"download": "Selector"},
            "2 new connections matched",
        ),
        (
            [make_connection("new", "example.com", ["node"])],
            {"node": "VLESS"},
            "could not discover a Selector",
        ),
    ],
)
def test_discovery_rejects_ambiguous_or_missing_selector(
    monkeypatch: pytest.MonkeyPatch,
    connections: list[dict[str, object]],
    proxy_types: dict[str, str],
    message: str,
) -> None:
    """Fail when the live route cannot identify one selector."""

    with pytest.raises(RuntimeError, match=message):
        make_discovery_manager(
            monkeypatch,
            connections,
            proxy_types,
        )


def test_connection_validation_rejects_malformed_metadata() -> None:
    """Reject malformed controller connection payloads clearly."""

    with pytest.raises(ValueError, match="metadata to be an object"):
        ranker.connection_matches_host(
            {"id": "new", "metadata": None},
            "example.com",
        )


def test_connections_propagates_controller_authorization_failure() -> None:
    """Expose an unauthorized controller instead of hiding discovery errors."""

    request = ranker.httpx.Request(
        "GET",
        "http://127.0.0.1:9091/connections",
    )
    response = ranker.httpx.Response(401, request=request)
    controller = object.__new__(ranker.MihomoController)
    controller._client = ControllerClient(response)

    with pytest.raises(ranker.httpx.HTTPStatusError):
        controller.connections()


def test_connections_rejects_malformed_controller_payload() -> None:
    """Reject a controller response whose connections field is not a list."""

    request = ranker.httpx.Request(
        "GET",
        "http://127.0.0.1:9091/connections",
    )
    response = ranker.httpx.Response(
        200,
        json={"connections": "invalid"},
        request=request,
    )
    controller = object.__new__(ranker.MihomoController)
    controller._client = ControllerClient(response)

    with pytest.raises(ValueError, match="connections to be a list"):
        controller.connections()


def test_connections_treats_null_as_an_empty_snapshot() -> None:
    """Accept Mihomo's null representation when no connections are active."""

    request = ranker.httpx.Request(
        "GET",
        "http://127.0.0.1:9091/connections",
    )
    response = ranker.httpx.Response(
        200,
        json={"connections": None},
        request=request,
    )
    controller = object.__new__(ranker.MihomoController)
    controller._client = ControllerClient(response)

    assert controller.connections() == []


def test_stability_precedes_peak_throughput() -> None:
    """Rank a consistently successful node above a flaky faster node."""

    stable = ranker.NodeScore(
        name="stable-0.1倍",
        order=1,
        latency_successes=3,
        latency_attempts=3,
        latencies_ms=(100.0, 110.0, 120.0),
        throughput_successes=2,
        throughput_attempts=2,
        throughputs_mib_s=(2.0, 2.1),
    )
    flaky = ranker.NodeScore(
        name="flaky-0.1倍",
        order=0,
        latency_successes=3,
        latency_attempts=3,
        latencies_ms=(50.0, 55.0, 60.0),
        throughput_successes=1,
        throughput_attempts=2,
        throughputs_mib_s=(20.0,),
    )

    assert sorted([flaky, stable], key=ranker.NodeScore.ranking_key)[0] \
        == stable


def test_equal_stability_prefers_higher_median_throughput() -> None:
    """Use measured speed after successful-sample counts are equal."""

    slow = ranker.NodeScore(
        "slow-0.1倍",
        0,
        3,
        3,
        (50.0, 55.0, 60.0),
        2,
        2,
        (2.0, 2.0),
    )
    fast = ranker.NodeScore(
        "fast-0.1倍",
        1,
        3,
        3,
        (100.0, 110.0, 120.0),
        2,
        2,
        (5.0, 6.0),
    )

    assert sorted([slow, fast], key=ranker.NodeScore.ranking_key)[0] \
        == fast


def test_bounded_byte_counter_never_exceeds_limit() -> None:
    """Stop a speed-test response at its configured byte budget."""

    chunks = [b"a" * 6, b"b" * 6, b"c" * 6]

    assert ranker.count_bounded_bytes(chunks, 10) == 10


def test_real_proxy_latency_probe_uses_one_byte_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep fallback repository probes bounded to response headers."""

    requests = install_discovery_client(monkeypatch)
    moments = iter([10.0, 10.25])
    monkeypatch.setattr(
        ranker.time,
        "monotonic",
        lambda: next(moments),
    )

    delay = ranker.measure_proxy_latency(
        "http://127.0.0.1:7893",
        "https://example.com/probe",
        8.0,
        {"Authorization": "Bearer hidden"},
    )

    assert delay == 250.0
    assert requests == [
        (
            "GET",
            "https://example.com/probe",
            {
                "Authorization": "Bearer hidden",
                "Range": "bytes=0-0",
            },
        )
    ]


def test_initial_speed_test_has_a_small_fixed_budget() -> None:
    """Prevent ranking traffic and timeout from delaying real downloads."""

    maximum_bytes = (
        ranker.SHORTLIST_SIZE
        * ranker.THROUGHPUT_SAMPLE_COUNT
        * ranker.THROUGHPUT_SAMPLE_BYTES
    )

    assert maximum_bytes <= 2 * ranker.MIB
    assert ranker.THROUGHPUT_SAMPLE_COUNT == 1
    assert ranker.MAX_THROUGHPUT_PROBE_SECONDS <= 8.0
    assert ranker.MAX_REAL_LATENCY_PROBE_SECONDS <= 8.0
    assert ranker.RANKING_TTL_SECONDS >= 6 * 60 * 60


def test_benchmark_selects_fastest_equally_stable_node() -> None:
    """Exercise the complete deterministic shortlist and selection path."""

    members = ["slow-0.1倍", "fast-0.1倍"]
    controller = FakeController(members)
    manager = make_manager(controller)
    calls: list[tuple[Any, ...]] = []

    def throughput(*args: Any) -> float:
        calls.append(args)
        if controller.selected == "fast-0.1倍":
            return 5.0
        return 1.0

    manager.throughput_fn = throughput
    winner = manager.benchmark()

    assert winner.name == "fast-0.1倍"
    assert controller.selected == "fast-0.1倍"
    assert len(calls) == len(members)
    assert all(
        call[2] == ranker.THROUGHPUT_SAMPLE_BYTES
        for call in calls
    )
    assert all(
        call[3] <= ranker.MAX_THROUGHPUT_PROBE_SECONDS
        for call in calls
    )


def test_healthy_controller_probes_skip_real_proxy_fallback() -> None:
    """Avoid extra repository requests when controller probes are usable."""

    members = ["one", "two", "three"]
    manager = make_manager(
        FakeController(members),
        node_marker="",
    )

    def unexpected_latency(*args: Any) -> float:
        raise AssertionError(
            "real proxy fallback ran after healthy controller probes."
        )

    manager.latency_fn = unexpected_latency

    assert manager.benchmark().name == "one"


def test_real_proxy_fallback_recovers_controller_probe_failures() -> None:
    """Use the downloader path when Mihomo delay checks falsely fail."""

    members = ["one", "two", "three", "four"]
    controller = FakeController(members)
    manager = make_manager(controller, node_marker="")
    calls: list[tuple[str, float]] = []

    def unavailable_delay(
        name: str,
        url: str,
        timeout: float,
    ) -> float:
        del name, url, timeout
        raise TimeoutError("synthetic controller timeout")

    def proxy_latency(
        proxy_url: str,
        url: str,
        timeout: float,
        headers: dict[str, str],
    ) -> float:
        del proxy_url, url, headers
        calls.append((controller.selected, timeout))
        return float(members.index(controller.selected) + 10)

    controller.delay = unavailable_delay
    manager.latency_fn = proxy_latency

    winner = manager.benchmark()

    assert winner.name == "one"
    assert [name for name, _ in calls] == members[:ranker.SHORTLIST_SIZE]
    assert all(
        timeout <= ranker.MAX_REAL_LATENCY_PROBE_SECONDS
        for _, timeout in calls
    )


def test_real_proxy_fallback_preserves_clear_unavailable_error() -> None:
    """Report failure when neither probe mechanism reaches a repository."""

    controller = FakeController(["one", "two"])
    manager = make_manager(controller, node_marker="")

    def unavailable(*args: Any) -> float:
        del args
        raise TimeoutError("synthetic repository timeout")

    controller.delay = unavailable
    manager.latency_fn = unavailable

    with pytest.raises(
        ranker.MihomoUnavailableError,
        match="no permitted Mihomo node passed the repository probes",
    ):
        manager.benchmark()


def test_failover_uses_real_transfer_without_controller_delay() -> None:
    """Do not reject failover nodes using an incompatible delay endpoint."""

    members = ["failed", "next"]
    controller = FakeController(members)
    manager = make_manager(controller, node_marker="")
    manager.ranked_nodes = [
        ranker.NodeScore(
            name=name,
            order=order,
            latency_successes=1,
            latency_attempts=1,
            latencies_ms=(10.0 + order,),
            throughput_successes=1,
            throughput_attempts=1,
            throughputs_mib_s=(1.0,),
        )
        for order, name in enumerate(members)
    ]

    def unavailable_delay(
        name: str,
        url: str,
        timeout: float,
    ) -> float:
        del name, url, timeout
        raise AssertionError("failover called the controller delay endpoint")

    controller.delay = unavailable_delay

    assert manager.failover() == "next"
    assert controller.selected == "next"
