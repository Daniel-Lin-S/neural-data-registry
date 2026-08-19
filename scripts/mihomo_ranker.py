"""Rank and select Mihomo nodes for resilient HTTPS transfers.

Input
-----
The module receives an explicit Mihomo controller URL, optional selector
group, optional node-name marker, proxy URL, probe URL, and HTTPS URL suitable
for bounded range tests.

Output
------
Only direct nodes are tested. When a marker is configured, eligible node names
must contain it. The selected node is changed through the Mihomo controller
after deterministic stability, throughput, and latency ranking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import ssl
import statistics
import sys
import time
from typing import Callable, Iterable
from urllib.parse import quote, urlparse

import httpx


LATENCY_PROBE_COUNT = 2
LATENCY_PROBE_WORKERS = 8
SHORTLIST_SIZE = 3
THROUGHPUT_SAMPLE_COUNT = 1
THROUGHPUT_SAMPLE_BYTES = 512 * 1024
FAILOVER_SAMPLE_BYTES = 256 * 1024
MAX_THROUGHPUT_PROBE_SECONDS = 8.0
MAX_REAL_LATENCY_PROBE_SECONDS = 8.0
DISCOVERY_RANGE_BYTES = 1
RANKING_TTL_SECONDS = 6 * 60 * 60
NODE_COOLDOWN_SECONDS = 30 * 60
MIB = 1024 * 1024
GROUP_PROXY_TYPES = frozenset(
    {
        "fallback",
        "loadbalance",
        "relay",
        "selector",
        "urltest",
    }
)

TimeFunction = Callable[[], float]
LatencyFunction = Callable[..., float]


class MihomoUnavailableError(RuntimeError):
    """Indicate that no permitted Mihomo node is currently usable."""


@dataclass(frozen=True)
class MihomoConfig:
    """Store explicit Mihomo ranking configuration.

    Parameters
    ----------
    controller_url : str
        Mihomo external-controller HTTP URL.
    group_name : str or None
        Selector group dedicated to the download, optional. When omitted, the
        selector is discovered from the speed-test connection chain.
    node_marker : str
        Literal substring required in eligible node names. An empty value
        allows every direct node in the selector.
    speed_test_url : str
        Large HTTPS file used for bounded range tests.
    probe_timeout : float
        Timeout in seconds for controller and throughput probes.
    secret : str or None
        Mihomo controller secret, optional.
    """

    controller_url: str
    group_name: str | None
    node_marker: str
    speed_test_url: str
    probe_timeout: float
    secret: str | None


@dataclass(frozen=True)
class LatencyProfile:
    """Store repeated latency-probe results for one node."""

    name: str
    order: int
    successes: int
    attempts: int
    latencies_ms: tuple[float, ...]

    @property
    def success_ratio(self) -> float:
        """Return the successful fraction of latency probes."""

        return self.successes / self.attempts

    @property
    def median_latency_ms(self) -> float:
        """Return median latency, or infinity when every probe failed."""

        if not self.latencies_ms:
            return math.inf
        return statistics.median(self.latencies_ms)


@dataclass(frozen=True)
class NodeScore:
    """Store deterministic stability and speed metrics for one node."""

    name: str
    order: int
    latency_successes: int
    latency_attempts: int
    latencies_ms: tuple[float, ...]
    throughput_successes: int
    throughput_attempts: int
    throughputs_mib_s: tuple[float, ...]

    @property
    def overall_success_ratio(self) -> float:
        """Return success ratio across latency and throughput probes."""

        successes = self.latency_successes + self.throughput_successes
        attempts = self.latency_attempts + self.throughput_attempts
        return successes / attempts

    @property
    def median_throughput_mib_s(self) -> float:
        """Return median measured throughput in MiB/s."""

        if not self.throughputs_mib_s:
            return 0.0
        return statistics.median(self.throughputs_mib_s)

    @property
    def p95_latency_ms(self) -> float:
        """Return the nearest-rank p95 latency in milliseconds."""

        if not self.latencies_ms:
            return math.inf
        values = sorted(self.latencies_ms)
        index = max(0, math.ceil(0.95 * len(values)) - 1)
        return values[index]

    def ranking_key(self) -> tuple[float, ...]:
        """Return the deterministic stability-first ordering key."""

        return (
            -float(self.throughput_successes),
            -self.overall_success_ratio,
            -self.median_throughput_mib_s,
            self.p95_latency_ms,
            float(self.order),
        )


class MihomoController:
    """Access only the Mihomo proxy inspection and selection APIs."""

    def __init__(self, config: MihomoConfig) -> None:
        """Create a controller client that ignores ambient proxies."""

        headers = {}
        if config.secret:
            headers["Authorization"] = f"Bearer {config.secret}"
        self._client = httpx.Client(
            base_url=config.controller_url.rstrip("/"),
            headers=headers,
            timeout=config.probe_timeout,
            trust_env=False,
        )

    def close(self) -> None:
        """Close the controller HTTP connection pool."""

        self._client.close()

    def proxy(self, name: str) -> dict[str, object]:
        """Return one proxy or policy-group description."""

        path = f"/proxies/{quote(name, safe='')}"
        response = self._client.get(path)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                "expected Mihomo proxy response to be an object, but got "
                f"{type(payload).__name__}."
            )
        return payload

    def connections(self) -> list[dict[str, object]]:
        """Return validated active Mihomo connection objects."""

        response = self._client.get("/connections")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                "expected Mihomo connections response to be an object, "
                f"but got {type(payload).__name__}."
            )
        connections = payload.get("connections")
        if connections is None:
            return []
        if not isinstance(connections, list):
            raise ValueError(
                "expected Mihomo connections to be a list, but got "
                f"{type(connections).__name__}."
            )
        if not all(isinstance(item, dict) for item in connections):
            raise ValueError(
                "expected every Mihomo connection to be an object."
            )
        return connections

    def delay(self, name: str, url: str, timeout: float) -> float:
        """Measure one node against an HTTPS URL through Mihomo."""

        path = f"/proxies/{quote(name, safe='')}/delay"
        response = self._client.get(
            path,
            params={
                "url": url,
                "timeout": int(timeout * 1000),
                "expected": "200-399",
            },
        )
        response.raise_for_status()
        delay = response.json().get("delay")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise ValueError(
                "expected Mihomo delay to be numeric, but got "
                f"{delay!r}."
            )
        if delay <= 0:
            raise ValueError(
                f"expected Mihomo delay to be positive, but got {delay}."
            )
        return float(delay)

    def select(self, group_name: str, node_name: str) -> None:
        """Select and verify a direct node in a Mihomo selector."""

        path = f"/proxies/{quote(group_name, safe='')}"
        response = self._client.put(path, json={"name": node_name})
        response.raise_for_status()
        selected = self.proxy(group_name).get("now")
        if selected != node_name:
            raise RuntimeError(
                f"expected Mihomo to select {node_name!r}, but got "
                f"{selected!r}."
            )


def create_tls_context() -> ssl.SSLContext:
    """Create the TLS 1.2 context used by bounded speed tests."""

    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def measure_throughput(
    proxy_url: str,
    url: str,
    byte_limit: int,
    timeout: float,
    request_headers: dict[str, str] | None,
) -> float:
    """Measure a bounded streamed transfer through the selected node.

    Parameters
    ----------
    proxy_url : str
        Explicit local Mihomo proxy URL.
    url : str
        Large HTTPS file suitable for range requests.
    byte_limit : int
        Maximum number of response bytes consumed.
    timeout : float
        Request timeout in seconds.
    request_headers : dict or None
        Additional request headers, optional. Their values are never printed.

    Returns
    -------
    float
        Measured transfer rate in MiB/s.
    """

    headers = dict(request_headers or {})
    headers["Range"] = f"bytes=0-{byte_limit - 1}"

    transferred = 0
    start = time.monotonic()
    with httpx.Client(
        proxy=proxy_url,
        verify=create_tls_context(),
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        trust_env=False,
    ) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            transferred = count_bounded_bytes(
                response.iter_bytes(),
                byte_limit,
            )

    elapsed = time.monotonic() - start
    if transferred <= 0:
        raise RuntimeError("speed test returned no response body bytes.")
    if elapsed <= 0:
        raise RuntimeError(
            f"expected a positive speed-test duration, but got {elapsed}."
        )
    return transferred / MIB / elapsed


def measure_proxy_latency(
    proxy_url: str,
    url: str,
    timeout: float,
    request_headers: dict[str, str] | None,
) -> float:
    """Measure response latency through the active selected proxy node.

    Parameters
    ----------
    proxy_url : str
        Explicit local Mihomo proxy URL.
    url : str
        HTTPS repository endpoint used by the real downloader.
    timeout : float
        Request timeout in seconds.
    request_headers : dict or None
        Additional request headers, optional. Their values are never printed.

    Returns
    -------
    float
        Time to a successful response in milliseconds.
    """

    headers = dict(request_headers or {})
    headers["Range"] = f"bytes=0-{DISCOVERY_RANGE_BYTES - 1}"
    start = time.monotonic()
    with httpx.Client(
        proxy=proxy_url,
        verify=create_tls_context(),
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        trust_env=False,
    ) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
    elapsed = time.monotonic() - start
    if elapsed <= 0:
        raise RuntimeError(
            f"expected a positive proxy-probe duration, but got {elapsed}."
        )
    return elapsed * 1000


def count_bounded_bytes(chunks: Iterable[bytes], byte_limit: int) -> int:
    """Count streamed bytes without consuming beyond a fixed limit."""

    transferred = 0
    for chunk in chunks:
        remaining = byte_limit - transferred
        if remaining <= 0:
            break
        transferred += min(len(chunk), remaining)
        if transferred >= byte_limit:
            break
    return transferred


def connection_ids(connections: Iterable[dict[str, object]]) -> set[str]:
    """Return validated connection identifiers."""

    identifiers = set()
    for connection in connections:
        identifier = connection.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(
                "expected every Mihomo connection to have a non-empty "
                "string ID."
            )
        identifiers.add(identifier)
    return identifiers


def connection_matches_host(
    connection: dict[str, object],
    hostname: str,
) -> bool:
    """Return whether a connection targets the requested hostname."""

    metadata = connection.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            "expected Mihomo connection metadata to be an object."
        )
    host_fields = ("host", "sniffHost", "destinationIP")
    hosts = {
        str(metadata.get(field, "")).lower()
        for field in host_fields
        if metadata.get(field)
    }
    return hostname.lower() in hosts


def connection_chain(connection: dict[str, object]) -> list[str]:
    """Return a validated non-empty proxy chain."""

    chain = connection.get("chains")
    if (
        not isinstance(chain, list)
        or not chain
        or not all(isinstance(name, str) and name for name in chain)
    ):
        raise ValueError(
            "expected the matching Mihomo connection to have a non-empty "
            "string chain."
        )
    return chain


class MihomoNodeManager:
    """Benchmark, rank, and fail over among permitted direct nodes."""

    def __init__(
        self,
        config: MihomoConfig,
        proxy_url: str,
        probe_url: str,
        request_headers: dict[str, str] | None = None,
        *,
        controller: MihomoController | None = None,
        time_fn: TimeFunction = time.monotonic,
        latency_fn: LatencyFunction = measure_proxy_latency,
        throughput_fn: Callable[..., float] = measure_throughput,
    ) -> None:
        """Create a process-local ranking manager."""

        self.config = config
        self.proxy_url = proxy_url
        self.probe_url = probe_url
        self.request_headers = dict(request_headers or {})
        owns_controller = controller is None
        self.controller = controller or MihomoController(config)
        self.time_fn = time_fn
        self.latency_fn = latency_fn
        self.throughput_fn = throughput_fn
        self.ranked_nodes: list[NodeScore] = []
        self.ranked_at: float | None = None
        self.cooldowns: dict[str, float] = {}
        try:
            self.group_name = config.group_name or self._discover_group()
        except Exception:
            if owns_controller:
                self.controller.close()
            raise

    def close(self) -> None:
        """Close controller resources."""

        self.controller.close()

    def _discover_group(self) -> str:
        """Discover the outermost selector used by the speed-test URL."""

        parsed = urlparse(self.config.speed_test_url)
        hostname = parsed.hostname
        if parsed.scheme != "https" or hostname is None:
            raise ValueError(
                "expected an HTTPS speed-test URL with a hostname, but got "
                f"{self.config.speed_test_url!r}."
            )
        existing = connection_ids(self.controller.connections())
        headers = dict(self.request_headers)
        headers["Range"] = (
            f"bytes=0-{DISCOVERY_RANGE_BYTES - 1}"
        )
        with httpx.Client(
            proxy=self.proxy_url,
            verify=create_tls_context(),
            follow_redirects=True,
            timeout=httpx.Timeout(self.config.probe_timeout),
            trust_env=False,
        ) as client:
            with client.stream(
                "GET",
                self.config.speed_test_url,
                headers=headers,
            ) as response:
                response.raise_for_status()
                response_hostname = response.url.host or hostname
                current = self.controller.connections()
        connection_ids(current)

        matches = [
            connection
            for connection in current
            if connection.get("id") not in existing
            and connection_matches_host(connection, response_hostname)
        ]
        if not matches:
            raise RuntimeError(
                "could not discover the Mihomo selector because no new "
                "connection matched speed-test host "
                f"{response_hostname!r}."
            )
        if len(matches) != 1:
            raise RuntimeError(
                "could not discover the Mihomo selector because "
                f"{len(matches)} new connections matched speed-test host "
                f"{response_hostname!r}."
            )

        selectors = []
        for name in connection_chain(matches[0]):
            proxy_type = str(
                self.controller.proxy(name).get("type", "")
            ).lower()
            if proxy_type == "selector":
                selectors.append(name)
        if not selectors:
            raise RuntimeError(
                "could not discover a Selector in the Mihomo connection "
                f"chain for {response_hostname!r}."
            )
        group_name = selectors[-1]
        print(f"Mihomo group discovered: {group_name}")
        return group_name

    def eligible_nodes(self) -> list[str]:
        """Return direct group members allowed by the optional marker."""

        group = self.controller.proxy(self.group_name)
        if str(group.get("type", "")).lower() != "selector":
            raise ValueError(
                "expected the Mihomo download group to be a Selector, "
                f"but got {group.get('type')!r}."
            )
        members = group.get("all")
        if not isinstance(members, list):
            raise ValueError("expected Mihomo selector members to be a list.")

        eligible = []
        for member in members:
            if not isinstance(member, str):
                continue
            if (
                self.config.node_marker
                and self.config.node_marker not in member
            ):
                continue
            proxy_type = str(
                self.controller.proxy(member).get("type", "")
            ).lower()
            if proxy_type in GROUP_PROXY_TYPES:
                continue
            eligible.append(member)

        if not eligible:
            if self.config.node_marker:
                raise ValueError(
                    "the Mihomo group contains no direct nodes matching "
                    f"{self.config.node_marker!r}."
                )
            raise ValueError(
                "the Mihomo group contains no direct nodes."
            )
        return eligible

    def _available(self, name: str) -> bool:
        """Return whether a node is outside its failure cooldown."""

        cooldown_until = self.cooldowns.get(name, 0.0)
        return cooldown_until <= self.time_fn()

    def _latency_profile(self, name: str, order: int) -> LatencyProfile:
        """Run repeated repository probes for one candidate."""

        latencies = []
        for _ in range(LATENCY_PROBE_COUNT):
            try:
                delay = self.controller.delay(
                    name,
                    self.probe_url,
                    self.config.probe_timeout,
                )
            except Exception:
                continue
            latencies.append(delay)
        return LatencyProfile(
            name=name,
            order=order,
            successes=len(latencies),
            attempts=LATENCY_PROBE_COUNT,
            latencies_ms=tuple(latencies),
        )

    def _real_latency_profile(
        self,
        name: str,
        order: int,
    ) -> LatencyProfile:
        """Probe one selected node through the real local proxy path."""

        latencies = []
        try:
            self.controller.select(self.group_name, name)
            delay = self.latency_fn(
                self.proxy_url,
                self.probe_url,
                min(
                    self.config.probe_timeout,
                    MAX_REAL_LATENCY_PROBE_SECONDS,
                ),
                self.request_headers,
            )
        except Exception:
            pass
        else:
            latencies.append(delay)
        return LatencyProfile(
            name=name,
            order=order,
            successes=len(latencies),
            attempts=1,
            latencies_ms=tuple(latencies),
        )

    def _supplement_latency_profiles(
        self,
        candidates: list[str],
        profiles: list[LatencyProfile],
    ) -> list[LatencyProfile]:
        """Fill an incomplete shortlist using bounded real proxy probes."""

        target_size = min(SHORTLIST_SIZE, len(candidates))
        passing = [profile for profile in profiles if profile.successes]
        if len(passing) >= target_size:
            return passing

        print(
            "WARNING: Mihomo controller probes produced only "
            f"{len(passing)} usable node(s); validating candidates through "
            "the real proxy path.",
            file=sys.stderr,
        )
        passing_names = {profile.name for profile in passing}
        for order, name in enumerate(candidates):
            if name in passing_names:
                continue
            profile = self._real_latency_profile(name, order)
            if not profile.successes:
                continue
            passing.append(profile)
            passing_names.add(name)
            if len(passing) >= target_size:
                break
        return passing

    def _throughput_samples(
        self,
        node_name: str,
        byte_limit: int,
        sample_count: int,
    ) -> tuple[float, ...]:
        """Collect bounded throughput samples on one selected node."""

        self.controller.select(self.group_name, node_name)
        samples = []
        for _ in range(sample_count):
            try:
                speed = self.throughput_fn(
                    self.proxy_url,
                    self.config.speed_test_url,
                    byte_limit,
                    min(
                        self.config.probe_timeout,
                        MAX_THROUGHPUT_PROBE_SECONDS,
                    ),
                    self.request_headers,
                )
            except Exception:
                continue
            samples.append(speed)
        return tuple(samples)

    def benchmark(self) -> NodeScore:
        """Benchmark all eligible nodes and select the best candidate."""

        candidates = [
            name for name in self.eligible_nodes() if self._available(name)
        ]
        if not candidates:
            raise MihomoUnavailableError(
                "all permitted Mihomo nodes are currently in cooldown."
            )

        indexed = list(enumerate(candidates))
        with ThreadPoolExecutor(
            max_workers=LATENCY_PROBE_WORKERS
        ) as executor:
            profiles = list(
                executor.map(
                    lambda item: self._latency_profile(item[1], item[0]),
                    indexed,
                )
            )
        passing = self._supplement_latency_profiles(
            candidates,
            profiles,
        )
        passing.sort(
            key=lambda profile: (
                -profile.success_ratio,
                profile.median_latency_ms,
                profile.order,
            )
        )
        shortlist = passing[:SHORTLIST_SIZE]
        if not shortlist:
            raise MihomoUnavailableError(
                "no permitted Mihomo node passed the repository probes."
            )

        scores = []
        for profile in shortlist:
            try:
                samples = self._throughput_samples(
                    profile.name,
                    THROUGHPUT_SAMPLE_BYTES,
                    THROUGHPUT_SAMPLE_COUNT,
                )
            except Exception:
                samples = ()
            scores.append(
                NodeScore(
                    name=profile.name,
                    order=profile.order,
                    latency_successes=profile.successes,
                    latency_attempts=profile.attempts,
                    latencies_ms=profile.latencies_ms,
                    throughput_successes=len(samples),
                    throughput_attempts=THROUGHPUT_SAMPLE_COUNT,
                    throughputs_mib_s=samples,
                )
            )

        successful = [
            score for score in scores if score.throughput_successes
        ]
        if not successful:
            raise MihomoUnavailableError(
                "no permitted Mihomo node passed a bounded speed test."
            )
        successful.sort(key=NodeScore.ranking_key)
        self.ranked_nodes = successful
        self.ranked_at = self.time_fn()
        winner = successful[0]
        self.controller.select(self.group_name, winner.name)
        self._print_ranking(successful)
        return winner

    def _print_ranking(self, scores: list[NodeScore]) -> None:
        """Print ranking metrics without controller credentials."""

        marker = self.config.node_marker or "all direct nodes"
        print(f"Mihomo node ranking (filter: {marker}):")
        for index, score in enumerate(scores, 1):
            print(
                f"  {index}. {score.name} | "
                f"success={score.overall_success_ratio:.0%} | "
                f"speed={score.median_throughput_mib_s:.2f} MiB/s | "
                f"p95={score.p95_latency_ms:.0f} ms"
            )
        print(f"Selected node: {scores[0].name}")

    def rankings_stale(self) -> bool:
        """Return whether current ranking data requires refreshing."""

        if self.ranked_at is None:
            return True
        return self.time_fn() - self.ranked_at >= RANKING_TTL_SECONDS

    def prepare_attempt(self) -> str:
        """Ensure a permitted ranked node is selected before a request."""

        if not self.ranked_nodes or self.rankings_stale():
            return self.benchmark().name

        group = self.controller.proxy(self.group_name)
        selected = group.get("now")
        ranked_names = {score.name for score in self.ranked_nodes}
        if (
            isinstance(selected, str)
            and selected in ranked_names
            and (
                not self.config.node_marker
                or self.config.node_marker in selected
            )
            and self._available(selected)
        ):
            return selected

        for score in self.ranked_nodes:
            if self._available(score.name):
                self.controller.select(
                    self.group_name,
                    score.name,
                )
                return score.name
        return self.benchmark().name

    def failover(self) -> str | None:
        """Cool down the failed node and select the next ranked candidate."""

        group = self.controller.proxy(self.group_name)
        failed = group.get("now")
        if isinstance(failed, str):
            self.cooldowns[failed] = (
                self.time_fn() + NODE_COOLDOWN_SECONDS
            )
            print(f"Mihomo node cooldown: {failed}", file=sys.stderr)

        for score in self.ranked_nodes:
            if not self._available(score.name):
                continue
            try:
                samples = self._throughput_samples(
                    score.name,
                    FAILOVER_SAMPLE_BYTES,
                    1,
                )
            except Exception:
                continue
            if samples:
                print(
                    f"Mihomo failover selected: {score.name} "
                    f"({samples[0]:.2f} MiB/s)",
                    file=sys.stderr,
                )
                return score.name

        self.ranked_nodes = []
        self.ranked_at = None
        return None
