"""Tests for deterministic Mihomo node ranking and filtering."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "mihomo_ranker.py"


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


def test_benchmark_selects_fastest_equally_stable_node() -> None:
    """Exercise the complete deterministic shortlist and selection path."""

    members = ["slow-0.1倍", "fast-0.1倍"]
    controller = FakeController(members)
    manager = make_manager(controller)

    def throughput(*args: Any) -> float:
        del args
        if controller.selected == "fast-0.1倍":
            return 5.0
        return 1.0

    manager.throughput_fn = throughput
    winner = manager.benchmark()

    assert winner.name == "fast-0.1倍"
    assert controller.selected == "fast-0.1倍"
