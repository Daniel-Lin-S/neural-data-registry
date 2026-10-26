"""Exercise Figshare identity, manifest validation, retries and file resume."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "download_helpers"))

import download_figshare as figshare
import download_retry
import download_http_manifest
from functools import partial
from download_http_manifest import download_file
from neural_data_registry import service
from neural_data_registry.config import Settings
from neural_data_registry.db.models import Dataset
from neural_data_registry.enums import Provider
from neural_data_registry.storage import IngestionConflictError

DATA = b"test file contents"
ARTICLE_ID = "7272617"
SOURCE = (
    "https://janelia.figshare.com/articles/dataset/"
    "Whole-brain_light-sheet_imaging_data/7272617"
)


@pytest.fixture
def config(tmp_path: Path) -> figshare.DownloadConfig:
    """Create isolated transfer settings with unlimited retries."""
    return figshare.DownloadConfig(
        ARTICLE_ID, None, SOURCE, str(tmp_path / "fish"),
        "https://api.figshare.com/v2", 1, 30.0, False, 0,
        1.0, 2.0, None,
    )


def payload() -> dict:
    """Return a complete immutable article fixture."""
    return {
        "id": int(ARTICLE_ID), "version": 4,
        "files": [{
            "name": "fish.mat", "size": len(DATA),
            "computed_md5": hashlib.md5(DATA).hexdigest(),
            "download_url": "https://api.figshare.com/file/123",
        }],
    }


@pytest.mark.parametrize("value, version", [
    (ARTICLE_ID, None), (SOURCE, None), (SOURCE + "/4", "4"),
    ("https://api.figshare.com/v2/articles/7272617/versions/4", "4"),
])
def test_article_identity(value: str, version: str | None) -> None:
    assert figshare.parse_article(value) == (ARTICLE_ID, version)


@pytest.mark.parametrize("value", [
    "0", "-1", SOURCE + "/0", "https://example.com/articles/7272617",
    "https://figshare.com.evil.test/articles/7272617",
])
def test_invalid_identity(value: str) -> None:
    with pytest.raises(ValueError):
        figshare.parse_article(value)


@pytest.mark.parametrize("field, value", [
    ("name", "../escape"), ("name", "/absolute"),
    ("size", 0), ("size", float("nan")), ("size", True),
    ("computed_md5", "bad"), ("is_link_only", True),
    ("download_url", "file:///tmp/data"),
])
def test_invalid_file_metadata(field: str, value: object) -> None:
    metadata = payload()
    metadata["files"][0][field] = value
    with pytest.raises((ValueError, TypeError)):
        figshare.parse_manifest(metadata)


@pytest.mark.parametrize("name", [
    "fish.mat", "fish.mat.part", "fish.mat/subfile", ".figshare-manifest.json"
])
def test_manifest_collisions(name: str) -> None:
    metadata = payload()
    other = dict(metadata["files"][0], name=name)
    metadata["files"].append(other)
    with pytest.raises(ValueError, match="Conflicting"):
        figshare.parse_manifest(metadata)


def test_empty_manifest() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        figshare.parse_manifest({"files": []})


def test_unlimited_retry_and_resume(config, monkeypatch) -> None:
    """Repeated failures preserve the offset and eventually publish MD5 data."""
    root = Path(config.destination)
    root.mkdir()
    partial_file = root / "fish.mat.part"
    partial_file.write_bytes(DATA[:5])
    calls = []
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Range"))
        if len(calls) <= 12:
            raise httpx.ReadTimeout("interrupted transfer", request=request)
        return httpx.Response(
            206, content=DATA[5:],
            headers={"Content-Range": f"bytes 5-{len(DATA)-1}/{len(DATA)}"},
        )

    monkeypatch.setattr(
        download_http_manifest, "run_with_retries",
        partial(download_retry.run_with_retries, sleep_fn=delays.append),
    )
    record = figshare.parse_manifest(payload())[0]
    factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(config, record, None, factory, "Figshare")
    assert result.read_bytes() == DATA
    assert not partial_file.exists()
    assert calls == ["bytes=5-"] * 13
    assert len(delays) == 12
    assert max(delays) <= config.retry_max_delay
    download_file(config, record, None, factory, "Figshare")
    assert len(calls) == 13


def test_server_ignoring_range(config) -> None:
    root = Path(config.destination)
    root.mkdir()
    (root / "fish.mat.part").write_bytes(DATA[:5])
    factory = lambda: httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=DATA)
    ))
    record = figshare.parse_manifest(payload())[0]
    result = download_file(config, record, None, factory, "Figshare")
    assert result.read_bytes() == DATA


def test_full_run_pins_manifest_and_reuses_files(
    config, tmp_path, monkeypatch
) -> None:
    settings = Settings(data_root=tmp_path / "registry-root")
    monkeypatch.setattr(figshare, "get_settings", lambda: settings)
    monkeypatch.setattr(
        figshare, "configure_diagnostics", lambda *args: tmp_path / "log"
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if "/articles/" in request.url.path:
            return httpx.Response(200, json=payload())
        return httpx.Response(200, content=DATA)

    monkeypatch.setattr(figshare, "create_client", lambda config:
        httpx.Client(transport=httpx.MockTransport(handler)))
    figshare.run(config)
    figshare.run(config)
    assert len(requests) == 2
    manifest = Path(config.destination) / figshare.MANIFEST_NAME
    assert json.loads(manifest.read_text())["version"] == 4
    assert not (settings.ingestion_lock_dir / "registry-intake.lock").exists()
    with pytest.raises(ValueError, match="version"):
        figshare.run(replace(config, version="3"))


@pytest.mark.parametrize("conflict", ["name", "url", "path"])
def test_registered_conflict_before_network(
    conflict, config, tmp_path, monkeypatch
) -> None:
    settings = Settings(data_root=tmp_path / "registry-root")
    monkeypatch.setattr(figshare, "get_settings", lambda: settings)
    with service.session(settings) as db:
        db.add(Dataset(
            name="fish" if conflict == "name" else "another-name",
            provider=Provider.OTHER,
            source_url=(
                "https://figshare.com/articles/7272617/2"
                if conflict == "url" else "https://example.com/another"
            ),
            storage_path=(
                config.destination if conflict == "path"
                else str(tmp_path / "registered")
            ),
        ))
        db.commit()
    monkeypatch.setattr(figshare, "create_client", lambda config:
        pytest.fail("network must not run on identity conflict"))
    with pytest.raises(service.DatasetConflictError, match="storage path"):
        figshare.run(config)
    assert not Path(config.destination).exists()


def test_second_preflight_and_concurrent_exclusion(tmp_path) -> None:
    settings = Settings(data_root=tmp_path / "registry-root")
    destination = tmp_path / "fish"
    with service.external_download_intake(
        "fish", SOURCE, destination, settings
    ):
        with pytest.raises(IngestionConflictError):
            with service.external_download_intake(
                "other", "7272617", tmp_path / "other", settings
            ):
                pytest.fail("a concurrent intake must never start")


def test_shell_default_unlimited_and_no_proxy(tmp_path) -> None:
    import os
    stub = tmp_path / "python"
    stub.write_text(
        '#!/bin/sh\n'
        'echo "$DOWNLOAD_RETRY_ATTEMPTS|$DOWNLOAD_PROXY_URL|'
        '${HTTPS_PROXY-unset}"\n'
    )
    stub.chmod(0o755)
    env = dict(os.environ, DOWNLOAD_PYTHON=str(stub), HTTPS_PROXY="bad")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/download_figshare.sh"),
         "--repo", ARTICLE_ID, "--dest", str(tmp_path / "fish"),
         "--no-proxy"], env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "0||unset" in result.stdout


@pytest.mark.parametrize("code, output, error_type", [
    (0, '{"name": "fish", "storage_path": "/registered"}',
     FileExistsError),
    (1, "database unavailable", RuntimeError),
    (2, "unknown query argument", RuntimeError),
])
def test_standalone_preflight_fails_closed(
    tmp_path, monkeypatch, code, output, error_type
) -> None:
    """Registry conflicts and failures must prevent network access."""
    import download_standalone as standalone
    monkeypatch.setattr(standalone.shutil, "which", lambda name: "/bin/cli")
    monkeypatch.setattr(standalone.subprocess, "run", lambda *a, **kw:
        subprocess.CompletedProcess(a[0], code, output, ""))
    with pytest.raises(error_type):
        standalone.check_registered_identities(
            "fish", [SOURCE], tmp_path / "fish"
        )


def test_standalone_checks_before_and_after_locks(tmp_path, monkeypatch):
    """The shared source lock rejects a differently named parallel request."""
    import download_standalone as standalone
    monkeypatch.setattr(standalone, "LOCK_DIRECTORY", tmp_path)
    checks = []
    monkeypatch.setattr(standalone, "check_registered_identities",
                        lambda *args: checks.append(args))
    with standalone.standalone_download_intake(
        "fish", "figshare:7272617", [SOURCE], tmp_path / "fish"
    ):
        assert len(checks) == 2
        with pytest.raises(RuntimeError, match="already running"):
            with standalone.standalone_download_intake(
                "other", "figshare:7272617", [SOURCE], tmp_path / "other"
            ):
                pytest.fail("the same article must not download twice")
    with standalone.standalone_download_intake(
        "fish", "figshare:7272617", [SOURCE], tmp_path / "fish"
    ):
        pass


def test_process_exit_releases_standalone_lock(tmp_path, monkeypatch):
    """A separate process sees contention and succeeds after owner exit."""
    import download_standalone as standalone
    monkeypatch.setattr(standalone, "LOCK_DIRECTORY", tmp_path)
    program = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "import download_standalone as s; "
        "s.LOCK_DIRECTORY = Path(sys.argv[2]); "
        "guard = s.identity_lock('source:figshare:7272617'); "
        "guard.__enter__()"
    )
    command = [sys.executable, "-c", program,
               str(ROOT / "download_helpers"), str(tmp_path)]
    with standalone.identity_lock("source:figshare:7272617"):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "already running" in result.stderr
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with standalone.identity_lock("source:figshare:7272617"):
        pass


@pytest.mark.parametrize("invalid_metadata", [False, True])
def test_mihomo_lifecycle_and_retry_wiring(
    config, tmp_path, monkeypatch, invalid_metadata
):
    """Metadata and transfers share failover, and always close the manager."""
    from mihomo_ranker import MihomoConfig
    settings = Settings(data_root=tmp_path / "registry-root")
    monkeypatch.setattr(figshare, "get_settings", lambda: settings)
    diagnostics = []
    monkeypatch.setattr(figshare, "configure_diagnostics", lambda *args:
        diagnostics.append(args[1]) or tmp_path / "log")
    events = []

    class Manager:
        def prepare_attempt(self):
            events.append("prepare")
            return "cheap-0.1倍"

        def failover(self):
            events.append("failover")
            return "backup-0.1倍"

        def close(self):
            events.append("close")

    monkeypatch.setattr(figshare, "MihomoNodeManager", lambda *args: Manager())
    attempts = []

    def handler(request):
        attempts.append(str(request.url))
        if len(attempts) == 1:
            raise httpx.ReadTimeout("temporary failure", request=request)
        if "/articles/" in request.url.path:
            metadata = payload()
            if invalid_metadata:
                metadata["version"] = None
            return httpx.Response(200, json=metadata)
        return httpx.Response(200, content=DATA)

    monkeypatch.setattr(figshare, "create_client", lambda config:
        httpx.Client(transport=httpx.MockTransport(handler)))
    config = replace(config, retry_base_delay=0.001, mihomo=MihomoConfig(
        "http://127.0.0.1:9091", "download", "0.1倍",
        "https://example.com/file", 8.0, "test-secret", True,
    ))
    if invalid_metadata:
        with pytest.raises(ValueError, match="version"):
            figshare.run(config)
    else:
        figshare.run(config)
        assert (Path(config.destination) / "fish.mat").read_bytes() == DATA
        assert events.count("prepare") == 3
    assert "failover" in events
    assert events[-1] == "close"
    assert "test-secret" not in repr(diagnostics)
