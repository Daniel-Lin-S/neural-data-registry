from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from neural_data_registry import cli
from neural_data_registry.config import Settings, get_settings
from neural_data_registry.enums import Provider
from neural_data_registry.main import create_app
from neural_data_registry.service import dataset_dict, find_datasets, ingest_local, session


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    """Provide an isolated temporary registry configuration."""
    root = tmp_path / "neural_data"
    return Settings(data_root=root, database_url=f"sqlite:///{root / 'registry' / 'registry.db'}")


def mock_dataset(tmp_path: Path, label: str) -> Path:
    """Create a minimal local MEG dataset fixture beneath pytest's temporary path."""
    source = tmp_path / label
    source.mkdir()
    (source / "dataset_description.json").write_text('{"Name": "Mock"}')
    (source / "meg.fif").write_bytes(b"mock meg data")
    return source


def test_global_data_root_comes_from_environment(monkeypatch, tmp_path):
    """Verify the global root and default SQLite URL are derived from NDR_DATA_ROOT."""
    expected_root = tmp_path / "global-neural-data"
    monkeypatch.setenv("NDR_DATA_ROOT", str(expected_root))
    monkeypatch.delenv("NDR_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    loaded = get_settings()
    assert loaded.data_root == expected_root
    assert loaded.resolved_database_url == f"sqlite:///{expected_root / 'registry' / 'registry.db'}"
    get_settings.cache_clear()



def ingest_mock(config: Settings, tmp_path: Path, *, name="THINGS-MEG", url="https://openneuro.org/datasets/ds004212", version="3.0.0"):
    """Ingest a mock OpenNeuro MEG dataset for tests requiring a registry record."""
    return ingest_local(mock_dataset(tmp_path, f"source-{name}-{version}"), name, Provider.OPENNEURO, url, version, ["MEG"], config)

def test_local_ingestion_references_mock_dataset_by_default(config, tmp_path):
    """Ensure default local ingestion references data in place and writes a manifest."""
    source = mock_dataset(tmp_path, "source")
    item = ingest_local(source, "THINGS-MEG", Provider.OPENNEURO, "https://openneuro.org/datasets/ds004212", "3.0.0", ["MEG"], config)
    data = dataset_dict(item)
    assert data["status"] == "available"
    assert data["size_bytes"] == len(b"mock meg data") + len('{"Name": "Mock"}')
    assert data["storage_mode"] == "reference"
    assert source.exists()
    assert Path(data["storage_path"]) == source.resolve()
    assert (Path(data["storage_path"]) / "meg.fif").is_file()
    assert (config.registry_dir / f"{item.id}.json").is_file()


def test_local_ingestion_can_move_mock_dataset(config, tmp_path):
    """Ensure explicit move mode relocates files into the managed datasets tree."""
    source = mock_dataset(tmp_path, "move-source")
    item = ingest_local(source, "MOVE-MEG", Provider.LOCAL, None, "1.0.0", ["MEG"], config, storage_mode="move")
    assert item.storage_mode.value == "move"
    assert not source.exists()
    assert Path(item.storage_path).is_relative_to(config.datasets_dir)
    assert (Path(item.storage_path) / "meg.fif").is_file()


def test_rejects_repeated_name_with_existing_managed_path(config, tmp_path):
    """Reject a second dataset using the same name and point to the existing copy."""
    existing = ingest_mock(config, tmp_path)
    with pytest.raises(RuntimeError, match="dataset name is already registered") as error:
        ingest_local(mock_dataset(tmp_path, "different-source"), "things-meg", Provider.LOCAL, None, "1.0.0", [], config)
    assert existing.id in str(error.value)
    assert existing.storage_path in str(error.value)


def test_rejects_repeated_url_with_existing_managed_path(config, tmp_path):
    """Reject a second dataset using the same source URL and point to the existing copy."""
    existing = ingest_mock(config, tmp_path)
    with pytest.raises(RuntimeError, match="source URL/path is already registered") as error:
        ingest_local(mock_dataset(tmp_path, "different-source"), "Other name", Provider.OPENNEURO, "https://openneuro.org/datasets/ds004212", "4.0.0", [], config)
    assert existing.storage_path in str(error.value)


def test_rejects_missing_local_source(config, tmp_path):
    """Reject ingestion requests whose declared local source does not exist."""
    with pytest.raises(ValueError, match="not a directory"):
        ingest_local(tmp_path / "missing", "Missing", Provider.LOCAL, None, "1.0.0", [], config)


def test_queries_by_name_url_and_modality(config, tmp_path):
    """Verify registry search works for name, source URL, and modality filters."""
    item = ingest_mock(config, tmp_path)
    with session(config) as db:
        assert [x.id for x in find_datasets(db, query="things")] == [item.id]
        assert [x.id for x in find_datasets(db, url=item.source_url)] == [item.id]
        assert [x.id for x in find_datasets(db, modality="meg")] == [item.id]
        assert find_datasets(db, query="absent") == []


def test_all_core_api_routes(config, tmp_path):
    """Exercise health, dataset lookup, local-ingest, duplicate, and error API responses."""
    client = TestClient(create_app(config))
    assert client.get("/health").json() == {"status": "ok"}
    source = mock_dataset(tmp_path, "api-source")
    created = client.post("/ingest/local", json={"source": str(source), "name": "THINGS-MEG", "provider": "openneuro", "url": "https://openneuro.org/datasets/ds004212", "version": "3.0.0", "modalities": ["MEG"]})
    assert created.status_code == 201
    item = created.json()
    assert client.get("/datasets", params={"query": "THINGS"}).json() == [item]
    assert client.get("/datasets", params={"url": item["source_url"]}).json() == [item]
    assert client.get("/datasets", params={"modality": "MEG"}).json() == [item]
    assert client.get(f"/datasets/{item['dataset_id']}").json() == item
    assert client.get("/datasets/no-such-id").status_code == 404
    duplicate = client.post("/ingest/local", json={"source": str(mock_dataset(tmp_path, "api-duplicate")), "name": "things-meg", "provider": "local", "version": "1.0.0"})
    assert duplicate.status_code == 409
    assert item["storage_path"] in duplicate.json()["detail"]
    assert item["storage_mode"] == "reference"
    assert source.exists()
    assert client.post("/ingest/local", json={"source": str(tmp_path / "missing"), "name": "Missing", "version": "1"}).status_code == 400


def test_cli_query_and_list(config, tmp_path, monkeypatch):
    """Verify query and modality-list CLI commands render registered datasets."""
    ingest_mock(config, tmp_path)
    monkeypatch.setattr(cli, "session", lambda: session(config))
    runner = CliRunner()
    assert runner.invoke(cli.app, ["query", "THINGS-MEG"]).exit_code == 0
    assert "THINGS-MEG" in runner.invoke(cli.app, ["list", "--modality", "MEG"]).output
