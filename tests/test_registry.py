from __future__ import annotations

from pathlib import Path
import json
import subprocess
from unittest.mock import Mock

import pytest
from sqlalchemy import inspect
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from neural_data_registry import cli
from neural_data_registry import health as health_service
from neural_data_registry.config import Settings, get_settings
from neural_data_registry.db.models import Base, Dataset, DatasetAlias, HealthCheckHistory, IngestionJob
from neural_data_registry.db.session import create_database, get_session_factory
from neural_data_registry.enums import DatasetStatus, Provider, StorageMode
from neural_data_registry.health import (
    maybe_launch_cooldown_check,
    request_health_check,
    run_health_checks,
)
from neural_data_registry.main import create_app
from neural_data_registry.provider import base as provider_base
from neural_data_registry.service import (
    add_name_aliases,
    download as download_dataset,
    resolve_dataset,
    resolve_download_version,
    infer_url_version,
)
from neural_data_registry.storage import directory_size, ensure_layout, process_lock
from neural_data_registry.storage import dataset_destination, ingestion_lock
from neural_data_registry.service import dataset_dict, find_datasets, ingest_local, session, update_dataset_metadata


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
    item = ingest_local(source, "MOVE-MEG", Provider.OTHER, None, "1.0.0", ["MEG"], config, storage_mode="move")
    assert item.storage_mode.value == "move"
    assert not source.exists()
    assert Path(item.storage_path).is_relative_to(config.datasets_dir)
    assert (Path(item.storage_path) / "meg.fif").is_file()


def test_storage_transition_api_moves_reference_and_preserves_metadata(config, tmp_path):
    """Move a registered reference without replacing its dataset identity."""
    source = mock_dataset(tmp_path, "transition-move-source")
    item = ingest_local(
        source,
        "Transition move",
        Provider.OPENNEURO,
        "https://openneuro.org/datasets/transition-move",
        "1",
        ["meg"],
        config,
        name_aliases=["Transition alias"],
    )

    client = TestClient(create_app(config))
    response = client.post(
        f"/datasets/{item.id}/storage-transition", json={"storage_mode": "move"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == item.id
    assert data["source_url"] == item.source_url
    assert data["aliases"] == ["Transition alias"]
    assert data["storage_mode"] == "move"
    assert not source.exists()
    assert Path(data["storage_path"]).is_relative_to(config.datasets_dir)
    assert (Path(data["storage_path"]) / "meg.fif").is_file()
    assert json.loads((config.registry_dir / f"{item.id}.json").read_text()) == data
    assert client.post(
        f"/datasets/{item.id}/storage-transition", json={"storage_mode": "copy"}
    ).status_code == 409
    with session(config) as db:
        job = (
            db.query(IngestionJob)
            .filter_by(dataset_id=item.id, mode="storage-transition-move")
            .first()
        )
        assert job.mode == "storage-transition-move"
        assert job.status.value == "succeeded"


def test_storage_transition_api_copies_reference_and_preserves_source(config, tmp_path):
    """Copying a reference keeps the registered source directory intact."""
    source = mock_dataset(tmp_path, "transition-copy-source")
    item = ingest_local(source, "Transition copy", Provider.OTHER, None, "1", ["meg"], config)

    response = TestClient(create_app(config)).post(
        f"/datasets/{item.id}/storage-transition", json={"storage_mode": "copy"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["storage_mode"] == "copy"
    assert source.exists()
    assert Path(data["storage_path"]).is_relative_to(config.datasets_dir)
    assert (Path(data["storage_path"]) / "meg.fif").read_bytes() == b"mock meg data"


def test_storage_transition_api_rejects_invalid_or_unavailable_targets(config, tmp_path):
    """Reject invalid modes, missing records, and missing reference paths safely."""
    client = TestClient(create_app(config))
    source = mock_dataset(tmp_path, "transition-invalid-source")
    item = ingest_local(source, "Transition invalid", Provider.OTHER, None, "1", ["meg"], config)

    assert client.post("/datasets/no-such-id/storage-transition", json={"storage_mode": "move"}).status_code == 404
    assert client.post(f"/datasets/{item.id}/storage-transition", json={"storage_mode": "reference"}).status_code == 400
    assert client.post(f"/datasets/{item.id}/storage-transition", json={"storage_mode": "other"}).status_code == 400

    for child in source.iterdir():
        child.unlink()
    source.rmdir()
    response = client.post(f"/datasets/{item.id}/storage-transition", json={"storage_mode": "move"})
    assert response.status_code == 400
    with session(config) as db:
        unchanged = db.get(Dataset, item.id)
        assert unchanged.storage_mode is StorageMode.REFERENCE
        assert unchanged.storage_path == str(source)


def test_storage_transition_rejects_destination_conflict_and_intake_lock(config, tmp_path):
    """Keep the record unchanged when destination or concurrent intake conflicts."""
    source = mock_dataset(tmp_path, "transition-conflict-source")
    item = ingest_local(source, "Transition conflict", Provider.OTHER, None, "1", ["meg"], config)
    destination = dataset_destination(item.id, item.name, item.version, config)
    destination.mkdir(parents=True)
    client = TestClient(create_app(config))

    response = client.post(f"/datasets/{item.id}/storage-transition", json={"storage_mode": "move"})
    assert response.status_code == 409
    assert source.exists()
    with session(config) as db:
        assert db.get(Dataset, item.id).storage_mode is StorageMode.REFERENCE

    destination.rmdir()
    with ingestion_lock("registry-intake", config):
        response = client.post(f"/datasets/{item.id}/storage-transition", json={"storage_mode": "move"})
    assert response.status_code == 409
    assert source.exists()


def test_rejects_repeated_name_with_existing_managed_path(config, tmp_path):
    """Reject a second dataset using the same name and point to the existing copy."""
    existing = ingest_mock(config, tmp_path)
    with pytest.raises(RuntimeError, match="dataset name is already registered") as error:
        ingest_local(mock_dataset(tmp_path, "different-source"), "things-meg", Provider.OTHER, None, "1.0.0", [], config)
    assert existing.id in str(error.value)
    assert existing.storage_path in str(error.value)


def test_rejects_repeated_url_with_existing_managed_path(config, tmp_path):
    """Reject a second dataset using the same source URL and point to the existing copy."""
    existing = ingest_mock(config, tmp_path)
    with pytest.raises(RuntimeError, match="source URL/path is already registered") as error:
        ingest_local(mock_dataset(tmp_path, "different-source"), "Other name", Provider.OPENNEURO, "https://openneuro.org/datasets/ds004212", "4.0.0", [], config)
    assert existing.storage_path in str(error.value)


def test_ingest_preflights_conflicts_before_validating_the_source(config, tmp_path):
    """A duplicate name or URL stops local intake before source-file work."""
    existing = ingest_mock(config, tmp_path)
    missing_source = tmp_path / "must-not-be-processed"

    with pytest.raises(RuntimeError, match="dataset name is already registered"):
        ingest_local(
            missing_source, existing.name, Provider.OTHER, None, "1.0.0", [], config
        )
    with pytest.raises(RuntimeError, match="source URL/path is already registered"):
        ingest_local(
            missing_source,
            "Different dataset",
            Provider.OTHER,
            existing.source_url,
            "1.0.0",
            [],
            config,
        )
    assert not missing_source.exists()


@pytest.mark.parametrize(
    ("name", "url", "reason"),
    [
        (
            "THINGS-MEG",
            "https://openneuro.org/datasets/ds999999/versions/1.0.0",
            "dataset name",
        ),
        (
            "Different dataset",
            "https://openneuro.org/datasets/ds004212",
            "source URL/path",
        ),
    ],
)
def test_download_preflights_name_and_url_before_provider_work(
    config, tmp_path, monkeypatch, name, url, reason
):
    """Duplicate downloads do not create a workspace, log, or provider request."""
    ingest_mock(config, tmp_path)
    calls = []
    monkeypatch.setattr(
        "neural_data_registry.service.download_from_url",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match=reason):
        download_dataset(url, "1.0.0", config, name=name, modalities=["meg"])

    assert calls == []
    assert list(config.incoming_dir.iterdir()) == []
    assert list(config.logs_dir.glob("download-*.log")) == []


def test_download_api_conflict_is_preflighted(config, tmp_path, monkeypatch):
    """The public download endpoint returns a conflict before provider work."""
    existing = ingest_mock(config, tmp_path)
    calls = []
    monkeypatch.setattr(
        "neural_data_registry.service.download_from_url",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    response = TestClient(create_app(config)).post(
        "/download",
        json={
            "url": existing.source_url,
            "version": "1.0.0",
            "name": "Different dataset",
            "modalities": ["meg"],
        },
    )

    assert response.status_code == 409
    assert "source URL/path is already registered" in response.json()["detail"]
    assert calls == []
    assert list(config.incoming_dir.iterdir()) == []




def test_rejects_missing_local_source(config, tmp_path):
    """Reject ingestion requests whose declared local source does not exist."""
    with pytest.raises(ValueError, match="not a directory"):
        ingest_local(tmp_path / "missing", "Missing", Provider.OTHER, None, "1.0.0", [], config)


def test_queries_by_name_url_and_modality(config, tmp_path):
    """Verify registry search works for name, source URL, and modality filters."""
    item = ingest_mock(config, tmp_path)
    with session(config) as db:
        assert [x.id for x in find_datasets(db, query="things")] == [item.id]
        assert [x.id for x in find_datasets(db, url=item.source_url)] == [item.id]
        assert [x.id for x in find_datasets(db, modality="meg")] == [item.id]
        assert find_datasets(db, query="absent") == []



def test_segment_bound_url_matching_for_queries_and_api(config, tmp_path):
    """Match remote URLs by host and first path segment, never provider root."""
    root = ingest_local(
        mock_dataset(tmp_path, "osf-root"), "OSF ROOT", Provider.OTHER,
        "https://osf.io/pq7vb/", "1", ["meg"], config,
    )
    nested = ingest_local(
        mock_dataset(tmp_path, "osf-nested"), "OSF NESTED", Provider.OTHER,
        "https://osf.io/pq7vb/files/osfstorage", "1", ["eeg"], config,
    )
    other = ingest_local(
        mock_dataset(tmp_path, "osf-other"), "OSF OTHER", Provider.OTHER,
        "https://osf.io/otherid/files", "1", ["meg"], config,
    )
    with session(config) as db:
        stored_root = db.get(Dataset, root.id)
        stored_root.aliases.append(DatasetAlias(kind="url", value="https://osf.io/aliasid/files"))
        stored_root.aliases.append(DatasetAlias(kind="path", value="/registered/local/path"))
        db.commit()
    with session(config) as db:
        assert {item.id for item in find_datasets(db, url="https://osf.io/pq7vb/?download=1#files")} == {root.id, nested.id}
        assert {item.id for item in find_datasets(db, url="https://osf.io/pq7vb/files/osfstorage")} == {root.id, nested.id}
        assert [item.id for item in find_datasets(db, url="https://osf.io/")] == []
        assert [item.id for item in find_datasets(db, url="https://osf.io/otherid/")] == [other.id]
        assert [item.id for item in find_datasets(db, url="https://osf.io/aliasid/")] == [root.id]
        assert [item.id for item in find_datasets(db, url="/registered/local/path")] == [root.id]
        assert find_datasets(db, url="/registered/local/path/child") == []
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_dataset(db, url="https://osf.io/pq7vb/")
    response = TestClient(create_app(config)).get(
        "/datasets", params={"url": "https://osf.io/pq7vb/files/osfstorage"}
    )
    assert {row["dataset_id"] for row in response.json()} == {root.id, nested.id}


def test_name_aliases_are_searchable_and_protected_during_local_intake(config, tmp_path):
    """Record intake aliases, resolve them, and reserve them as dataset names."""
    item = ingest_local(
        mock_dataset(tmp_path, "alias-source"),
        "THINGS-MEG",
        Provider.OPENNEURO,
        "https://openneuro.org/datasets/ds004212",
        "3.0.0",
        ["meg"],
        config,
        name_aliases=["THINGS", "THINGS vision"],
    )

    assert dataset_dict(item)["aliases"] == ["THINGS", "THINGS vision"]
    with session(config) as db:
        assert [row.id for row in find_datasets(db, query="vision")] == [item.id]
        assert resolve_dataset(db, name="things") is not None
        assert resolve_dataset(db, "THINGS vision").id == item.id

    duplicate_source = tmp_path / "must-not-be-read"
    with pytest.raises(RuntimeError, match="dataset name is already registered"):
        ingest_local(
            duplicate_source,
            "things",
            Provider.OTHER,
            None,
            "1",
            [],
            config,
        )
    assert not duplicate_source.exists()


def test_existing_dataset_alias_command_and_list_search(config, tmp_path, monkeypatch):
    """Add aliases after registration and use them through brainctl list."""
    item = ingest_mock(config, tmp_path)
    updated = add_name_aliases(item.id, ["Things dataset", "Visual things"], config)
    assert dataset_dict(updated)["aliases"] == ["Things dataset", "Visual things"]

    monkeypatch.setattr(cli, "session", lambda: session(config))
    monkeypatch.setattr(cli, "add_name_aliases", lambda identifier, aliases: add_name_aliases(identifier, aliases, config))
    monkeypatch.setattr(cli, "maybe_launch_cooldown_check", lambda: False)
    monkeypatch.setattr(cli, "console", cli.Console(width=160))
    runner = CliRunner()

    result = runner.invoke(cli.app, ["list", "--query", "visual"], terminal_width=160)
    assert result.exit_code == 0
    assert "THINGS-MEG" in result.output

    added = runner.invoke(cli.app, ["alias", item.id, "--alias", "MEG things"])
    assert added.exit_code == 0
    assert "MEG things" in added.output
    with session(config) as db:
        assert resolve_dataset(db, "meg THINGS").id == item.id


def test_aliases_can_be_provided_to_download(config, monkeypatch):
    """Persist aliases supplied before a download begins."""
    def fake_download(url, version, destination, **kwargs):
        (destination / "meg.fif").write_bytes(b"mock meg data")

    monkeypatch.setattr("neural_data_registry.service.download_from_url", fake_download)
    item = download_dataset(
        "https://openneuro.org/datasets/ds004212",
        "3.0.0",
        config,
        name="THINGS-MEG",
        modalities=["meg"],
        name_aliases=["THINGS", "Object vision"],
    )

    assert dataset_dict(item)["aliases"] == ["Object vision", "THINGS"]
    with session(config) as db:
        assert [row.id for row in find_datasets(db, query="object")] == [item.id]


def test_aliases_are_accepted_by_intake_api(config, tmp_path):
    """Keep API and CLI/local intake metadata equivalent."""
    source = mock_dataset(tmp_path, "api-alias-source")
    response = TestClient(create_app(config)).post(
        "/ingest/local",
        json={
            "source": str(source),
            "name": "API aliases",
            "version": "1",
            "modalities": ["meg"],
            "aliases": ["API alias"],
        },
    )

    assert response.status_code == 201
    assert response.json()["aliases"] == ["API alias"]
    found = TestClient(create_app(config)).get("/datasets", params={"query": "alias"})
    assert [row["dataset_id"] for row in found.json()] == [response.json()["dataset_id"]]


def test_all_core_api_routes(config, tmp_path):
    """Exercise health, dataset lookup, local-ingest, duplicate, and error API responses."""
    client = TestClient(create_app(config))
    assert client.get("/health").json() == {"status": "ok"}
    source = mock_dataset(tmp_path, "api-source")
    created = client.post("/ingest/local", json={"source": str(source), "name": "THINGS-MEG", "provider": "openneuro", "url": "https://openneuro.org/datasets/ds004212", "version": "3.0.0", "modalities": ["meg"]})
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


def test_cli_query_prompts_for_a_segment_url_match(config, tmp_path, monkeypatch):
    """Let users choose one canonical name when a URL matches several datasets."""
    root = ingest_local(
        mock_dataset(tmp_path, "cli-osf-root"), "CLI OSF ROOT", Provider.OTHER,
        "https://osf.io/pq7vb/", "1", ["meg"], config,
    )
    nested = ingest_local(
        mock_dataset(tmp_path, "cli-osf-nested"), "CLI OSF NESTED", Provider.OTHER,
        "https://osf.io/pq7vb/files/osfstorage", "1", ["eeg"], config,
    )
    monkeypatch.setattr(cli, "session", lambda: session(config))
    monkeypatch.setattr(cli, "maybe_launch_cooldown_check", lambda: False)
    monkeypatch.setattr(cli, "request_health_check", lambda dataset_id: request_health_check(dataset_id, config))
    monkeypatch.setattr(cli, "console", cli.Console(width=160))
    result = CliRunner().invoke(
        cli.app, ["query", "--url", "https://osf.io/pq7vb/"],
        input="not a dataset\nCLI OSF NESTED\n",
    )
    assert result.exit_code == 0
    assert "Multiple matching datasets found" in result.output
    assert "Name" in result.output and "Modalities" in result.output
    assert "Source URL" in result.output and "Size" in result.output
    assert "Enter one of the displayed canonical names" in result.output
    assert str(Path(nested.storage_path).resolve()) in result.output
    assert str(Path(root.storage_path).resolve()) not in result.output
    positional = CliRunner().invoke(
        cli.app, ["query", "https://osf.io/pq7vb/files/osfstorage"],
        input="CLI OSF ROOT\n",
    )
    assert positional.exit_code == 0
    assert str(Path(root.storage_path).resolve()) in positional.output


def test_cli_query_and_list(config, tmp_path, monkeypatch):
    """Verify query and modality-list CLI commands render registered datasets."""
    item = ingest_mock(config, tmp_path)
    monkeypatch.setattr(cli, "session", lambda: session(config))
    monkeypatch.setattr(cli, "maybe_launch_cooldown_check", lambda: False)
    monkeypatch.setattr(
        cli,
        "request_health_check",
        lambda dataset_id: request_health_check(dataset_id, config),
    )
    monkeypatch.setattr(cli, "console", cli.Console(width=160))
    runner = CliRunner()
    query_result = runner.invoke(cli.app, ["query", "THINGS-MEG"])
    assert query_result.exit_code == 0
    assert query_result.output.strip() == str(Path(item.storage_path).resolve())
    assert runner.invoke(cli.app, ["query", item.id]).output.strip() == str(Path(item.storage_path).resolve())
    assert runner.invoke(cli.app, ["query", "--url", item.source_url]).output.strip() == str(Path(item.storage_path).resolve())
    result = runner.invoke(
        cli.app, ["list", "--modality", "meg"], terminal_width=160
    )
    assert result.exit_code == 0
    assert "THINGS-MEG" in result.output
    assert "Storage Mode" not in result.output
    assert "reference" not in result.output


def test_create_database_reconciles_missing_columns_across_registry(config):
    """Synchronize old SQLite tables with all columns in the current models."""
    create_database(config)
    engine = get_session_factory(config.resolved_database_url).kw["bind"]
    with engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE datasets DROP COLUMN storage_mode")
        connection.exec_driver_sql("ALTER TABLE ingestion_jobs DROP COLUMN message")
        connection.exec_driver_sql(
            "INSERT INTO datasets "
            "(id, name, provider, version, modalities, size_bytes, status) "
            "VALUES ('legacy-id', 'Legacy dataset', 'LOCAL', 'unknown', '', 0, 'AVAILABLE')"
        )
        connection.exec_driver_sql(
            "INSERT INTO ingestion_jobs (id, dataset_id, status, mode) "
            "VALUES ('legacy-job', 'legacy-id', 'SUCCEEDED', 'local')"
        )

    create_database(config)

    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        actual = {column["name"] for column in inspector.get_columns(table.name)}
        assert actual == set(table.columns.keys())

    with session(config) as db:
        dataset = db.get(Dataset, "legacy-id")
        assert dataset is not None
        assert dataset.storage_mode.value == "reference"
        job = db.get(IngestionJob, "legacy-job")
        assert job is not None
        assert job.message is None


def test_legacy_dataset_fields_are_preserved_but_do_not_block_new_rows(
    config, tmp_path, monkeypatch
):
    """Retain retired SQL data without exposing or requiring its old field."""
    config.registry_dir.mkdir(parents=True)
    engine = get_session_factory(config.resolved_database_url).kw["bind"]
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE datasets (
                id VARCHAR(36) NOT NULL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                provider VARCHAR(9) NOT NULL,
                source_url VARCHAR(2048),
                modalities TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status VARCHAR(11) NOT NULL,
                storage_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                retired_required_field TEXT NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO datasets (
                id, name, provider, source_url, modalities,
                size_bytes, status, storage_path, retired_required_field
            ) VALUES (
                'old-dataset', 'Old dataset', 'LOCAL', 'file:///old',
                'meg', 7, 'AVAILABLE', '/old', 'legacy-secret'
            )
            """
        )

    create_database(config)

    columns = {
        column["name"]: column for column in inspect(engine).get_columns("datasets")
    }
    assert "storage_mode" in columns
    assert columns["retired_required_field"]["nullable"] is True

    with session(config) as db:
        old_dataset = db.get(Dataset, "old-dataset")
        assert old_dataset is not None
        old_data = dataset_dict(old_dataset)
        assert old_data["version"] == "unknown"
        assert old_data["storage_mode"] == "reference"
        assert "retired_required_field" not in old_data

    source = mock_dataset(tmp_path, "new-source")
    new_dataset = ingest_local(
        source,
        "New dataset",
        Provider.OTHER,
        None,
        "1",
        ["meg"],
        config,
        storage_mode=StorageMode.REFERENCE,
    )

    with engine.connect() as connection:
        old_retired, new_retired = connection.exec_driver_sql(
            """
            SELECT
                MAX(CASE WHEN id = 'old-dataset' THEN retired_required_field END),
                MAX(CASE WHEN id = ? THEN retired_required_field END)
            FROM datasets
            """,
            (new_dataset.id,),
        ).one()
    assert old_retired == "legacy-secret"
    assert new_retired is None

    monkeypatch.setattr(cli, "session", lambda: session(config))
    monkeypatch.setattr(cli, "console", cli.Console(width=160))
    result = CliRunner().invoke(cli.app, ["list"], terminal_width=160)
    assert result.exit_code == 0
    assert "Storage Mode" not in result.output
    assert "reference" not in result.output
    assert "retired_required_field" not in result.output
    assert "legacy-secret" not in result.output



def test_aliases_upgrade_a_legacy_dataset_without_removing_metadata(config):
    """Create the alias table for an old registry while preserving its values."""
    config.registry_dir.mkdir(parents=True)
    engine = get_session_factory(config.resolved_database_url).kw["bind"]
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE datasets (
                id VARCHAR(36) NOT NULL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                provider VARCHAR(9) NOT NULL,
                source_url VARCHAR(2048),
                version VARCHAR(128) NOT NULL DEFAULT 'unknown',
                modalities TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status VARCHAR(11) NOT NULL,
                storage_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO datasets (
                id, name, provider, source_url, version, modalities,
                size_bytes, status, storage_path
            ) VALUES (
                'legacy-alias-id', 'Legacy alias dataset', 'LOCAL',
                'file:///legacy', '1', 'meg', 7, 'AVAILABLE', '/legacy'
            )
            """
        )

    create_database(config)
    updated = add_name_aliases("legacy-alias-id", ["Legacy MEG"], config)

    assert dataset_dict(updated)["aliases"] == ["Legacy MEG"]
    with session(config) as db:
        legacy = resolve_dataset(db, "legacy meg")
        assert legacy is not None
        assert legacy.source_url == "file:///legacy"
        assert legacy.storage_path == "/legacy"


def test_layout_consolidates_download_workspace_in_incoming(config):
    """Create one not-ready workspace and do not recreate the staging tree."""
    ensure_layout(config)
    assert config.incoming_dir.is_dir()
    assert not (config.data_root / "staging").exists()


def test_datalad_download_uses_mirror_proxy_and_fetches_content(tmp_path, monkeypatch):
    """Use DataLad clone/get with the requested branch, mirror, and proxy."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Mock()

    monkeypatch.setattr(
        provider_base.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"datalad", "git-annex"} else None,
    )
    monkeypatch.setattr(provider_base.subprocess, "run", fake_run)
    destination = tmp_path / "incoming" / "dataset"
    provider_base.download_from_url(
        "https://openneuro.org/datasets/ds004212",
        "3.0.0",
        destination,
        proxy="https://proxy.example:8080",
        mirror="https://mirror.example/{dataset_id}.git",
    )

    assert calls[0][0] == [
        "/usr/bin/datalad",
        "clone",
        "https://mirror.example/ds004212.git",
        str(destination),
        "--branch",
        "3.0.0",
    ]
    assert calls[1][0] == [
        "/usr/bin/datalad",
        "get",
        "--recursive",
        ".",
    ]
    assert calls[1][1]["cwd"] == destination
    for _, kwargs in calls:
        assert kwargs["env"]["HTTPS_PROXY"] == "https://proxy.example:8080"
        assert kwargs["env"]["https_proxy"] == "https://proxy.example:8080"


def test_failed_download_remains_in_incoming(config, monkeypatch):
    """Retain a partial download in incoming instead of moving it to quarantine."""
    attempts = []

    def fail_download(url, version, destination, **kwargs):
        attempts.append(destination)
        (destination / "partial-file").write_text("partial")
        raise RuntimeError("download failed")

    monkeypatch.setattr("neural_data_registry.service.download_from_url", fail_download)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="download failed"):
            download_dataset(
                "https://openneuro.org/datasets/ds004212",
                "3.0.0",
                config,
                name="THINGS-MEG",
                modalities=["meg"],
            )

    partial = config.incoming_dir / "download-openneuro-ds004212-3.0.0"
    log_path = config.logs_dir / f"{partial.name}.log"
    assert attempts == [partial, partial]
    assert (partial / "partial-file").is_file()
    assert "FAILED RuntimeError: download failed" in log_path.read_text()
    assert not any(config.quarantine_dir.iterdir())


@pytest.mark.parametrize(
    ("url", "provider"),
    [
        ("https://physionet.org/content/example/1.0.0/", Provider.PHYSIONET),
        ("https://neurovault.org/collections/1234/", Provider.NEUROVAULT),
        ("https://www.kaggle.com/datasets/example/dataset", Provider.KAGGLE),
        ("https://www.synapse.org/Synapse:syn51549340", Provider.SYNAPSE),
    ],
)
def test_new_providers_are_recognized_but_not_downloaded(url, provider):
    """Recognize new provider URLs while keeping automatic downloads disabled."""
    assert provider_base.provider_for_url(url) is provider
    with pytest.raises(provider_base.ProviderDownloadError, match="not configured"):
        provider_base.download_from_url(url, "1.0.0", Path("/tmp/incoming"))


def test_download_requires_explicit_metadata(config):
    """Reject downloads that would otherwise create blank registry metadata."""
    with pytest.raises(ValueError, match="dataset name"):
        download_dataset(
            "https://openneuro.org/datasets/ds004212",
            "1.0.0",
            config,
            name=" ",
            modalities=["meg"],
        )
    with pytest.raises(ValueError, match="At least one modality"):
        download_dataset(
            "https://openneuro.org/datasets/ds004212",
            "1.0.0",
            config,
            name="THINGS-MEG",
            modalities=[],
        )


def test_ingest_local_detects_provider_and_physionet_version_from_url(config, tmp_path):
    """A local registration takes provider and version metadata from its URL."""
    item = ingest_local(
        mock_dataset(tmp_path, "physionet-local"),
        "EEG Motor Movement",
        Provider.OPENNEURO,
        "https://www.physionet.org/content/eegmmidb/1.0.0/",
        None,
        ["eeg"],
        config,
    )
    assert item.provider is Provider.PHYSIONET
    assert item.version == "1.0.0"


def test_ingest_local_uses_other_for_unknown_url(config, tmp_path):
    item = ingest_local(
        mock_dataset(tmp_path, "unknown-url"),
        "Unknown URL",
        Provider.OPENNEURO,
        "https://example.invalid/dataset",
        None,
        ["eeg"],
        config,
    )
    assert item.provider is Provider.OTHER


def test_physionet_missing_url_version_warns():
    with pytest.warns(UserWarning, match="Could not infer a version"):
        assert infer_url_version("https://physionet.org/content/eegmmidb/") is None


def test_download_rejects_unknown_provider_before_workspace(config):
    with pytest.raises(ValueError, match="Cannot identify a supported provider"):
        download_dataset(
            "https://example.invalid/dataset", None, config,
            name="Unknown URL", modalities=["eeg"],
        )
    assert not config.incoming_dir.exists()


def test_download_version_is_inferred_or_required():
    """Infer OpenNeuro numeric versions and require versions elsewhere."""
    assert resolve_download_version(
        "https://openneuro.org/datasets/ds007338/versions/1.0.0"
    ) == "1.0.0"
    assert resolve_download_version(
        "https://openneuro.org/datasets/ds007338/versions/1.0.0",
        "main",
    ) == "main"
    assert resolve_download_version(
        "https://www.physionet.org/content/eegmmidb/1.0.0/"
    ) == "1.0.0"
    with pytest.raises(ValueError, match="version is required"):
        resolve_download_version("https://dandiarchive.org/dandiset/000001/1.0.0")

def test_datalad_resume_skips_clone(tmp_path, monkeypatch):
    """Resume an existing DataLad workspace with get instead of cloning again."""
    destination = tmp_path / "incoming" / "dataset"
    (destination / ".git").mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Mock()

    monkeypatch.setattr(
        provider_base.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"datalad", "git-annex"} else None,
    )
    monkeypatch.setattr(provider_base.subprocess, "run", fake_run)
    provider_base.download_from_url(
        "https://openneuro.org/datasets/ds004212", "3.0.0", destination
    )
    assert [command for command, _ in calls] == [
        ["/usr/bin/datalad", "get", "--recursive", "."]
    ]
    assert calls[0][1]["cwd"] == destination


def test_datalad_requires_git_annex(tmp_path, monkeypatch):
    """Report the missing system git-annex dependency before cloning."""
    monkeypatch.setattr(
        provider_base,
        "_find_command",
        lambda name: "/usr/bin/datalad" if name == "datalad" else None,
    )
    with pytest.raises(provider_base.ProviderDownloadError, match="git-annex"):
        provider_base.download_from_url(
            "https://openneuro.org/datasets/ds004212",
            "3.0.0",
            tmp_path / "incoming" / "dataset",
        )


def register_dataset_path(config: Settings, path: Path, *, name: str = "Health fixture") -> Dataset:
    """Register a row at an arbitrary path without mutating test data."""
    create_database(config)
    with session(config) as db:
        item = Dataset(
            name=name,
            provider=Provider.LOCAL,
            version="1",
            modalities="meg",
            size_bytes=0,
            status=DatasetStatus.AVAILABLE,
            storage_path=str(path),
            storage_mode=StorageMode.REFERENCE,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item


def test_missing_health_check_logs_history_filters_and_recovers(config, tmp_path):
    """Mark an absent path missing, warn through GET, and recover automatically."""
    path = tmp_path / "later-restored"
    item = register_dataset_path(config, path)

    report = request_health_check(item.id, config)

    assert report.status is DatasetStatus.MISSING
    assert "does not exist" in report.warning
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        assert stored.status is DatasetStatus.MISSING
        assert find_datasets(db) == []
        assert [entry.id for entry in find_datasets(db, show_all=True)] == [item.id]
        histories = db.query(HealthCheckHistory).filter_by(dataset_id=item.id).all()
        assert histories[-1].result == "missing"
        assert histories[-1].resulting_status is DatasetStatus.MISSING

    log_path = config.logs_dir / "critical_errors.log"
    assert item.id in log_path.read_text()
    assert '"result": "missing"' in log_path.read_text()
    assert '"status": "missing"' in (
        config.registry_dir / f"{item.id}.json"
    ).read_text()

    client = TestClient(create_app(config))
    assert client.get("/datasets").json() == []
    shown = client.get("/datasets", params={"show_all": True}).json()
    assert shown[0]["status"] == "missing"
    response = client.get(f"/datasets/{item.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "missing"
    assert "health_warning" in response.json()

    path.mkdir()
    (path / "data.bin").write_bytes(b"restored")
    recovered = request_health_check(item.id, config)
    assert recovered.status is DatasetStatus.AVAILABLE
    with session(config) as db:
        assert db.get(Dataset, item.id).status is DatasetStatus.AVAILABLE
        assert (
            db.query(HealthCheckHistory).filter_by(dataset_id=item.id).count()
            == 3
        )


def test_hidden_repository_metadata_does_not_count_as_payload(config, tmp_path):
    """A directory containing only hidden metadata is still missing."""
    path = tmp_path / "metadata-only"
    (path / ".git" / "annex").mkdir(parents=True)
    item = register_dataset_path(config, path, name="Metadata only")

    report = request_health_check(item.id, config)

    assert report.status is DatasetStatus.MISSING
    assert "no non-hidden payload" in report.warning


def test_datalad_check_repairs_missing_annex_content(config, tmp_path, monkeypatch):
    """Verify, retrieve, and recheck a damaged DataLad checkout."""
    source = mock_dataset(tmp_path, "datalad-repair")
    (source / ".git" / "annex").mkdir(parents=True)
    item = ingest_local(
        source,
        "Repairable DataLad",
        Provider.OPENNEURO,
        "https://openneuro.org/datasets/ds000001",
        "1",
        ["meg"],
        config,
    )
    repaired = False
    calls = []

    def fake_run(command, **kwargs):
        nonlocal repaired
        calls.append(command)
        if command[:2] == ["/usr/bin/datalad", "get"]:
            repaired = True
        missing = (
            command[:2] == ["/usr/bin/git-annex", "find"] and not repaired
        )
        return subprocess.CompletedProcess(
            command, 0, stdout="meg.fif\n" if missing else "", stderr=""
        )

    monkeypatch.setattr(
        provider_base, "_find_command", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(health_service.subprocess, "run", fake_run)

    assert run_health_checks([item.id], config) is True

    assert ["/usr/bin/datalad", "get", "--recursive", "."] in calls
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        history = (
            db.query(HealthCheckHistory)
            .filter_by(dataset_id=item.id)
            .order_by(HealthCheckHistory.started_at.desc())
            .first()
        )
        assert stored.status is DatasetStatus.AVAILABLE
        assert history.result == "healthy"
        assert history.repair_attempted is True
        assert history.repair_succeeded is True


def test_datalad_network_failure_does_not_hide_known_local_damage(
    config, tmp_path, monkeypatch
):
    """Keep locally missing content broken when its attempted retrieval fails."""
    source = mock_dataset(tmp_path, "datalad-network-failure")
    (source / ".git" / "annex").mkdir(parents=True)
    item = ingest_local(
        source,
        "Broken DataLad",
        Provider.OPENNEURO,
        "https://openneuro.org/datasets/ds000002",
        "1",
        ["meg"],
        config,
    )

    def fake_run(command, **kwargs):
        if command[:2] == ["/usr/bin/datalad", "get"]:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="temporary network failure"
            )
        if command[:2] == ["/usr/bin/git-annex", "find"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="meg.fif\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        provider_base, "_find_command", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(health_service.subprocess, "run", fake_run)

    run_health_checks([item.id], config)

    with session(config) as db:
        stored = db.get(Dataset, item.id)
        history = db.query(HealthCheckHistory).filter_by(dataset_id=item.id).one()
        assert stored.status is DatasetStatus.BROKEN
        assert history.result == "broken"
        assert history.repair_attempted is True
        assert history.repair_succeeded is False
        assert "network failure" in history.message
    assert "network failure" in (
        config.logs_dir / "critical_errors.log"
    ).read_text()


def test_query_defers_datalad_repair_to_background(config, tmp_path, monkeypatch):
    """Return promptly and expose repair state without running DataLad inline."""
    source = mock_dataset(tmp_path, "datalad-background")
    (source / ".git" / "annex").mkdir(parents=True)
    item = ingest_local(
        source,
        "Background DataLad",
        Provider.OPENNEURO,
        "https://openneuro.org/datasets/ds000003",
        "1",
        ["meg"],
        config,
    )
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        stored.status = DatasetStatus.BROKEN
        db.commit()

    launches = []
    monkeypatch.setattr(
        health_service,
        "launch_health_worker",
        lambda config, **kwargs: launches.append(kwargs) or True,
    )
    monkeypatch.setattr(
        health_service.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("DataLad ran in the query process"),
    )

    report = request_health_check(item.id, config)

    assert report.status is DatasetStatus.BROKEN
    assert report.repair_in_progress is True
    assert "try again later" in report.warning.lower()
    assert launches[0]["dataset_id"] == item.id
    with session(config) as db:
        history = db.get(HealthCheckHistory, report.history_id)
        assert history.result == "running"


def test_global_health_worker_lock_skips_overlapping_check(config, tmp_path):
    """Only one process may run deep checks at a time."""
    path = mock_dataset(tmp_path, "lock-source")
    item = register_dataset_path(config, path, name="Lock fixture")

    with process_lock("registry-health-worker", config) as acquired:
        assert acquired is True
        assert run_health_checks([item.id], config) is False


def test_first_invocation_health_scan_obeys_environment_cooldown(
    config, tmp_path, monkeypatch
):
    """Launch one all-dataset worker per virtual environment per cooldown."""
    launches = []
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    monkeypatch.setattr(
        health_service,
        "launch_health_worker",
        lambda config, **kwargs: launches.append(kwargs) or True,
    )

    assert maybe_launch_cooldown_check(config) is True
    assert maybe_launch_cooldown_check(config) is False
    assert launches == [{"all_datasets": True}]


def test_cli_list_show_all_and_one_shot_health_command(
    config, tmp_path, monkeypatch
):
    """Hide unhealthy rows by default and expose explicit health commands."""
    item = register_dataset_path(config, tmp_path / "absent", name="Hidden missing")
    request_health_check(item.id, config)
    monkeypatch.setattr(cli, "session", lambda: session(config))
    monkeypatch.setattr(cli, "maybe_launch_cooldown_check", lambda: False)
    monkeypatch.setattr(cli, "console", cli.Console(width=160))
    checked = []
    monkeypatch.setattr(
        cli, "run_health_checks", lambda ids=None: checked.append(ids) or True
    )
    runner = CliRunner()

    hidden = runner.invoke(cli.app, ["list"], terminal_width=160)
    shown = runner.invoke(cli.app, ["list", "--show-all"], terminal_width=160)
    one_shot = runner.invoke(cli.app, ["health-check", item.id])

    assert "Hidden missing" not in hidden.output
    assert "Hidden missing" in shown.output
    assert "missing" in shown.output
    assert one_shot.exit_code == 0
    assert "Health check found problems" in one_shot.output
    assert "Hidden missing" in one_shot.output
    assert "critical_errors.log" in one_shot.output
    assert checked == [[item.id]]
    assert cli._interval_seconds("24h") == 86400


def test_annex_command_error_preserves_available_status(
    config, tmp_path, monkeypatch
):
    """Do not classify command or repository failures as broken datasets."""
    source = mock_dataset(tmp_path, "datalad-command-error")
    (source / ".git" / "annex").mkdir(parents=True)
    item = ingest_local(
        source,
        "DataLad command error",
        Provider.OPENNEURO,
        "https://openneuro.org/datasets/ds000004",
        "1",
        ["meg"],
        config,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 128, stdout="", stderr="fatal: damaged git metadata"
        )

    monkeypatch.setattr(
        provider_base, "_find_command", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(health_service.subprocess, "run", fake_run)

    run_health_checks([item.id], config)

    assert calls == [["/usr/bin/git-annex", "find", "--not", "--in=here"]]
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        history = db.query(HealthCheckHistory).filter_by(dataset_id=item.id).one()
        assert stored.status is DatasetStatus.AVAILABLE
        assert history.result == "error"

        assert history.resulting_status is DatasetStatus.AVAILABLE

def test_directory_size_deduplicates_symlink_targets_and_ignores_broken_links(tmp_path):
    """Count readable logical payload once even when it has multiple paths."""
    source = tmp_path / "linked-payload"
    source.mkdir()
    annex = source / ".git" / "annex" / "objects"
    annex.mkdir(parents=True)
    payload = annex / "payload.bin"
    payload.write_bytes(b"annex payload")
    (source / "visible.bin").symlink_to(payload)
    (source / "second.bin").write_bytes(b"other payload")
    (source / "broken.bin").symlink_to(source / "missing.bin")

    assert directory_size(source) == len(b"annex payload") + len(b"other payload")


def test_ingestion_persists_inode_deduplicated_logical_size(config, tmp_path):
    """Ingestion stores each git-annex payload once, not once per symlink."""
    source = mock_dataset(tmp_path, "deduplicated-ingest")
    annex = source / ".git" / "annex" / "objects"
    annex.mkdir(parents=True)
    payload = annex / "payload.bin"
    payload.write_bytes(b"annex payload")
    (source / "visible.bin").symlink_to(payload)

    item = ingest_local(
        source, "Deduplicated ingest", Provider.OTHER, None, "1", ["meg"], config
    )

    assert item.size_bytes == directory_size(source)


def test_worker_health_check_refreshes_stale_size_and_manifest(config, tmp_path):
    """Deep health checks backfill logical sizes for registered datasets."""
    source = mock_dataset(tmp_path, "size-backfill")
    item = ingest_local(source, "Size backfill", Provider.OTHER, None, "1", ["meg"], config)
    expected_size = directory_size(source)
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        stored.size_bytes = 1
        db.commit()

    assert run_health_checks([item.id], config) is True
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        assert stored.size_bytes == expected_size
    manifest = json.loads((config.registry_dir / f"{item.id}.json").read_text())
    assert manifest["size_bytes"] == expected_size


def test_query_time_health_check_does_not_recalculate_size(config, tmp_path, monkeypatch):
    """Fast query checks retain their status-only behavior."""
    source = mock_dataset(tmp_path, "query-size")
    item = ingest_local(source, "Query size", Provider.OTHER, None, "1", ["meg"], config)

    def fail_size_scan(path):
        raise AssertionError(f"unexpected size scan for {path}")

    monkeypatch.setattr(health_service, "directory_size", fail_size_scan)
    report = request_health_check(item.id, config)
    assert report.status is DatasetStatus.AVAILABLE


def test_health_size_failure_preserves_existing_size_and_availability(config, tmp_path, monkeypatch):
    """A failed worker scan reports an error without mutating dataset availability."""
    source = mock_dataset(tmp_path, "size-failure")
    item = ingest_local(source, "Size failure", Provider.OTHER, None, "1", ["meg"], config)
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        stored.size_bytes = 123
        db.commit()

    def fail_size_scan(path):
        raise OSError("read failure")

    monkeypatch.setattr(health_service, "directory_size", fail_size_scan)
    assert run_health_checks([item.id], config) is True
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        history = db.query(HealthCheckHistory).filter_by(dataset_id=item.id).one()
        assert stored.size_bytes == 123
        assert stored.status is DatasetStatus.AVAILABLE
        assert history.result == "error"
    manifest = json.loads((config.registry_dir / f"{item.id}.json").read_text())
    assert manifest["size_bytes"] == 123


def test_successful_datalad_health_check_refreshes_size(config, tmp_path, monkeypatch):
    """Verified DataLad datasets receive the same worker-only size refresh."""
    source = mock_dataset(tmp_path, "datalad-size")
    (source / ".git" / "annex").mkdir(parents=True)
    item = ingest_local(source, "DataLad size", Provider.OPENNEURO, "https://openneuro.org/datasets/ds000005", "1", ["meg"], config)
    expected_size = directory_size(source)
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        stored.size_bytes = 1
        db.commit()

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(provider_base, "_find_command", lambda name: "/usr/bin/git-annex")
    monkeypatch.setattr(health_service.subprocess, "run", fake_run)
    assert run_health_checks([item.id], config) is True
    with session(config) as db:
        stored = db.get(Dataset, item.id)
        history = db.query(HealthCheckHistory).filter_by(dataset_id=item.id).one()
        assert stored.size_bytes == expected_size
        assert history.result == "healthy"

def test_update_metadata_enriches_missing_values_and_refreshes_manifest(config, tmp_path):
    """Metadata updates fill local-record gaps while retaining the path identity."""
    item = ingest_local(
        mock_dataset(tmp_path, "update-missing"),
        "Update missing",
        Provider.OTHER,
        None,
        None,
        [],
        config,
    )
    original_path = item.source_url

    updated = update_dataset_metadata(
        item.id,
        url="https://openneuro.org/datasets/ds009999",
        version="1.0.0",
        modalities=["meg", "eeg", "meg"],
        aliases=["Update alias", "Update alias"],
        config=config,
    )

    data = dataset_dict(updated)
    assert data["source_url"] == "https://openneuro.org/datasets/ds009999"
    assert data["provider"] == "openneuro"
    assert data["version"] == "1.0.0"
    assert data["modalities"] == ["eeg", "meg"]
    assert data["aliases"] == ["Update alias"]
    assert json.loads((config.registry_dir / f"{item.id}.json").read_text()) == data
    with session(config) as db:
        assert resolve_dataset(db, original_path).id == item.id
        assert resolve_dataset(db, data["source_url"]).id == item.id


def test_update_metadata_requires_force_and_never_replaces_url(config, tmp_path):
    """Existing scalar metadata is guarded and remote URLs remain immutable."""
    item = ingest_mock(config, tmp_path)
    with pytest.raises(ValueError, match="force-replace"):
        update_dataset_metadata(item.id, version="4.0.0", config=config)
    updated = update_dataset_metadata(
        item.id, version="4.0.0", provider=Provider.DANDI,
        force_replace=True, config=config,
    )
    assert updated.version == "4.0.0"
    assert updated.provider is Provider.DANDI
    with pytest.raises(ValueError, match="cannot be replaced"):
        update_dataset_metadata(
            item.id, url="https://openneuro.org/datasets/ds000001", config=config
        )


def test_update_metadata_cli_and_api_statuses(config, tmp_path, monkeypatch):
    """CLI resolves identifiers and PATCH reports the documented error codes."""
    item = ingest_local(
        mock_dataset(tmp_path, "update-cli"), "Update CLI", Provider.OTHER,
        None, None, [], config,
    )
    other = ingest_local(
        mock_dataset(tmp_path, "update-conflict"), "Update conflict", Provider.OTHER,
        None, None, [], config,
    )
    monkeypatch.setattr(cli, "session", lambda: session(config))
    monkeypatch.setattr(
        cli, "update_dataset_metadata",
        lambda identifier, **kwargs: update_dataset_metadata(identifier, config=config, **kwargs),
    )
    monkeypatch.setattr(cli, "maybe_launch_cooldown_check", lambda: False)
    result = CliRunner().invoke(
        cli.app,
        ["update", "Update CLI", "--version", "1", "--modality", "meg", "--alias", "CLI alias"],
    )
    assert result.exit_code == 0
    assert '"version": "1"' in result.output
    assert '"aliases": [' in result.output

    client = TestClient(create_app(config))
    assert client.patch("/datasets/no-such-id", json={"version": "1"}).status_code == 404
    assert client.patch(f"/datasets/{item.id}", json={"version": "2"}).status_code == 400
    assert client.patch(f"/datasets/{item.id}", json={"url": other.source_url}).status_code == 409
    response = client.patch(
        f"/datasets/{item.id}", json={"modalities": ["eeg", "meg"], "aliases": ["API alias"]}
    )
    assert response.status_code == 200
    assert response.json()["modalities"] == ["eeg", "meg"]
    assert response.json()["aliases"] == ["API alias", "CLI alias"]
