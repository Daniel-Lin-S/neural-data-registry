from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from neural_data_registry.config import Settings
from neural_data_registry.health import request_health_check
from neural_data_registry.db.models import Dataset
from neural_data_registry.enums import Modality, Provider, StorageMode
from neural_data_registry.service import DatasetConflictError, dataset_dict, download, find_datasets, ingest_local, session


class LocalIngestionRequest(BaseModel):
    """Request body for registering a local dataset."""
    source: Path
    name: str = Field(min_length=1)
    provider: Provider = Provider.OTHER
    url: str | None = None
    version: str | None = None
    modalities: list[Modality] = Field(default_factory=list)
    storage_mode: StorageMode = StorageMode.REFERENCE


class DownloadRequest(BaseModel):
    """Request body for downloading and registering a provider dataset."""
    url: str
    version: str | None = None
    name: str = Field(min_length=1)
    modalities: list[Modality] = Field(min_length=1)
    proxy: str | None = None
    mirror: str | None = None


def create_app(config: Settings | None = None) -> FastAPI:
    """Create the registry API application.

    Parameters
    ----------
    config : Settings or None, optional
        Configuration for the application and database.

    Returns
    -------
    fastapi.FastAPI
        Configured API application.
    """
    api = FastAPI(title="Neural Data Registry", version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/datasets")
    def datasets(query: str | None = None, url: str | None = None, modality: str | None = None, provider: str | None = None, show_all: bool = False) -> list[dict]:
        with session(config) as db:
            return [dataset_dict(item) for item in find_datasets(db, query, url, modality, provider, show_all=show_all)]

    @api.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str) -> dict:
        with session(config) as db:
            item = db.get(Dataset, dataset_id)
            if not item:
                raise HTTPException(status_code=404, detail="Dataset not found")
        report = request_health_check(dataset_id, config)
        with session(config) as db:
            item = db.get(Dataset, dataset_id)
            data = dataset_dict(item)
        if report.warning:
            data["health_warning"] = report.warning
        if report.repair_in_progress:
            data["repair_in_progress"] = True
        return data

    @api.post("/ingest/local", status_code=201)
    def ingest_local_dataset(request: LocalIngestionRequest, response: Response) -> dict:
        if request.storage_mode is StorageMode.COPY:
            response.headers["Warning"] = "299 - \"copy mode uses additional disk space; use only when SOURCE may be cleaned in the future\""
        try:
            item = ingest_local(
                request.source, request.name, request.provider, request.url,
                request.version, request.modalities, config, request.storage_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return dataset_dict(item)
    @api.post("/download", status_code=202)
    def download_dataset(request: DownloadRequest) -> dict:
        try:
            return dataset_dict(download(request.url, request.version, config, name=request.name, modalities=request.modalities, proxy=request.proxy, mirror=request.mirror))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DatasetConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return api


app = create_app()
