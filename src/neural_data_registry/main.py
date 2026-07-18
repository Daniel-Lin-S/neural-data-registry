from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from neural_data_registry.config import Settings
from neural_data_registry.health import request_health_check
from neural_data_registry.db.models import Dataset
from neural_data_registry.enums import Modality, Provider, StorageMode
from neural_data_registry.service import DatasetConflictError, dataset_dict, download, find_datasets, ingest_local, session
from neural_data_registry.service import (
    DatasetNotFoundError,
    transition_reference_storage,
    update_dataset_metadata,
)


class LocalIngestionRequest(BaseModel):
    """Request body for registering a local dataset."""
    source: Path
    name: str = Field(min_length=1)
    provider: Provider = Provider.OTHER
    url: str | None = None
    version: str | None = None
    modalities: list[Modality] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    storage_mode: StorageMode = StorageMode.REFERENCE


class DownloadRequest(BaseModel):
    """Request body for downloading and registering a provider dataset."""
    url: str
    version: str | None = None
    name: str = Field(min_length=1)
    modalities: list[Modality] = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    proxy: str | None = None
    mirror: str | None = None


class StorageTransitionRequest(BaseModel):
    """Request a supported transition from reference to managed storage."""
    storage_mode: Literal[StorageMode.MOVE, StorageMode.COPY]


class DatasetUpdateRequest(BaseModel):
    """Metadata fields that may enrich an existing dataset."""
    url: str | None = None
    provider: Provider | None = None
    version: str | None = None
    modalities: list[Modality] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    force_replace: bool = False


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

    @api.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        """Use the API's standard bad-request status for invalid inputs."""
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

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

    @api.patch("/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, request: DatasetUpdateRequest) -> dict:
        try:
            return dataset_dict(
                update_dataset_metadata(
                    dataset_id,
                    url=request.url,
                    provider=request.provider,
                    version=request.version,
                    modalities=request.modalities,
                    aliases=request.aliases,
                    force_replace=request.force_replace,
                    config=config,
                )
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DatasetConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/datasets/{dataset_id}/storage-transition")
    def storage_transition(
        dataset_id: str, request: StorageTransitionRequest
    ) -> dict:
        try:
            return dataset_dict(
                transition_reference_storage(dataset_id, request.storage_mode, config)
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @api.post("/ingest/local", status_code=201)
    def ingest_local_dataset(request: LocalIngestionRequest, response: Response) -> dict:
        if request.storage_mode is StorageMode.COPY:
            response.headers["Warning"] = "299 - \"copy mode uses additional disk space; use only when SOURCE may be cleaned in the future\""
        try:
            item = ingest_local(
                request.source, request.name, request.provider, request.url,
                request.version, request.modalities, config, request.storage_mode,
                name_aliases=request.aliases,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return dataset_dict(item)
    @api.post("/download", status_code=202)
    def download_dataset(request: DownloadRequest) -> dict:
        try:
            return dataset_dict(download(request.url, request.version, config, name=request.name, modalities=request.modalities, proxy=request.proxy, mirror=request.mirror, name_aliases=request.aliases))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DatasetConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return api


app = create_app()
