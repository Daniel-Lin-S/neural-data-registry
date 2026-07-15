from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neural_data_registry.config import Settings
from neural_data_registry.db.models import Dataset
from neural_data_registry.enums import Provider, StorageMode
from neural_data_registry.service import dataset_dict, download, find_datasets, ingest_local, session


class LocalIngestionRequest(BaseModel):
    source: Path
    name: str = Field(min_length=1)
    provider: Provider = Provider.LOCAL
    url: str | None = None
    version: str | None = None
    modalities: list[str] = Field(default_factory=list)
    storage_mode: StorageMode = StorageMode.REFERENCE


class DownloadRequest(BaseModel):
    url: str
    version: str = "latest"


def create_app(config: Settings | None = None) -> FastAPI:
    """Create the API application, optionally with an isolated configuration."""
    api = FastAPI(title="Neural Data Registry", version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/datasets")
    def datasets(query: str | None = None, url: str | None = None, modality: str | None = None) -> list[dict]:
        with session(config) as db:
            return [dataset_dict(item) for item in find_datasets(db, query, url, modality)]

    @api.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str) -> dict:
        with session(config) as db:
            item = db.get(Dataset, dataset_id)
            if not item:
                raise HTTPException(status_code=404, detail="Dataset not found")
            return dataset_dict(item)

    @api.post("/ingest/local", status_code=201)
    def ingest_local_dataset(request: LocalIngestionRequest) -> dict:
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
            return dataset_dict(download(request.url, request.version, config))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return api


app = create_app()
