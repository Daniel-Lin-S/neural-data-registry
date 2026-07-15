from __future__ import annotations
from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
from rich.table import Table
from neural_data_registry.enums import Provider, StorageMode
from neural_data_registry.service import dataset_dict, download as download_dataset, find_datasets, ingest_local, session

app = typer.Typer(help="Search, list, ingest, and download managed neuroscience datasets.", add_completion=False)
console = Console()
def display(items):
    table = Table("Dataset ID", "Name", "Provider", "Version", "Modalities", "Size", "Status")
    for item in items:
        row = dataset_dict(item); table.add_row(row["dataset_id"], row["name"], row["provider"], row["version"], ", ".join(row["modalities"]) or "-", str(row["size_bytes"]), row["status"])
    console.print(table)

@app.command()
def query(query: Annotated[str | None, typer.Argument(help="Dataset name or alias fragment to search for.")] = None, url: str | None = typer.Option(None, "--url", help="Exact source URL to look up.")):
    """Search datasets by name or exact source URL.

    Provide a positional name fragment, or use --url. Results show the dataset
    ID, provider, version, modalities, size, and registry status.
    """
    if not query and not url: raise typer.BadParameter("Provide a name query or --url")
    with session() as db: display(find_datasets(db, query=query, url=url))
@app.command("list")
def list_datasets(modality: str | None = typer.Option(None, "--modality", help="Restrict results to a modality, such as MEG or EEG.")):
    """List registered datasets, optionally filtered by modality.

    The table contains dataset ID, name, provider, version, modalities, size,
    and current status.
    """
    with session() as db: display(find_datasets(db, modality=modality))
@app.command("ingest-local")
def ingest_local_command(source: Path = typer.Argument(..., help="Existing local dataset directory to register."), name: str = typer.Option(..., "--name", help="Canonical dataset name to register."), provider: Provider = typer.Option(Provider.LOCAL, "--provider", help="Dataset provider: openneuro, dandi, nemar, or local."), url: str | None = typer.Option(None, "--url", help="Optional canonical source URL for the dataset."), version: str | None = typer.Option(None, "--version", help="Optional dataset version; defaults to unknown for unversioned local data."), modality: list[str] = typer.Option([], "--modality", help="Dataset modality (eeg, meg, ieeg, fmri, fnris, pet, smri, dmri, ephys, or other); repeat for multiple modalities."), storage_mode: StorageMode = typer.Option(StorageMode.REFERENCE, "--storage-mode", help="Reference files in place (default) or move into managed storage.")):
    """Register an already-downloaded local dataset.

    SOURCE must be a directory. By default, its files remain in place and
    the registry stores a reference path. Use --storage-mode move to relocate
    files under NDR_DATA_ROOT/datasets. Duplicate names or source URLs are
    rejected with the existing storage path. The created record is printed as JSON.
    """
    console.print_json(data=dataset_dict(ingest_local(source, name, provider, url, version, modality, storage_mode=storage_mode)))
@app.command()
def download(url: str = typer.Option(..., "--url", help="Provider dataset URL; the provider is detected from its host."), version: str = typer.Option("latest", "--version", help="Version to download, or 'latest' (the default).")):
    """Download a supported provider dataset and ingest it automatically.

    The URL identifies the provider. OpenNeuro downloads require git. Failed
    downloads are moved to NDR_DATA_ROOT/quarantine; successful records are
    printed as JSON.
    """
    console.print_json(data=dataset_dict(download_dataset(url, version)))
