from __future__ import annotations
from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
from rich.table import Table
from neural_data_registry.enums import Modality, Provider, StorageMode
from neural_data_registry.service import (
    dataset_dict,
    dataset_output_fields,
    download as download_dataset,
    find_datasets,
    ingest_local,
    resolve_dataset,
    session,
)

app = typer.Typer(help="Search, list, ingest, and download managed neuroscience datasets.", add_completion=False)
console = Console()

def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} PB"

def _format_dataset_cell(field: str, value) -> str:
    if field == "size_bytes":
        return format_size(value or 0)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "-"
    if value is None or value == "":
        return "-"
    return str(value)


def display(items):
    table = Table(title="Datasets", show_lines=True, padding=(0, 0))
    fields = dataset_output_fields(cli=True)
    for field, label in fields:
        table.add_column(
            label,
            overflow="fold",
            justify="right" if field == "size_bytes" else "left",
            no_wrap=field == "size_bytes",
        )

    for item in items:
        row = dataset_dict(item)
        table.add_row(
            *[_format_dataset_cell(field, row.get(field)) for field, _ in fields]
        )
    console.print(table)

@app.command()
def query(
    query: Annotated[str | None, typer.Argument(help="Exact dataset ID, name, or source URL.")] = None,
    name: str | None = typer.Option(None, "--name", help="Exact dataset name to look up."),
    url: str | None = typer.Option(None, "--url", help="Exact source URL to look up.")
):
    """Print the absolute storage path for one dataset."""
    try:
        with session() as db:
            item = resolve_dataset(db, query, name=name, url=url)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if item is None:
        raise typer.BadParameter("No dataset matched the supplied ID, name, or URL")
    if not item.storage_path:
        raise typer.BadParameter("The matched dataset has no storage path")
    console.print(Path(item.storage_path).expanduser().resolve())
@app.command("list")
def list_datasets(modality: Modality | None = typer.Option(None, "--modality", help="Restrict results to a modality, such as meg or eeg."), provider: Provider | None = typer.Option(None, "--provider", help="Restrict results to a provider.")):
    """List registered datasets, optionally filtered by modality.

    The table contains dataset ID, name, provider, version, modalities, size,
    and current status.
    """
    with session() as db: display(find_datasets(db, modality=modality.value if modality else None, provider=provider.value if provider else None))
@app.command("ingest-local")
def ingest_local_command(source: Path = typer.Argument(..., help="Existing local dataset directory to register."), name: str = typer.Option(..., "--name", help="Canonical dataset name to register."), provider: Provider = typer.Option(Provider.LOCAL, "--provider", help="Dataset provider: openneuro, dandi, nemar, physionet, neurovault, kaggle, or local."), url: str | None = typer.Option(None, "--url", help="Optional canonical source URL for the dataset."), version: str | None = typer.Option(None, "--version", help="Optional dataset version; defaults to unknown for unversioned local data."), modality: list[Modality] = typer.Option([], "--modality", help="Dataset modality (eeg, meg, ieeg, fmri, fnris, pet, smri, dmri, ephys, or other); repeat for multiple modalities."), storage_mode: StorageMode = typer.Option(StorageMode.REFERENCE, "--storage-mode", help="Reference files in place (default) or move into managed storage.")):
    """Register an already-downloaded local dataset.

    SOURCE must be a directory. By default, its files remain in place and
    the registry stores a reference path. Use --storage-mode move to relocate
    files under NDR_DATA_ROOT/datasets. Duplicate names or source URLs are
    rejected with the existing storage path. The created record is printed as JSON.
    """
    console.print_json(
        data=dataset_dict(
            ingest_local(
                source,
                name,
                provider,
                url,
                version,
                [item.value for item in modality],
                storage_mode=storage_mode,
            )
        )
    )
@app.command()
def download(url: str = typer.Option(..., "--url", help="Provider dataset URL; the provider is detected from its host."), name: str = typer.Option(..., "--name", help="Dataset name to register."), modality: list[Modality] = typer.Option(..., "--modality", help="Dataset modality; repeat for multiple modalities."), version: str | None = typer.Option(None, "--version", help="Version or branch; required unless an OpenNeuro URL contains /versions/x.y.z."), proxy: str | None = typer.Option(None, "--proxy", help="Proxy URL for this download."), mirror: str | None = typer.Option(None, "--mirror", help="Mirror URL, URL base, or template containing {dataset_id}.")):
    """Download a supported provider dataset and ingest it automatically.

    NAME and at least one --modality are required so the registry record has
    explicit metadata. Install the package with the download extra to enable
    DataLad-backed downloads. Failed downloads remain in NDR_DATA_ROOT/incoming.
    """
    try:
        item = download_dataset(url, version, name=name, modalities=[item.value for item in modality], proxy=proxy, mirror=mirror)
    except RuntimeError as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(data=dataset_dict(item))
