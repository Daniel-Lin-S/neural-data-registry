from __future__ import annotations
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Annotated
import typer
from rich.console import Console
from rich.table import Table
from neural_data_registry.config import get_settings
from neural_data_registry.db.models import Dataset, HealthCheckHistory
from neural_data_registry.health import (
    maybe_launch_cooldown_check,
    request_health_check,
    run_health_checks,
)
from neural_data_registry.enums import Modality, Provider, StorageMode
from neural_data_registry.service import (
    add_name_aliases,
    dataset_dict,
    dataset_output_fields,
    download as download_dataset,
    find_datasets,
    ingest_local,
    resolve_dataset,
    session,
    update_dataset_metadata,
)

app = typer.Typer(help="Search, list, ingest, and download managed neuroscience datasets.", add_completion=False)
console = Console()


@app.callback()
def startup_health_check(ctx: typer.Context) -> None:
    """Launch the cooldown-limited health scan on normal CLI invocations."""
    if ctx.invoked_subcommand in {"health-check", "health-scheduler"}:
        return
    try:
        maybe_launch_cooldown_check()
    except Exception:
        # A best-effort startup check must not block normal registry commands.
        return

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


def _health_problems(dataset_ids: list[str]):
    """Return the latest problem result for each dataset in a health run."""
    if not dataset_ids:
        return []
    with session() as db:
        datasets = {
            item.id: item
            for item in db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
        }
        histories = (
            db.query(HealthCheckHistory)
            .filter(HealthCheckHistory.dataset_id.in_(dataset_ids))
            .order_by(HealthCheckHistory.started_at.desc(), HealthCheckHistory.id.desc())
            .all()
        )
    latest = {}
    for history in histories:
        latest.setdefault(history.dataset_id, history)
    return [
        (datasets[dataset_id], history)
        for dataset_id, history in latest.items()
        if history.result in {"missing", "broken", "error"}
        and dataset_id in datasets
    ]

@app.command()
def query(
    query: Annotated[str | None, typer.Argument(help="Dataset ID, name, or source URL.")] = None,
    name: str | None = typer.Option(None, "--name", help="Exact dataset name to look up."),
    url: str | None = typer.Option(None, "--url", help="Source URL to look up by dataset path segment.")
):
    """Print a matched dataset storage path, prompting when a URL is ambiguous."""
    positional_url = (
        query
        if query and urlparse(query).scheme and urlparse(query).hostname
        else None
    )
    url_lookup = url or positional_url
    try:
        with session() as db:
            if url_lookup:
                matches = find_datasets(db, url=url_lookup)
            else:
                item = resolve_dataset(db, query, name=name)
                matches = [item] if item else []
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not matches:
        raise typer.BadParameter("No dataset matched the supplied ID, name, or URL")
    if len(matches) > 1:
        table = Table(title="Multiple matching datasets", show_lines=True, padding=(0, 0))
        for label in ("Name", "Modalities", "Source URL", "Size"):
            table.add_column(label, overflow="fold")
        for match in matches:
            data = dataset_dict(match)
            table.add_row(
                str(data["name"]),
                _format_dataset_cell("modalities", data.get("modalities")),
                _format_dataset_cell("source_url", data.get("source_url")),
                _format_dataset_cell("size_bytes", data.get("size_bytes")),
            )
        console.print("[yellow]Multiple matching datasets found. Select one by canonical name.[/yellow]")
        console.print(table)
        by_name = {item.name: item for item in matches}
        while True:
            selected_name = typer.prompt("Canonical name")
            item = by_name.get(selected_name)
            if item:
                break
            console.print("[red]Enter one of the displayed canonical names.[/red]")
    else:
        item = matches[0]
    report = request_health_check(item.id)
    if report.warning:
        typer.echo(f"Warning: {report.warning}", err=True)
    if not item.storage_path:
        raise typer.BadParameter("The matched dataset has no storage path")
    console.print(Path(item.storage_path).expanduser().resolve())


@app.command("list")
def list_datasets(
    query: str | None = typer.Option(None, "--query", "-q", help="Search canonical dataset names and user aliases."),
    modality: Modality | None = typer.Option(None, "--modality", help="Restrict results to a modality, such as meg or eeg."),
    provider: Provider | None = typer.Option(None, "--provider", help="Restrict results to a provider."),
    show_all: bool = typer.Option(False, "--show-all", help="Include missing and broken datasets."),
):
    """List registered datasets, optionally filtered by modality.

    The table contains dataset ID, name, provider, version, modalities, size,
    and current status.
    """
    with session() as db:
        display(
            find_datasets(
                db,
                query=query,
                modality=modality.value if modality else None,
                provider=provider.value if provider else None,
                show_all=show_all,
            )
        )
@app.command("ingest-local")
def ingest_local_command(source: Path = typer.Argument(..., help="Existing local dataset directory to register."), name: str = typer.Option(..., "--name", help="Canonical dataset name to register."), alias: list[str] = typer.Option([], "--alias", help="Searchable alternate dataset name; repeat for multiple aliases."), provider: Provider = typer.Option(Provider.OTHER, "--provider", help="Dataset provider: openneuro, dandi, nemar, physionet, neurovault, kaggle, or other."), url: str | None = typer.Option(None, "--url", help="Optional canonical source URL for the dataset."), version: str | None = typer.Option(None, "--version", help="Optional dataset version; defaults to unknown for unversioned local data."), modality: list[Modality] = typer.Option([], "--modality", help="Dataset modality (eeg, meg, ieeg, fmri, fnris, pet, smri, dmri, ephys, or other); repeat for multiple modalities."), storage_mode: StorageMode = typer.Option(StorageMode.REFERENCE, "--storage-mode", help="Reference files in place (default), move into managed storage, or copy into managed storage (leaves a duplicate; use only when SOURCE may be cleaned later).")):
    """Register an already-downloaded local dataset.

    SOURCE must be a directory. By default, its files remain in place and
    the registry stores a reference path. Use --storage-mode move to relocate
    files under NDR_DATA_ROOT/datasets. Duplicate names or source URLs are
    rejected with the existing storage path. The created record is printed as JSON.
    """
    if storage_mode is StorageMode.COPY:
        console.print("[yellow]Warning: copy mode uses additional disk space because SOURCE and the managed copy are both retained. Use it only when SOURCE may be cleaned in the future.[/yellow]")
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
                name_aliases=alias,
            )
        )
    )
@app.command()
def download(url: str = typer.Option(..., "--url", help="Provider dataset URL; the provider is detected from its host."), name: str = typer.Option(..., "--name", help="Dataset name to register."), alias: list[str] = typer.Option([], "--alias", help="Searchable alternate dataset name; repeat for multiple aliases."), modality: list[Modality] = typer.Option(..., "--modality", help="Dataset modality; repeat for multiple modalities."), version: str | None = typer.Option(None, "--version", help="Version or branch; required unless an OpenNeuro URL contains /versions/x.y.z."), proxy: str | None = typer.Option(None, "--proxy", help="Proxy URL for this download."), mirror: str | None = typer.Option(None, "--mirror", help="Mirror URL, URL base, or template containing {dataset_id}.")):
    """Download a supported provider dataset and ingest it automatically.

    NAME and at least one --modality are required so the registry record has
    explicit metadata. Install the package with the download extra to enable
    DataLad-backed downloads. Failed downloads remain in NDR_DATA_ROOT/incoming.
    """
    try:
        item = download_dataset(url, version, name=name, modalities=[item.value for item in modality], proxy=proxy, mirror=mirror, name_aliases=alias)
    except RuntimeError as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(data=dataset_dict(item))


@app.command("alias")
def add_alias_command(
    dataset: Annotated[str, typer.Argument(help="Dataset ID, canonical name, existing alias, or source URL.")],
    alias: list[str] = typer.Option(..., "--alias", help="Searchable alternate dataset name; repeat for multiple aliases."),
):
    """Add searchable name aliases to an existing dataset."""
    try:
        item = add_name_aliases(dataset, alias)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=dataset_dict(item))


@app.command("update")
def update_command(
    dataset: Annotated[str, typer.Argument(help="Dataset ID, canonical name, existing alias, or source URL.")],
    url: str | None = typer.Option(None, "--url", help="Canonical remote source URL to add when missing."),
    provider: Provider | None = typer.Option(None, "--provider", help="Dataset provider; a supplied URL determines this value."),
    version: str | None = typer.Option(None, "--version", help="Dataset version to add or replace."),
    modality: list[Modality] = typer.Option([], "--modality", help="Dataset modality to append; repeat for multiple modalities."),
    alias: list[str] = typer.Option([], "--alias", help="Searchable alternate dataset name to append; repeat for multiple aliases."),
    force_replace: bool = typer.Option(False, "--force-replace", help="Allow replacement of an existing provider or version."),
):
    """Add missing metadata to a registered dataset.

    Modalities and aliases are appended. Existing provider or version values
    require --force-replace; an existing canonical source URL cannot change.
    """
    try:
        item = update_dataset_metadata(
            dataset,
            url=url,
            provider=provider,
            version=version,
            modalities=[item.value for item in modality],
            aliases=alias,
            force_replace=force_replace,
        )
    except (ValueError, RuntimeError, LookupError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=dataset_dict(item))


def _interval_seconds(value: str) -> float:
    """Parse scheduler intervals such as 30m, 24h, or 1d."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    normalized = value.strip().lower()
    if len(normalized) < 2 or normalized[-1] not in units:
        raise typer.BadParameter("Interval must end in s, m, h, or d (for example 24h)")
    try:
        seconds = float(normalized[:-1]) * units[normalized[-1]]
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid interval: {value}") from exc
    if seconds <= 0:
        raise typer.BadParameter("Interval must be greater than zero")
    return seconds


@app.command("health-check")
def health_check_command(
    dataset: Annotated[str | None, typer.Argument(help="Dataset ID, name, or URL.")] = None,
    all_datasets: bool = typer.Option(False, "--all", help="Check every registered dataset."),
):
    """Run a one-shot health check, including DataLad repair when needed."""
    if all_datasets == (dataset is not None):
        raise typer.BadParameter("Provide exactly one DATASET or --all")
    dataset_ids = None
    if dataset is not None:
        with session() as db:
            item = resolve_dataset(db, dataset)
        if item is None:
            raise typer.BadParameter("No dataset matched the supplied identifier")
        dataset_ids = [item.id]
    else:
        with session() as db:
            dataset_ids = [item.id for item in db.query(Dataset).all()]
    if not run_health_checks(dataset_ids):
        console.print("[yellow]Skipped: another health worker is already running.[/yellow]")
        raise typer.Exit(code=2)
    problems = _health_problems(dataset_ids)
    if not problems:
        console.print("[green]Health check completed.[/green]")
        return
    console.print("[red]Health check found problems:[/red]")
    for item, history in problems:
        detail = history.message or "No further details were recorded."
        console.print(f"[red]- {item.name} ({history.result}): {detail}[/red]")
    log_path = get_settings().logs_dir / "critical_errors.log"
    console.print(f"[red]See the critical error log: {log_path}[/red]")


@app.command("health-scheduler")
def health_scheduler_command(
    interval: str = typer.Option("24h", "--interval", help="Delay between checks, such as 30m, 24h, or 1d."),
):
    """Run opt-in periodic all-dataset health checks until interrupted."""
    seconds = _interval_seconds(interval)
    console.print(f"Health scheduler started; interval={interval}.")
    try:
        while True:
            if not run_health_checks():
                console.print("[yellow]Skipped: another health worker is running.[/yellow]")
            time.sleep(seconds)
    except KeyboardInterrupt:
        console.print("Health scheduler stopped.")
