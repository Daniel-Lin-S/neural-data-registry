# Neural Data Registry

`brainctl` is a small control plane for curating and ingesting neuroscience datasets
under a managed storage root.

## Data Root Layout

The application expects this directory structure under a configured root path:

```text
{NDR_DATA_ROOT}/
  datasets/      # Prepared datasets ready for use
  incoming/      # Manual uploads (not ready for use)
  staging/       # Temporary workspace used during ingestion or automatic download
  quarantine/    # Failed or suspicious ingestion attempts
  registry/      # Registry database backups and dataset manifests
  logs/          # Records of ingestion operations
```

Users may move files directly to `incoming`, but should NOT directly modify other folders.

## Configuration

Set the global dataset root with the required `NDR_DATA_ROOT` environment
variable. Every CLI or API process that should use the same registry must receive
the same value:

```bash
export NDR_DATA_ROOT=/data/neural_data
# Optional database override; otherwise SQLite is stored under NDR_DATA_ROOT.
export NDR_DATABASE_URL=sqlite:////data/neural_data/registry/registry.db
```

These values can also be placed in a `.env` file in the working directory. For a long-running service, configure them in the service manager or deployment environment. If `NDR_DATABASE_URL` is omitted, the application uses `$NDR_DATA_ROOT/registry/registry.db`.

## Installation

```bash
pip install -e .

# install with development tools (debugging etc.)
pip install -e '.[dev]'
```

## CLI commands

All commands read `NDR_DATA_ROOT` and operate on the same registry database.

### `brainctl query`

Searches registered datasets by a name/alias fragment or an exact source URL.

```bash
brainctl query "THINGS-MEG"
brainctl query --url "https://openneuro.org/datasets/ds004212"
```

`QUERY` is a case-insensitive name fragment. `--url` performs an exact source-URL
lookup. Provide at least one of them. Results are shown as a table with dataset
ID, name, provider, version, modalities, size, and status.

### `brainctl list`

Lists all registered datasets, optionally narrowed to one modality.

```bash
brainctl list --modality MEG
```

`--modality` accepts a value such as `MEG`, `EEG`, or `fMRI`; omitting it lists
every registry record. The output table has the same fields as `query`.

### `brainctl ingest-local`

Registers a dataset that has already been downloaded to a local directory. Usage example:

```bash
brainctl ingest-local /data/legacy/things-meg \
  --name THINGS-MEG \
  --provider openneuro \
  --url "https://openneuro.org/datasets/ds004212" \
  --version 3.0.0 \
  --modality MEG
```

`SOURCE` must be an existing directory. `--name` and `--version` are required.
`--provider` accepts `openneuro`, `dandi`, `nemar`, or `local`; it defaults to `local`. `--url` records the canonical remote URL when one exists.
Repeat `--modality` to register multiple modalities. By default (`--storage-mode reference`), the command leaves `SOURCE` where it is and records its absolute path. Use `--storage-mode move` to relocate `SOURCE` into `$NDR_DATA_ROOT/datasets`. Both modes write a manifest and print the new record as JSON.
It rejects a duplicate canonical name or source URL and reports the existing storage path.

### `brainctl download`

Detects the provider from a dataset URL, downloads into staging, and ingests the
result automatically.

```bash
brainctl download --url "https://openneuro.org/datasets/ds004212" --version latest
```

`--url` is required. `--version` defaults to `latest`; provide another value to
request a provider version or branch. Automatic downloads currently support
OpenNeuro and require `git`. DANDI and NEMAR URLs are recognised but report that
their download clients are not configured. Failed downloads are moved to

## API

Run the API service:

```bash
uvicorn neural_data_registry.main:app --reload
```

Interactive docs are available at `http://localhost:8000/docs`.
