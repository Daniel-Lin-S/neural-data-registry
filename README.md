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

Queries the storage path of a registered dataset by its ID (internal to this registry), canonical name, or source URL and prints only its absolute storage path. The three forms are equivalent:

```bash
brainctl query --name THINGS-MEG
brainctl query --url "https://openneuro.org/datasets/ds004212/versions/3.0.0"  # remote URL
brainctl query 220cb6c2-cc2f-409d-be24-5abb018da87d  # internal ID
```

IDs, names, and source URLs are unique. Query does not list or summarize records; use `brainctl list` for that.

### `brainctl list`

Lists all registered datasets as a structured summary, optionally narrowed to one modality or provider.

```bash
brainctl list --modality MEG
brainctl list --provider openneuro
```

`--modality` accepts a value such as `MEG`, `EEG`, or `fMRI`; `--provider` accepts
`openneuro`, `dandi`, `nemar`, or `local`. Omitting both lists every registry record.
The summary includes dataset ID, name, provider, version, modalities, size, and status.

### `brainctl ingest-local`

Registers a dataset that has already been downloaded to a local directory, or a dataset that is uploaded manually. IMPORTANT: please use to check whether your dataset already exists using `brainctl query` before ingesting.

Usage example:

```bash
brainctl ingest-local /data/legacy/things-meg \
  --name THINGS-MEG \
  --provider openneuro \
  --url "https://openneuro.org/datasets/ds004212" \
  --version 3.0.0 \
  --modality MEG
```

`SOURCE` must be an existing directory. `--name` and `--version` are required.
`--provider` accepts `openneuro`, `dandi`, `nemar`, or `local`; it defaults to `local`.
`--url` records the canonical remote URL when one exists.
Repeat `--modality` to register multiple modalities.
By default (`--storage-mode reference`), the command leaves `SOURCE` where it is and records its absolute path. Use `--storage-mode move` to relocate `SOURCE` into `$NDR_DATA_ROOT/datasets`.

It rejects a duplicate canonical name or source URL and reports the existing storage path.

### `brainctl download`

Detects the provider from a dataset URL, downloads into staging, and ingests the
result automatically. IMPORTANT: please use to check whether your dataset already exists using `brainctl query` before downloading.

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
