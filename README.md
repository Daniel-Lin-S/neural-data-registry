# Neural Data Registry

`brainctl` is a small control plane for curating and ingesting neuroscience datasets
under a managed storage root.

## Data Root Layout

The application expects this directory structure under a configured root path:

```text
{NDR_DATA_ROOT}/
  datasets/      # Prepared datasets ready for use
  incoming/      # Manual uploads and incomplete downloads (not ready for use)
  quarantine/    # Failed or suspicious ingestion attempts
  registry/      # Registry database backups and dataset manifests
  logs/          # Records of ingestion operations and download diagnostics
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

# install the download workflow
pip install -e '.[download]'

# install both download and development tools
pip install -e '.[download,dev]'
```
The `download` extra installs both DataLad and `git-annex` (`>=10.20230126`).

## CLI commands

All commands read `NDR_DATA_ROOT` and operate on the same registry database.

### `brainctl query`

Queries the storage path of a registered dataset by its ID (internal to this registry), canonical name, name alias, or source URL and prints only its absolute storage path. The forms are equivalent:

```bash
brainctl query --name THINGS-MEG  # a user-defined name alias can also be searched (e.g., THINGS_MEG)
brainctl query --url "https://openneuro.org/datasets/ds004212/versions/3.0.0"  # remote URL
brainctl query 220cb6c2-cc2f-409d-be24-5abb018da87d  # internal ID
```

IDs, names, and source URLs are unique. Query does not list or summarize records; use `brainctl list` for that.

### `brainctl list`

Lists all registered datasets as a structured summary, optionally narrowed to one modality or provider.

```bash
brainctl list --modality MEG
brainctl list --provider openneuro
brainctl list --query THINGS_MEG  # searches canonical names and aliases
```

`--modality` accepts a value such as `MEG`, `EEG`, or `fMRI`;
`--provider` accepts
`openneuro`, `dandi`, `nemar`, `physionet`, `neurovault`, `kaggle`, `synapse`, or `other`.
Missing and broken datasets are hidden by default; use `brainctl list --show-all`
to include every status. The summary includes dataset ID, name, provider, version,
modalities, size, and status.

### Health checks

Run a synchronous one-shot check for one dataset or the entire registry:

```bash
brainctl health-check THINGS-MEG
brainctl health-check --all
```

For opt-in recurring checks, run the long-lived scheduler:

```bash
brainctl health-scheduler --interval 24h
```

Intervals accept `s`, `m`, `h`, or `d` suffixes. The scheduler is not started or installed as a service automatically. Normal `brainctl` usage launches an all-dataset background scan at most once every 24 hours per virtual environment.

Every check is stored in the SQL `health_check_history` table. Only missing, broken, or operational-error results are appended to `$NDR_DATA_ROOT/logs/critical_errors.log`. One global process lock prevents overlapping deep checks.

For DataLad datasets, only files reported by `git annex find --not --in=here`
cause a `BROKEN` status. Repository, tool, command, remote, and network errors
are recorded but do not change the dataset status by themselves.

Checks are automatically performed when `brainctl query` is called or `GET /datasets/{id}` is called.

### `brainctl ingest-local`

Registers a dataset that has already been downloaded to a local directory, or a dataset that is uploaded manually. IMPORTANT: please use to check whether your dataset already exists using `brainctl query` before ingesting.

Usage example:

```bash
brainctl ingest-local /data/legacy/things-meg \
  --name THINGS-MEG \
  --alias THINGS_MEG \
  --alias "THINGS object vision" \
  --url "https://openneuro.org/datasets/ds004212" \
  --version 3.0.0 \
  --modality MEG
```

`SOURCE` must be an existing directory. `--name` and `--version` are required.
`--provider` accepts `openneuro`, `dandi`, `nemar`, `physionet`, `neurovault`, `kaggle`, `synapse`, or `other`; it defaults to `other`. URLs automatically determine the provider and, where present, the version.
`--url` records the canonical remote URL when one exists.
Repeat `--modality` to register multiple modalities. Repeat `--alias` to
register searchable alternate names alongside the canonical `--name`.
By default (`--storage-mode reference`), the command leaves `SOURCE` where it is and records its absolute path. Use `--storage-mode move` to relocate `SOURCE` into `$NDR_DATA_ROOT/datasets`. Use `--storage-mode copy` to preserve `SOURCE` while creating a managed duplicate; this consumes additional disk space and should only be used when `SOURCE` may be cleaned in the future. (DO NOT use `move` mode if your source location is still used by other codes)
Before validating or moving `SOURCE`, it rejects a duplicate canonical name or
source URL/path and reports the existing storage path.

### `brainctl download`

Detects the provider from a dataset URL, downloads into `incoming`, and ingests the
result automatically. Failed or incomplete downloads remain in `incoming` until they
are successfully completed or manually removed.
Before creating an `incoming` workspace or contacting a provider, it rejects a
duplicate canonical name or URL and reports the existing storage path. A
duplicate is never downloaded or processed again.


```bash
brainctl download --url "https://openneuro.org/datasets/ds007338/versions/1.0.0" --name EXAMPLE-MEG --alias EXAMPLE --modality MEG
```

`--url` is required. `--version` is optional only when an OpenNeuro URL contains a version such as `/versions/1.0.0`; otherwise provide it manually. An explicit `--version` may name a provider branch or tag.

Install `neural-data-registry[download]` to enable downloads. Automatic downloads currently support OpenNeuro; DANDI, NEMAR, PhysioNet, NeuroVault, and Kaggle URLs are recognised but their download clients are not configured yet.

Each attempt appends a log to `$NDR_DATA_ROOT/logs/download-*.log`, including the full DataLad stdout and stderr on failure. Rerun the same command to resume a valid partial DataLad workspace in `incoming`; an empty workspace is retried as a new clone.

The API uses required `name` and `modalities` fields for the same reason, so registered downloaded datasets always carry explicit metadata. Both intake API requests also accept an optional `aliases` array.

### `brainctl alias`

Aliases are globally unique (case-insensitive)
and cannot reuse another dataset’s canonical name or alias.

```bash
brainctl alias THINGS-MEG --alias THINGS_MEG --alias "THINGS object vision"
```

Aliases are append-only: repeat the command to add another name. Use
`brainctl list --query NAME` to search by either a canonical name or alias.

Use `--proxy https://proxy.example:8080` to configure a download proxy. Use `--mirror https://mirror.example/{dataset_id}.git` (or a mirror base URL) to select a mirror URL. The API accepts the same `proxy` and `mirror` fields, and deployment defaults can be set with `NDR_DOWNLOAD_PROXY` and `NDR_DOWNLOAD_MIRROR`.

### `brainctl update`

Add metadata to a dataset already registered in the registry:

```bash
brainctl update THINGS-MEG --version 3.0.0 --modality MEG --alias THINGS_MEG
```

Repeat `--modality` and `--alias` to append values. A missing canonical URL can
be added with `--url`; its provider is detected from the URL. Existing provider
and version values are protected unless `--force-replace` is supplied. A
canonical source URL is never replaced.

## API

Run the API service:

```bash
uvicorn neural_data_registry.main:app --reload
```

Interactive docs are available at `http://localhost:8000/docs`.

## Neural data providers

Common providers for neural data are included:

- `openneuro`: Open BIDS-formatted neuroimaging and electrophysiology. https://openneuro.org/
- `nemar`: Gateway for human neuroelectromagnetic data, mirrored/linked with OpenNeuro, BIDS-formatted, with HED/event annotations and quality metadata. https://nemar.org/discover
- `dandi`: Systems-neuroscience data platform with neurophysiology, electrophysiology, optophysiology. Formatted in NWB/BIDS. https://dandiarchive.org/
- `physionet`: EEG, sleep PSG, ECG, ICU signals. https://physionet.org/content/
- `neurovault`: https://neurovault.org/ Derived neuroimaging maps: fMRI/PET statistical maps, parcellations, atlases.
- `kaggle`: Kaggle datasets and competitions, typically downloaded for machine-learning workflows. https://www.kaggle.com/
- `other`: Any other dataset downloaded manually from arbitrary websites or requested from labs.

## API server and common requests

Run the API on a local port:

```bash
python -m uvicorn neural_data_registry.main:app --host 127.0.0.1 --port 8000
```

Leave the server running, then use another terminal for API requests. The
examples below assume `http://127.0.0.1:8000`.

```bash
# Check that the service is available.
curl http://127.0.0.1:8000/health

# List datasets; optional filters include query, url, modality, provider, and show_all.
curl 'http://127.0.0.1:8000/datasets?query=THINGS-MEG'

# Get one registered dataset by its ID. This also triggers a health check.
curl http://127.0.0.1:8000/datasets/220cb6c2-cc2f-409d-be24-5abb018da87d
```

Register an existing local dataset with `POST /ingest/local`:

```bash
curl -X POST http://127.0.0.1:8000/ingest/local \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "/data/legacy/things-meg",
    "name": "THINGS-MEG",
    "url": "https://openneuro.org/datasets/ds004212",
    "version": "3.0.0",
    "modalities": ["meg"],
    "aliases": ["THINGS_MEG"],
    "storage_mode": "reference"
  }'
```

Download and register a provider dataset with `POST /download`:

```bash
curl -X POST http://127.0.0.1:8000/download \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://openneuro.org/datasets/ds007338/versions/1.0.0",
    "name": "EXAMPLE-MEG",
    "modalities": ["meg"],
    "aliases": ["EXAMPLE"]
  }'
```

To move or copy a dataset that was previously registered with
`storage_mode: "reference"`, send a storage-transition request:

```bash
curl -X POST http://127.0.0.1:8000/datasets/220cb6c2-cc2f-409d-be24-5abb018da87d/storage-transition \
  -H 'Content-Type: application/json' \
  -d '{"storage_mode": "move"}'
```

The intake `POST` endpoints reject duplicate canonical names and canonical
URLs or paths with HTTP 409, before processing data or contacting a provider.
