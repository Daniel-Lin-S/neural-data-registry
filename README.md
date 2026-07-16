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

`--modality` accepts a value such as `MEG`, `EEG`, or `fMRI`;
`--provider` accepts
`openneuro`, `dandi`, `nemar`, `physionet`, `neurovault`, `kaggle`, or `other`. Omitting both lists every registry record.
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
`--provider` accepts `openneuro`, `dandi`, `nemar`, `physionet`, `neurovault`, `kaggle`, or `other`; it defaults to `other`.
`--url` records the canonical remote URL when one exists.
Repeat `--modality` to register multiple modalities.
By default (`--storage-mode reference`), the command leaves `SOURCE` where it is and records its absolute path. Use `--storage-mode move` to relocate `SOURCE` into `$NDR_DATA_ROOT/datasets`. (DO NOT use `move` mode if your source location is still used by other codes)
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
brainctl download --url "https://openneuro.org/datasets/ds007338/versions/1.0.0" --name EXAMPLE-MEG --modality MEG
```

`--url` is required. `--version` is optional only when an OpenNeuro URL contains a version such as `/versions/1.0.0`; otherwise provide it manually. An explicit `--version` may name a provider branch or tag.

Install `neural-data-registry[download]` to enable downloads. Automatic downloads currently support OpenNeuro; DANDI, NEMAR, PhysioNet, NeuroVault, and Kaggle URLs are recognised but their download clients are not configured yet.
Each attempt appends a log to `$NDR_DATA_ROOT/logs/download-*.log`, including the
full DataLad stdout and stderr on failure. Rerun the same command to resume a valid
partial DataLad workspace in `incoming`; an empty workspace is retried as a new clone.


The download command does not infer these fields from the provider URL: `--name` and at least one repeated `--modality` are required. The API uses required `name` and `modalities` fields for the same reason, so registered downloaded datasets always carry explicit metadata.
Use `--proxy https://proxy.example:8080` to configure a download proxy. Use `--mirror https://mirror.example/{dataset_id}.git` (or a mirror base URL) to select a mirror URL. The API accepts the same `proxy` and `mirror` fields, and deployment defaults can be set with `NDR_DOWNLOAD_PROXY` and `NDR_DOWNLOAD_MIRROR`.

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
