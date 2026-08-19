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
export NDR_DATA_ROOT=/ABSOLUTE/PATH/TO/NEURAL_DATA
# Optional database override; otherwise SQLite is stored under NDR_DATA_ROOT.
export NDR_DATABASE_URL=sqlite:////ABSOLUTE/PATH/TO/NEURAL_DATA/registry/registry.db
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

## Hugging Face dataset downloads

Install the Hub client and its chunked Xet transport in the download
environment:

```bash
python -m pip install -U huggingface_hub hf_xet httpx
```

The repository and absolute destination are always explicit. Xet, loopback
proxy host, one file worker, a 300-second timeout, and a 300-second maximum
retry delay are operational defaults:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_huggingface.sh \
  --repo OWNER/DATASET \
  --dest /ABSOLUTE/PATH/TO/DESTINATION \
  --proxy-port PROXY_PORT
```

Authentication is inherited from `HF_TOKEN` or `hf auth login`. Re-running
the same command against the same destination reuses completed and partial
files. Use `--transport http` for the TLS 1.2 HTTP path, `--dry-run` to inspect
pending files, or `--retry-attempts 0` to retry transient failures until
Ctrl-C.

For unstable Mihomo routes, the downloader can rank a selector by repeated
HTTPS success, bounded throughput, and latency. Supplying
`--mihomo-speed-test-url` activates ranking. Supply the controller explicitly;
the selector is discovered from the live route used by the speed-test URL.
The group and node marker remain optional overrides:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_huggingface.sh \
  --repo OWNER/DATASET \
  --dest /ABSOLUTE/PATH/TO/DESTINATION \
  --proxy-port PROXY_PORT \
  --retry-attempts 0 \
  --mihomo-controller http://127.0.0.1:CONTROLLER_PORT \
  --mihomo-speed-test-url LARGE_HTTPS_FILE_URL
```

Every Hub and CDN domain must route through the selector. Without
`--mihomo-node-marker`, every direct node in the selector is eligible. With a
marker, the downloader never selects an unmarked node. If the speed-test URL
is omitted, ranking and failover are skipped while ordinary proxying remains
active. The initial throughput test consumes at most 1.5 MiB,
uses one sample for each of at most three candidates, limits each sample to
eight seconds, and caches the ranking for six hours. Export an optional
controller address as `MIHOMO_CONTROLLER`, an optional selector override as
`MIHOMO_GROUP`, and an optional controller secret as `MIHOMO_SECRET`; secret
values are never printed. CLI values override their environment counterparts.

Every continued command line must end with `\` as its final character. A
missing continuation, or spaces after it, causes the next option to run as a
separate command and produces errors such as
`--mihomo-group: command not found`.

## OSF dataset downloads

Install the HTTP client used by the OSF downloader:

```bash
python -m pip install -U httpx
```

### Usage

Pass either an OSF node ID or project URL and an absolute destination. Direct
access is the default, so no Mihomo setting is required:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_osf.sh \
  --repo https://osf.io/ag3kj/overview \
  --dest /ABSOLUTE/PATH/TO/DESTINATION
```

To send requests through a local Mihomo mixed port, add only the proxy port:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_osf.sh \
  --repo https://osf.io/ag3kj/overview \
  --dest /ABSOLUTE/PATH/TO/DESTINATION \
  --proxy-port PROXY_PORT
```

The default storage provider is `osfstorage`. Re-running the command verifies
completed file sizes and resumes sibling `.part` files. Use `--dry-run` to
inspect the manifest without creating the destination, `--storage NAME` for a
different provider, or `--retry-attempts 0` for retries until Ctrl-C. Export
`OSF_TOKEN` for private projects; its value is never printed.

Ranked Mihomo failover is optional. Supply the speed URL and controller to
activate it, and keep one file worker so switching nodes cannot interrupt
another active file. Group and node-marker options are optional overrides:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_osf.sh \
  --repo PROJECT_ID_OR_URL \
  --dest /ABSOLUTE/PATH/TO/DESTINATION \
  --proxy-port PROXY_PORT \
  --retry-attempts 0 \
  --mihomo-controller http://127.0.0.1:CONTROLLER_PORT \
  --mihomo-speed-test-url LARGE_OSF_HTTPS_FILE_URL
```

The `api.osf.io` and `files.osf.io` domains must route through the named
selector. Without `--mihomo-node-marker`, all direct selector members are
eligible. Without a speed-test URL, ranking is skipped and `--proxy-port`
still provides ordinary proxy access. OSF ranking uses the same bounded
1.5 MiB initial benchmark and six-hour cache as Hugging Face ranking. The
`MIHOMO_CONTROLLER`, `MIHOMO_GROUP`, and `MIHOMO_SECRET` environment variables
have the same behavior as in the Hugging Face downloader.

The controller delay API is used only as a fast ranking hint. If it is
incompatible with the repository endpoint or returns too few usable nodes,
the ranker automatically checks candidates through the same local HTTPS proxy
path used by the download. These checks are range-bounded and require no
additional option or Mihomo configuration-file path.

## Zenodo dataset downloads

Use the same common downloader options with a Zenodo record ID or URL:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_zenodo.sh \
  --repo https://zenodo.org/records/583331 \
  --dest /ABSOLUTE/PATH/TO/DESTINATION \
  --proxy-port PROXY_PORT
```

Existing files are verified against Zenodo metadata and interrupted transfers
resume from sibling `.part` files. `--dry-run`, unlimited retries, and optional
Mihomo ranking behave as in the OSF downloader.

## OpenNeuro dataset downloads

Install the `download` extra, then pass an accession number, dataset URL, or
version URL. A version URL pins the corresponding DataLad snapshot tag:

```bash
/ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_openneuro.sh \
  --repo https://openneuro.org/datasets/ds005261/versions/2.0.0 \
  --dest /ABSOLUTE/PATH/TO/DESTINATION \
  --proxy-port PROXY_PORT
```

DataLad and git-annex reuse existing content on retries. Public snapshots use
the remotes configured by OpenNeuro; private access may additionally require
OpenNeuro credentials and its git-annex special-remote helper.

## Download diagnostics

Every provider downloader writes a unique sanitized diagnostic log beneath
`/ABSOLUTE/PATH/TO/REPOSITORY/logs/downloads/`. This directory is ignored by
git. The absolute log path is printed when a run starts and whenever it fails.
Retry attempts, exception chains, dependency versions, interruptions, and
Mihomo failovers are recorded without authentication secrets or signed URL
queries.

## CLI commands

All commands read `NDR_DATA_ROOT` and operate on the same registry database.

### `brainctl query`

Queries the storage path of a registered dataset by its ID (internal to this registry), canonical name, name alias, or source URL. Remote URLs match by normalized scheme and host plus their first non-empty path segment, so an OSF dataset root and a nested OSF file URL can resolve together. A unique match prints only its absolute storage path; multiple URL matches display their details and prompt for a canonical name.

```bash
brainctl query --name THINGS-MEG  # a user-defined name alias can also be searched (e.g., THINGS_MEG)
brainctl query --url "https://openneuro.org/datasets/ds004212/versions/3.0.0"  # remote URL
brainctl query 220cb6c2-cc2f-409d-be24-5abb018da87d  # internal ID
```

Query a read-only metadata field with `--field`; repeat it to return a JSON
object containing multiple fields. For example:

```bash
brainctl query THINGS-MEG --field aliases
brainctl query THINGS-MEG --field name --field modalities
```

`aliases`, `url_aliases`, and `path_aliases` expose their respective stored
alias categories.

IDs and names are unique. URL segment matches can be ambiguous, in which case `brainctl query` asks which displayed canonical dataset to use.

### `brainctl list`

Lists all registered datasets as a structured summary, optionally narrowed to one modality or provider.

```bash
brainctl list --modality meg
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
brainctl ingest-local /path/to/things-meg \
  --name THINGS-MEG \
  --alias THINGS_MEG \
  --alias "THINGS object vision" \
  --url "https://openneuro.org/datasets/ds004212" \
  --version 3.0.0 \
  --modality meg
```

`SOURCE` must be an existing directory. `--name` and `--version` are required.
`--provider` accepts `openneuro`, `dandi`, `nemar`, `physionet`, `neurovault`, `kaggle`, `synapse`, or `other`; it defaults to `other`. URLs automatically determine the provider and, where present, the version.
`--url` records the canonical remote URL when one exists.
Repeat `--modality` to register multiple modalities. Repeat `--alias` to
register searchable alternate names alongside the canonical `--name`.
By default (`--storage-mode reference`), the command asks for confirmation,
then leaves `SOURCE` where it is and records its absolute path. It makes the
tree publicly readable and traversable without changing ownership, existing
owner or group permissions, or POSIX ACLs. An already-public source owned by
another user can be registered read-only. If publication needs a permission
change that the caller cannot make, registration fails before changing the
tree. Use `--storage-mode move` to relocate `SOURCE` into
`$NDR_DATA_ROOT/datasets`. Use `--storage-mode copy` to preserve `SOURCE`
while creating a managed duplicate.
Before validating or moving `SOURCE`, it rejects a duplicate canonical name or
source URL/path and reports the existing storage path.

### Protected CLI local ingestion

Install this service account so that `ingest-local` is delegated to the locally configured service account. The service account alone can write the registry, manifests, logs, and managed datasets; ordinary accounts cannot write those paths directly. (installation requires admin)

Installation steps:

1. Copy `deployment/reference_ingest.env.example` to `deployment/reference_ingest.local.env` and replace every placeholder.

```bash
# make the file root-owned
sudo chown root:root /path/to/registry/deployment/reference_ingest.local.env
sudo chmod 600 /path/to/registry/deployment/reference_ingest.local.env
```
2. User an administrator accont to make the local file root-owned with mode `600`,
   then run:

   ```bash
   sudo $REPOSITORY_ROOT/deployment/install_protected_ingest.sh \
     $LOCAL_REFERENCE_INGEST_CONFIG
   ```

The installation process `install_protected_ingest.sh` should be re-run when:

- source codes under `src` changes
- python dependencies changes
- `deployment/` files changes
- $LOCAL_REFERENCE_INGEST_CONFIG changes

### `brainctl download`

Detects the provider from a dataset URL, downloads into `incoming`, and ingests the
result automatically. Failed or incomplete downloads remain in `incoming` until they
are successfully completed or manually removed.
Before creating an `incoming` workspace or contacting a provider, it rejects a
duplicate canonical name or URL and reports the existing storage path. A
duplicate is never downloaded or processed again.


```bash
brainctl download --url "https://openneuro.org/datasets/ds007338/versions/1.0.0" --name EXAMPLE-MEG --alias EXAMPLE --modality meg
```

Check the selected HTTP(S) source without downloading or registering anything:

```bash
brainctl download --url "https://openneuro.org/datasets/ds007338" \\
  --mirror "https://mirror.example/{dataset_id}.git" --check-connection
```

The check uses the same `--proxy`, `--mirror`, and deployment defaults as a
download. It creates no incoming workspace, log, job, or dataset record. It
reports that total dataset size is unavailable when Git/DataLad metadata cannot
provide a trustworthy annexed-content total before cloning.

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
brainctl update THINGS-MEG --version 3.0.0 --modality meg --alias THINGS_MEG
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
    "source": "/path/to/things-meg",
    "name": "THINGS-MEG",
    "url": "https://openneuro.org/datasets/ds004212",
    "version": "3.0.0",
    "modalities": ["meg"],

    "aliases": ["THINGS_MEG"],
    "storage_mode": "copy"
  }'
```

The API rejects reference mode because public reference ingestion requires the
interactive `brainctl ingest-local` confirmation.

Download and register a provider dataset with `POST /download`:

Check a download source without creating a download workspace with
`POST /download/check`:

```bash
curl -X POST http://127.0.0.1:8000/download/check \\
  -H 'Content-Type: application/json' \\
  -d '{
    "url": "https://openneuro.org/datasets/ds007338",
    "mirror": "https://mirror.example/{dataset_id}.git"
  }'
```

The endpoint returns HTTP 502 when the configured HTTP(S) source cannot be
reached.

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
curl -X POST http://127.0.0.1:8000/datasets/{DATASET_ID}/storage-transition \
  -H 'Content-Type: application/json' \
  -d '{"storage_mode": "move"}'
```

The intake `POST` endpoints reject duplicate canonical names and canonical
URLs or paths with HTTP 409, before processing data or contacting a provider.
