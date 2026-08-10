#!/usr/bin/env bash

set -Eeuo pipefail


# ============================================================
# Hugging Face dataset downloader
#
# Examples:
#
# Direct Hugging Face + local proxy:
#
#   ./download_huggingface.sh \
#       --dest /data/LibriBrain \
#       --proxy-port 7893
#
#
# Mainland China mirror + proxy:
#
#   ./download_huggingface.sh \
#       --dest /data/LibriBrain \
#       --mirror https://hf-mirror.com \
#       --proxy-port 7893
#
#
# Direct download without proxy:
#
#   ./download_huggingface.sh \
#       --dest /data/LibriBrain
#
# ============================================================


# -----------------------------
# Default configuration
# -----------------------------

REPO_ID="pnpl/LibriBrain"

DEST=""

HF_ENDPOINT="https://huggingface.co"

USE_PROXY=0

PROXY_HOST="127.0.0.1"
PROXY_PORT="7893"
PROXY_SCHEME="http"

MAX_WORKERS=4
TIMEOUT=60

DRY_RUN=0


# -----------------------------
# Help
# -----------------------------

show_help()
{
cat <<EOF

Usage:

  $0 --dest PATH [OPTIONS]


Required:

  --dest PATH
        Destination directory for dataset.


Optional:

  --repo REPO_ID
        Hugging Face dataset repository.

        Default:
        pnpl/LibriBrain


  --mirror URL
        Hugging Face endpoint.

        Example:
        https://hf-mirror.com

        Default:
        https://huggingface.co


  --proxy-port PORT
        Enable proxy with given port.

        Example:
        7893


  --proxy-host HOST
        Proxy host.

        Default:
        127.0.0.1


  --proxy-scheme SCHEME
        Proxy protocol.

        Default:
        http


  --no-proxy
        Disable proxy.


  --max-workers N
        Number of parallel file downloads.

        Default:
        4


  --timeout SEC
        HTTP timeout.

        Default:
        60


  --dry-run
        Show files without downloading.


  -h, --help
        Show this message.

EOF
}


# -----------------------------
# Parse arguments
# -----------------------------

while [[ $# -gt 0 ]]; do

    case "$1" in

        --dest)
            DEST="$2"
            shift 2
            ;;

        --repo)
            REPO_ID="$2"
            shift 2
            ;;

        --mirror)
            HF_ENDPOINT="$2"
            shift 2
            ;;

        --proxy-port)
            USE_PROXY=1
            PROXY_PORT="$2"
            shift 2
            ;;

        --proxy-host)
            USE_PROXY=1
            PROXY_HOST="$2"
            shift 2
            ;;

        --proxy-scheme)
            USE_PROXY=1
            PROXY_SCHEME="$2"
            shift 2
            ;;

        --no-proxy)
            USE_PROXY=0
            shift
            ;;

        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;

        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;

        --dry-run)
            DRY_RUN=1
            shift
            ;;

        -h|--help)
            show_help
            exit 0
            ;;

        *)
            echo "Unknown option: $1"
            echo
            show_help
            exit 1
            ;;

    esac

done


# -----------------------------
# Validate
# -----------------------------

if [[ -z "${DEST}" ]]; then

    echo "ERROR: --dest is required."
    echo

    show_help

    exit 1

fi


# -----------------------------
# Check dependencies
# -----------------------------

if ! python - <<'PY' >/dev/null 2>&1
import httpx
import huggingface_hub
PY
then

    echo "ERROR: Required Python packages are missing."
    echo
    echo "Install with:"
    echo
    echo "  pip install -U huggingface_hub httpx"

    exit 1

fi


# -----------------------------
# Environment
# -----------------------------

export HF_ENDPOINT="${HF_ENDPOINT}"

export HF_HUB_DOWNLOAD_TIMEOUT="${TIMEOUT}"


# Important:
#
# The TLS 1.2 workaround below customizes Hugging Face's
# HTTPX client. hf_xet has a separate network stack, so
# disable it to ensure file requests use this HTTP client.
export HF_HUB_DISABLE_XET=1


# -----------------------------
# Proxy
# -----------------------------

if [[ "${USE_PROXY}" == "1" ]]; then

    PROXY_URL="${PROXY_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"

    export HTTP_PROXY="${PROXY_URL}"
    export HTTPS_PROXY="${PROXY_URL}"

    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"

else

    unset HTTP_PROXY HTTPS_PROXY || true
    unset http_proxy https_proxy || true

fi


# -----------------------------
# Create destination
# -----------------------------

mkdir -p "${DEST}"


# -----------------------------
# Print configuration
# -----------------------------

echo
echo "=============================================="
echo " Hugging Face Dataset Download"
echo "=============================================="

echo "Repository : ${REPO_ID}"
echo "Destination: ${DEST}"
echo "Endpoint   : ${HF_ENDPOINT}"

if [[ "${USE_PROXY}" == "1" ]]; then
    echo "Proxy      : ${PROXY_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"
else
    echo "Proxy      : disabled"
fi

echo "Workers    : ${MAX_WORKERS}"
echo "Timeout    : ${TIMEOUT}s"
echo "TLS        : TLS 1.2 only"
echo "Xet        : disabled"
echo

echo "Interrupted downloads can be resumed by rerunning"
echo "the same command."
echo


# -----------------------------
# Export configuration for Python
# -----------------------------

export DOWNLOAD_REPO_ID="${REPO_ID}"
export DOWNLOAD_DEST="${DEST}"
export DOWNLOAD_ENDPOINT="${HF_ENDPOINT}"
export DOWNLOAD_MAX_WORKERS="${MAX_WORKERS}"
export DOWNLOAD_TIMEOUT="${TIMEOUT}"
export DOWNLOAD_DRY_RUN="${DRY_RUN}"


# -----------------------------
# Download
# -----------------------------

python - <<'PY'
import os
import ssl

import httpx
from huggingface_hub import (
    set_client_factory,
    snapshot_download,
)


repo_id = os.environ["DOWNLOAD_REPO_ID"]
dest = os.environ["DOWNLOAD_DEST"]
endpoint = os.environ["DOWNLOAD_ENDPOINT"]

max_workers = int(os.environ["DOWNLOAD_MAX_WORKERS"])
timeout = float(os.environ["DOWNLOAD_TIMEOUT"])

dry_run = os.environ["DOWNLOAD_DRY_RUN"] == "1"


# ------------------------------------------------------------
# TLS configuration
#
# HTTPX default TLS negotiation was unstable through the
# current Mihomo route, while forcing TLS 1.2 was stable.
# ------------------------------------------------------------

ssl_context = ssl.create_default_context()

ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2


# ------------------------------------------------------------
# Hugging Face HTTP client
# ------------------------------------------------------------

def hf_client_factory():

    return httpx.Client(
        verify=ssl_context,

        # HTTP_PROXY / HTTPS_PROXY are configured by the
        # parent shell script.
        trust_env=True,

        follow_redirects=True,

        timeout=httpx.Timeout(timeout),
    )


set_client_factory(hf_client_factory)


# ------------------------------------------------------------
# Download repository
# ------------------------------------------------------------

result = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",

    local_dir=dest,

    endpoint=endpoint,

    max_workers=max_workers,

    dry_run=dry_run,
)


# ------------------------------------------------------------
# Dry-run output
# ------------------------------------------------------------

if dry_run:

    pending = [
        item
        for item in result
        if item.will_download
    ]

    total_bytes = sum(
        item.file_size
        for item in pending
    )

    print()
    print("Dry-run summary")
    print("----------------------------------------")
    print(f"Files to download : {len(pending)}")
    print(f"GiB to download   : {total_bytes / 1024**3:.2f}")

    print()

    for item in pending:

        print(
            f"{item.file_size / 1024**2:10.2f} MiB  "
            f"{item.filename}"
        )

else:

    print()
    print(f"Download location: {result}")

PY