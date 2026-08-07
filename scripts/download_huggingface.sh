#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Hugging Face dataset downloader
#
# Example:
#
# Mainland China + HF mirror + local proxy:
#
#   ./download_hf_dataset.sh \
#       --dest /data/LibriBrain \
#       --mirror https://hf-mirror.com \
#       --proxy-port 7893
#
#
# Direct download:
#
#   ./download_hf_dataset.sh \
#       --dest /data/LibriBrain
#
#
# Disable proxy explicitly:
#
#   ./download_hf_dataset.sh \
#       --dest /data/LibriBrain \
#       --no-proxy
#
# ============================================================


# -----------------------------
# Default configuration
# -----------------------------

REPO_ID="pnpl/LibriBrain"

DEST=""

# Official HF by default
HF_ENDPOINT="https://huggingface.co"

# Proxy disabled by default
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
        Enable HTTP proxy with given port.

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

        Example:
        socks5


  --no-proxy
        Disable proxy.


  --max-workers N
        Number of parallel download workers.

        Default:
        4


  --timeout SEC
        Download timeout.

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
# Check hf CLI
# -----------------------------

if ! command -v hf >/dev/null 2>&1; then

    echo "ERROR: Hugging Face CLI not found."

    echo
    echo "Install with:"
    echo
    echo "  pip install -U huggingface_hub"

    exit 1

fi



# -----------------------------
# Environment setup
# -----------------------------

export HF_ENDPOINT="${HF_ENDPOINT}"

export HF_HUB_DOWNLOAD_TIMEOUT="${TIMEOUT}"



if [[ "${USE_PROXY}" == "1" ]]; then

    PROXY_URL="${PROXY_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"

    export HTTP_PROXY="${PROXY_URL}"
    export HTTPS_PROXY="${PROXY_URL}"

    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"

else

    unset HTTP_PROXY HTTPS_PROXY
    unset http_proxy https_proxy

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
echo

echo "Interrupted downloads can be resumed by rerunning"
echo "the same command."
echo


# -----------------------------
# Build command
# -----------------------------

CMD=(

    hf download

    "${REPO_ID}"

    --repo-type dataset

    --local-dir "${DEST}"

    --max-workers "${MAX_WORKERS}"

)



if [[ "${DRY_RUN}" == "1" ]]; then

    CMD+=(--dry-run)

fi



# -----------------------------
# Execute
# -----------------------------

"${CMD[@]}"