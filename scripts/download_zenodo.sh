#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash download_zenodo.sh [record_id] [output_directory] [--port VALUE]
#
# Examples:
#   bash download_zenodo.sh 583331 ~/datasets/zenodo_583331
#   bash download_zenodo.sh 583331 ~/datasets/zenodo_583331 --port 7893
#   bash download_zenodo.sh 583331 ~/datasets/zenodo_583331 --port none

RECORD_ID="${1:-583331}"
OUTPUT_DIR="${2:-zenodo_${RECORD_ID}}"
API_URL="https://zenodo.org/api/records/${RECORD_ID}"
PROXY_SETTING="${ZENODO_PROXY_PORT:-}"
if (( $# > 2 )); then
    case "${3}" in
        --port)
            if (( $# != 4 )); then
                echo "--port requires one value." >&2
                exit 1
            fi
            PROXY_SETTING="${4}"
            ;;
        --port=*)
            if (( $# != 3 )); then
                echo "--port=VALUE cannot be combined with extra arguments." >&2
                exit 1
            fi
            PROXY_SETTING="${3#--port=}"
            ;;
        *)
            echo "Expected --port VALUE after the output directory." >&2
            exit 1
            ;;
    esac
fi
CURL_ARGS=(
    --fail
    --location
    --retry 10
    --retry-delay 5
    --retry-all-errors
)
WGET_ARGS=(
    -c
    --tries=20
    --waitretry=10
    --timeout=60
    --read-timeout=60
)
CURL_PROXY_ARGS=()
WGET_PROXY_ARGS=()

case "${PROXY_SETTING,,}" in
    "")
        ;;
    none|no-proxy)
        CURL_PROXY_ARGS+=(--noproxy "*")
        WGET_PROXY_ARGS+=(--execute use_proxy=no)
        ;;
    [0-9]*)
        if ! [[ "${PROXY_SETTING}" =~ ^[0-9]+$ ]] ||
            (( PROXY_SETTING < 1 || PROXY_SETTING > 65535 )); then
            echo "Proxy port must be an integer from 1 to 65535." >&2
            exit 1
        fi

        PROXY_URL="http://127.0.0.1:${PROXY_SETTING}"
        CURL_PROXY_ARGS+=(--proxy "${PROXY_URL}")
        WGET_PROXY_ARGS+=(
            --execute use_proxy=yes
            --execute "http_proxy=${PROXY_URL}"
            --execute "https_proxy=${PROXY_URL}"
        )
        ;;
    *)
        echo "Proxy setting must be a port number, none, or no-proxy." >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

echo "Retrieving metadata for Zenodo record ${RECORD_ID}..."

CURL_OUTPUT_ARGS=(--output record.json)
curl "${CURL_ARGS[@]}" "${CURL_PROXY_ARGS[@]}" "${API_URL}"     "${CURL_OUTPUT_ARGS[@]}"

# Zenodo has used two slightly different JSON layouts over time.
# This parser supports both the legacy file-list representation and
# the newer files.entries representation.
python3 - <<'PY'
import json

with open("record.json", "r", encoding="utf-8") as f:
    record = json.load(f)

files = record.get("files", [])

if isinstance(files, dict):
    entries = files.get("entries", {})
    file_objects = list(entries.values())
else:
    file_objects = files

for item in file_objects:
    name = item.get("key") or item.get("filename")
    size = int(item.get("size", 0))
    checksum = item.get("checksum", "")

    links = item.get("links", {})
    url = links.get("content") or links.get("self")

    if name and url:
        print(f"{name}	{size}	{url}	{checksum}")
PY

if [[ ! -s files.tsv ]]; then
    echo "No downloadable files were found in the API response." >&2
    exit 1
fi

echo
echo "Files to download:"
python3 - <<'PY'
def human_size(n):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

total = 0

with open("files.tsv", encoding="utf-8") as f:
    for line in f:
        name, size, _, _ = line.rstrip().split(chr(9), 3)
        size = int(size)
        total += size
        print(f"  {human_size(size):>12}  {name}")

print()
print(f"Total: {human_size(total)}")
PY

echo
while IFS=$'\t' read -r name size url checksum; do
    echo "Downloading: ${name}"

    # -c continues a partially downloaded file.
    WGET_OUTPUT_ARGS=(--output-document="${name}" "${url}")
    wget "${WGET_ARGS[@]}" "${WGET_PROXY_ARGS[@]}"         "${WGET_OUTPUT_ARGS[@]}"
done < files.tsv

echo
echo "Downloads completed."
