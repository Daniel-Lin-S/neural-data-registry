#!/usr/bin/env bash

set -Eeuo pipefail

# Download a public Figshare article using the shared HTTP interface.
# Usage: /ABSOLUTE/REPOSITORY/scripts/download_figshare.sh --repo ID_OR_URL
#        --dest /ABSOLUTE/DESTINATION --no-proxy
readonly PROVIDER_SUPPORTS_STANDALONE="1"
readonly PROVIDER_SUPPORTS_MIHOMO_FALLBACK="1"
readonly PROVIDER_LABEL="Figshare"
readonly PROVIDER_DEFAULT_ENDPOINT="https://api.figshare.com/v2"
readonly PROVIDER_REPO_HELP="Public Figshare article ID or article/version URL."
readonly PROVIDER_RESUME_MESSAGE="Resume .part files; verify size and MD5."
readonly PROVIDER_PYTHON_MODULE="download_figshare.py"
readonly PROVIDER_DEFAULT_RETRY_ATTEMPTS="0"
PROVIDER_SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)"
readonly PROVIDER_SCRIPT_DIR
export PYTHONPATH="${PROVIDER_SCRIPT_DIR}/../src${PYTHONPATH:+:${PYTHONPATH}}"
source "${PROVIDER_SCRIPT_DIR}/download_provider_common.sh"
