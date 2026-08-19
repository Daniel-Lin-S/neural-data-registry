#!/usr/bin/env bash

set -Eeuo pipefail

# Download one Zenodo record with the shared downloader interface.
#
# Usage:
#   /ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_zenodo.sh \
#     --repo ID_OR_URL --dest /ABSOLUTE/DESTINATION [OPTIONS]
#
# Options include --mirror, proxy or --no-proxy, --max-workers, --timeout,
# retry controls, Mihomo controls, and --dry-run. Run with --help for complete
# argument details.

readonly PROVIDER_LABEL="Zenodo"
readonly PROVIDER_DEFAULT_ENDPOINT="https://zenodo.org/api"
readonly PROVIDER_REPO_HELP="Zenodo record ID or record URL."
readonly PROVIDER_RESUME_MESSAGE="Interrupted downloads resume from "\
"sibling .part files."
readonly PROVIDER_PYTHON_MODULE="download_zenodo.py"
PROVIDER_SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)"
readonly PROVIDER_SCRIPT_DIR

source "${PROVIDER_SCRIPT_DIR}/download_provider_common.sh"
