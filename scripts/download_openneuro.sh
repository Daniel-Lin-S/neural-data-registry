#!/usr/bin/env bash

set -Eeuo pipefail

# Download one OpenNeuro snapshot with the shared downloader interface.
#
# Usage:
#   /ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_openneuro.sh \
#     --repo ID_OR_VERSION_URL --dest /ABSOLUTE/DESTINATION [OPTIONS]
#
# Options include --mirror, proxy or --no-proxy, --max-workers, --timeout,
# retry controls, Mihomo controls, and --dry-run. Run with --help for complete
# argument details.

readonly PROVIDER_LABEL="OpenNeuro"
readonly PROVIDER_DEFAULT_ENDPOINT="https://github.com/OpenNeuroDatasets"
readonly PROVIDER_REPO_HELP="OpenNeuro dataset ID, dataset URL, or version URL."
readonly PROVIDER_RESUME_MESSAGE="Interrupted DataLad downloads resume "\
"in the same dataset."
readonly PROVIDER_PYTHON_MODULE="download_openneuro.py"
PROVIDER_SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)"
readonly PROVIDER_SCRIPT_DIR

source "${PROVIDER_SCRIPT_DIR}/download_provider_common.sh"
