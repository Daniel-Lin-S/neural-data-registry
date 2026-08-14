#!/usr/bin/env bash

set -Eeuo pipefail


# Download a Hugging Face dataset snapshot into a local directory.
#
# Input:
#   A dataset repository ID, destination directory, endpoint, optional proxy,
#   transport mode, concurrency, timeout, and retry settings.
#
# Output:
#   The repository files beneath --dest. Existing partial downloads are kept
#   and reused by later attempts and later invocations.
#
# For long downloads through Mihomo, pin one stable node. Do not use an
# automatic URL-test group that may change nodes during active transfers.


readonly DEFAULT_ENDPOINT="https://huggingface.co"
readonly DEFAULT_PROXY_SCHEME="http"
readonly DEFAULT_MAX_WORKERS="2"
readonly DEFAULT_TIMEOUT="300"
readonly DEFAULT_TRANSPORT="http"
readonly DEFAULT_RETRY_ATTEMPTS="8"
readonly DEFAULT_XET_RANGE_CONCURRENCY="2"
readonly RETRY_BASE_DELAY="5"
readonly RETRY_MAX_DELAY="60"

REPO_ID=""
DEST=""
HF_ENDPOINT="${DEFAULT_ENDPOINT}"
USE_PROXY=0
PROXY_HOST=""
PROXY_PORT=""
PROXY_SCHEME="${DEFAULT_PROXY_SCHEME}"
MAX_WORKERS="${DEFAULT_MAX_WORKERS}"
TIMEOUT="${DEFAULT_TIMEOUT}"
TRANSPORT="${DEFAULT_TRANSPORT}"
RETRY_ATTEMPTS="${DEFAULT_RETRY_ATTEMPTS}"
XET_RANGE_CONCURRENCY="${DEFAULT_XET_RANGE_CONCURRENCY}"
DRY_RUN=0


show_help()
{
    cat <<EOF

Usage:

  $0 --repo REPO_ID --dest PATH [OPTIONS]

Required:

  --repo REPO_ID
        Hugging Face dataset repository.

  --dest PATH
        Destination directory for the dataset.

Optional:

  --mirror URL
        Hugging Face endpoint.
        Default: ${DEFAULT_ENDPOINT}

  --proxy-port PORT
        Enable the proxy with this port.
        Required with --proxy-host.

  --proxy-host HOST
        Enable the proxy with this host.
        Required with --proxy-port.

  --proxy-scheme SCHEME
        Enable the proxy with http, https, socks5, or socks5h.
        Default when enabled: ${DEFAULT_PROXY_SCHEME}

  --no-proxy
        Disable the proxy and ignore ambient proxy variables.

  --max-workers N
        Number of files downloaded concurrently.
        Default: ${DEFAULT_MAX_WORKERS}

  --timeout SEC
        HTTP download timeout in seconds.
        Default: ${DEFAULT_TIMEOUT}

  --transport MODE
        File transport: http or xet.
        Default: ${DEFAULT_TRANSPORT}

  --retry-attempts N
        Maximum complete-snapshot attempts.
        Default: ${DEFAULT_RETRY_ATTEMPTS}

  --xet-range-concurrency N
        Concurrent range requests per file in Xet mode.
        Default: ${DEFAULT_XET_RANGE_CONCURRENCY}

  --dry-run
        Show pending files without downloading.

  -h, --help
        Show this message.

EOF
}


fail()
{
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}


require_option_value()
{
    local option="$1"
    local value="${2-}"

    if [[ -z "${value}" || "${value}" == --* ]]; then
        fail "${option} requires a value."
    fi
}


is_positive_integer()
{
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}


validate_configuration()
{
    [[ -n "${REPO_ID}" ]] || fail "--repo is required."
    [[ -n "${DEST}" ]] || fail "--dest is required."
    [[ "${DEST}" == /* ]] \
        || fail "--dest must be an absolute path."

    if [[ ! "${HF_ENDPOINT}" =~ ^https?://[^[:space:]]+$ ]]; then
        fail "--mirror must be an HTTP or HTTPS URL."
    fi

    case "${PROXY_SCHEME}" in
        http|https|socks5|socks5h)
            ;;
        *)
            fail "--proxy-scheme must be http, https, socks5, or socks5h."
            ;;
    esac

    case "${TRANSPORT}" in
        http|xet)
            ;;
        *)
            fail "--transport must be http or xet."
            ;;
    esac

    if [[ "${USE_PROXY}" == "1" ]]; then
        [[ -n "${PROXY_HOST}" ]] \
            || fail "--proxy-host is required when proxying is enabled."
        [[ -n "${PROXY_PORT}" ]] \
            || fail "--proxy-port is required when proxying is enabled."
        is_positive_integer "${PROXY_PORT}" \
            || fail "--proxy-port must be a positive integer."
        if (( PROXY_PORT > 65535 )); then
            fail "--proxy-port must not exceed 65535."
        fi
    fi
    is_positive_integer "${MAX_WORKERS}" \
        || fail "--max-workers must be a positive integer."
    is_positive_integer "${TIMEOUT}" \
        || fail "--timeout must be a positive integer."
    is_positive_integer "${RETRY_ATTEMPTS}" \
        || fail "--retry-attempts must be a positive integer."
    is_positive_integer "${XET_RANGE_CONCURRENCY}" \
        || fail "--xet-range-concurrency must be a positive integer."
}


configure_proxy()
{
    if [[ "${USE_PROXY}" == "1" ]]; then
        local proxy_url
        proxy_url="${PROXY_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"

        export HTTP_PROXY="${proxy_url}"
        export HTTPS_PROXY="${proxy_url}"
        export http_proxy="${proxy_url}"
        export https_proxy="${proxy_url}"
        return
    fi

    unset HTTP_PROXY HTTPS_PROXY || true
    unset http_proxy https_proxy || true
}


configure_transport()
{
    if [[ "${TRANSPORT}" == "http" ]]; then
        export HF_HUB_DISABLE_XET=1
        unset HF_XET_NUM_CONCURRENT_RANGE_GETS || true
        return
    fi

    export HF_HUB_DISABLE_XET=0
    export HF_XET_NUM_CONCURRENT_RANGE_GETS="${XET_RANGE_CONCURRENCY}"
}


check_dependencies()
{
    if python - <<'PY' >/dev/null 2>&1
import httpx
import huggingface_hub
PY
    then
        return
    fi

    fail "Install required packages with: "\
"pip install -U huggingface_hub httpx"
}


print_configuration()
{
    printf '\n%s\n' '=============================================='
    printf '%s\n' ' Hugging Face Dataset Download'
    printf '%s\n' '=============================================='
    printf 'Repository    : %s\n' "${REPO_ID}"
    printf 'Destination   : %s\n' "${DEST}"
    printf 'Endpoint      : %s\n' "${HF_ENDPOINT}"

    if [[ "${USE_PROXY}" == "1" ]]; then
        printf 'Proxy         : %s://%s:%s\n' \
            "${PROXY_SCHEME}" "${PROXY_HOST}" "${PROXY_PORT}"
    else
        printf '%s\n' 'Proxy         : disabled'
    fi

    printf 'Transport     : %s\n' "${TRANSPORT}"
    printf 'Workers       : %s\n' "${MAX_WORKERS}"
    printf 'Timeout       : %ss\n' "${TIMEOUT}"
    printf 'Retry attempts: %s\n' "${RETRY_ATTEMPTS}"
    printf '%s\n' 'HTTP TLS      : TLS 1.2 only'

    if [[ "${TRANSPORT}" == "xet" ]]; then
        printf 'Xet ranges    : %s\n' "${XET_RANGE_CONCURRENCY}"
    fi

    printf '\n%s\n\n' \
        'Interrupted downloads resume from existing partial files.'
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            require_option_value "$1" "${2-}"
            DEST="$2"
            shift 2
            ;;
        --repo)
            require_option_value "$1" "${2-}"
            REPO_ID="$2"
            shift 2
            ;;
        --mirror)
            require_option_value "$1" "${2-}"
            HF_ENDPOINT="$2"
            shift 2
            ;;
        --proxy-port)
            require_option_value "$1" "${2-}"
            USE_PROXY=1
            PROXY_PORT="$2"
            shift 2
            ;;
        --proxy-host)
            require_option_value "$1" "${2-}"
            USE_PROXY=1
            PROXY_HOST="$2"
            shift 2
            ;;
        --proxy-scheme)
            require_option_value "$1" "${2-}"
            USE_PROXY=1
            PROXY_SCHEME="$2"
            shift 2
            ;;
        --no-proxy)
            USE_PROXY=0
            shift
            ;;
        --max-workers)
            require_option_value "$1" "${2-}"
            MAX_WORKERS="$2"
            shift 2
            ;;
        --timeout)
            require_option_value "$1" "${2-}"
            TIMEOUT="$2"
            shift 2
            ;;
        --transport)
            require_option_value "$1" "${2-}"
            TRANSPORT="$2"
            shift 2
            ;;
        --retry-attempts)
            require_option_value "$1" "${2-}"
            RETRY_ATTEMPTS="$2"
            shift 2
            ;;
        --xet-range-concurrency)
            require_option_value "$1" "${2-}"
            XET_RANGE_CONCURRENCY="$2"
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
            fail "Unknown option: $1"
            ;;
    esac
done


validate_configuration
configure_proxy
configure_transport

export HF_ENDPOINT
export HF_HUB_DOWNLOAD_TIMEOUT="${TIMEOUT}"

check_dependencies

mkdir -p "${DEST}"
DEST="$(cd -- "${DEST}" && pwd -P)"

print_configuration

export DOWNLOAD_REPO_ID="${REPO_ID}"
export DOWNLOAD_DEST="${DEST}"
export DOWNLOAD_ENDPOINT="${HF_ENDPOINT}"
export DOWNLOAD_MAX_WORKERS="${MAX_WORKERS}"
export DOWNLOAD_TIMEOUT="${TIMEOUT}"
export DOWNLOAD_DRY_RUN="${DRY_RUN}"
export DOWNLOAD_TRANSPORT="${TRANSPORT}"
export DOWNLOAD_RETRY_ATTEMPTS="${RETRY_ATTEMPTS}"
export DOWNLOAD_RETRY_BASE_DELAY="${RETRY_BASE_DELAY}"
export DOWNLOAD_RETRY_MAX_DELAY="${RETRY_MAX_DELAY}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python "${SCRIPT_DIR}/download_huggingface.py"
