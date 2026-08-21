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
#
# Usage:
#   /ABSOLUTE/PATH/TO/REPOSITORY/scripts/download_huggingface.sh \
#     --repo OWNER/DATASET --dest /ABSOLUTE/DESTINATION [OPTIONS]
#
# Common options include --mirror, proxy or --no-proxy, --max-workers,
# --timeout, retry controls, Mihomo controls, and --dry-run. Hugging Face also
# accepts --transport and --xet-range-concurrency. Run with --help for details.


readonly DEFAULT_ENDPOINT="https://huggingface.co"
readonly DEFAULT_PROXY_HOST="127.0.0.1"
readonly DEFAULT_PROXY_SCHEME="http"
readonly DEFAULT_MAX_WORKERS="1"
readonly DEFAULT_TIMEOUT="300"
readonly DEFAULT_TRANSPORT="xet"
readonly DEFAULT_RETRY_ATTEMPTS="8"
readonly DEFAULT_RETRY_BASE_DELAY="5"
readonly DEFAULT_RETRY_MAX_DELAY="300"
readonly DEFAULT_XET_RANGE_CONCURRENCY="16"
readonly DEFAULT_MIHOMO_PROBE_TIMEOUT="8"

REPO_ID=""
DEST=""
HF_ENDPOINT="${DEFAULT_ENDPOINT}"
USE_PROXY=0
PROXY_HOST="${DEFAULT_PROXY_HOST}"
PROXY_PORT=""
PROXY_SCHEME="${DEFAULT_PROXY_SCHEME}"
PROXY_URL=""
MAX_WORKERS="${DEFAULT_MAX_WORKERS}"
TIMEOUT="${DEFAULT_TIMEOUT}"
TRANSPORT="${DEFAULT_TRANSPORT}"
RETRY_ATTEMPTS="${DEFAULT_RETRY_ATTEMPTS}"
RETRY_BASE_DELAY="${DEFAULT_RETRY_BASE_DELAY}"
RETRY_MAX_DELAY="${DEFAULT_RETRY_MAX_DELAY}"
XET_RANGE_CONCURRENCY="${DEFAULT_XET_RANGE_CONCURRENCY}"
MIHOMO_CONTROLLER="${MIHOMO_CONTROLLER-}"
MIHOMO_GROUP="${MIHOMO_GROUP-}"
MIHOMO_NODE_MARKER=""
MIHOMO_SPEED_TEST_URL=""
MIHOMO_PROBE_TIMEOUT="${DEFAULT_MIHOMO_PROBE_TIMEOUT}"
MIHOMO_ENABLED=0
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
        Proxy host used when --proxy-port is provided.
        Default: ${DEFAULT_PROXY_HOST}

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
        Maximum complete-snapshot attempts; 0 retries until Ctrl-C.
        Default: ${DEFAULT_RETRY_ATTEMPTS}

  --retry-base-delay SEC
        Initial complete-snapshot retry delay.
        Default: ${DEFAULT_RETRY_BASE_DELAY}

  --retry-max-delay SEC
        Maximum complete-snapshot retry delay.
        Default: ${DEFAULT_RETRY_MAX_DELAY}

  --xet-range-concurrency N
        Concurrent range requests per file in Xet mode.
        Default: ${DEFAULT_XET_RANGE_CONCURRENCY}

  --mihomo-controller URL
        External-controller URL for ranked node selection.
        Required with --mihomo-speed-test-url unless MIHOMO_CONTROLLER is set.

  --mihomo-group NAME
        Optional selector override containing eligible direct nodes. When
        omitted, discover the selector used by the speed-test URL.

  --mihomo-node-marker TEXT
        Optional literal filter for eligible node names. When omitted, all
        direct nodes in the selector are eligible.

  --mihomo-speed-test-url URL
        HTTPS file used for bounded throughput tests. Supplying this option
        activates ranking; omitting it skips ranking.

  --mihomo-probe-timeout SEC
        Timeout for node probes and bounded speed tests.
        Default: ${DEFAULT_MIHOMO_PROBE_TIMEOUT}

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


is_non_negative_integer()
{
    [[ "$1" =~ ^[0-9]+$ ]]
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
    is_non_negative_integer "${RETRY_ATTEMPTS}" \
        || fail "--retry-attempts must be a non-negative integer."
    is_positive_integer "${RETRY_BASE_DELAY}" \
        || fail "--retry-base-delay must be a positive integer."
    is_positive_integer "${RETRY_MAX_DELAY}" \
        || fail "--retry-max-delay must be a positive integer."
    is_positive_integer "${XET_RANGE_CONCURRENCY}" \
        || fail "--xet-range-concurrency must be a positive integer."
    if [[ -n "${MIHOMO_SPEED_TEST_URL}" ]]; then
        [[ "${USE_PROXY}" == "1" ]] \
            || fail "Mihomo ranking requires --proxy-port."
        [[ -n "${MIHOMO_CONTROLLER}" ]] \
            || fail "Set --mihomo-controller or MIHOMO_CONTROLLER for ranking."
        [[ "${MIHOMO_CONTROLLER}" =~ ^https?://[^[:space:]]+$ ]] \
            || fail "--mihomo-controller must be an HTTP or HTTPS URL."
        [[ "${MIHOMO_SPEED_TEST_URL}" =~ ^https://[^[:space:]]+$ ]] \
            || fail "--mihomo-speed-test-url must be an HTTPS URL."
        is_positive_integer "${MIHOMO_PROBE_TIMEOUT}" \
            || fail "--mihomo-probe-timeout must be a positive integer."
        MIHOMO_ENABLED=1
    fi
}


configure_proxy()
{
    if [[ "${USE_PROXY}" == "1" ]]; then
        PROXY_URL="${PROXY_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"

        export HTTP_PROXY="${PROXY_URL}"
        export HTTPS_PROXY="${PROXY_URL}"
        export ALL_PROXY="${PROXY_URL}"
        export http_proxy="${PROXY_URL}"
        export https_proxy="${PROXY_URL}"
        export all_proxy="${PROXY_URL}"
        unset NO_PROXY no_proxy || true
        return
    fi

    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
    unset http_proxy https_proxy all_proxy || true
}


configure_transport()
{
    if [[ "${TRANSPORT}" == "http" ]]; then
        export HF_HUB_DISABLE_XET=1
        unset HF_XET_NUM_CONCURRENT_RANGE_GETS || true
        unset HF_XET_FIXED_DOWNLOAD_CONCURRENCY || true
        return
    fi

    export HF_HUB_DISABLE_XET=0
    export HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=1
    unset HF_XET_FIXED_DOWNLOAD_CONCURRENCY || true
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
    printf 'Retry delay   : %s-%ss\n' \
        "${RETRY_BASE_DELAY}" "${RETRY_MAX_DELAY}"
    printf '%s\n' 'HTTP TLS      : TLS 1.2 only'

    if [[ "${TRANSPORT}" == "xet" ]]; then
        printf 'Xet ranges    : %s\n' "${XET_RANGE_CONCURRENCY}"
    fi

    if [[ "${MIHOMO_ENABLED}" == "1" ]]; then
        printf 'Mihomo API   : %s\n' "${MIHOMO_CONTROLLER}"
        if [[ -n "${MIHOMO_GROUP}" ]]; then
            printf 'Mihomo group : %s\n' "${MIHOMO_GROUP}"
        fi
        if [[ -n "${MIHOMO_NODE_MARKER}" ]]; then
            printf 'Node filter  : %s\n' "${MIHOMO_NODE_MARKER}"
        else
            printf '%s\n' 'Node filter  : all direct nodes'
        fi
        if [[ -n "${MIHOMO_SECRET-}" ]]; then
            printf '%s\n' 'Mihomo auth  : configured'
        else
            printf '%s\n' 'Mihomo auth  : not configured'
        fi
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
        --retry-base-delay)
            require_option_value "$1" "${2-}"
            RETRY_BASE_DELAY="$2"
            shift 2
            ;;
        --retry-max-delay)
            require_option_value "$1" "${2-}"
            RETRY_MAX_DELAY="$2"
            shift 2
            ;;
        --xet-range-concurrency)
            require_option_value "$1" "${2-}"
            XET_RANGE_CONCURRENCY="$2"
            shift 2
            ;;
        --mihomo-controller)
            require_option_value "$1" "${2-}"
            MIHOMO_CONTROLLER="$2"
            shift 2
            ;;
        --mihomo-group)
            require_option_value "$1" "${2-}"
            MIHOMO_GROUP="$2"
            shift 2
            ;;
        --mihomo-node-marker)
            require_option_value "$1" "${2-}"
            MIHOMO_NODE_MARKER="$2"
            shift 2
            ;;
        --mihomo-speed-test-url)
            require_option_value "$1" "${2-}"
            MIHOMO_SPEED_TEST_URL="$2"
            shift 2
            ;;
        --mihomo-probe-timeout)
            require_option_value "$1" "${2-}"
            MIHOMO_PROBE_TIMEOUT="$2"
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
export DOWNLOAD_XET_RANGE_CONCURRENCY="${XET_RANGE_CONCURRENCY}"
export DOWNLOAD_RETRY_ATTEMPTS="${RETRY_ATTEMPTS}"
export DOWNLOAD_RETRY_BASE_DELAY="${RETRY_BASE_DELAY}"
export DOWNLOAD_RETRY_MAX_DELAY="${RETRY_MAX_DELAY}"
export DOWNLOAD_PROXY_URL="${PROXY_URL}"
export DOWNLOAD_MIHOMO_CONTROLLER="${MIHOMO_CONTROLLER}"
export DOWNLOAD_MIHOMO_GROUP="${MIHOMO_GROUP}"
export DOWNLOAD_MIHOMO_NODE_MARKER="${MIHOMO_NODE_MARKER}"
export DOWNLOAD_MIHOMO_SPEED_TEST_URL="${MIHOMO_SPEED_TEST_URL}"
export DOWNLOAD_MIHOMO_PROBE_TIMEOUT="${MIHOMO_PROBE_TIMEOUT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
python "${REPOSITORY_ROOT}/download_helpers/download_huggingface.py"
