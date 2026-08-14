#!/usr/bin/env bash

set -Eeuo pipefail


# Download one OSF storage tree into a local directory.
#
# Input:
#   An OSF project ID or URL, absolute destination, optional storage provider,
#   proxy, concurrency, timeout, retry, and Mihomo ranking settings.
#
# Output:
#   The advertised OSF file tree beneath --dest. Partial bodies use .part
#   files and are retained for retry and resume operations.


readonly DEFAULT_ENDPOINT="https://api.osf.io/v2"
readonly DEFAULT_STORAGE="osfstorage"
readonly DEFAULT_PROXY_HOST="127.0.0.1"
readonly DEFAULT_PROXY_SCHEME="http"
readonly DEFAULT_MAX_WORKERS="1"
readonly DEFAULT_TIMEOUT="300"
readonly DEFAULT_RETRY_ATTEMPTS="8"
readonly DEFAULT_RETRY_BASE_DELAY="5"
readonly DEFAULT_RETRY_MAX_DELAY="300"
readonly DEFAULT_MIHOMO_PROBE_TIMEOUT="15"

PROJECT=""
DEST=""
OSF_ENDPOINT="${DEFAULT_ENDPOINT}"
STORAGE="${DEFAULT_STORAGE}"
USE_PROXY=0
PROXY_HOST="${DEFAULT_PROXY_HOST}"
PROXY_PORT=""
PROXY_SCHEME="${DEFAULT_PROXY_SCHEME}"
PROXY_URL=""
MAX_WORKERS="${DEFAULT_MAX_WORKERS}"
TIMEOUT="${DEFAULT_TIMEOUT}"
RETRY_ATTEMPTS="${DEFAULT_RETRY_ATTEMPTS}"
RETRY_BASE_DELAY="${DEFAULT_RETRY_BASE_DELAY}"
RETRY_MAX_DELAY="${DEFAULT_RETRY_MAX_DELAY}"
MIHOMO_CONTROLLER=""
MIHOMO_GROUP=""
MIHOMO_NODE_MARKER=""
MIHOMO_SPEED_TEST_URL=""
MIHOMO_PROBE_TIMEOUT="${DEFAULT_MIHOMO_PROBE_TIMEOUT}"
MIHOMO_ENABLED=0
DRY_RUN=0


show_help()
{
    cat <<EOF

Usage:

  $0 --project ID_OR_URL --dest PATH [OPTIONS]

Required:

  --project ID_OR_URL
        OSF node ID or project URL, such as ag3kj or
        https://osf.io/ag3kj/overview.

  --dest PATH
        Absolute destination directory for the OSF file tree.

Optional:

  --api-base URL
        OSF-compatible API v2 base URL.
        Default: ${DEFAULT_ENDPOINT}

  --storage NAME
        OSF storage provider to download.
        Default: ${DEFAULT_STORAGE}

  --proxy-port PORT
        Enable an explicit proxy with this port. Omit for direct access.

  --proxy-host HOST
        Proxy host used when --proxy-port is provided.
        Default: ${DEFAULT_PROXY_HOST}

  --proxy-scheme SCHEME
        Proxy scheme: http, https, socks5, or socks5h.
        Default when enabled: ${DEFAULT_PROXY_SCHEME}

  --no-proxy
        Disable the explicit proxy.

  --max-workers N
        Number of files downloaded concurrently.
        Default: ${DEFAULT_MAX_WORKERS}

  --timeout SEC
        Metadata and file request timeout in seconds.
        Default: ${DEFAULT_TIMEOUT}

  --retry-attempts N
        Attempts per manifest or file; 0 retries until Ctrl-C.
        Default: ${DEFAULT_RETRY_ATTEMPTS}

  --retry-base-delay SEC
        Initial exponential retry delay.
        Default: ${DEFAULT_RETRY_BASE_DELAY}

  --retry-max-delay SEC
        Maximum exponential retry delay.
        Default: ${DEFAULT_RETRY_MAX_DELAY}

  --mihomo-controller URL
        External-controller URL for ranked node selection. Required only
        when both --mihomo-group and --mihomo-speed-test-url are set.

  --mihomo-group NAME
        Selector containing eligible direct nodes. Omit to skip ranking.

  --mihomo-node-marker TEXT
        Optional literal filter for eligible node names. When omitted, all
        direct nodes in the selector are eligible.

  --mihomo-speed-test-url URL
        Large HTTPS file used for bounded throughput tests. Omit to skip
        ranking.

  --mihomo-probe-timeout SEC
        Timeout for node probes and bounded speed tests.
        Default: ${DEFAULT_MIHOMO_PROBE_TIMEOUT}

  --dry-run
        List pending files and sizes without creating --dest.

  -h, --help
        Show this message.

Authentication:

  Export OSF_TOKEN for a private project. The token is never printed.

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
    [[ -n "${PROJECT}" ]] || fail "--project is required."
    [[ -n "${DEST}" ]] || fail "--dest is required."
    [[ "${DEST}" == /* ]] \
        || fail "--dest must be an absolute path."

    if [[ ! "${OSF_ENDPOINT}" =~ ^https?://[^[:space:]]+$ ]]; then
        fail "--api-base must be an HTTP or HTTPS URL."
    fi
    [[ "${STORAGE}" =~ ^[A-Za-z0-9]+$ ]] \
        || fail "--storage must be alphanumeric."

    case "${PROXY_SCHEME}" in
        http|https|socks5|socks5h)
            ;;
        *)
            fail "--proxy-scheme must be http, https, socks5, or socks5h."
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
    if [[ -n "${MIHOMO_GROUP}" && -n "${MIHOMO_SPEED_TEST_URL}" ]]; then
        [[ -n "${MIHOMO_CONTROLLER}" ]] \
            || fail "Mihomo ranking requires --mihomo-controller."
        [[ "${USE_PROXY}" == "1" ]] \
            || fail "Mihomo ranking requires --proxy-port."
        [[ "${MAX_WORKERS}" == "1" ]] \
            || fail "Mihomo ranking requires --max-workers 1."
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
    fi
}


check_dependencies()
{
    if python - <<'PY' >/dev/null 2>&1
import httpx
PY
    then
        return
    fi

    fail "Install the required package with: pip install -U httpx"
}


print_configuration()
{
    printf '\n%s\n' '=============================================='
    printf '%s\n' ' OSF Dataset Download'
    printf '%s\n' '=============================================='
    printf 'Project       : %s\n' "${PROJECT}"
    printf 'Storage       : %s\n' "${STORAGE}"
    printf 'Destination   : %s\n' "${DEST}"
    printf 'API base      : %s\n' "${OSF_ENDPOINT}"

    if [[ "${USE_PROXY}" == "1" ]]; then
        printf 'Proxy         : %s://%s:%s\n' \
            "${PROXY_SCHEME}" "${PROXY_HOST}" "${PROXY_PORT}"
    else
        printf '%s\n' 'Proxy         : disabled'
    fi

    printf 'Workers       : %s\n' "${MAX_WORKERS}"
    printf 'Timeout       : %ss\n' "${TIMEOUT}"
    printf 'Retry attempts: %s\n' "${RETRY_ATTEMPTS}"
    printf 'Retry delay   : %s-%ss\n' \
        "${RETRY_BASE_DELAY}" "${RETRY_MAX_DELAY}"
    printf '%s\n' 'HTTP TLS      : TLS 1.2 only'

    if [[ "${MIHOMO_ENABLED}" == "1" ]]; then
        printf 'Mihomo API    : %s\n' "${MIHOMO_CONTROLLER}"
        printf 'Mihomo group  : %s\n' "${MIHOMO_GROUP}"
        if [[ -n "${MIHOMO_NODE_MARKER}" ]]; then
            printf 'Node filter   : %s\n' "${MIHOMO_NODE_MARKER}"
        else
            printf '%s\n' 'Node filter   : all direct nodes'
        fi
        if [[ -n "${MIHOMO_SECRET-}" ]]; then
            printf '%s\n' 'Mihomo auth   : configured'
        else
            printf '%s\n' 'Mihomo auth   : not configured'
        fi
    fi

    printf '\n%s\n\n' \
        'Interrupted downloads resume from sibling .part files.'
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)
            require_option_value "$1" "${2-}"
            PROJECT="$2"
            shift 2
            ;;
        --dest)
            require_option_value "$1" "${2-}"
            DEST="$2"
            shift 2
            ;;
        --api-base)
            require_option_value "$1" "${2-}"
            OSF_ENDPOINT="$2"
            shift 2
            ;;
        --storage)
            require_option_value "$1" "${2-}"
            STORAGE="$2"
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
check_dependencies
print_configuration

export DOWNLOAD_PROJECT="${PROJECT}"
export DOWNLOAD_DEST="${DEST}"
export DOWNLOAD_ENDPOINT="${OSF_ENDPOINT}"
export DOWNLOAD_STORAGE="${STORAGE}"
export DOWNLOAD_MAX_WORKERS="${MAX_WORKERS}"
export DOWNLOAD_TIMEOUT="${TIMEOUT}"
export DOWNLOAD_DRY_RUN="${DRY_RUN}"
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
python "${SCRIPT_DIR}/download_osf.py"
