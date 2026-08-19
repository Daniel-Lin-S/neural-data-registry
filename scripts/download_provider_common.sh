#!/usr/bin/env bash

# Parse and validate the shared HTTP-provider downloader interface.
#
# Input:
#   Provider constants set by a thin entrypoint and user CLI options.
#
# Output:
#   Validated DOWNLOAD_* environment variables passed to the provider's
#   Python companion module.

readonly DEFAULT_PROXY_HOST="127.0.0.1"
readonly DEFAULT_PROXY_SCHEME="http"
readonly DEFAULT_MAX_WORKERS="1"
readonly DEFAULT_TIMEOUT="300"
readonly DEFAULT_RETRY_ATTEMPTS="8"
readonly DEFAULT_RETRY_BASE_DELAY="5"
readonly DEFAULT_RETRY_MAX_DELAY="300"
readonly DEFAULT_MIHOMO_PROBE_TIMEOUT="8"

REPO_ID=""
DEST=""
ENDPOINT="${PROVIDER_DEFAULT_ENDPOINT}"
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

  $0 --repo ID_OR_URL --dest PATH [OPTIONS]

Required:

  --repo ID_OR_URL
        ${PROVIDER_REPO_HELP}

  --dest PATH
        Absolute destination directory for the dataset.

Optional:

  --mirror URL
        ${PROVIDER_LABEL} endpoint.
        Default: ${PROVIDER_DEFAULT_ENDPOINT}

  --proxy-port PORT
        Enable the proxy with this port.

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
        Provider operation timeout in seconds.
        Default: ${DEFAULT_TIMEOUT}

  --retry-attempts N
        Maximum attempts per operation; 0 retries until Ctrl-C.
        Default: ${DEFAULT_RETRY_ATTEMPTS}

  --retry-base-delay SEC
        Initial exponential retry delay.
        Default: ${DEFAULT_RETRY_BASE_DELAY}

  --retry-max-delay SEC
        Maximum exponential retry delay.
        Default: ${DEFAULT_RETRY_MAX_DELAY}

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
        Show pending content without downloading.

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
    [[ "${ENDPOINT}" =~ ^https?://[^[:space:]]+$ ]] \
        || fail "--mirror must be an HTTP or HTTPS URL."

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

    if [[ -n "${MIHOMO_SPEED_TEST_URL}" ]]; then
        [[ "${USE_PROXY}" == "1" ]] \
            || fail "Mihomo ranking requires --proxy-port."
        [[ -n "${MIHOMO_CONTROLLER}" ]] \
            || fail "Set --mihomo-controller or MIHOMO_CONTROLLER for ranking."
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
        export HTTP_PROXY="${PROXY_URL}"
        export HTTPS_PROXY="${PROXY_URL}"
        export ALL_PROXY="${PROXY_URL}"
        export http_proxy="${PROXY_URL}"
        export https_proxy="${PROXY_URL}"
        export all_proxy="${PROXY_URL}"
        return
    fi

    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
    unset http_proxy https_proxy all_proxy || true
}


print_configuration()
{
    printf '\n%s\n' '=============================================='
    printf ' %s Dataset Download\n' "${PROVIDER_LABEL}"
    printf '%s\n' '=============================================='
    printf 'Repository    : %s\n' "${REPO_ID}"
    printf 'Destination   : %s\n' "${DEST}"
    printf 'Endpoint      : %s\n' "${ENDPOINT}"

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

    if [[ "${MIHOMO_ENABLED}" == "1" ]]; then
        printf 'Mihomo API    : %s\n' "${MIHOMO_CONTROLLER}"
        if [[ -n "${MIHOMO_GROUP}" ]]; then
            printf 'Mihomo group  : %s\n' "${MIHOMO_GROUP}"
        fi
        if [[ -n "${MIHOMO_NODE_MARKER}" ]]; then
            printf 'Node filter   : %s\n' "${MIHOMO_NODE_MARKER}"
        else
            printf '%s\n' 'Node filter   : all direct nodes'
        fi
    fi
    printf '\n%s\n\n' "${PROVIDER_RESUME_MESSAGE}"
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo|--dest|--mirror|--proxy-port|--proxy-host|--proxy-scheme|\
        --max-workers|--timeout|--retry-attempts|--retry-base-delay|\
        --retry-max-delay|--mihomo-controller|--mihomo-group|\
        --mihomo-node-marker|--mihomo-speed-test-url|\
        --mihomo-probe-timeout)
            require_option_value "$1" "${2-}"
            case "$1" in
                --repo) REPO_ID="$2" ;;
                --dest) DEST="$2" ;;
                --mirror) ENDPOINT="$2" ;;
                --proxy-port) USE_PROXY=1; PROXY_PORT="$2" ;;
                --proxy-host) USE_PROXY=1; PROXY_HOST="$2" ;;
                --proxy-scheme) USE_PROXY=1; PROXY_SCHEME="$2" ;;
                --max-workers) MAX_WORKERS="$2" ;;
                --timeout) TIMEOUT="$2" ;;
                --retry-attempts) RETRY_ATTEMPTS="$2" ;;
                --retry-base-delay) RETRY_BASE_DELAY="$2" ;;
                --retry-max-delay) RETRY_MAX_DELAY="$2" ;;
                --mihomo-controller) MIHOMO_CONTROLLER="$2" ;;
                --mihomo-group) MIHOMO_GROUP="$2" ;;
                --mihomo-node-marker) MIHOMO_NODE_MARKER="$2" ;;
                --mihomo-speed-test-url) MIHOMO_SPEED_TEST_URL="$2" ;;
                --mihomo-probe-timeout) MIHOMO_PROBE_TIMEOUT="$2" ;;
            esac
            shift 2
            ;;
        --no-proxy)
            USE_PROXY=0
            PROXY_PORT=""
            shift
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
print_configuration

export DOWNLOAD_REPO_ID="${REPO_ID}"
export DOWNLOAD_DEST="${DEST}"
export DOWNLOAD_ENDPOINT="${ENDPOINT}"
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

readonly PROVIDER_REPOSITORY_ROOT="$({
    cd -- "${PROVIDER_SCRIPT_DIR}/.." && pwd -P
})"
python \
    "${PROVIDER_REPOSITORY_ROOT}/download_helpers/${PROVIDER_PYTHON_MODULE}"
