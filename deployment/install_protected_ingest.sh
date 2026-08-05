#!/usr/bin/env bash

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    printf "Run this installer as root.\n" >&2
    exit 1
fi
if [[ $# -ne 1 ]]; then
    printf "Usage: %s LOCAL_CONFIG\n" "$0" >&2
    exit 2
fi

readonly CONFIG_FILE=$1
readonly CONFIG_MODE=600
readonly ROOT_UID=0
readonly ROOT_SQUASHED_UID=65534
readonly TEMPLATE_DIR=$(cd "$(dirname "$0")" && pwd)
if [[ ! -f $CONFIG_FILE ]]; then
    printf "Local configuration does not exist: %s\n" "$CONFIG_FILE" >&2
    exit 2
fi
config_owner_uid=$(stat -c "%u" "$CONFIG_FILE")
config_mode=$(stat -c "%a" "$CONFIG_FILE")
if [[ $config_mode != $CONFIG_MODE ]] ||
    { [[ $config_owner_uid != $ROOT_UID ]] &&
      [[ $config_owner_uid != $ROOT_SQUASHED_UID ]]; }; then
    printf "Local configuration must have mode 600 and be owned by root " >&2
    printf "or this server's root-squashed account.\n" >&2
    exit 2
fi
source "$CONFIG_FILE"

required_values=(
    NDR_DATA_ROOT
    NDR_HELPER_PATH
    NDR_PUBLIC_BRAINCTL_PATH
    NDR_RUNTIME_ROOT
    NDR_SAFE_PATH
    NDR_SERVICE_GROUP
    NDR_SERVICE_HOME
    NDR_SERVICE_SHELL
    NDR_SERVICE_USER
    NDR_SOURCE_ROOTS
    NDR_SUDOERS_PATH
)
for variable_name in "${required_values[@]}"; do
    if [[ -z ${!variable_name:-} ]]; then
        printf "Missing required local setting: %s\n" "$variable_name" >&2
        exit 2
    fi
done

NDR_DATABASE_URL=${NDR_DATABASE_URL:-sqlite:///$NDR_DATA_ROOT/registry/registry.db}
readonly CLI_PATH="$NDR_RUNTIME_ROOT/venv/bin/brainctl"
readonly COMMAND_HELPER_PATH="${NDR_HELPER_PATH%/*}/ndr-brainctl"
readonly ENV_PATH=$(command -v env)
readonly SUDO_PATH=$(command -v sudo)

require_command() {
    local command_name=$1
    if ! command -v "$command_name" >/dev/null; then
        printf "Required command is unavailable: %s\n" "$command_name" >&2
        exit 1
    fi
}

escape_for_sed() {
    printf "%s" "$1" | sed "s/[&|\\]/\\&/g"
}

render_template() {
    local template_path=$1
    local output_path=$2
    sed \
        -e "s|@CLI_PATH@|$(escape_for_sed "$CLI_PATH")|g" \
        -e "s|@DATA_ROOT@|$(escape_for_sed "$NDR_DATA_ROOT")|g" \
        -e "s|@DATABASE_URL@|$(escape_for_sed "$NDR_DATABASE_URL")|g" \
        -e "s|@ENV_PATH@|$(escape_for_sed "$ENV_PATH")|g" \
        -e "s|@HELPER_PATH@|$(escape_for_sed "$NDR_HELPER_PATH")|g" \
        -e "s|@COMMAND_HELPER_PATH@|$(escape_for_sed \
            "$COMMAND_HELPER_PATH")|g" \
        -e "s|@SAFE_PATH@|$(escape_for_sed "$NDR_SAFE_PATH")|g" \
        -e "s|@SERVICE_HOME@|$(escape_for_sed "$NDR_SERVICE_HOME")|g" \
        -e "s|@SERVICE_USER@|$(escape_for_sed "$NDR_SERVICE_USER")|g" \
        -e "s|@SOURCE_ROOTS@|$(escape_for_sed "$NDR_SOURCE_ROOTS")|g" \
        -e "s|@SUDO_PATH@|$(escape_for_sed "$SUDO_PATH")|g" \
        "$template_path" > "$output_path"
}

install_managed_directory() {
    local path=$1
    install -d -o "$NDR_SERVICE_USER" -g "$NDR_SERVICE_GROUP" -m 2750 \
        "$path"
}

protect_control_path() {
    local path=$1
    install_managed_directory "$path"
    chown -R -h "$NDR_SERVICE_USER:$NDR_SERVICE_GROUP" "$path"
    find "$path" -xdev -type f -exec chmod u=rwX,g=rX,o= {} +
    find "$path" -xdev -type d -exec chmod 2750 {} +
}


require_command find
require_command getent
require_command install
require_command mktemp
require_command python3
require_command visudo

if ! getent group "$NDR_SERVICE_GROUP" >/dev/null; then
    groupadd --system "$NDR_SERVICE_GROUP"
fi
if ! id -u "$NDR_SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --gid "$NDR_SERVICE_GROUP" \
        --home-dir "$NDR_SERVICE_HOME" --create-home \
        --shell "$NDR_SERVICE_SHELL" "$NDR_SERVICE_USER"
fi

brainctl_temp=$(mktemp)
helper_temp=$(mktemp)
command_helper_temp=$(mktemp)
sudoers_temp=$(mktemp)
trap "rm -f \"$brainctl_temp\" \"$helper_temp\" \
    \"$command_helper_temp\" \"$sudoers_temp\"" EXIT
render_template "$TEMPLATE_DIR/brainctl.template" "$brainctl_temp"
render_template "$TEMPLATE_DIR/ndr-ingest-local.template" "$helper_temp"
render_template "$TEMPLATE_DIR/ndr-brainctl.template" \
    "$command_helper_temp"
render_template "$TEMPLATE_DIR/ndr-ingest-local.sudoers.template" "$sudoers_temp"
chmod 0755 "$brainctl_temp" "$helper_temp" "$command_helper_temp"
chmod 0440 "$sudoers_temp"
visudo -cf "$sudoers_temp"

printf "Installing isolated registry runtime...\n"
install -d -o root -g root -m 0755 "$NDR_RUNTIME_ROOT"
python3 -m venv "$NDR_RUNTIME_ROOT/venv"
"$NDR_RUNTIME_ROOT/venv/bin/pip" install --upgrade pip
"$NDR_RUNTIME_ROOT/venv/bin/pip" install "$TEMPLATE_DIR/.."
chown -R root:root "$NDR_RUNTIME_ROOT"
chmod -R go-w "$NDR_RUNTIME_ROOT"

printf "Protecting managed registry storage...\n"
install -d -o "$NDR_SERVICE_USER" -g "$NDR_SERVICE_GROUP" -m 0750 \
    "$NDR_SERVICE_HOME"
install -d -o "$NDR_SERVICE_USER" -g "$NDR_SERVICE_GROUP" -m 0711 \
    "$NDR_DATA_ROOT"
for directory_name in datasets incoming quarantine; do
    install_managed_directory "$NDR_DATA_ROOT/$directory_name"
done
for directory_name in registry logs; do
    protect_control_path "$NDR_DATA_ROOT/$directory_name"
done

printf "Validating source roots without traversing them...\n"
IFS=: read -r -a source_roots <<< "$NDR_SOURCE_ROOTS"
for source_root in "${source_roots[@]}"; do
    if [[ ! -d $source_root ]]; then
        printf "Configured source root is missing: %s\n" "$source_root" >&2
        exit 1
    fi
done

install -d -o root -g root -m 0755 \
    "$(dirname "$NDR_PUBLIC_BRAINCTL_PATH")" \
    "$(dirname "$NDR_HELPER_PATH")" \
    "$(dirname "$NDR_SUDOERS_PATH")"
install -o root -g root -m 0755 "$brainctl_temp" \
    "$NDR_PUBLIC_BRAINCTL_PATH"
install -o root -g root -m 0755 "$helper_temp" "$NDR_HELPER_PATH"
install -o root -g root -m 0755 "$command_helper_temp" \
    "$COMMAND_HELPER_PATH"
install -o root -g root -m 0440 "$sudoers_temp" "$NDR_SUDOERS_PATH"
