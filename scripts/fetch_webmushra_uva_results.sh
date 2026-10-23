#!/usr/bin/env bash

set -euo pipefail

DEFAULT_HOST="illc-aml-organ.science.uva.nl"
DEFAULT_USER="njaffe"
DEFAULT_PORT="22"
DEFAULT_PODMAN_USER="podman"

HOST="${DEFAULT_HOST}"
DEPLOY_USER="${DEFAULT_USER}"
PODMAN_USER="${DEFAULT_PODMAN_USER}"
PODMAN_DIR=""
PORT="${DEFAULT_PORT}"
IDENTITY_FILE=""
VERBOSE=0
PODMAN_DIR_EXPLICIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_ROOT="${PROJECT_ROOT}/outputs"

default_podman_dir() {
  printf '/home/%s/webMUSHRA' "$1"
}

timestamp_now() {
  date '+%Y-%m-%d_%H-%M'
}

log() {
  printf '%s\n' "$*"
}

log_verbose() {
  if [[ "${VERBOSE}" -eq 1 ]]; then
    printf '%s\n' "$*"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Fetch the remote webMUSHRA results/ directory from the UvA server into a new local
timestamped folder under outputs/.

Options:
  --host HOST             SSH host. Default: ${DEFAULT_HOST}
  --user USER             SSH user. Default: ${DEFAULT_USER}
  --remote-dir PATH       Podman-user webMUSHRA repo directory. Default: /home/<podman-user>/webMUSHRA
  --podman-user USER      Account that owns/runs webMUSHRA. Default: ${DEFAULT_PODMAN_USER}
  --port PORT             SSH port. Default: ${DEFAULT_PORT}
  --identity PATH         SSH private key to use explicitly.
  --verbose               Show more SSH/rsync output.
  -h, --help              Show this help text.

Examples:
  $(basename "$0")
  $(basename "$0") --verbose
  $(basename "$0") --identity ~/.ssh/id_rsa
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      DEPLOY_USER="$2"
      shift 2
      ;;
    --remote-dir)
      PODMAN_DIR="$2"
      PODMAN_DIR_EXPLICIT=1
      shift 2
      ;;
    --podman-user)
      PODMAN_USER="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --identity)
      IDENTITY_FILE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PODMAN_DIR_EXPLICIT}" -eq 0 ]]; then
  PODMAN_DIR="$(default_podman_dir "${PODMAN_USER}")"
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not installed." >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but not installed." >&2
  exit 1
fi

if [[ -n "${IDENTITY_FILE}" && ! -f "${IDENTITY_FILE}" ]]; then
  echo "SSH identity file not found: ${IDENTITY_FILE}" >&2
  exit 1
fi

SSH_TARGET="${DEPLOY_USER}@${HOST}"
SSH_OPTS=(-p "${PORT}" -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
SSH_LOG_LEVEL="ERROR"
RSYNC_PROGRESS_FLAGS=(--stats)

if [[ -n "${IDENTITY_FILE}" ]]; then
  SSH_OPTS+=(-i "${IDENTITY_FILE}" -o IdentitiesOnly=yes)
fi

if [[ "${VERBOSE}" -eq 1 ]]; then
  SSH_OPTS+=(-v)
  SSH_LOG_LEVEL="VERBOSE"
  RSYNC_PROGRESS_FLAGS=(--progress --stats)
fi

SSH_OPTS+=(-o "LogLevel=${SSH_LOG_LEVEL}")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"

timestamp="$(timestamp_now)"
dest_dir="${DEST_ROOT}/webmushra_results_${timestamp}"
remote_staging_dir="/tmp/webmushra_results_staging"
mkdir -p "${dest_dir}"

log "Fetch SSH target: ${SSH_TARGET}"
log "Fetch source: ${PODMAN_USER}:${PODMAN_DIR}/results/"
log "Fetch destination: ${dest_dir}/results/"

mkdir -p "${dest_dir}/results"

# Write the remote helper script to a local temp file
remote_helper="$(mktemp "${TMPDIR:-/tmp}/webmushra-fetch-helper.XXXXXX")"
trap 'rm -f "$remote_helper"' EXIT

cat > "$remote_helper" <<'HELPER_EOF'
set -euo pipefail

handoff_dir="/tmp/webmushra-fetch-handoff"

# Write the podman copy script
podman_copy_script="$(mktemp /tmp/webmushra-fetch-copy.XXXXXX)"
cat > "$podman_copy_script" <<'PODMAN_EOF'
set -euo pipefail

if [[ ! -d "$PODMAN_DIR/results" ]]; then
  echo "Remote results directory not found: $PODMAN_DIR/results" >&2
  exit 1
fi

rm -rf "$HANDOFF_DIR"
mkdir -p "$HANDOFF_DIR/results"
cp -r "$PODMAN_DIR/results/." "$HANDOFF_DIR/results/"
chmod -R a+rX "$HANDOFF_DIR"
PODMAN_EOF
chmod a+rx "$podman_copy_script"

# Write the podman cleanup script
podman_cleanup_script="$(mktemp /tmp/webmushra-fetch-cleanup.XXXXXX)"
cat > "$podman_cleanup_script" <<'PODMAN_CLEANUP_EOF'
set -euo pipefail
rm -rf "$HANDOFF_DIR"
PODMAN_CLEANUP_EOF
chmod a+rx "$podman_cleanup_script"

# Run as podman to copy results into the handoff dir
copy_cmd="$(printf 'PODMAN_DIR=%q HANDOFF_DIR=%q bash %q' "$PODMAN_DIR" "$handoff_dir" "$podman_copy_script")"
copy_stdin="$(mktemp /tmp/webmushra-fetch-su-copy.XXXXXX)"
printf '%s\n' "$copy_cmd" > "$copy_stdin"
if ! sudo su - "$PODMAN_USER" < "$copy_stdin"; then
  echo "Failed to switch to ${PODMAN_USER} with sudo su and copy webMUSHRA results." >&2
  rm -f "$podman_copy_script" "$podman_cleanup_script" "$copy_stdin"
  exit 1
fi
rm -f "$podman_copy_script" "$copy_stdin"

# Copy from podman-owned handoff to njaffe-owned staging for rsync
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -r "$handoff_dir/results" "$STAGING_DIR/results"

# Clean up the podman-owned handoff dir via sudo su - podman
cleanup_cmd="$(printf 'HANDOFF_DIR=%q bash %q' "$handoff_dir" "$podman_cleanup_script")"
cleanup_stdin="$(mktemp /tmp/webmushra-fetch-su-cleanup.XXXXXX)"
printf '%s\n' "$cleanup_cmd" > "$cleanup_stdin"
sudo su - "$PODMAN_USER" < "$cleanup_stdin" || true
rm -f "$podman_cleanup_script" "$cleanup_stdin"
HELPER_EOF
chmod 700 "$remote_helper"

# Upload the helper script and execute it remotely with a TTY for the su password prompt
remote_helper_path="/home/${DEPLOY_USER}/webmushra-fetch-helper.sh"
log_verbose "Uploading fetch helper to ${SSH_TARGET}:${remote_helper_path}"
rsync -e "${RSYNC_RSH}" "$remote_helper" "${SSH_TARGET}:${remote_helper_path}"

remote_env="$(printf 'PODMAN_USER=%q PODMAN_DIR=%q STAGING_DIR=%q' "$PODMAN_USER" "$PODMAN_DIR" "$remote_staging_dir")"
remote_run="$(printf '%s bash %q; status=$?; rm -f %q; exit $status' "$remote_env" "$remote_helper_path" "$remote_helper_path")"
log_verbose "Running: ssh -tt ${SSH_OPTS[*]} ${SSH_TARGET} ${remote_run}"
ssh -tt "${SSH_OPTS[@]}" "${SSH_TARGET}" "$remote_run"

# Rsync the staged results to the local destination
rsync_args=(
  -az
  --itemize-changes
  "${RSYNC_PROGRESS_FLAGS[@]}"
  --exclude=.DS_Store
  -e "${RSYNC_RSH}"
)

rsync "${rsync_args[@]}" "${SSH_TARGET}:${remote_staging_dir}/results/" "${dest_dir}/results/"

# Clean up the remote staging directory
ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "rm -rf $(printf '%q' "${remote_staging_dir}")"

log "Results fetched into ${dest_dir}"
open "${dest_dir}"
