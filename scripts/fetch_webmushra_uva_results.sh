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
remote_staging_dir="/home/${DEPLOY_USER}/webmushra_results_staging"
mkdir -p "${dest_dir}"

log "Fetch SSH target: ${SSH_TARGET}"
log "Fetch source: ${PODMAN_USER}:${PODMAN_DIR}/results/"
log "Fetch destination: ${dest_dir}/results/"

mkdir -p "${dest_dir}/results"

# Stage results from the podman user into a directory readable by the deploy user
stage_cmd=(
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}"
  "PODMAN_USER=$(printf '%q' "${PODMAN_USER}") PODMAN_DIR=$(printf '%q' "${PODMAN_DIR}") STAGING_DIR=$(printf '%q' "${remote_staging_dir}") bash -s"
)

"${stage_cmd[@]}" <<'EOF'
set -euo pipefail

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR/results"
chmod 777 "$STAGING_DIR/results"

podman_script="$(mktemp /tmp/webmushra-copy-results.XXXXXX)"
cat > "$podman_script" <<'PODMAN_EOF'
set -euo pipefail

if [[ ! -d "$PODMAN_DIR/results" ]]; then
  echo "Remote results directory not found: $PODMAN_DIR/results" >&2
  exit 1
fi

cp -a "$PODMAN_DIR/results/." "$STAGING_DIR/results/"
chmod -R a+rX "$STAGING_DIR/results"
PODMAN_EOF
chmod a+rx "$podman_script"

podman_cmd="$(printf 'PODMAN_DIR=%q STAGING_DIR=%q bash %q' "$PODMAN_DIR" "$STAGING_DIR" "$podman_script")"
if ! su - "$PODMAN_USER" -c "$podman_cmd"; then
  echo "Failed to switch to ${PODMAN_USER} with su and copy webMUSHRA results." >&2
  rm -f "$podman_script"
  exit 1
fi
rm -f "$podman_script"
EOF

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
