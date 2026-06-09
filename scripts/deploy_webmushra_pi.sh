#!/usr/bin/env bash

set -euo pipefail

DEFAULT_HOST="pi-media-server.local"
DEFAULT_USER="pi"
DEFAULT_PORT="22"

HOST="${DEFAULT_HOST}"
DEPLOY_USER="${DEFAULT_USER}"
REMOTE_DIR=""
PORT="${DEFAULT_PORT}"
IDENTITY_FILE=""
SYNC_RESULTS=0
SKIP_DEPLOY=0
DRY_RUN=0
VERBOSE=0
REMOTE_DIR_EXPLICIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/src/webMUSHRA"

default_remote_dir() {
  printf '/home/%s/webMUSHRA' "$1"
}

log() {
  printf '%s\n' "$*"
}

log_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}

log_verbose() {
  if [[ "${VERBOSE}" -eq 1 ]]; then
    printf '%s\n' "$*"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Sync src/webMUSHRA to a remote host and redeploy the Docker container there.

Options:
  --host HOST             SSH host. Default: ${DEFAULT_HOST}
  --user USER             SSH user. Default: ${DEFAULT_USER}
  --remote-dir PATH       Remote app directory. Default: /home/<user>/webMUSHRA
  --port PORT             SSH port. Default: ${DEFAULT_PORT}
  --identity PATH         SSH private key to use explicitly.
  --sync-results          Also sync local results/ to the remote host.
  --skip-deploy           Only rsync files. Do not run Docker Compose remotely.
  --dry-run               Show what would happen without changing the remote host.
  --verbose               Show commands and more SSH/rsync output.
  -h, --help              Show this help text.

Examples:
  $(basename "$0")
  $(basename "$0") --remote-dir /home/pi/webMUSHRA
  $(basename "$0") --sync-results
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
      REMOTE_DIR="$2"
      REMOTE_DIR_EXPLICIT=1
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
    --sync-results)
      SYNC_RESULTS=1
      shift
      ;;
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ "${REMOTE_DIR_EXPLICIT}" -eq 0 ]]; then
  REMOTE_DIR="$(default_remote_dir "${DEPLOY_USER}")"
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory not found: ${SOURCE_DIR}" >&2
  exit 1
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

log "Deploy target: ${SSH_TARGET}:${REMOTE_DIR}"
log "Source: ${SOURCE_DIR}"

if [[ "${SYNC_RESULTS}" -eq 0 ]]; then
  log "Remote results/: preserved"
else
  log "Remote results/: synced from local"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "Dry run: enabled"
fi

if [[ "${VERBOSE}" -eq 1 ]]; then
  log "Verbose logging: enabled"
fi

if [[ -n "${IDENTITY_FILE}" ]]; then
  log "SSH identity: ${IDENTITY_FILE}"
fi

mkdir_cmd=(
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}"
  "REMOTE_DIR=$(printf '%q' "${REMOTE_DIR}") bash -s"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log_step "1/3" "[dry-run] Would ensure remote directories exist."
else
  log_step "1/3" "Ensuring remote directories exist..."
  log_verbose "Running: ssh ${SSH_OPTS[*]} ${SSH_TARGET} REMOTE_DIR=$(printf '%q' "${REMOTE_DIR}") bash -s"
  "${mkdir_cmd[@]}" <<'EOF'
set -euo pipefail
mkdir -p "$REMOTE_DIR" "$REMOTE_DIR/configs" "$REMOTE_DIR/results"
EOF
  log "Remote directories ready."
fi

rsync_args=(
  -az
  --delete
  --itemize-changes
  "${RSYNC_PROGRESS_FLAGS[@]}"
  --filter='P .git/***'
  --exclude=.git
  --exclude=.git/
  --exclude=node_modules/
  --exclude=.DS_Store
  -e "${RSYNC_RSH}"
)

if [[ "${SYNC_RESULTS}" -eq 0 ]]; then
  rsync_args+=(--exclude=results/)
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  rsync_args+=(--dry-run)
fi

log_step "2/3" "Syncing files with rsync..."
log_verbose "Running: rsync ${rsync_args[*]} ${SOURCE_DIR}/ ${SSH_TARGET}:${REMOTE_DIR}/"
rsync "${rsync_args[@]}" "${SOURCE_DIR}/" "${SSH_TARGET}:${REMOTE_DIR}/"
log "Sync complete."

if [[ "${SKIP_DEPLOY}" -eq 1 ]]; then
  log_step "3/3" "Skipping remote Docker deployment."
  exit 0
fi

remote_deploy_cmd=(
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}"
  "REMOTE_DIR=$(printf '%q' "${REMOTE_DIR}") DRY_RUN=$(printf '%q' "${DRY_RUN}") bash -s"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log_step "3/3" "[dry-run] Would run docker compose up -d --build --remove-orphans on the remote host."
  exit 0
fi

log_step "3/3" "Deploying container remotely with Docker Compose..."
log_verbose "Running: ssh ${SSH_OPTS[*]} ${SSH_TARGET} REMOTE_DIR=$(printf '%q' "${REMOTE_DIR}") bash -s"
"${remote_deploy_cmd[@]}" <<'EOF'
set -euo pipefail
cd "$REMOTE_DIR"

fix_results_permissions() {
  mkdir -p results

  if command -v sudo >/dev/null 2>&1; then
    if sudo chown -R 33:33 results 2>/dev/null || sudo chown -R www-data:www-data results 2>/dev/null; then
      sudo find results -type d -exec chmod 2775 {} +
      sudo find results -type f -exec chmod 664 {} +
      return 0
    fi
  fi

  echo "Warning: could not chown results to the web server user; falling back to permissive chmod." >&2
  chmod -R a+rwX results
}

fix_results_permissions

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose up -d --build --remove-orphans
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose up -d --build --remove-orphans
elif command -v podman-compose >/dev/null 2>&1; then
  podman-compose up -d --build --remove-orphans
else
  echo "Docker Compose is not available on the remote host." >&2
  exit 1
fi
EOF

log
log "Deployment finished. Expected app URL: http://${HOST}:8080"
