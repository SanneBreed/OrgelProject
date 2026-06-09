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
VERBOSE=0
REMOTE_DIR_EXPLICIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_ROOT="${PROJECT_ROOT}/outputs"

default_remote_dir() {
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

Fetch the remote webMUSHRA results/ directory from the Pi into a new local
timestamped folder under outputs/.

Options:
  --host HOST             SSH host. Default: ${DEFAULT_HOST}
  --user USER             SSH user. Default: ${DEFAULT_USER}
  --remote-dir PATH       Remote webMUSHRA app directory. Default: /home/<user>/webMUSHRA
  --port PORT             SSH port. Default: ${DEFAULT_PORT}
  --identity PATH         SSH private key to use explicitly.
  --verbose               Show more SSH/scp output.
  -h, --help              Show this help text.

Examples:
  $(basename "$0")
  $(basename "$0") --verbose
  $(basename "$0") --identity ~/.ssh/id_rsa
  $(basename "$0") --remote-dir /home/pi/webMUSHRA
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

if ! command -v scp >/dev/null 2>&1; then
  echo "scp is required but not installed." >&2
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
SCP_OPTS=(-P "${PORT}" -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
SSH_LOG_LEVEL="ERROR"

if [[ -n "${IDENTITY_FILE}" ]]; then
  SSH_OPTS+=(-i "${IDENTITY_FILE}" -o IdentitiesOnly=yes)
  SCP_OPTS+=(-i "${IDENTITY_FILE}" -o IdentitiesOnly=yes)
fi

if [[ "${VERBOSE}" -eq 1 ]]; then
  SSH_OPTS+=(-v)
  SCP_OPTS+=(-v)
  SSH_LOG_LEVEL="VERBOSE"
fi

SSH_OPTS+=(-o "LogLevel=${SSH_LOG_LEVEL}")
SCP_OPTS+=(-o "LogLevel=${SSH_LOG_LEVEL}")

timestamp="$(timestamp_now)"
dest_dir="${DEST_ROOT}/webmushra_pi_results_${timestamp}"
mkdir -p "${dest_dir}"

log "Fetch source: ${SSH_TARGET}:${REMOTE_DIR}/results/"
log "Fetch destination: ${dest_dir}/results/"

mkdir -p "${dest_dir}/results"
scp -r "${SCP_OPTS[@]}" "${SSH_TARGET}:${REMOTE_DIR}/results/." "${dest_dir}/results/"

log "Results fetched into ${dest_dir}"
