#!/usr/bin/env bash

set -euo pipefail

DEFAULT_HOST="illc-aml-organ.science.uva.nl"
DEFAULT_USER="njaffe"
DEFAULT_PORT="22"
DEFAULT_PODMAN_USER="podman"
WEBMUSHRA_REPO_URL="https://github.com/w4iei/webMUSHRA.git"

HOST="${DEFAULT_HOST}"
DEPLOY_USER="${DEFAULT_USER}"
STAGING_DIR=""
PODMAN_USER="${DEFAULT_PODMAN_USER}"
PODMAN_DIR=""
PORT="${DEFAULT_PORT}"
IDENTITY_FILE=""
SYNC_RESULTS=0
SKIP_WAV=0
SKIP_DEPLOY=0
DRY_RUN=0
VERBOSE=0
STAGING_DIR_EXPLICIT=0
PODMAN_DIR_EXPLICIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/src/webMUSHRA/configs"
RESULTS_SOURCE_DIR="${PROJECT_ROOT}/src/webMUSHRA/results"

default_staging_dir() {
  printf '/home/%s/webmushra' "$1"
}

default_podman_dir() {
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

Sync local src/webMUSHRA/configs to the UvA server and redeploy the Docker container there.

By default this syncs only the configs tree, including local config-side audio assets,
while preserving the remote results/ directory. The podman user creates or updates the
webMUSHRA checkout from ${WEBMUSHRA_REPO_URL}, then overlays the staged configs.

Options:
  --host HOST             SSH host. Default: ${DEFAULT_HOST}
  --user USER             SSH user. Default: ${DEFAULT_USER}
  --staging-dir PATH      Writable SSH-user staging directory. Default: /home/<user>/webmushra
  --remote-dir PATH       Podman-user webMUSHRA repo directory. Default: /home/<podman-user>/webMUSHRA
  --podman-user USER      Account used to run Docker/Podman. Default: ${DEFAULT_PODMAN_USER}
  --port PORT             SSH port. Default: ${DEFAULT_PORT}
  --identity PATH         SSH private key to use explicitly.
  --sync-results          Also sync local results/ to the remote host.
  --skip-wav              Do not sync .wav files in configs/; use existing staged remote .wav files.
  --skip-deploy           Only copy files. Do not run Docker Compose remotely.
  --dry-run               Show what would happen without changing the remote host.
  --verbose               Show commands and more SSH/scp output.
  -h, --help              Show this help text.

Examples:
  $(basename "$0")
  $(basename "$0") --verbose
  $(basename "$0") --identity ~/.ssh/id_rsa
  $(basename "$0") --skip-deploy
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
    --staging-dir)
      STAGING_DIR="$2"
      STAGING_DIR_EXPLICIT=1
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
    --sync-results)
      SYNC_RESULTS=1
      shift
      ;;
    --skip-wav)
      SKIP_WAV=1
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

if [[ "${STAGING_DIR_EXPLICIT}" -eq 0 ]]; then
  STAGING_DIR="$(default_staging_dir "${DEPLOY_USER}")"
fi

if [[ "${PODMAN_DIR_EXPLICIT}" -eq 0 ]]; then
  PODMAN_DIR="$(default_podman_dir "${PODMAN_USER}")"
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

log "Deploy SSH target: ${SSH_TARGET}"
log "Staging target: ${STAGING_DIR}"
log "Podman target: ${PODMAN_USER}:${PODMAN_DIR}"
log "Podman repo: ${WEBMUSHRA_REPO_URL}"
log "Source configs: ${SOURCE_DIR}"

if [[ "${SKIP_WAV}" -eq 1 ]]; then
  log "Config audio: using existing staged remote .wav files"
fi

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
  "STAGING_DIR=$(printf '%q' "${STAGING_DIR}") bash -s"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log_step "1/3" "[dry-run] Would ensure remote directories exist."
else
  log_step "1/3" "Ensuring remote directories exist..."
  log_verbose "Running: ssh ${SSH_OPTS[*]} ${SSH_TARGET} STAGING_DIR=$(printf '%q' "${STAGING_DIR}") bash -s"
  "${mkdir_cmd[@]}" <<'EOF'
set -euo pipefail
mkdir -p "$STAGING_DIR" "$STAGING_DIR/configs" "$STAGING_DIR/results"
EOF
  log "Remote directories ready."
fi

rsync_args=(
  -az
  --delete
  --itemize-changes
  "${RSYNC_PROGRESS_FLAGS[@]}"
  --exclude=.DS_Store
  -e "${RSYNC_RSH}"
)

if [[ "${SKIP_WAV}" -eq 1 ]]; then
  rsync_args+=(--exclude='*.wav' --exclude='*.WAV')
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  rsync_args+=(--dry-run)
fi

if [[ "${SKIP_WAV}" -eq 1 ]]; then
  log_step "2/3" "Syncing configs/ without .wav files with rsync..."
else
  log_step "2/3" "Syncing configs/ with rsync..."
fi
log_verbose "Running: rsync ${rsync_args[*]} ${SOURCE_DIR}/ ${SSH_TARGET}:${STAGING_DIR}/configs/"
rsync "${rsync_args[@]}" "${SOURCE_DIR}/" "${SSH_TARGET}:${STAGING_DIR}/configs/"
log "Configs sync complete."

if [[ "${SYNC_RESULTS}" -eq 1 ]]; then
  if [[ ! -d "${RESULTS_SOURCE_DIR}" ]]; then
    echo "Local results directory not found: ${RESULTS_SOURCE_DIR}" >&2
    exit 1
  fi

  log_step "2/3" "Syncing results/ with rsync..."
  rsync_results_args=(
    -az
    --delete
    --itemize-changes
    "${RSYNC_PROGRESS_FLAGS[@]}"
    --exclude=.DS_Store
    -e "${RSYNC_RSH}"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    rsync_results_args+=(--dry-run)
  fi
  log_verbose "Running: rsync ${rsync_results_args[*]} ${RESULTS_SOURCE_DIR}/ ${SSH_TARGET}:${STAGING_DIR}/results/"
  rsync "${rsync_results_args[@]}" "${RESULTS_SOURCE_DIR}/" "${SSH_TARGET}:${STAGING_DIR}/results/"
  log "Results sync complete."
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  if [[ "${SKIP_DEPLOY}" -eq 1 ]]; then
    log_step "3/3" "[dry-run] Would hand off staged files through /tmp using sudo su - ${PODMAN_USER} and skip Docker Compose."
  else
    log_step "3/3" "[dry-run] Would hand off staged files through /tmp using sudo su - ${PODMAN_USER}, then run docker compose."
  fi
  exit 0
fi

if [[ "${SKIP_DEPLOY}" -eq 1 ]]; then
  log_step "3/3" "Handing staged files to ${PODMAN_USER} without running Docker Compose..."
else
  log_step "3/3" "Handing staged files to ${PODMAN_USER} and deploying remotely..."
fi
remote_helper="$(mktemp "${TMPDIR:-/tmp}/webmushra-uva-remote-deploy.XXXXXX")"
cleanup_remote_helper() {
  rm -f "$remote_helper"
}
trap cleanup_remote_helper EXIT

cat > "$remote_helper" <<'EOF'
set -euo pipefail

handoff_dir="$(mktemp -d /tmp/webmushra-deploy.XXXXXX)"
cleanup() {
  rm -rf "$handoff_dir"
}
trap cleanup EXIT

mkdir -p "$handoff_dir/configs"
cp -a "$STAGING_DIR/configs/." "$handoff_dir/configs/"

if [[ "$SYNC_RESULTS" -eq 1 ]]; then
  mkdir -p "$handoff_dir/results"
  cp -a "$STAGING_DIR/results/." "$handoff_dir/results/"
fi

validate_audio_refs() {
  configs_root="$1"
  app_root="$(dirname "$configs_root")"
  missing_refs="$(mktemp /tmp/webmushra-missing-audio.XXXXXX)"

  find "$configs_root" -maxdepth 1 -type f \( -name 'default.yaml' -o -name 'marcussen_batch_*.yaml' \) -print \
    | while IFS= read -r yaml_file; do
        awk '
          {
            line = $0
            while (match(line, /configs\/resources\/audio\/marcussen_batches\/[^"]+\.wav/)) {
              print substr(line, RSTART, RLENGTH)
              line = substr(line, RSTART + RLENGTH)
            }
          }
        ' "$yaml_file"
      done \
    | sort -u \
    | while IFS= read -r audio_ref; do
        if [[ ! -f "$app_root/$audio_ref" ]]; then
          printf '%s\n' "$audio_ref"
        fi
      done > "$missing_refs"

  if [[ -s "$missing_refs" ]]; then
    echo "Generated configs reference WAV files that are missing from the deploy payload." >&2
    echo "First missing references:" >&2
    head -20 "$missing_refs" >&2
    echo "If you used --skip-wav after regenerating batches, rerun once without --skip-wav." >&2
    rm -f "$missing_refs"
    exit 1
  fi

  rm -f "$missing_refs"
}

validate_audio_refs "$handoff_dir/configs"

chmod -R a+rX "$handoff_dir"

podman_script="$handoff_dir/deploy_as_podman.sh"
cat > "$podman_script" <<'PODMAN_EOF'
set -euo pipefail

prepare_checkout() {
  parent_dir="$(dirname "$PODMAN_DIR")"
  mkdir -p "$parent_dir"

  if [[ -d "$PODMAN_DIR/.git" ]]; then
    cd "$PODMAN_DIR"
    if command -v git >/dev/null 2>&1; then
      current_branch="$(git symbolic-ref --short HEAD 2>/dev/null || printf 'master')"
      git fetch origin
      git reset --hard "origin/${current_branch}"
    else
      echo "git is required for updating the webMUSHRA checkout." >&2
      exit 1
    fi
    return 0
  fi

  previous_dir=""
  if [[ -d "$PODMAN_DIR" ]]; then
    if find "$PODMAN_DIR" -mindepth 1 -maxdepth 1 | grep -q .; then
      previous_dir="${PODMAN_DIR}.preclone.$(date +%Y%m%d%H%M%S)"
      mv "$PODMAN_DIR" "$previous_dir"
    else
      rmdir "$PODMAN_DIR"
    fi
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "git is required for cloning webMUSHRA." >&2
    exit 1
  fi

  git clone "$WEBMUSHRA_REPO_URL" "$PODMAN_DIR"

  if [[ "$SYNC_RESULTS" -eq 0 && -n "$previous_dir" && -d "$previous_dir/results" ]]; then
    rm -rf "$PODMAN_DIR/results"
    cp -a "$previous_dir/results" "$PODMAN_DIR/results"
  fi
}

prepare_checkout
cd "$PODMAN_DIR"

configure_container_port() {
  if [[ ! -f docker-compose.yml ]]; then
    echo "docker-compose.yml not found in $PODMAN_DIR after clone/update." >&2
    exit 1
  fi

  if [[ ! -f Dockerfile ]]; then
    echo "Dockerfile not found in $PODMAN_DIR after clone/update." >&2
    exit 1
  fi

  tmp_compose="$(mktemp /tmp/webmushra-compose.XXXXXX)"
  awk '
    {
      gsub(/"?8080:80"?/, "\"8080:8080\"")
      gsub(/\.\/configs:\/var\/www\/html\/configs(:[A-Za-z,]+)?/, "./configs:/var/www/html/configs:Z")
      gsub(/\.\/results:\/var\/www\/html\/results(:[A-Za-z,]+)?/, "./results:/var/www/html/results:Z")
      print
    }
  ' docker-compose.yml > "$tmp_compose"
  mv "$tmp_compose" docker-compose.yml

  tmp_dockerfile="$(mktemp /tmp/webmushra-dockerfile.XXXXXX)"
  awk '
    /VU_ORGEL_APACHE_8080_BEGIN/ {
      skip = 1
      next
    }
    /VU_ORGEL_APACHE_8080_END/ {
      skip = 0
      next
    }
    skip {
      next
    }
    /^FROM / {
      print
      print "# VU_ORGEL_APACHE_8080_BEGIN"
      print "RUN sed -i '\''s/^Listen 80$/Listen 8080/'\'' /etc/apache2/ports.conf \\"
      print "    && sed -i '\''s/<VirtualHost \\*:80>/<VirtualHost *:8080>/'\'' /etc/apache2/sites-available/000-default.conf"
      print "# VU_ORGEL_APACHE_8080_END"
      next
    }
    { print }
  ' Dockerfile > "$tmp_dockerfile"
  mv "$tmp_dockerfile" Dockerfile
}

configure_container_port

rm -rf configs
cp -a "$HANDOFF_DIR/configs" "$PODMAN_DIR/configs"

if [[ "$SYNC_RESULTS" -eq 1 ]]; then
  rm -rf results
  cp -a "$HANDOFF_DIR/results" "$PODMAN_DIR/results"
else
  mkdir -p results
fi

chmod -R a+rX .
chmod -R a+rwX results

if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
  exit 0
fi

run_compose() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose build --no-cache
    docker compose up -d --force-recreate --remove-orphans
    return 0
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose build --no-cache
    docker-compose up -d --force-recreate --remove-orphans
    return 0
  fi

  if command -v podman-compose >/dev/null 2>&1; then
    podman-compose \
      --podman-build-args="--storage-opt ignore_chown_errors=true" \
      build --no-cache
    podman-compose up -d --force-recreate --remove-orphans
    return 0
  fi

  echo "Docker Compose is not available for the podman user." >&2
  exit 1
}

run_compose
PODMAN_EOF
chmod a+rx "$podman_script"

podman_cmd="$(printf 'WEBMUSHRA_REPO_URL=%q PODMAN_DIR=%q HANDOFF_DIR=%q SYNC_RESULTS=%q SKIP_DEPLOY=%q bash %q' "$WEBMUSHRA_REPO_URL" "$PODMAN_DIR" "$handoff_dir" "$SYNC_RESULTS" "$SKIP_DEPLOY" "$podman_script")"
su_stdin="$handoff_dir/su_podman_stdin.sh"
printf '%s\n' "$podman_cmd" > "$su_stdin"
if ! sudo su - "$PODMAN_USER" < "$su_stdin"; then
  echo "Failed to switch to ${PODMAN_USER} with sudo su and deploy webMUSHRA." >&2
  exit 1
fi
EOF
chmod 700 "$remote_helper"

remote_helper_path="${STAGING_DIR}/$(basename "$remote_helper").sh"
log_verbose "Running: rsync -e '${RSYNC_RSH}' ${remote_helper} ${SSH_TARGET}:${remote_helper_path}"
rsync -e "${RSYNC_RSH}" "$remote_helper" "${SSH_TARGET}:${remote_helper_path}"

remote_deploy_env="$(printf 'STAGING_DIR=%q PODMAN_USER=%q PODMAN_DIR=%q WEBMUSHRA_REPO_URL=%q SYNC_RESULTS=%q SKIP_DEPLOY=%q DRY_RUN=%q' "$STAGING_DIR" "$PODMAN_USER" "$PODMAN_DIR" "$WEBMUSHRA_REPO_URL" "$SYNC_RESULTS" "$SKIP_DEPLOY" "$DRY_RUN")"
remote_deploy_run="$(printf '%s bash %q; status=$?; rm -f %q; exit $status' "$remote_deploy_env" "$remote_helper_path" "$remote_helper_path")"
log_verbose "Running: ssh -tt ${SSH_OPTS[*]} ${SSH_TARGET} ${remote_deploy_run}"
ssh -tt "${SSH_OPTS[@]}" "${SSH_TARGET}" "$remote_deploy_run"

log
if [[ "${SKIP_DEPLOY}" -eq 1 ]]; then
  log "Files copied into ${PODMAN_USER}:${PODMAN_DIR}; Docker Compose was skipped."
else
  log "Deployment finished. Expected app URL: http://${HOST}"
fi
